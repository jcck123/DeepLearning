# GenAI is only used as an auxiliary tool to improve code efficiency and optimize bugs.

import argparse
import bisect
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


DEFAULT_PROCESSED_DIR = Path(r"D:\CHB-MIT-Data\processed")
DEFAULT_SPLIT_FILE = Path(r"D:\CHB-MIT-Data\splits\patient_split_balanced.json")
DEFAULT_OUTPUT_DIR = Path(r"E:\CHB-MIT\runs\cnn_lstm")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


class CHBMITSequenceDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        patient_ids: list[str],
        seq_len: int,
        seq_stride: int,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if seq_stride < 1:
            raise ValueError("seq_stride must be >= 1")

        self.processed_dir = Path(processed_dir)
        self.patient_ids = list(patient_ids)
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.patient_meta = {}
        self.cumulative_lengths = []
        self.samples = 0
        self._window_cache = {}
        self._label_cache = {}

        for patient_id in self.patient_ids:
            patient_dir = self.processed_dir / patient_id
            windows_path = patient_dir / "windows.npy"
            labels_path = patient_dir / "labels.npy"
            meta_path = patient_dir / "meta.json"

            if not windows_path.exists() or not labels_path.exists():
                raise FileNotFoundError(f"Missing windows/labels for {patient_id} in {patient_dir}")

            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                n_windows = int(meta["n_windows"])
            else:
                n_windows = int(np.load(labels_path, mmap_mode="r").shape[0])

            if n_windows >= self.seq_len:
                n_sequences = 1 + (n_windows - self.seq_len) // self.seq_stride
            else:
                n_sequences = 0

            self.patient_meta[patient_id] = {
                "windows_path": windows_path,
                "labels_path": labels_path,
                "n_windows": n_windows,
                "n_sequences": n_sequences,
            }
            self.samples += n_sequences
            self.cumulative_lengths.append(self.samples)

    def __len__(self) -> int:
        return self.samples

    def _windows(self, patient_id: str):
        if patient_id not in self._window_cache:
            self._window_cache[patient_id] = np.load(
                self.patient_meta[patient_id]["windows_path"], mmap_mode="r"
            )
        return self._window_cache[patient_id]

    def _labels(self, patient_id: str):
        if patient_id not in self._label_cache:
            self._label_cache[patient_id] = np.load(
                self.patient_meta[patient_id]["labels_path"], mmap_mode="r"
            )
        return self._label_cache[patient_id]

    def _resolve_index(self, index: int) -> tuple[str, int]:
        patient_pos = bisect.bisect_right(self.cumulative_lengths, index)
        patient_id = self.patient_ids[patient_pos]
        patient_start = 0 if patient_pos == 0 else self.cumulative_lengths[patient_pos - 1]
        local_sequence_index = index - patient_start
        return patient_id, local_sequence_index

    def _end_index(self, patient_id: str, local_sequence_index: int) -> int:
        return (self.seq_len - 1) + local_sequence_index * self.seq_stride

    def all_labels(self) -> np.ndarray:
        labels = []
        for patient_id in self.patient_ids:
            n_windows = self.patient_meta[patient_id]["n_windows"]
            if n_windows < self.seq_len:
                continue
            end_indices = np.arange(self.seq_len - 1, n_windows, self.seq_stride, dtype=np.int64)
            labels.append(np.asarray(self._labels(patient_id)[end_indices], dtype=np.int64))
        if not labels:
            return np.array([], dtype=np.int64)
        return np.concatenate(labels, axis=0)

    def __getitem__(self, index: int):
        patient_id, local_sequence_index = self._resolve_index(index)
        end_index = self._end_index(patient_id, local_sequence_index)
        start_index = end_index - self.seq_len + 1

        sequence = np.array(
            self._windows(patient_id)[start_index : end_index + 1],
            dtype=np.float32,
            copy=True,
        )
        label = float(self._labels(patient_id)[end_index])

        return (
            torch.from_numpy(sequence),
            torch.tensor(label, dtype=torch.float32),
            patient_id,
            int(end_index),
        )


class WindowCNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 18, embedding_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.proj(x)


class CNNLSTMSeizurePredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = WindowCNNEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, samples = x.shape
        x = x.reshape(batch_size * seq_len, channels, samples)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(batch_size, seq_len, -1)
        outputs, _ = self.lstm(embeddings)
        logits = self.classifier(outputs[:, -1, :])
        return logits.squeeze(1)


def safe_metric(fn, *args, default=float("nan"), **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return default


def format_seconds(seconds: float) -> str:
    if seconds != seconds or seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def gpu_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.memory_allocated(device) / (1024 ** 3)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    return to_builtin(
        {
            "auroc": safe_metric(roc_auc_score, labels, probs),
            "auprc": safe_metric(average_precision_score, labels, probs),
            "brier": safe_metric(brier_score_loss, labels, probs),
            "precision": safe_metric(precision_score, labels, preds, zero_division=0),
            "recall": safe_metric(recall_score, labels, preds, zero_division=0),
            "f1": safe_metric(f1_score, labels, preds, zero_division=0),
            "threshold": threshold,
            "positive_rate": float(np.mean(labels == 1)),
            "predicted_positive_rate": float(np.mean(preds == 1)),
        }
    )


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.round(probs, 4))
    if len(candidates) == 0:
        return 0.5, float("nan")

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        preds = (probs >= threshold).astype(np.int64)
        score = safe_metric(f1_score, labels, preds, zero_division=0, default=-1.0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    stage_name: str,
    epoch: int,
    total_epochs: int,
    current_lr: float | None = None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_probs = []
    all_logits = []
    all_patients = []
    all_window_indices = []
    running_positive = 0.0

    autocast_enabled = device.type == "cuda"
    total_steps = len(loader)
    if max_batches is not None:
        total_steps = min(total_steps, max_batches)

    progress = tqdm(
        total=total_steps,
        desc=f"{stage_name} {epoch:02d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (inputs, labels, patient_ids, window_indices) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if train_mode:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        logits_np = logits.detach().cpu().numpy()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        running_positive += float(labels_np.mean()) * batch_size
        all_probs.append(probs)
        all_logits.append(logits_np)
        all_labels.append(labels_np)
        all_patients.extend(patient_ids)
        all_window_indices.extend(int(idx) for idx in window_indices)

        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg": f"{(total_loss / max(1, total_examples)):.4f}",
            "pos": f"{(running_positive / max(1, total_examples)):.3f}",
        }
        if current_lr is not None:
            postfix["lr"] = f"{current_lr:.2e}"
        mem_gb = gpu_memory_gb(device)
        if mem_gb is not None:
            postfix["gpu_gb"] = f"{mem_gb:.2f}"

        progress.set_postfix(postfix)
        progress.update(1)

    progress.close()

    labels_np = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    probs_np = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    logits_np = np.concatenate(all_logits) if all_logits else np.array([], dtype=np.float32)
    avg_loss = total_loss / max(1, total_examples)

    return {
        "loss": avg_loss,
        "labels": labels_np.astype(np.int64),
        "probs": probs_np.astype(np.float64),
        "logits": logits_np.astype(np.float64),
        "patients": all_patients,
        "window_indices": all_window_indices,
    }


def save_predictions(
    path: Path,
    patient_ids: list[str],
    window_indices: list[int],
    labels: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    threshold: float,
):
    preds = (probs >= threshold).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["patient_id", "window_end_index", "label", "logit", "probability", "prediction"]
        )
        for row in zip(
            patient_ids,
            window_indices,
            labels.tolist(),
            logits.tolist(),
            probs.tolist(),
            preds.tolist(),
        ):
            writer.writerow(row)


def build_loaders(
    processed_dir: Path,
    split_file: Path,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    seq_len: int,
    seq_stride: int,
):
    split_payload = json.loads(split_file.read_text())
    split = split_payload["split"]

    datasets = {
        split_name: CHBMITSequenceDataset(processed_dir, split[split_name], seq_len, seq_stride)
        for split_name in ("train", "val", "test")
    }

    train_labels = datasets["train"].all_labels()
    if train_labels.size == 0:
        raise RuntimeError("No train sequences were produced. Try lowering --seq-len.")

    train_sampler = None
    train_shuffle = True
    if balanced_sampling:
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = np.zeros(2, dtype=np.float64)
        for cls, count in enumerate(class_counts):
            class_weights[cls] = 1.0 / max(1, count)
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )
        train_shuffle = False

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
    }

    split_stats = {}
    for split_name, dataset in datasets.items():
        labels = dataset.all_labels()
        split_stats[split_name] = {
            "n_sequences": int(len(dataset)),
            "n_preictal": int(labels.sum()),
            "n_interictal": int((labels == 0).sum()),
            "positive_ratio": float(labels.mean()) if labels.size > 0 else 0.0,
        }

    train_pos = split_stats["train"]["n_preictal"]
    train_neg = split_stats["train"]["n_interictal"]
    pos_weight = train_neg / max(1, train_pos)

    return loaders, split_payload, split_stats, pos_weight


def maybe_init_wandb(args, split_stats: dict):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it with `pip install wandb`.")

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=tags or None,
        notes=args.wandb_notes or None,
        mode=args.wandb_mode,
        config=to_builtin(
            {
                **vars(args),
                "model_name": "cnn_lstm",
                "split_stats": split_stats,
            }
        ),
    )
    wandb.define_metric("epoch")
    for prefix in ("train/*", "val/*", "best/*", "final_test/*"):
        wandb.define_metric(prefix, step_metric="epoch")
    return run


def wandb_metric_group(prefix: str, loss: float, metrics: dict) -> dict:
    return {
        f"{prefix}/loss": loss,
        f"{prefix}/auprc": metrics["auprc"],
        f"{prefix}/auroc": metrics["auroc"],
        f"{prefix}/f1": metrics["f1"],
        f"{prefix}/precision": metrics["precision"],
        f"{prefix}/recall": metrics["recall"],
        f"{prefix}/brier": metrics["brier"],
        f"{prefix}/threshold": metrics["threshold"],
        f"{prefix}/positive_rate": metrics["positive_rate"],
        f"{prefix}/predicted_positive_rate": metrics["predicted_positive_rate"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN+LSTM on processed CHB-MIT windows")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="CHB-MIT")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="cnn-lstm,chb-mit")
    parser.add_argument("--wandb-notes", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-artifact", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaders, split_payload, split_stats, pos_weight = build_loaders(
        args.processed_dir,
        args.split_file,
        args.batch_size,
        args.num_workers,
        args.balanced_sampling,
        args.seq_len,
        args.seq_stride,
    )

    run = maybe_init_wandb(args, split_stats)

    model = CNNLSTMSeizurePredictor(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    serialized_args = to_builtin(vars(args))

    criterion = (
        nn.BCEWithLogitsLoss()
        if args.balanced_sampling
        else nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_state = None
    best_val_auprc = -1.0
    best_threshold = 0.5
    best_epoch = -1
    history = []
    stale_epochs = 0

    train_patients = split_payload["split"]["train"]
    val_patients = split_payload["split"]["val"]
    test_patients = split_payload["split"]["test"]

    print(f"Device: {device}")
    print(f"Train patients: {train_patients}")
    print(f"Val patients:   {val_patients}")
    print(f"Test patients:  {test_patients}")
    print(
        "Sequences -> "
        f"train: {split_stats['train']['n_sequences']}, "
        f"val: {split_stats['val']['n_sequences']}, "
        f"test: {split_stats['test']['n_sequences']}"
    )
    print(
        "Positive ratio -> "
        f"train: {split_stats['train']['positive_ratio']:.4f}, "
        f"val: {split_stats['val']['positive_ratio']:.4f}, "
        f"test: {split_stats['test']['positive_ratio']:.4f}"
    )
    print(f"Sequence length: {args.seq_len} windows")
    print(f"Train pos_weight: {pos_weight:.3f}")
    print(f"Balanced sampling: {args.balanced_sampling}")
    if run is not None:
        run_url = getattr(run, "url", None)
        if run_url:
            print(f"W&B run: {run_url}")
        else:
            print(f"W&B mode: {args.wandb_mode}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        if epoch > 1 and history:
            avg_epoch_sec = sum(item["elapsed_sec"] for item in history) / len(history)
            remaining_sec = avg_epoch_sec * (args.epochs - epoch + 1)
            print(
                f"\nEpoch {epoch:02d}/{args.epochs} | "
                f"lr={epoch_lr:.2e} | est. remaining {format_seconds(remaining_sec)}"
            )
        else:
            print(f"\nEpoch {epoch:02d}/{args.epochs} | lr={epoch_lr:.2e}")

        train_epoch = run_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            scaler,
            stage_name="Train",
            epoch=epoch,
            total_epochs=args.epochs,
            current_lr=epoch_lr,
            max_batches=args.max_train_batches,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_epoch = run_epoch(
            model,
            loaders["val"],
            None,
            criterion,
            device,
            None,
            stage_name="Val",
            epoch=epoch,
            total_epochs=args.epochs,
            max_batches=args.max_val_batches,
        )

        current_threshold, val_best_f1 = find_best_threshold(val_epoch["labels"], val_epoch["probs"])
        train_metrics = classification_metrics(train_epoch["labels"], train_epoch["probs"], current_threshold)
        val_metrics = classification_metrics(val_epoch["labels"], val_epoch["probs"], current_threshold)
        val_metrics["best_f1_on_val"] = val_best_f1

        epoch_record = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - start_time, 2),
            "train_loss": train_epoch["loss"],
            "val_loss": val_epoch["loss"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        scheduler.step(val_metrics["auprc"])

        elapsed_sec = epoch_record["elapsed_sec"]
        avg_epoch_sec = sum(item["elapsed_sec"] for item in history) / len(history)
        remaining_sec = avg_epoch_sec * max(0, args.epochs - epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_epoch['loss']:.4f} val_loss={val_epoch['loss']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f} val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} thr={current_threshold:.4f} | "
            f"time={format_seconds(elapsed_sec)} eta={format_seconds(remaining_sec)}"
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_threshold = current_threshold
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_epoch["loss"],
                "val_metrics": val_metrics,
                "threshold": best_threshold,
                "args": serialized_args,
            }
            torch.save(best_state, args.out_dir / "best_model.pt")
        else:
            stale_epochs += 1

        if run is not None:
            best_metrics = {
                "auprc": best_state["val_metrics"]["auprc"],
                "auroc": best_state["val_metrics"]["auroc"],
                "f1": best_state["val_metrics"]["f1"],
                "precision": best_state["val_metrics"]["precision"],
                "recall": best_state["val_metrics"]["recall"],
                "brier": best_state["val_metrics"]["brier"],
                "threshold": best_threshold,
                "positive_rate": best_state["val_metrics"]["positive_rate"],
                "predicted_positive_rate": best_state["val_metrics"]["predicted_positive_rate"],
            }
            log_payload = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "runtime/epoch_sec": elapsed_sec,
                "runtime/eta_sec": remaining_sec,
                "best/epoch": best_epoch,
            }
            log_payload.update(wandb_metric_group("train", train_epoch["loss"], train_metrics))
            log_payload.update(wandb_metric_group("val", val_epoch["loss"], val_metrics))
            log_payload.update(
                wandb_metric_group(
                    "best",
                    best_state["val_loss"],
                    best_metrics,
                )
            )
            run.log(log_payload, step=epoch)

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    checkpoint = torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_final = run_epoch(
        model,
        loaders["val"],
        None,
        criterion,
        device,
        None,
        stage_name="Val",
        epoch=best_epoch,
        total_epochs=args.epochs,
        max_batches=args.max_val_batches,
    )
    test_final = run_epoch(
        model,
        loaders["test"],
        None,
        criterion,
        device,
        None,
        stage_name="Test",
        epoch=best_epoch,
        total_epochs=args.epochs,
        max_batches=args.max_test_batches,
    )

    final_val_metrics = classification_metrics(val_final["labels"], val_final["probs"], best_threshold)
    final_test_metrics = classification_metrics(test_final["labels"], test_final["probs"], best_threshold)

    summary = {
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_val_auprc": best_val_auprc,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "history": history,
        "args": serialized_args,
        "split_stats": split_stats,
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(to_builtin(summary), indent=2))
    save_predictions(
        args.out_dir / "val_predictions.csv",
        val_final["patients"],
        val_final["window_indices"],
        val_final["labels"],
        val_final["logits"],
        val_final["probs"],
        best_threshold,
    )
    save_predictions(
        args.out_dir / "test_predictions.csv",
        test_final["patients"],
        test_final["window_indices"],
        test_final["labels"],
        test_final["logits"],
        test_final["probs"],
        best_threshold,
    )

    print("\nBest checkpoint summary")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Val AUPRC:      {final_val_metrics['auprc']:.4f}")
    print(f"  Val AUROC:      {final_val_metrics['auroc']:.4f}")
    print(f"  Test AUPRC:     {final_test_metrics['auprc']:.4f}")
    print(f"  Test AUROC:     {final_test_metrics['auroc']:.4f}")
    print(f"  Test F1:        {final_test_metrics['f1']:.4f}")
    print(f"  Test Recall:    {final_test_metrics['recall']:.4f}")
    print(f"  Test Precision: {final_test_metrics['precision']:.4f}")
    print(f"\nArtifacts saved to {args.out_dir}")

    if run is not None:
        final_log_step = history[-1]["epoch"] if history else best_epoch
        run.log(
            {
                "epoch": final_log_step,
                "final_test/auprc": final_test_metrics["auprc"],
                "final_test/auroc": final_test_metrics["auroc"],
                "final_test/f1": final_test_metrics["f1"],
                "final_test/precision": final_test_metrics["precision"],
                "final_test/recall": final_test_metrics["recall"],
                "final_test/brier": final_test_metrics["brier"],
                "final_test/threshold": best_threshold,
                "final_test/positive_rate": final_test_metrics["positive_rate"],
                "final_test/predicted_positive_rate": final_test_metrics["predicted_positive_rate"],
            },
            step=final_log_step,
        )
        run.summary["best_epoch"] = best_epoch
        run.summary["best_threshold"] = best_threshold
        run.summary["final_val_auprc"] = final_val_metrics["auprc"]
        run.summary["final_val_auroc"] = final_val_metrics["auroc"]
        run.summary["final_test_auprc"] = final_test_metrics["auprc"]
        run.summary["final_test_auroc"] = final_test_metrics["auroc"]
        run.summary["final_test_f1"] = final_test_metrics["f1"]
        if args.wandb_log_artifact:
            artifact = wandb.Artifact(f"{run.name or 'cnn-lstm'}-artifacts", type="model")
            artifact.add_file(str(args.out_dir / "best_model.pt"))
            artifact.add_file(str(metrics_path))
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
