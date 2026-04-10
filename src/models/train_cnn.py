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


DEFAULT_PROCESSED_DIR = Path(r"D:\CHB-MIT-Data\processed")
DEFAULT_SPLIT_FILE = Path(r"D:\CHB-MIT-Data\splits\patient_split.json")
DEFAULT_OUTPUT_DIR = Path(r"E:\CHB-MIT\runs\cnn1d")


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


class CHBMITWindowDataset(Dataset):
    def __init__(self, processed_dir: Path, patient_ids: list[str]):
        self.processed_dir = Path(processed_dir)
        self.patient_ids = list(patient_ids)
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

            self.patient_meta[patient_id] = {
                "windows_path": windows_path,
                "labels_path": labels_path,
                "n_windows": n_windows,
            }
            self.samples += n_windows
            self.cumulative_lengths.append(self.samples)

    def __len__(self) -> int:
        return self.samples

    def all_labels(self) -> np.ndarray:
        labels = []
        for patient_id in self.patient_ids:
            labels.append(np.array(self._labels(patient_id), dtype=np.int64, copy=False))
        return np.concatenate(labels, axis=0)

    def _resolve_index(self, index: int) -> tuple[str, int]:
        patient_pos = bisect.bisect_right(self.cumulative_lengths, index)
        patient_id = self.patient_ids[patient_pos]
        patient_start = 0 if patient_pos == 0 else self.cumulative_lengths[patient_pos - 1]
        local_index = index - patient_start
        return patient_id, local_index

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

    def __getitem__(self, index: int):
        patient_id, local_index = self._resolve_index(index)
        window = np.array(self._windows(patient_id)[local_index], dtype=np.float32, copy=True)
        label = float(self._labels(patient_id)[local_index])
        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.float32),
            patient_id,
            int(local_index),
        )


class CNN1DSeizurePredictor(nn.Module):
    def __init__(self, in_channels: int = 18, dropout: float = 0.2):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


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
    metrics = {
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
    return to_builtin(metrics)


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
    scaler: torch.cuda.amp.GradScaler | None,
    stage_name: str,
    epoch: int,
    total_epochs: int,
    current_lr: float | None = None,
    max_batches: int | None = None,
):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_probs = []
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
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        running_positive += float(labels_np.mean()) * batch_size
        all_probs.append(probs)
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
    avg_loss = total_loss / max(1, total_examples)

    return {
        "loss": avg_loss,
        "labels": labels_np.astype(np.int64),
        "probs": probs_np.astype(np.float64),
        "patients": all_patients,
        "window_indices": all_window_indices,
    }


def save_predictions(path: Path, patient_ids: list[str], window_indices: list[int], labels, probs, threshold: float):
    preds = (probs >= threshold).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "window_index", "label", "probability", "prediction"])
        for row in zip(patient_ids, window_indices, labels.tolist(), probs.tolist(), preds.tolist()):
            writer.writerow(row)


def build_loaders(
    processed_dir: Path,
    split_file: Path,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
):
    split_payload = json.loads(split_file.read_text())
    split = split_payload["split"]
    patient_stats = split_payload["patient_stats"]

    datasets = {
        split_name: CHBMITWindowDataset(processed_dir, split[split_name])
        for split_name in ("train", "val", "test")
    }

    train_sampler = None
    train_shuffle = True
    if balanced_sampling:
        train_labels = datasets["train"].all_labels()
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

    train_pos = sum(patient_stats[pid]["n_preictal"] for pid in split["train"])
    train_neg = sum(patient_stats[pid]["n_interictal"] for pid in split["train"])
    pos_weight = train_neg / max(1, train_pos)

    return loaders, split_payload, pos_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D CNN on processed CHB-MIT windows")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaders, split_payload, pos_weight = build_loaders(
        args.processed_dir,
        args.split_file,
        args.batch_size,
        args.num_workers,
        args.balanced_sampling,
    )

    model = CNN1DSeizurePredictor(dropout=args.dropout).to(device)
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

    train_stats = split_payload["patient_stats"]
    train_patients = split_payload["split"]["train"]
    val_patients = split_payload["split"]["val"]
    test_patients = split_payload["split"]["test"]
    train_windows = sum(train_stats[pid]["n_windows"] for pid in train_patients)
    val_windows = sum(train_stats[pid]["n_windows"] for pid in val_patients)
    test_windows = sum(train_stats[pid]["n_windows"] for pid in test_patients)

    print(f"Device: {device}")
    print(f"Train patients: {train_patients}")
    print(f"Val patients:   {val_patients}")
    print(f"Test patients:  {test_patients}")
    print(f"Windows -> train: {train_windows}, val: {val_windows}, test: {test_windows}")
    print(f"Train pos_weight: {pos_weight:.3f}")
    print(f"Balanced sampling: {args.balanced_sampling}")

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
                "val_metrics": val_metrics,
                "threshold": best_threshold,
                "args": serialized_args,
            }
            torch.save(best_state, args.out_dir / "best_model.pt")
        else:
            stale_epochs += 1

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
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(to_builtin(summary), indent=2))
    save_predictions(
        args.out_dir / "val_predictions.csv",
        val_final["patients"],
        val_final["window_indices"],
        val_final["labels"],
        val_final["probs"],
        best_threshold,
    )
    save_predictions(
        args.out_dir / "test_predictions.csv",
        test_final["patients"],
        test_final["window_indices"],
        test_final["labels"],
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


if __name__ == "__main__":
    main()
