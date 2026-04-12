#!/usr/bin/env python3
"""
Train seizure prediction models (Sequence-of-Windows approach).

Architecture:
  1. Each 30s window is processed by CNN to extract a feature vector
  2. Feature vectors from N consecutive windows form a sequence fed into LSTM
  3. LSTM captures temporal trends across windows
  4. The output of the last time step is used for prediction

Example: seq_len=10 (10 consecutive 30s windows = 5 minutes):
  Window sequence: [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]
                                                          ↓
  CNN features:    [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
                                                          ↓
  LSTM:            Observes trends over first 9 windows → predicts label for window 10

Usage:
  # Quick test
  python train.py \
    --processed-dir /Volumes/T9/data/processed \
    --split-file /Volumes/T9/data/splits/patient_split.json \
    --save-dir /Volumes/T9/models \
    --epochs 2 --batch-size 32

  # Full training
  python train.py \
    --processed-dir /Volumes/T9/data/processed \
    --split-file /Volumes/T9/data/splits/patient_split.json \
    --save-dir /Volumes/T9/models
"""
# GenAI is only used as an auxiliary tool to improve code efficiency and optimize bugs.


import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── Dataset: Sequence of Windows ────────────────────────────────────

class SeqEEGDataset(Dataset):
    """
    Builds sequence samples from consecutive windows.

    Each sample = seq_len consecutive windows, label = label of the last window.
    Windows within one patient are ordered chronologically; no cross-patient splicing.
    """

    def __init__(self, patient_ids, processed_dir, seq_len=10):
        self.seq_len = seq_len
        self.samples = []  # list of (patient_idx, start_idx)

        self.all_windows = []  # per-patient windows mmap
        self.all_labels = []

        for pid in patient_ids:
            w_path = Path(processed_dir) / pid / "windows.npy"
            l_path = Path(processed_dir) / pid / "labels.npy"
            if not w_path.exists():
                print(f"  Warning: {pid} not found, skipping")
                continue

            w = np.load(w_path, mmap_mode='r')
            l = np.load(l_path, mmap_mode='r')
            p_idx = len(self.all_windows)
            self.all_windows.append(w)
            self.all_labels.append(l)

            # Generate all possible consecutive sequences from this patient's windows
            n = len(l)
            for start in range(n - seq_len + 1):
                self.samples.append((p_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_idx, start = self.samples[idx]
        end = start + self.seq_len

        # (seq_len, 18, 7680)
        x = torch.from_numpy(
            self.all_windows[p_idx][start:end].copy()
        )
        # Label = label of the last window
        y = torch.tensor(
            int(self.all_labels[p_idx][end - 1]), dtype=torch.long
        )
        return x, y

    def get_label_counts(self):
        """Count label distribution."""
        labels = [int(self.all_labels[p][s + self.seq_len - 1])
                  for p, s in self.samples]
        labels = np.array(labels)
        return int(np.sum(labels == 1)), int(np.sum(labels == 0))

    def get_sample_weights(self):
        """Compute sample weights for balanced sampling."""
        labels = np.array([int(self.all_labels[p][s + self.seq_len - 1])
                           for p, s in self.samples])
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        w_pos = len(labels) / (2 * max(1, n_pos))
        w_neg = len(labels) / (2 * max(1, n_neg))
        weights = np.where(labels == 1, w_pos, w_neg)
        return torch.from_numpy(weights).float()


# ── Models ──────────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """CNN processes a single window (18, 7680) -> feature vector (feature_dim,)"""

    def __init__(self, in_channels=18, filters=(32, 64, 128)):
        super().__init__()
        layers = []
        ch = in_channels
        for f in filters:
            layers.extend([
                nn.Conv1d(ch, f, kernel_size=5, padding=2),
                nn.BatchNorm1d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),  # Aggressive downsampling: 7680->1920->480->120
            ])
            ch = f
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # (128, 120) → (128, 1)

    def forward(self, x):
        # x: (B, 18, 7680)
        f = self.cnn(x)      # (B, 128, 120)
        f = self.pool(f)      # (B, 128, 1)
        f = f.squeeze(-1)     # (B, 128)
        return f


class SeqBaselineCNNLSTM(nn.Module):
    """
    Sequence model (Baseline): CNN extracts per-window features -> LSTM models window sequence -> classification.
    Output: logit (B, 1)
    """

    def __init__(self, in_channels=18, cnn_filters=(32, 64, 128),
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, cnn_filters)
        feature_dim = cnn_filters[-1]

        self.lstm = nn.LSTM(feature_dim, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, 18, 7680)
        B, S, C, T = x.shape

        # CNN: merge batch and seq_len dimensions
        x_flat = x.view(B * S, C, T)          # (B*S, 18, 7680)
        features = self.cnn(x_flat)             # (B*S, 128)
        features = features.view(B, S, -1)      # (B, S, 128)

        # LSTM: model the window sequence
        lstm_out, _ = self.lstm(features)        # (B, S, H)

        # Predict using the last time step
        logit = self.head(lstm_out[:, -1, :])    # (B, 1)
        return logit


class SeqProbabilisticCNNLSTM(nn.Module):
    """
    Sequence model (Probabilistic): same as above + dual-head output for mu and log_var.
    Output: mu (B, 1), log_var (B, 1)
    """

    def __init__(self, in_channels=18, cnn_filters=(32, 64, 128),
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, cnn_filters)
        feature_dim = cnn_filters[-1]

        self.lstm = nn.LSTM(feature_dim, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)

        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))
        self.logvar_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        B, S, C, T = x.shape
        x_flat = x.view(B * S, C, T)
        features = self.cnn(x_flat)
        features = features.view(B, S, -1)
        lstm_out, _ = self.lstm(features)
        h = self.shared(lstm_out[:, -1, :])
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var


# ── Loss ────────────────────────────────────────────────────────────

class HeteroscedasticBCELoss(nn.Module):
    """Sample logits from predicted Gaussian -> BCE, jointly learning probability and uncertainty."""
    def __init__(self, n_samples=10):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, mu, log_var, target):
        target = target.float().unsqueeze(-1)
        log_var = torch.clamp(log_var, -10, 10)
        sigma = torch.exp(0.5 * log_var)

        eps = torch.randn(self.n_samples, *mu.shape, device=mu.device)
        sampled = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

        bce = nn.functional.binary_cross_entropy_with_logits(
            sampled, target.unsqueeze(0).expand_as(sampled), reduction='none')
        return bce.mean()


# ── Training ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, is_prob):
    model.train()
    total_loss = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if is_prob:
            mu, log_var = model(x)
            loss = criterion(mu, log_var, y)
        else:
            logit = model(x)
            loss = criterion(logit.squeeze(-1), y.float())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)
        n += len(x)

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_prob):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if is_prob:
            mu, log_var = model(x)
            loss = criterion(mu, log_var, y)
            probs = torch.sigmoid(mu).squeeze(-1)
        else:
            logit = model(x)
            loss = criterion(logit.squeeze(-1), y.float())
            probs = torch.sigmoid(logit).squeeze(-1)

        total_loss += loss.item() * len(x)
        n += len(x)
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.0

    return total_loss / max(1, n), auroc


def train_model(model, train_loader, val_loader, cfg, device, is_prob, save_dir, name):
    save_path = Path(save_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)

    criterion = HeteroscedasticBCELoss() if is_prob else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'lr': []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} | {n_params:,} params | device={device}")
    print(f"  seq_len={cfg['seq_len']} windows = {cfg['seq_len']*30}s context")
    print(f"{'=' * 60}")

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device, is_prob)
        val_loss, val_auroc = evaluate(model, val_loader, criterion,
                                        device, is_prob)

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['lr'].append(lr)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
            marker = " *"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"auroc={val_auroc:.4f} lr={lr:.6f} "
              f"({elapsed:.0f}s){marker}")

        if patience_counter >= cfg['patience']:
            print(f"  Early stopping (no improvement for {cfg['patience']} epochs)")
            break

    torch.save(model.state_dict(), save_path / "final_model.pt")

    # Save config and history
    history['config'] = cfg
    with open(save_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  -> {save_path}")
    return history


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train seizure prediction (sequence model)")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length: number of consecutive windows per sample (default: 10 = 5min)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--only", choices=["baseline", "probabilistic"], default=None)
    args = parser.parse_args()

    cfg = {
        'seq_len': args.seq_len,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
    }

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load split
    with open(args.split_file) as f:
        split = json.load(f)['split']

    print(f"Train: {split['train']}")
    print(f"Val:   {split['val']}")
    print(f"Test:  {split['test']}")
    print(f"Seq length: {cfg['seq_len']} windows = {cfg['seq_len']*30}s")

    # Load data
    print("\nLoading data...")
    train_ds = SeqEEGDataset(split['train'], args.processed_dir, seq_len=cfg['seq_len'])
    val_ds = SeqEEGDataset(split['val'], args.processed_dir, seq_len=cfg['seq_len'])

    n_pos_train, n_neg_train = train_ds.get_label_counts()
    n_pos_val, n_neg_val = val_ds.get_label_counts()

    print(f"  Train: {len(train_ds)} sequences "
          f"(preictal={n_pos_train}, interictal={n_neg_train})")
    print(f"  Val:   {len(val_ds)} sequences "
          f"(preictal={n_pos_val}, interictal={n_neg_val})")

    # Weighted sampling
    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=False)

    torch.manual_seed(42)
    np.random.seed(42)

    # Train Baseline
    if args.only is None or args.only == "baseline":
        model = SeqBaselineCNNLSTM().to(device)
        train_model(model, train_loader, val_loader, cfg, device,
                    is_prob=False, save_dir=args.save_dir, name="baseline")

    # Train Probabilistic
    if args.only is None or args.only == "probabilistic":
        torch.manual_seed(42)
        model = SeqProbabilisticCNNLSTM().to(device)
        train_model(model, train_loader, val_loader, cfg, device,
                    is_prob=True, save_dir=args.save_dir, name="probabilistic")

    print("\nDone.")


if __name__ == "__main__":
    main()