#!/usr/bin/env python3
"""
Evaluate trained seizure prediction models on the held-out test set.

Loads saved model weights, produces final metrics (AUROC, AUPRC, F1,
Brier, ECE), runs MC Dropout uncertainty estimation, applies probability
calibration, and generates visual results.

Usage:
  python test.py \
    --model-path saved_models/best_model.pt \
    --processed-dir /path/to/processed \
    --split-file /path/to/patient_split_balanced.json \
    --out-dir results/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Import shared components from training code
# ---------------------------------------------------------------------------
from train import (
    CHBMITSequenceDataset,
    CNNLSTMSeizurePredictor,
    find_best_threshold,
    safe_metric,
    set_seed,
    to_builtin,
)


# ── Helpers ────────────────────────────────────────────────────────────


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0 - eps)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def classification_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
    return to_builtin({
        "auroc": safe_metric(roc_auc_score, labels, probs),
        "auprc": safe_metric(average_precision_score, labels, probs),
        "brier": safe_metric(brier_score_loss, labels, probs),
        "precision": safe_metric(precision_score, labels, preds, zero_division=0),
        "recall": safe_metric(recall_score, labels, preds, zero_division=0),
        "f1": safe_metric(f1_score, labels, preds, zero_division=0),
        "threshold": threshold,
    })


def expected_calibration_error(labels, probs, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(ece)


# ── Inference ──────────────────────────────────────────────────────────


def deterministic_predict(model, loader, device):
    model.eval()
    all_labels, all_logits, all_probs = [], [], []
    all_patients, all_window_indices = [], []

    with torch.no_grad():
        for inputs, labels, patient_ids, window_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_patients.extend(patient_ids)
            all_window_indices.extend(int(i) for i in window_indices)

    return {
        "labels": np.concatenate(all_labels).astype(np.int64),
        "logits": np.concatenate(all_logits).astype(np.float64),
        "probs": np.concatenate(all_probs).astype(np.float64),
        "patients": all_patients,
        "window_indices": all_window_indices,
    }


def mc_dropout_predict(model, loader, device, mc_samples=20):
    from torch import nn
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()

    all_labels, all_patients, all_window_indices = [], [], []
    all_mean_probs, all_std_probs = [], []
    all_pred_entropy, all_mutual_info = [], []

    with torch.no_grad():
        for inputs, labels, patient_ids, window_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits_samples = []
            for _ in range(mc_samples):
                logits = model(inputs)
                logits_samples.append(logits.cpu().numpy().astype(np.float64))

            logits_stack = np.stack(logits_samples, axis=0)
            probs_stack = 1.0 / (1.0 + np.exp(-logits_stack))

            mean_probs = probs_stack.mean(axis=0)
            std_probs = probs_stack.std(axis=0)
            pred_entropy = binary_entropy(mean_probs)
            expected_entropy = binary_entropy(probs_stack).mean(axis=0)
            mutual_info = pred_entropy - expected_entropy

            all_labels.append(labels.numpy())
            all_patients.extend(patient_ids)
            all_window_indices.extend(int(i) for i in window_indices)
            all_mean_probs.append(mean_probs)
            all_std_probs.append(std_probs)
            all_pred_entropy.append(pred_entropy)
            all_mutual_info.append(mutual_info)

    model.eval()
    return {
        "labels": np.concatenate(all_labels).astype(np.int64),
        "mean_probs": np.concatenate(all_mean_probs),
        "std_probs": np.concatenate(all_std_probs),
        "predictive_entropy": np.concatenate(all_pred_entropy),
        "mutual_information": np.concatenate(all_mutual_info),
        "patients": all_patients,
        "window_indices": all_window_indices,
    }


# ── Calibration ────────────────────────────────────────────────────────


def fit_temperature(val_logits, val_labels):
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.float32)
    log_T = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T) + 1e-6
        loss = criterion(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).detach().item())


def calibrate_probs(logits, temperature):
    return 1.0 / (1.0 + np.exp(-(logits / temperature)))


# ── Visualization ──────────────────────────────────────────────────────


def save_roc_pr_curves(out_dir, labels, probs, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    ax1.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve{title_suffix}")
    ax1.legend()

    prec, rec, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    ax2.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    baseline = labels.mean()
    ax2.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve{title_suffix}")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def save_reliability_diagram(out_dir, labels, probs_orig, probs_cal):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    orig_true, orig_pred = calibration_curve(labels, probs_orig, n_bins=10, strategy="uniform")
    cal_true, cal_pred = calibration_curve(labels, probs_cal, n_bins=10, strategy="uniform")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.plot(orig_pred, orig_true, marker="o", label="Original")
    ax.plot(cal_pred, cal_true, marker="o", label="Calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Test Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)


def save_predictions_csv(path, patients, window_indices, labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "window_end_index", "label", "probability", "prediction"])
        for row in zip(patients, window_indices, labels, probs, preds):
            writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved seizure prediction model")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="Directory with processed patient data (windows.npy / labels.npy)")
    parser.add_argument("--split-file", type=Path, required=True,
                        help="Path to patient_split_balanced.json")
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                        help="Output directory for metrics and figures")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="Number of MC Dropout forward passes")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    threshold = checkpoint["threshold"]

    seq_len = saved_args.get("seq_len", 6)
    seq_stride = saved_args.get("seq_stride", 2)

    if args.device == "auto":
        device = select_device("cuda")
    else:
        device = select_device(args.device)
    print(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────
    model = CNNLSTMSeizurePredictor(
        in_channels=saved_args.get("in_channels", 18),
        embedding_dim=saved_args.get("embedding_dim", 128),
        hidden_size=saved_args.get("hidden_size", 128),
        lstm_layers=saved_args.get("lstm_layers", 1),
        dropout=saved_args.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded (threshold={threshold:.4f}, seq_len={seq_len}, seq_stride={seq_stride})")

    # ── Load data ──────────────────────────────────────────────────────
    split_payload = json.loads(args.split_file.read_text())
    split = split_payload["split"]

    datasets = {}
    loaders = {}
    for name in ("val", "test"):
        ds = CHBMITSequenceDataset(args.processed_dir, split[name], seq_len, seq_stride)
        datasets[name] = ds
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        labels = ds.all_labels()
        n_pre = int(labels.sum())
        print(f"  {name}: {len(ds)} sequences ({n_pre} preictal, {len(labels) - n_pre} interictal)")

    # ── 1. Deterministic evaluation on test set ────────────────────────
    print("\n=== Deterministic Evaluation ===")
    val_pred = deterministic_predict(model, loaders["val"], device)
    test_pred = deterministic_predict(model, loaders["test"], device)

    test_metrics = classification_metrics(test_pred["labels"], test_pred["probs"], threshold)
    test_metrics["ece"] = expected_calibration_error(test_pred["labels"], test_pred["probs"])

    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Brier:     {test_metrics['brier']:.4f}")
    print(f"  ECE:       {test_metrics['ece']:.4f}")

    # ── 2. Temperature scaling calibration ─────────────────────────────
    print("\n=== Probability Calibration (Temperature Scaling) ===")
    temperature = fit_temperature(val_pred["logits"], val_pred["labels"])
    print(f"  Learned temperature: {temperature:.4f}")

    test_probs_cal = calibrate_probs(test_pred["logits"], temperature)
    cal_metrics = classification_metrics(test_pred["labels"], test_probs_cal, threshold)
    cal_metrics["ece"] = expected_calibration_error(test_pred["labels"], test_probs_cal)

    print(f"  Calibrated Brier: {cal_metrics['brier']:.4f} (was {test_metrics['brier']:.4f})")
    print(f"  Calibrated ECE:   {cal_metrics['ece']:.4f} (was {test_metrics['ece']:.4f})")

    # ── 3. MC Dropout uncertainty estimation ───────────────────────────
    print(f"\n=== MC Dropout (M={args.mc_samples}) ===")
    mc_pred = mc_dropout_predict(model, loaders["test"], device, mc_samples=args.mc_samples)
    mc_metrics = classification_metrics(mc_pred["labels"], mc_pred["mean_probs"], threshold)
    print(f"  MC AUROC: {mc_metrics['auroc']:.4f}")
    print(f"  MC AUPRC: {mc_metrics['auprc']:.4f}")
    print(f"  MC F1:    {mc_metrics['f1']:.4f}")

    # Error detection
    preds = (mc_pred["mean_probs"] >= threshold).astype(np.int64)
    errors = (preds != mc_pred["labels"]).astype(np.int64)
    print(f"\n  Error Detection:")
    for score_name in ("predictive_entropy", "std_probs", "mutual_information"):
        scores = mc_pred[score_name]
        err_auroc = safe_metric(roc_auc_score, errors, scores)
        err_auprc = safe_metric(average_precision_score, errors, scores)
        print(f"    {score_name:25s}  Err-AUROC={err_auroc:.3f}  Err-AUPRC={err_auprc:.3f}")

    # ── 4. Save results ────────────────────────────────────────────────
    print(f"\n=== Saving results to {args.out_dir} ===")

    summary = {
        "threshold": threshold,
        "temperature": temperature,
        "seq_len": seq_len,
        "seq_stride": seq_stride,
        "mc_samples": args.mc_samples,
        "deterministic_test": test_metrics,
        "calibrated_test": cal_metrics,
        "mc_dropout_test": mc_metrics,
    }
    (args.out_dir / "test_metrics.json").write_text(json.dumps(to_builtin(summary), indent=2))

    save_predictions_csv(
        args.out_dir / "test_predictions.csv",
        test_pred["patients"], test_pred["window_indices"],
        test_pred["labels"], test_pred["probs"], threshold,
    )

    # ── 5. Visualizations ──────────────────────────────────────────────
    save_roc_pr_curves(args.out_dir, test_pred["labels"], test_pred["probs"])
    save_reliability_diagram(args.out_dir, test_pred["labels"], test_pred["probs"], test_probs_cal)

    print(f"  test_metrics.json")
    print(f"  test_predictions.csv")
    print(f"  roc_pr_curves.png")
    print(f"  reliability_diagram.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
