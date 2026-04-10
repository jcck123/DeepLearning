import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_cnn_lstm import (
    CHBMITSequenceDataset,
    CNNLSTMSeizurePredictor,
    find_best_threshold,
    format_seconds,
    safe_metric,
    set_seed,
    to_builtin,
)


DEFAULT_RUN_DIR = Path(r"E:\CHB-MIT\runs\cnn_lstm_seq6_stride2_v1")
UNCERTAINTY_KEYS = ["std_probs", "predictive_entropy", "mutual_information"]
DEFAULT_TRIAGE_SCORE = "predictive_entropy"


def binary_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0 - eps)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def robust_classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int64)
    unique_labels = np.unique(labels)
    has_both_classes = unique_labels.size > 1
    has_positive = np.any(labels == 1)

    return to_builtin(
        {
            "auroc": safe_metric(roc_auc_score, labels, probs) if has_both_classes else float("nan"),
            "auprc": safe_metric(average_precision_score, labels, probs) if has_positive else float("nan"),
            "brier": safe_metric(brier_score_loss, labels, probs),
            "precision": safe_metric(precision_score, labels, preds, zero_division=0),
            "recall": safe_metric(recall_score, labels, preds, zero_division=0),
            "f1": safe_metric(f1_score, labels, preds, zero_division=0),
            "threshold": threshold,
            "positive_rate": float(np.mean(labels == 1)) if labels.size > 0 else float("nan"),
            "predicted_positive_rate": float(np.mean(preds == 1)) if preds.size > 0 else float("nan"),
        }
    )


def enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


def build_eval_loader(
    processed_dir: Path,
    split_file: Path,
    split_name: str,
    seq_len: int,
    seq_stride: int,
    batch_size: int,
    num_workers: int,
):
    split_payload = json.loads(Path(split_file).read_text())
    patient_ids = split_payload["split"][split_name]
    dataset = CHBMITSequenceDataset(processed_dir, patient_ids, seq_len, seq_stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, patient_ids


def deterministic_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage_name: str,
    max_batches: int | None = None,
):
    model.eval()
    all_labels = []
    all_logits = []
    all_probs = []
    all_patients = []
    all_window_indices = []
    autocast_enabled = device.type == "cuda"

    total_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(total=total_steps, desc=f"{stage_name} deterministic", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (inputs, labels, patient_ids, window_indices) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_patients.extend(patient_ids)
            all_window_indices.extend(int(idx) for idx in window_indices)
            progress.update(1)
    progress.close()

    return {
        "labels": np.concatenate(all_labels).astype(np.int64),
        "logits": np.concatenate(all_logits).astype(np.float64),
        "probs": np.concatenate(all_probs).astype(np.float64),
        "patients": all_patients,
        "window_indices": all_window_indices,
    }


def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    stage_name: str,
    max_batches: int | None = None,
):
    enable_dropout_only(model)
    all_labels = []
    all_patients = []
    all_window_indices = []
    all_mean_logits = []
    all_mean_probs = []
    all_std_probs = []
    all_q05_probs = []
    all_q95_probs = []
    all_pred_entropy = []
    all_expected_entropy = []
    all_mutual_information = []

    autocast_enabled = device.type == "cuda"
    total_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(total=total_steps, desc=f"{stage_name} MC-dropout", leave=False, dynamic_ncols=True)

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, labels, patient_ids, window_indices) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            logits_samples = []
            for _ in range(mc_samples):
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    logits = model(inputs)
                logits_samples.append(logits.detach().cpu().numpy().astype(np.float64))

            logits_stack = np.stack(logits_samples, axis=0)
            probs_stack = 1.0 / (1.0 + np.exp(-logits_stack))

            mean_logits = logits_stack.mean(axis=0)
            mean_probs = probs_stack.mean(axis=0)
            std_probs = probs_stack.std(axis=0)
            q05_probs = np.quantile(probs_stack, 0.05, axis=0)
            q95_probs = np.quantile(probs_stack, 0.95, axis=0)
            pred_entropy = binary_entropy(mean_probs)
            expected_entropy = binary_entropy(probs_stack).mean(axis=0)
            mutual_information = pred_entropy - expected_entropy

            all_labels.append(labels.numpy())
            all_patients.extend(patient_ids)
            all_window_indices.extend(int(idx) for idx in window_indices)
            all_mean_logits.append(mean_logits)
            all_mean_probs.append(mean_probs)
            all_std_probs.append(std_probs)
            all_q05_probs.append(q05_probs)
            all_q95_probs.append(q95_probs)
            all_pred_entropy.append(pred_entropy)
            all_expected_entropy.append(expected_entropy)
            all_mutual_information.append(mutual_information)

            elapsed = time.time() - start_time
            done = batch_idx + 1
            eta = (elapsed / done) * (total_steps - done) if done > 0 else float("nan")
            progress.set_postfix({"passes": mc_samples, "eta": format_seconds(eta)})
            progress.update(1)

    progress.close()
    model.eval()

    return {
        "labels": np.concatenate(all_labels).astype(np.int64),
        "patients": all_patients,
        "window_indices": all_window_indices,
        "mean_logits": np.concatenate(all_mean_logits).astype(np.float64),
        "mean_probs": np.concatenate(all_mean_probs).astype(np.float64),
        "std_probs": np.concatenate(all_std_probs).astype(np.float64),
        "q05_probs": np.concatenate(all_q05_probs).astype(np.float64),
        "q95_probs": np.concatenate(all_q95_probs).astype(np.float64),
        "predictive_entropy": np.concatenate(all_pred_entropy).astype(np.float64),
        "expected_entropy": np.concatenate(all_expected_entropy).astype(np.float64),
        "mutual_information": np.concatenate(all_mutual_information).astype(np.float64),
    }


def error_detection_metrics(labels: np.ndarray, probs: np.ndarray, uncertainty: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    errors = (preds != labels).astype(np.int64)
    return to_builtin(
        {
            "error_rate": float(errors.mean()),
            "mean_uncertainty_correct": float(uncertainty[errors == 0].mean()) if np.any(errors == 0) else float("nan"),
            "mean_uncertainty_incorrect": float(uncertainty[errors == 1].mean()) if np.any(errors == 1) else float("nan"),
            "error_detection_auroc": safe_metric(roc_auc_score, errors, uncertainty),
            "error_detection_auprc": safe_metric(average_precision_score, errors, uncertainty),
        }
    )


def selective_metrics(labels: np.ndarray, probs: np.ndarray, uncertainty: np.ndarray, class_threshold: float, uncertainty_threshold: float) -> dict:
    retain_mask = uncertainty <= uncertainty_threshold
    retained_labels = labels[retain_mask]
    retained_probs = probs[retain_mask]
    metrics = robust_classification_metrics(retained_labels, retained_probs, class_threshold) if retain_mask.any() else {}
    return to_builtin(
        {
            "coverage": float(retain_mask.mean()),
            "retained_samples": int(retain_mask.sum()),
            "uncertainty_threshold": float(uncertainty_threshold),
            **metrics,
        }
    )


def best_threshold_or_fallback(labels: np.ndarray, probs: np.ndarray, fallback_threshold: float) -> tuple[float, str]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return float(fallback_threshold), "fallback_empty_subset"
    if not np.any(labels == 1):
        return float(fallback_threshold), "fallback_no_positive"
    threshold, _ = find_best_threshold(labels, probs)
    return float(threshold), "retained_val_refit"


def selective_metrics_with_refit(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    val_uncertainty: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    test_uncertainty: np.ndarray,
    uncertainty_threshold: float,
    fallback_threshold: float,
) -> tuple[dict, dict]:
    val_mask = val_uncertainty <= uncertainty_threshold
    test_mask = test_uncertainty <= uncertainty_threshold

    retained_val_labels = val_labels[val_mask]
    retained_val_probs = val_probs[val_mask]
    retained_test_labels = test_labels[test_mask]
    retained_test_probs = test_probs[test_mask]

    chosen_threshold, threshold_source = best_threshold_or_fallback(
        retained_val_labels, retained_val_probs, fallback_threshold
    )

    val_metrics = robust_classification_metrics(retained_val_labels, retained_val_probs, chosen_threshold)
    test_metrics = robust_classification_metrics(retained_test_labels, retained_test_probs, chosen_threshold)

    val_row = {
        "coverage": float(val_mask.mean()),
        "retained_samples": int(val_mask.sum()),
        "uncertainty_threshold": float(uncertainty_threshold),
        "threshold_source": threshold_source,
        **val_metrics,
    }
    test_row = {
        "coverage": float(test_mask.mean()),
        "retained_samples": int(test_mask.sum()),
        "uncertainty_threshold": float(uncertainty_threshold),
        "threshold_source": threshold_source,
        **test_metrics,
    }
    return to_builtin(val_row), to_builtin(test_row)


def triage_decisions(
    probs: np.ndarray,
    uncertainty: np.ndarray,
    probability_threshold: float,
    uncertainty_threshold: float,
) -> np.ndarray:
    positive_screen = probs >= probability_threshold
    decisions = np.full(probs.shape, "no_alert", dtype=object)
    decisions[positive_screen] = "review"
    decisions[positive_screen & (uncertainty <= uncertainty_threshold)] = "alert"
    return decisions


def triage_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    probability_threshold: float,
    uncertainty_threshold: float,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    decisions = triage_decisions(probs, uncertainty, probability_threshold, uncertainty_threshold)

    positive_mask = labels == 1
    negative_mask = labels == 0
    alert_mask = decisions == "alert"
    review_mask = decisions == "review"
    no_alert_mask = decisions == "no_alert"
    escalated_mask = alert_mask | review_mask

    alert_tp = int(np.sum(alert_mask & positive_mask))
    alert_fp = int(np.sum(alert_mask & negative_mask))
    review_tp = int(np.sum(review_mask & positive_mask))
    review_fp = int(np.sum(review_mask & negative_mask))
    no_alert_tp = int(np.sum(no_alert_mask & positive_mask))
    no_alert_fp = int(np.sum(no_alert_mask & negative_mask))

    total_positive = int(np.sum(positive_mask))
    total_negative = int(np.sum(negative_mask))
    alert_count = int(np.sum(alert_mask))
    review_count = int(np.sum(review_mask))
    no_alert_count = int(np.sum(no_alert_mask))
    escalated_count = int(np.sum(escalated_mask))

    alert_precision = safe_divide(alert_tp, alert_count)
    alert_recall = safe_divide(alert_tp, total_positive)
    alert_f1 = safe_divide(2.0 * alert_precision * alert_recall, alert_precision + alert_recall)

    review_precision = safe_divide(review_tp, review_count)
    review_recall = safe_divide(review_tp, total_positive)
    escalated_precision = safe_divide(alert_tp + review_tp, escalated_count)
    escalated_recall = safe_divide(alert_tp + review_tp, total_positive)

    baseline_positive_mask = np.asarray(probs >= probability_threshold, dtype=bool)
    baseline_tp = int(np.sum(baseline_positive_mask & positive_mask))
    baseline_fp = int(np.sum(baseline_positive_mask & negative_mask))

    return to_builtin(
        {
            "probability_threshold": float(probability_threshold),
            "uncertainty_threshold": float(uncertainty_threshold),
            "positive_rate": float(np.mean(positive_mask)) if labels.size > 0 else float("nan"),
            "alert_rate": safe_divide(alert_count, labels.size),
            "review_rate": safe_divide(review_count, labels.size),
            "no_alert_rate": safe_divide(no_alert_count, labels.size),
            "escalated_rate": safe_divide(escalated_count, labels.size),
            "alert_count": alert_count,
            "review_count": review_count,
            "no_alert_count": no_alert_count,
            "alert_tp": alert_tp,
            "alert_fp": alert_fp,
            "review_tp": review_tp,
            "review_fp": review_fp,
            "no_alert_tp": no_alert_tp,
            "no_alert_fp": no_alert_fp,
            "alert_precision": alert_precision,
            "alert_recall": alert_recall,
            "alert_f1": alert_f1,
            "review_precision": review_precision,
            "review_recall": review_recall,
            "escalated_precision": escalated_precision,
            "escalated_recall": escalated_recall,
            "alert_false_positive_rate": safe_divide(alert_fp, total_negative),
            "review_false_positive_rate": safe_divide(review_fp, total_negative),
            "review_share_of_baseline_tp": safe_divide(review_tp, baseline_tp),
            "review_share_of_baseline_fp": safe_divide(review_fp, baseline_fp),
            "auto_alert_share_of_baseline_tp": safe_divide(alert_tp, baseline_tp),
            "auto_alert_share_of_baseline_fp": safe_divide(alert_fp, baseline_fp),
        }
    )


def choose_triage_uncertainty_threshold(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    val_uncertainty: np.ndarray,
    probability_threshold: float,
    max_review_rate: float,
    min_alert_recall_fraction: float,
) -> tuple[float, dict, list[dict]]:
    positive_screen_mask = val_probs >= probability_threshold
    screened_uncertainty = val_uncertainty[positive_screen_mask]
    if screened_uncertainty.size == 0:
        fallback_threshold = float(np.max(val_uncertainty))
        fallback_metrics = triage_metrics(
            val_labels,
            val_probs,
            val_uncertainty,
            probability_threshold=probability_threshold,
            uncertainty_threshold=fallback_threshold,
        )
        fallback_metrics["selection_reason"] = "fallback_no_positive_screen"
        return fallback_threshold, fallback_metrics, [fallback_metrics]

    quantile_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
    candidate_thresholds = np.unique(np.quantile(screened_uncertainty, quantile_grid))

    baseline_threshold = float(np.max(screened_uncertainty))
    baseline_metrics = triage_metrics(
        val_labels,
        val_probs,
        val_uncertainty,
        probability_threshold=probability_threshold,
        uncertainty_threshold=baseline_threshold,
    )
    min_alert_recall = baseline_metrics["alert_recall"] * float(min_alert_recall_fraction)

    sweep_rows = []
    best_choice = None
    best_rank = None
    for candidate_threshold in candidate_thresholds:
        row = triage_metrics(
            val_labels,
            val_probs,
            val_uncertainty,
            probability_threshold=probability_threshold,
            uncertainty_threshold=float(candidate_threshold),
        )
        row["meets_review_cap"] = bool(row["review_rate"] <= max_review_rate + 1e-12)
        row["meets_min_alert_recall"] = bool(row["alert_recall"] + 1e-12 >= min_alert_recall)
        row["is_feasible"] = bool(row["meets_review_cap"] and row["meets_min_alert_recall"])
        sweep_rows.append(row)

        rank = (
            1 if row["is_feasible"] else 0,
            row["alert_precision"],
            row["alert_f1"],
            row["alert_recall"],
            -row["review_rate"],
            -row["alert_false_positive_rate"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_choice = row

    assert best_choice is not None
    best_choice["selection_reason"] = (
        "best_feasible_alert_precision"
        if best_choice["is_feasible"]
        else "best_available_under_soft_constraints"
    )
    return float(best_choice["uncertainty_threshold"]), to_builtin(best_choice), to_builtin(sweep_rows)


def save_triage_predictions(
    path: Path,
    labels: np.ndarray,
    patients: list[str],
    window_indices: list[int],
    probs: np.ndarray,
    uncertainty: np.ndarray,
    probability_threshold: float,
    uncertainty_threshold: float,
) -> None:
    decisions = triage_decisions(probs, uncertainty, probability_threshold, uncertainty_threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_id",
                "window_end_index",
                "label",
                "mean_probability",
                "uncertainty",
                "probability_threshold",
                "uncertainty_threshold",
                "decision",
            ]
        )
        for row in zip(
            patients,
            window_indices,
            labels.tolist(),
            probs.tolist(),
            uncertainty.tolist(),
            [probability_threshold] * len(labels),
            [uncertainty_threshold] * len(labels),
            decisions.tolist(),
        ):
            writer.writerow(row)


def save_uq_predictions(path: Path, deterministic: dict, mc: dict, threshold: float) -> None:
    preds = (mc["mean_probs"] >= threshold).astype(np.int64)
    correct = (preds == mc["labels"]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_id",
                "window_end_index",
                "label",
                "deterministic_logit",
                "deterministic_probability",
                "mc_mean_logit",
                "mc_mean_probability",
                "mc_std_probability",
                "mc_q05_probability",
                "mc_q95_probability",
                "predictive_entropy",
                "expected_entropy",
                "mutual_information",
                "prediction",
                "correct",
            ]
        )
        for row in zip(
            mc["patients"],
            mc["window_indices"],
            mc["labels"].tolist(),
            deterministic["logits"].tolist(),
            deterministic["probs"].tolist(),
            mc["mean_logits"].tolist(),
            mc["mean_probs"].tolist(),
            mc["std_probs"].tolist(),
            mc["q05_probs"].tolist(),
            mc["q95_probs"].tolist(),
            mc["predictive_entropy"].tolist(),
            mc["expected_entropy"].tolist(),
            mc["mutual_information"].tolist(),
            preds.tolist(),
            correct.tolist(),
        ):
            writer.writerow(row)


def plot_uq_summary(path: Path, coverage_rows: list[dict], uncertainty: np.ndarray, errors: np.ndarray, title: str) -> None:
    coverages = [row["coverage"] for row in coverage_rows]
    auroc = [row.get("auroc", np.nan) for row in coverage_rows]
    auprc = [row.get("auprc", np.nan) for row in coverage_rows]
    f1 = [row.get("f1", np.nan) for row in coverage_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(coverages, auroc, marker="o", label="AUROC")
    axes[0].plot(coverages, auprc, marker="o", label="AUPRC")
    axes[0].plot(coverages, f1, marker="o", label="F1")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Coverage retained")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Selective prediction by uncertainty")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    correct_unc = uncertainty[errors == 0]
    incorrect_unc = uncertainty[errors == 1]
    axes[1].hist(correct_unc, bins=30, alpha=0.7, label="Correct", color="#1F6AA5")
    axes[1].hist(incorrect_unc, bins=30, alpha=0.7, label="Incorrect", color="#C84C3A")
    axes[1].set_xlabel("Uncertainty score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Uncertainty vs prediction correctness")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_comparison(path: Path, summary_by_score: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    score_names = list(summary_by_score.keys())

    error_det_auroc = [summary_by_score[name]["error_detection"]["test"]["error_detection_auroc"] for name in score_names]
    error_det_auprc = [summary_by_score[name]["error_detection"]["test"]["error_detection_auprc"] for name in score_names]
    x = np.arange(len(score_names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, error_det_auroc, width, label="Error-det AUROC", color="#1F6AA5")
    axes[0, 0].bar(x + width / 2, error_det_auprc, width, label="Error-det AUPRC", color="#D09129")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(score_names, rotation=10)
    axes[0, 0].set_title("Error detection by uncertainty score")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend()

    for name in score_names:
        rows = summary_by_score[name]["selective_prediction_refit"]["test"]
        coverages = [row["coverage"] for row in rows]
        axes[0, 1].plot(coverages, [row.get("auroc", np.nan) for row in rows], marker="o", label=name)
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_xlabel("Coverage retained")
    axes[0, 1].set_ylabel("AUROC")
    axes[0, 1].set_title("Selective prediction AUROC")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    for name in score_names:
        rows = summary_by_score[name]["selective_prediction_refit"]["test"]
        coverages = [row["coverage"] for row in rows]
        axes[1, 0].plot(coverages, [row.get("auprc", np.nan) for row in rows], marker="o", label=name)
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_xlabel("Coverage retained")
    axes[1, 0].set_ylabel("AUPRC")
    axes[1, 0].set_title("Selective prediction AUPRC")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    for name in score_names:
        rows = summary_by_score[name]["selective_prediction_refit"]["test"]
        coverages = [row["coverage"] for row in rows]
        axes[1, 1].plot(coverages, [row.get("f1", np.nan) for row in rows], marker="o", label=name)
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_xlabel("Coverage retained")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_title("Selective prediction F1")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle("MC Dropout uncertainty comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_triage_policy(path: Path, val_sweep: list[dict], test_sweep: list[dict], selected_threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for rows, axis, title in [
        (val_sweep, axes[0], "Validation triage sweep"),
        (test_sweep, axes[1], "Test triage sweep"),
    ]:
        review_rate = [row["review_rate"] for row in rows]
        axis.plot(review_rate, [row["alert_precision"] for row in rows], marker="o", label="Alert precision")
        axis.plot(review_rate, [row["alert_recall"] for row in rows], marker="o", label="Alert recall")
        axis.plot(review_rate, [row["alert_f1"] for row in rows], marker="o", label="Alert F1")
        axis.set_xlabel("Review rate")
        axis.set_ylabel("Metric value")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend()

        selected = next((row for row in rows if abs(row["uncertainty_threshold"] - selected_threshold) < 1e-12), None)
        if selected is not None:
            axis.scatter(
                [selected["review_rate"]],
                [selected["alert_precision"]],
                s=110,
                color="#C84C3A",
                zorder=5,
                label="Selected",
            )

    fig.suptitle("Alert / Review / No Alert triage policy")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MC Dropout UQ for a trained CNN+LSTM seizure predictor")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--uncertainty-key", type=str, default="all", choices=UNCERTAINTY_KEYS + ["all"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--triage-score", type=str, default=DEFAULT_TRIAGE_SCORE, choices=UNCERTAINTY_KEYS)
    parser.add_argument("--max-review-rate", type=float, default=0.05)
    parser.add_argument("--min-alert-recall-fraction", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else run_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    processed_dir = Path(train_args["processed_dir"])
    split_file = Path(train_args["split_file"])
    seq_len = int(train_args["seq_len"])
    seq_stride = int(train_args["seq_stride"])
    batch_size = int(args.batch_size or train_args["batch_size"])
    default_suffix = f"uq_mcdropout_mc{args.mc_samples}_all" if args.uncertainty_key == "all" else f"uq_mcdropout_mc{args.mc_samples}_{args.uncertainty_key}"
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / default_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available in the current Python environment. Falling back to CPU.")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = CNNLSTMSeizurePredictor(
        in_channels=18,
        embedding_dim=int(train_args["embedding_dim"]),
        hidden_size=int(train_args["hidden_size"]),
        lstm_layers=int(train_args["lstm_layers"]),
        dropout=float(train_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_loader, _ = build_eval_loader(processed_dir, split_file, "val", seq_len, seq_stride, batch_size, args.num_workers)
    _, test_loader, _ = build_eval_loader(processed_dir, split_file, "test", seq_len, seq_stride, batch_size, args.num_workers)

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Uncertainty score: {args.uncertainty_key}")

    val_det = deterministic_predict(model, val_loader, device, "Val", max_batches=args.max_batches)
    test_det = deterministic_predict(model, test_loader, device, "Test", max_batches=args.max_batches)
    val_mc = mc_dropout_predict(model, val_loader, device, args.mc_samples, "Val", max_batches=args.max_batches)
    test_mc = mc_dropout_predict(model, test_loader, device, args.mc_samples, "Test", max_batches=args.max_batches)

    checkpoint_threshold = float(checkpoint["threshold"])
    mc_threshold, _ = find_best_threshold(val_mc["labels"], val_mc["mean_probs"])

    deterministic_val_metrics = robust_classification_metrics(val_det["labels"], val_det["probs"], checkpoint_threshold)
    deterministic_test_metrics = robust_classification_metrics(test_det["labels"], test_det["probs"], checkpoint_threshold)
    mc_val_metrics = robust_classification_metrics(val_mc["labels"], val_mc["mean_probs"], mc_threshold)
    mc_test_metrics = robust_classification_metrics(test_mc["labels"], test_mc["mean_probs"], mc_threshold)

    target_coverages = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    score_names = UNCERTAINTY_KEYS if args.uncertainty_key == "all" else [args.uncertainty_key]
    summary_by_score = {}
    for score_name in score_names:
        val_uncertainty = val_mc[score_name]
        test_uncertainty = test_mc[score_name]

        val_error_detection = error_detection_metrics(val_mc["labels"], val_mc["mean_probs"], val_uncertainty, mc_threshold)
        test_error_detection = error_detection_metrics(test_mc["labels"], test_mc["mean_probs"], test_uncertainty, mc_threshold)

        selective_val_fixed = []
        selective_test_fixed = []
        selective_val_refit = []
        selective_test_refit = []
        for coverage in target_coverages:
            unc_threshold = float(np.quantile(val_uncertainty, coverage)) if coverage < 1.0 else float(np.max(val_uncertainty))

            fixed_val_row = {"target_coverage": coverage, **selective_metrics(val_mc["labels"], val_mc["mean_probs"], val_uncertainty, mc_threshold, unc_threshold)}
            fixed_test_row = {"target_coverage": coverage, **selective_metrics(test_mc["labels"], test_mc["mean_probs"], test_uncertainty, mc_threshold, unc_threshold)}
            selective_val_fixed.append(fixed_val_row)
            selective_test_fixed.append(fixed_test_row)

            refit_val_row, refit_test_row = selective_metrics_with_refit(
                val_mc["labels"],
                val_mc["mean_probs"],
                val_uncertainty,
                test_mc["labels"],
                test_mc["mean_probs"],
                test_uncertainty,
                uncertainty_threshold=unc_threshold,
                fallback_threshold=mc_threshold,
            )
            selective_val_refit.append({"target_coverage": coverage, **refit_val_row})
            selective_test_refit.append({"target_coverage": coverage, **refit_test_row})

        summary_by_score[score_name] = {
            "error_detection": {
                "val": val_error_detection,
                "test": test_error_detection,
            },
            "selective_prediction_fixed_threshold": {
                "val": selective_val_fixed,
                "test": selective_test_fixed,
            },
            "selective_prediction_refit": {
                "val": selective_val_refit,
                "test": selective_test_refit,
            },
        }

    triage_score_name = args.triage_score
    val_triage_uncertainty = val_mc[triage_score_name]
    test_triage_uncertainty = test_mc[triage_score_name]
    triage_uncertainty_threshold, triage_val_selected, triage_val_sweep = choose_triage_uncertainty_threshold(
        val_mc["labels"],
        val_mc["mean_probs"],
        val_triage_uncertainty,
        probability_threshold=mc_threshold,
        max_review_rate=float(args.max_review_rate),
        min_alert_recall_fraction=float(args.min_alert_recall_fraction),
    )
    triage_test_selected = triage_metrics(
        test_mc["labels"],
        test_mc["mean_probs"],
        test_triage_uncertainty,
        probability_threshold=mc_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )
    triage_test_sweep = [
        triage_metrics(
            test_mc["labels"],
            test_mc["mean_probs"],
            test_triage_uncertainty,
            probability_threshold=mc_threshold,
            uncertainty_threshold=row["uncertainty_threshold"],
        )
        for row in triage_val_sweep
    ]

    save_uq_predictions(out_dir / "val_uq.csv", val_det, val_mc, mc_threshold)
    save_uq_predictions(out_dir / "test_uq.csv", test_det, test_mc, mc_threshold)
    save_triage_predictions(
        out_dir / "val_triage_predictions.csv",
        val_mc["labels"],
        val_mc["patients"],
        val_mc["window_indices"],
        val_mc["mean_probs"],
        val_triage_uncertainty,
        probability_threshold=mc_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )
    save_triage_predictions(
        out_dir / "test_triage_predictions.csv",
        test_mc["labels"],
        test_mc["patients"],
        test_mc["window_indices"],
        test_mc["mean_probs"],
        test_triage_uncertainty,
        probability_threshold=mc_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )

    test_preds = (test_mc["mean_probs"] >= mc_threshold).astype(np.int64)
    test_errors = (test_preds != test_mc["labels"]).astype(np.int64)
    if len(score_names) == 1:
        score_name = score_names[0]
        plot_uq_summary(
            out_dir / "uq_summary.png",
            summary_by_score[score_name]["selective_prediction_refit"]["test"],
            test_mc[score_name],
            test_errors,
            title=f"MC Dropout UQ ({score_name})",
        )
    else:
        plot_uncertainty_comparison(out_dir / "uq_comparison.png", summary_by_score)
    plot_triage_policy(out_dir / "triage_policy.png", triage_val_sweep, triage_test_sweep, triage_uncertainty_threshold)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "mc_samples": args.mc_samples,
        "uncertainty_key": args.uncertainty_key,
        "checkpoint_threshold": checkpoint_threshold,
        "mc_threshold": mc_threshold,
        "deterministic": {
            "val": deterministic_val_metrics,
            "test": deterministic_test_metrics,
        },
        "mc_dropout": {
            "val": mc_val_metrics,
            "test": mc_test_metrics,
        },
        "triage_policy": {
            "score_name": triage_score_name,
            "constraints": {
                "max_review_rate": float(args.max_review_rate),
                "min_alert_recall_fraction": float(args.min_alert_recall_fraction),
            },
            "selected_uncertainty_threshold": triage_uncertainty_threshold,
            "val_selected": triage_val_selected,
            "test_selected": triage_test_selected,
            "val_sweep": triage_val_sweep,
            "test_sweep": triage_test_sweep,
        },
        "uncertainty_scores": summary_by_score,
    }

    (out_dir / "uq_metrics.json").write_text(json.dumps(to_builtin(summary), indent=2))

    print("\nMC Dropout summary")
    print(f"  MC threshold:           {mc_threshold:.4f}")
    print(f"  MC val AUROC/AUPRC:     {mc_val_metrics['auroc']:.4f} / {mc_val_metrics['auprc']:.4f}")
    print(f"  MC test AUROC/AUPRC:    {mc_test_metrics['auroc']:.4f} / {mc_test_metrics['auprc']:.4f}")
    print(f"  Triage score:           {triage_score_name}")
    print(f"  Triage uncertainty th:  {triage_uncertainty_threshold:.4f}")
    print(
        f"  Triage test alert P/R/F1: {triage_test_selected['alert_precision']:.4f} / "
        f"{triage_test_selected['alert_recall']:.4f} / {triage_test_selected['alert_f1']:.4f}"
    )
    print(
        f"  Triage test rates A/R/N: {triage_test_selected['alert_rate']:.4f} / "
        f"{triage_test_selected['review_rate']:.4f} / {triage_test_selected['no_alert_rate']:.4f}"
    )
    for score_name in score_names:
        test_error_detection = summary_by_score[score_name]["error_detection"]["test"]
        selective_rows = summary_by_score[score_name]["selective_prediction_refit"]["test"]
        best_refit_by_auprc = max(
            selective_rows,
            key=lambda row: row["auprc"] if row["auprc"] == row["auprc"] else float("-inf"),
        )
        print(f"  [{score_name}] error-det AUROC/AUPRC (test): {test_error_detection['error_detection_auroc']:.4f} / {test_error_detection['error_detection_auprc']:.4f}")
        print(
            f"  [{score_name}] best refit selective test AUPRC: {best_refit_by_auprc['auprc']:.4f}"
            f" at coverage={best_refit_by_auprc['coverage']:.3f}, threshold={best_refit_by_auprc['threshold']:.4f}"
        )
    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
