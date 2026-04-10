import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_RUN_DIR = Path(r"E:\CHB-MIT\runs\cnn_lstm_v2")


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


def safe_metric(fn, *args, default=float("nan"), **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return default


def clip_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(probs, eps, 1.0 - eps)


def prob_to_logit(probs: np.ndarray) -> np.ndarray:
    probs = clip_probs(probs)
    return np.log(probs / (1.0 - probs))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    probs = clip_probs(np.asarray(probs, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    return to_builtin(
        {
            "auroc": safe_metric(roc_auc_score, labels, probs),
            "auprc": safe_metric(average_precision_score, labels, probs),
            "brier": safe_metric(brier_score_loss, labels, probs),
            "ece": expected_calibration_error(labels, probs),
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


def load_predictions(path: Path) -> dict:
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    payload = {
        "patient_id": [row["patient_id"] for row in rows],
        "window_end_index": np.asarray([int(row["window_end_index"]) for row in rows], dtype=np.int64),
        "label": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        "probability": np.asarray([float(row["probability"]) for row in rows], dtype=np.float64),
    }
    if "logit" in rows[0]:
        payload["logit"] = np.asarray([float(row["logit"]) for row in rows], dtype=np.float64)
    return payload


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=200)
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature) + 1e-6
            loss = criterion(logits_t / temperature, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().cpu().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(logits / self.temperature)))


class PlattScaler:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self.model.fit(scores.reshape(-1, 1), labels)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicScaler:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        self.model.fit(probs, labels)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.transform(probs), dtype=np.float64)


def choose_method(requested_method: str, val_payload: dict) -> str:
    if requested_method != "auto":
        return requested_method
    return "temperature" if "logit" in val_payload else "platt"


def fit_calibrator(method: str, val_payload: dict):
    labels = val_payload["label"]
    probs = clip_probs(val_payload["probability"])

    if method == "temperature":
        if "logit" not in val_payload:
            raise ValueError("Temperature scaling requires logits, but the input CSV has no logit column.")
        calibrator = TemperatureScaler().fit(val_payload["logit"], labels)
        val_calibrated = calibrator.transform(val_payload["logit"])
        aux = {"temperature": calibrator.temperature}
        return calibrator, val_calibrated, aux

    if method == "platt":
        scores = val_payload["logit"] if "logit" in val_payload else prob_to_logit(probs)
        calibrator = PlattScaler().fit(scores, labels)
        val_calibrated = calibrator.transform(scores)
        coef = float(calibrator.model.coef_.ravel()[0])
        intercept = float(calibrator.model.intercept_.ravel()[0])
        aux = {"coef": coef, "intercept": intercept}
        return calibrator, val_calibrated, aux

    if method == "isotonic":
        calibrator = IsotonicScaler().fit(probs, labels)
        val_calibrated = calibrator.transform(probs)
        aux = {}
        return calibrator, val_calibrated, aux

    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(method: str, calibrator, payload: dict) -> np.ndarray:
    probs = clip_probs(payload["probability"])
    if method == "temperature":
        return calibrator.transform(payload["logit"])
    if method == "platt":
        scores = payload["logit"] if "logit" in payload else prob_to_logit(probs)
        return calibrator.transform(scores)
    if method == "isotonic":
        return calibrator.transform(probs)
    raise ValueError(f"Unsupported calibration method: {method}")


def save_calibrated_csv(path: Path, payload: dict, calibrated_probs: np.ndarray, threshold: float):
    preds = (calibrated_probs >= threshold).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        columns = ["patient_id", "window_end_index", "label"]
        if "logit" in payload:
            columns.append("logit")
        columns.extend(["original_probability", "calibrated_probability", "prediction"])
        writer.writerow(columns)

        for idx in range(len(payload["label"])):
            row = [
                payload["patient_id"][idx],
                int(payload["window_end_index"][idx]),
                int(payload["label"][idx]),
            ]
            if "logit" in payload:
                row.append(float(payload["logit"][idx]))
            row.extend(
                [
                    float(payload["probability"][idx]),
                    float(calibrated_probs[idx]),
                    int(preds[idx]),
                ]
            )
            writer.writerow(row)


def save_reliability_plot(path: Path, val_payload: dict, test_payload: dict, val_calibrated: np.ndarray, test_calibrated: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, split_name, payload, calibrated in [
        (axes[0], "Validation", val_payload, val_calibrated),
        (axes[1], "Test", test_payload, test_calibrated),
    ]:
        labels = payload["label"]
        original = clip_probs(payload["probability"])
        orig_true, orig_pred = calibration_curve(labels, original, n_bins=10, strategy="uniform")
        cal_true, cal_pred = calibration_curve(labels, calibrated, n_bins=10, strategy="uniform")

        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        ax.plot(orig_pred, orig_true, marker="o", label="Original")
        ax.plot(cal_pred, cal_true, marker="o", label="Calibrated")
        ax.set_title(f"{split_name} Reliability")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate saved CHB-MIT prediction probabilities")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "platt", "temperature", "isotonic"])
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / f"calibration_{args.method}")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_payload = load_predictions(run_dir / "val_predictions.csv")
    test_payload = load_predictions(run_dir / "test_predictions.csv")

    method = choose_method(args.method, val_payload)
    calibrator, val_calibrated, calibrator_params = fit_calibrator(method, val_payload)
    test_calibrated = apply_calibrator(method, calibrator, test_payload)

    original_val_threshold, _ = find_best_threshold(val_payload["label"], clip_probs(val_payload["probability"]))
    calibrated_val_threshold, _ = find_best_threshold(val_payload["label"], val_calibrated)

    summary = {
        "method_requested": args.method,
        "method_used": method,
        "calibrator_params": calibrator_params,
        "original": {
            "val": classification_metrics(
                val_payload["label"], clip_probs(val_payload["probability"]), original_val_threshold
            ),
            "test": classification_metrics(
                test_payload["label"], clip_probs(test_payload["probability"]), original_val_threshold
            ),
        },
        "calibrated": {
            "val": classification_metrics(val_payload["label"], val_calibrated, calibrated_val_threshold),
            "test": classification_metrics(test_payload["label"], test_calibrated, calibrated_val_threshold),
        },
    }

    (out_dir / "calibration_metrics.json").write_text(json.dumps(to_builtin(summary), indent=2))
    save_calibrated_csv(out_dir / "val_calibrated.csv", val_payload, val_calibrated, calibrated_val_threshold)
    save_calibrated_csv(out_dir / "test_calibrated.csv", test_payload, test_calibrated, calibrated_val_threshold)
    save_reliability_plot(
        out_dir / "reliability_diagram.png",
        val_payload,
        test_payload,
        val_calibrated,
        test_calibrated,
    )

    print(f"Calibration method: {method}")
    print("Original metrics")
    print(
        f"  Val  AUPRC={summary['original']['val']['auprc']:.4f} "
        f"AUROC={summary['original']['val']['auroc']:.4f} "
        f"Brier={summary['original']['val']['brier']:.4f} "
        f"ECE={summary['original']['val']['ece']:.4f}"
    )
    print(
        f"  Test AUPRC={summary['original']['test']['auprc']:.4f} "
        f"AUROC={summary['original']['test']['auroc']:.4f} "
        f"Brier={summary['original']['test']['brier']:.4f} "
        f"ECE={summary['original']['test']['ece']:.4f}"
    )
    print("Calibrated metrics")
    print(
        f"  Val  AUPRC={summary['calibrated']['val']['auprc']:.4f} "
        f"AUROC={summary['calibrated']['val']['auroc']:.4f} "
        f"Brier={summary['calibrated']['val']['brier']:.4f} "
        f"ECE={summary['calibrated']['val']['ece']:.4f}"
    )
    print(
        f"  Test AUPRC={summary['calibrated']['test']['auprc']:.4f} "
        f"AUROC={summary['calibrated']['test']['auroc']:.4f} "
        f"Brier={summary['calibrated']['test']['brier']:.4f} "
        f"ECE={summary['calibrated']['test']['ece']:.4f}"
    )
    print(f"Original best val threshold:   {original_val_threshold:.4f}")
    print(f"Calibrated best val threshold: {calibrated_val_threshold:.4f}")
    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
