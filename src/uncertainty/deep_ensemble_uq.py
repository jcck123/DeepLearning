import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train import find_best_threshold, to_builtin
from uq_mcdropout import (
    UNCERTAINTY_KEYS,
    binary_entropy,
    choose_triage_uncertainty_threshold,
    error_detection_metrics,
    plot_triage_policy,
    robust_classification_metrics,
    save_triage_predictions,
    triage_metrics,
)


DEFAULT_OUT_DIR = Path(r"E:\CHB-MIT\runs\deep_ensemble_seq6_stride2\ensemble_uq")


def read_prediction_csv(path: Path) -> dict:
    patients = []
    window_indices = []
    labels = []
    logits = []
    probs = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patients.append(row["patient_id"])
            window_indices.append(int(row["window_end_index"]))
            labels.append(int(float(row["label"])))
            logits.append(float(row["logit"]))
            probs.append(float(row["probability"]))
    return {
        "patients": patients,
        "window_indices": np.asarray(window_indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float64),
        "probs": np.asarray(probs, dtype=np.float64),
    }


def validate_alignment(reference: dict, current: dict, split_name: str, run_dir: Path) -> None:
    if reference["patients"] != current["patients"]:
        raise ValueError(f"{split_name} patient order mismatch in {run_dir}")
    if not np.array_equal(reference["window_indices"], current["window_indices"]):
        raise ValueError(f"{split_name} window order mismatch in {run_dir}")
    if not np.array_equal(reference["labels"], current["labels"]):
        raise ValueError(f"{split_name} label mismatch in {run_dir}")


def aggregate_member_predictions(run_dirs: list[Path], split_name: str) -> tuple[dict, list[dict]]:
    members = []
    reference = None
    for run_dir in run_dirs:
        payload = read_prediction_csv(run_dir / f"{split_name}_predictions.csv")
        if reference is None:
            reference = payload
        else:
            validate_alignment(reference, payload, split_name, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        members.append(
            {
                "run_dir": str(run_dir),
                "seed": metrics["args"]["seed"],
                "metrics": metrics[f"final_{split_name}_metrics"],
                "threshold": metrics["best_threshold"],
                "payload": payload,
            }
        )

    assert reference is not None
    logits_stack = np.stack([member["payload"]["logits"] for member in members], axis=0)
    probs_stack = np.stack([member["payload"]["probs"] for member in members], axis=0)
    mean_logits = logits_stack.mean(axis=0)
    mean_probs = probs_stack.mean(axis=0)
    std_probs = probs_stack.std(axis=0)
    q05_probs = np.quantile(probs_stack, 0.05, axis=0)
    q95_probs = np.quantile(probs_stack, 0.95, axis=0)
    pred_entropy = binary_entropy(mean_probs)
    expected_entropy = binary_entropy(probs_stack).mean(axis=0)
    mutual_information = pred_entropy - expected_entropy

    summary = {
        "patients": reference["patients"],
        "window_indices": reference["window_indices"],
        "labels": reference["labels"],
        "mean_logits": mean_logits,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "q05_probs": q05_probs,
        "q95_probs": q95_probs,
        "predictive_entropy": pred_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "member_probs": probs_stack,
        "member_logits": logits_stack,
    }
    member_info = [
        {
            "run_dir": member["run_dir"],
            "seed": member["seed"],
            "threshold": member["threshold"],
            "metrics": member["metrics"],
        }
        for member in members
    ]
    return summary, member_info


def save_ensemble_predictions(path: Path, aggregate: dict, threshold: float) -> None:
    preds = (aggregate["mean_probs"] >= threshold).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_id",
                "window_end_index",
                "label",
                "ensemble_mean_logit",
                "ensemble_mean_probability",
                "ensemble_std_probability",
                "ensemble_q05_probability",
                "ensemble_q95_probability",
                "predictive_entropy",
                "expected_entropy",
                "mutual_information",
                "prediction",
            ]
        )
        for row in zip(
            aggregate["patients"],
            aggregate["window_indices"].tolist(),
            aggregate["labels"].tolist(),
            aggregate["mean_logits"].tolist(),
            aggregate["mean_probs"].tolist(),
            aggregate["std_probs"].tolist(),
            aggregate["q05_probs"].tolist(),
            aggregate["q95_probs"].tolist(),
            aggregate["predictive_entropy"].tolist(),
            aggregate["expected_entropy"].tolist(),
            aggregate["mutual_information"].tolist(),
            preds.tolist(),
        ):
            writer.writerow(row)


def parse_run_dirs(raw: str) -> list[Path]:
    run_dirs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        run_dirs.append(Path(item))
    if not run_dirs:
        raise ValueError("At least one run directory must be provided.")
    return run_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multiple CNN+LSTM runs into a Deep Ensemble and evaluate UQ")
    parser.add_argument("--run-dirs", type=str, required=True, help="Comma-separated run directories")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--triage-score", type=str, default="auto", choices=UNCERTAINTY_KEYS + ["auto"])
    parser.add_argument("--max-review-rate", type=float, default=0.05)
    parser.add_argument("--min-alert-recall-fraction", type=float, default=0.5)
    return parser.parse_args()


def plot_ensemble_uncertainty_comparison(path: Path, score_summary: dict) -> None:
    score_names = list(score_summary.keys())
    err_auroc = [score_summary[name]["error_detection"]["test"]["error_detection_auroc"] for name in score_names]
    err_auprc = [score_summary[name]["error_detection"]["test"]["error_detection_auprc"] for name in score_names]
    correct_mean = [score_summary[name]["error_detection"]["test"]["mean_uncertainty_correct"] for name in score_names]
    incorrect_mean = [score_summary[name]["error_detection"]["test"]["mean_uncertainty_incorrect"] for name in score_names]

    x = np.arange(len(score_names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    axes[0].bar(x - width / 2, err_auroc, width, label="Error-det AUROC", color="#1F6AA5")
    axes[0].bar(x + width / 2, err_auprc, width, label="Error-det AUPRC", color="#D09129")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(score_names, rotation=12)
    axes[0].set_title("Deep Ensemble uncertainty comparison")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, correct_mean, width, label="Correct mean", color="#8AA4BE")
    axes[1].bar(x + width / 2, incorrect_mean, width, label="Incorrect mean", color="#C84C3A")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(score_names, rotation=12)
    axes[1].set_title("Uncertainty on correct vs incorrect test predictions")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dirs = [path.resolve() for path in parse_run_dirs(args.run_dirs)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Run dirs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")

    val_agg, members = aggregate_member_predictions(run_dirs, "val")
    test_agg, _ = aggregate_member_predictions(run_dirs, "test")

    ensemble_threshold, _ = find_best_threshold(val_agg["labels"], val_agg["mean_probs"])
    val_metrics = robust_classification_metrics(val_agg["labels"], val_agg["mean_probs"], ensemble_threshold)
    test_metrics = robust_classification_metrics(test_agg["labels"], test_agg["mean_probs"], ensemble_threshold)

    score_summary = {}
    for score_name in UNCERTAINTY_KEYS:
        score_summary[score_name] = {
            "error_detection": {
                "val": error_detection_metrics(
                    val_agg["labels"], val_agg["mean_probs"], val_agg[score_name], ensemble_threshold
                ),
                "test": error_detection_metrics(
                    test_agg["labels"], test_agg["mean_probs"], test_agg[score_name], ensemble_threshold
                ),
            }
        }

    if args.triage_score == "auto":
        triage_score_name = max(
            UNCERTAINTY_KEYS,
            key=lambda name: score_summary[name]["error_detection"]["val"]["error_detection_auprc"],
        )
    else:
        triage_score_name = args.triage_score

    triage_uncertainty_threshold, triage_val_selected, triage_val_sweep = choose_triage_uncertainty_threshold(
        val_agg["labels"],
        val_agg["mean_probs"],
        val_agg[triage_score_name],
        probability_threshold=ensemble_threshold,
        max_review_rate=float(args.max_review_rate),
        min_alert_recall_fraction=float(args.min_alert_recall_fraction),
    )
    triage_test_selected = triage_metrics(
        test_agg["labels"],
        test_agg["mean_probs"],
        test_agg[triage_score_name],
        probability_threshold=ensemble_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )
    triage_test_sweep = [
        triage_metrics(
            test_agg["labels"],
            test_agg["mean_probs"],
            test_agg[triage_score_name],
            probability_threshold=ensemble_threshold,
            uncertainty_threshold=row["uncertainty_threshold"],
        )
        for row in triage_val_sweep
    ]

    save_ensemble_predictions(args.out_dir / "val_ensemble_predictions.csv", val_agg, ensemble_threshold)
    save_ensemble_predictions(args.out_dir / "test_ensemble_predictions.csv", test_agg, ensemble_threshold)
    save_triage_predictions(
        args.out_dir / "val_triage_predictions.csv",
        val_agg["labels"],
        val_agg["patients"],
        val_agg["window_indices"].tolist(),
        val_agg["mean_probs"],
        val_agg[triage_score_name],
        probability_threshold=ensemble_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )
    save_triage_predictions(
        args.out_dir / "test_triage_predictions.csv",
        test_agg["labels"],
        test_agg["patients"],
        test_agg["window_indices"].tolist(),
        test_agg["mean_probs"],
        test_agg[triage_score_name],
        probability_threshold=ensemble_threshold,
        uncertainty_threshold=triage_uncertainty_threshold,
    )
    plot_ensemble_uncertainty_comparison(args.out_dir / "ensemble_uq_comparison.png", score_summary)
    plot_triage_policy(args.out_dir / "ensemble_triage_policy.png", triage_val_sweep, triage_test_sweep, triage_uncertainty_threshold)

    summary = {
        "run_dirs": [str(path) for path in run_dirs],
        "n_members": len(run_dirs),
        "member_info": members,
        "ensemble_threshold": ensemble_threshold,
        "ensemble_metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "uncertainty_scores": score_summary,
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
    }
    (args.out_dir / "ensemble_metrics.json").write_text(json.dumps(to_builtin(summary), indent=2))

    print("\nDeep Ensemble summary")
    print(f"  Members:                {len(run_dirs)}")
    print(f"  Ensemble threshold:     {ensemble_threshold:.4f}")
    print(f"  Val AUROC/AUPRC:        {val_metrics['auroc']:.4f} / {val_metrics['auprc']:.4f}")
    print(f"  Test AUROC/AUPRC:       {test_metrics['auroc']:.4f} / {test_metrics['auprc']:.4f}")
    for score_name in UNCERTAINTY_KEYS:
        err = score_summary[score_name]["error_detection"]["test"]
        print(
            f"  [{score_name}] test error-det AUROC/AUPRC: "
            f"{err['error_detection_auroc']:.4f} / {err['error_detection_auprc']:.4f}"
        )
    print(f"  Triage score:           {triage_score_name}")
    print(f"  Triage threshold:       {triage_uncertainty_threshold:.4f}")
    print(
        f"  Triage test alert P/R/F1: {triage_test_selected['alert_precision']:.4f} / "
        f"{triage_test_selected['alert_recall']:.4f} / {triage_test_selected['alert_f1']:.4f}"
    )
    print(
        f"  Triage test rates A/R/N: {triage_test_selected['alert_rate']:.4f} / "
        f"{triage_test_selected['review_rate']:.4f} / {triage_test_selected['no_alert_rate']:.4f}"
    )
    print(f"\nArtifacts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
