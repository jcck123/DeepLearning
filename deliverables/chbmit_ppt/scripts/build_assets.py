import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\CHB-MIT")
RUNS = ROOT / "runs"
ASSET_DIR = ROOT / "deliverables" / "chbmit_ppt" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

BALANCED_SPLIT = Path(r"D:\CHB-MIT-Data\splits\patient_split_balanced.json")
ORIGINAL_SPLIT = Path(r"D:\CHB-MIT-Data\splits\patient_split.json")

COLORS = {
    "cnn1d": "#1f6aa5",
    "cnnlstm_v1": "#b65d2f",
    "cnnlstm_tuned": "#18804b",
    "bad": "#c0392b",
    "good": "#1f7a52",
    "neutral": "#6b7280",
    "accent": "#f4b942",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_stats(split_payload: dict) -> dict:
    stats = split_payload["patient_stats"]
    split = split_payload["split"]
    output = {}
    for split_name in ("train", "val", "test"):
        pids = split[split_name]
        windows = sum(stats[pid]["n_windows"] for pid in pids)
        preictal = sum(stats[pid]["n_preictal"] for pid in pids)
        ratio = preictal / max(1, windows)
        output[split_name] = {
            "windows": windows,
            "preictal": preictal,
            "ratio": ratio,
        }
    return output


def save_summary_json():
    original = split_stats(load_json(ORIGINAL_SPLIT))
    balanced = split_stats(load_json(BALANCED_SPLIT))

    cnn1d = load_json(RUNS / "cnn1d_balanced_v1" / "metrics.json")
    cnnlstm_v1 = load_json(RUNS / "cnn_lstm_v2" / "metrics.json")
    cnnlstm_tuned = load_json(RUNS / "cnn_lstm_seq6_stride2_v1" / "metrics.json")
    cnnlstm_tuned_cal = load_json(
        RUNS / "cnn_lstm_seq6_stride2_v1" / "calibration_auto" / "calibration_metrics.json"
    )

    summary = {
        "dataset": {
            "patients": 24,
            "windows": 115330,
            "preictal": 7273,
            "interictal": 108057,
            "positive_ratio": 7273 / 115330,
            "channels": 18,
            "sfreq": 256,
            "window_sec": 30,
            "stride_sec": 30,
            "preictal_sec": 1800,
            "postictal_sec": 300,
        },
        "splits": {
            "original": original,
            "balanced": balanced,
        },
        "models": {
            "cnn1d": cnn1d,
            "cnnlstm_v1": cnnlstm_v1,
            "cnnlstm_tuned": cnnlstm_tuned,
            "cnnlstm_tuned_calibrated": cnnlstm_tuned_cal,
        },
    }
    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def build_split_chart(summary: dict):
    setup_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    splits = ["train", "val", "test"]
    x = np.arange(len(splits))
    width = 0.35

    orig = [summary["splits"]["original"][s]["ratio"] * 100 for s in splits]
    bal = [summary["splits"]["balanced"][s]["ratio"] * 100 for s in splits]

    bars1 = ax.bar(x - width / 2, orig, width, label="Original split", color=COLORS["bad"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, bal, width, label="Balanced split", color=COLORS["good"], alpha=0.9)

    ax.set_title("Positive Ratio Shift Before and After Split Repair")
    ax.set_ylabel("Positive ratio (%)")
    ax.set_xticks(x, [s.capitalize() for s in splits])
    ax.legend(frameon=False)
    ax.set_ylim(0, max(max(orig), max(bal)) * 1.25)

    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.25,
                f"{b.get_height():.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.savefig(ASSET_DIR / "split_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_training_curves():
    setup_style()
    runs = {
        "1D CNN": (load_json(RUNS / "cnn1d_balanced_v1" / "metrics.json"), COLORS["cnn1d"]),
        "CNN+LSTM v1": (load_json(RUNS / "cnn_lstm_v2" / "metrics.json"), COLORS["cnnlstm_v1"]),
        "CNN+LSTM tuned": (load_json(RUNS / "cnn_lstm_seq6_stride2_v1" / "metrics.json"), COLORS["cnnlstm_tuned"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    for name, (payload, color) in runs.items():
        history = payload["history"]
        epochs = [item["epoch"] for item in history]
        val_auprc = [item["val_metrics"]["auprc"] for item in history]
        val_auroc = [item["val_metrics"]["auroc"] for item in history]
        axes[0].plot(epochs, val_auprc, marker="o", linewidth=2, color=color, label=name)
        axes[1].plot(epochs, val_auroc, marker="o", linewidth=2, color=color, label=name)

    axes[0].set_title("Validation AUPRC by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUPRC")
    axes[1].set_title("Validation AUROC by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False, loc="best")

    fig.savefig(ASSET_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_model_comparison_chart():
    setup_style()
    models = [
        ("1D CNN", load_json(RUNS / "cnn1d_balanced_v1" / "metrics.json"), COLORS["cnn1d"]),
        ("CNN+LSTM v1", load_json(RUNS / "cnn_lstm_v2" / "metrics.json"), COLORS["cnnlstm_v1"]),
        ("CNN+LSTM tuned", load_json(RUNS / "cnn_lstm_seq6_stride2_v1" / "metrics.json"), COLORS["cnnlstm_tuned"]),
    ]
    metrics = ["auroc", "auprc", "f1"]
    metric_names = ["Test AUROC", "Test AUPRC", "Test F1"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, metric_names):
        values = [m[1]["final_test_metrics"][metric] for m in models]
        colors = [m[2] for m in models]
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(range(len(models)), [m[0] for m in models], rotation=20, ha="right")
        ax.set_ylim(0, max(values) * 1.25)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    fig.savefig(ASSET_DIR / "test_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_calibration_chart():
    setup_style()
    payload = load_json(RUNS / "cnn_lstm_seq6_stride2_v1" / "calibration_auto" / "calibration_metrics.json")

    original_val = payload["original"]["val"]
    calibrated_val = payload["calibrated"]["val"]
    original_test = payload["original"]["test"]
    calibrated_test = payload["calibrated"]["test"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    labels = ["Validation", "Test"]
    brier_before = [original_val["brier"], original_test["brier"]]
    brier_after = [calibrated_val["brier"], calibrated_test["brier"]]
    ece_before = [original_val["ece"], original_test["ece"]]
    ece_after = [calibrated_val["ece"], calibrated_test["ece"]]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, brier_before, width, label="Before", color=COLORS["bad"], alpha=0.85)
    bars2 = axes[0].bar(x + width / 2, brier_after, width, label="After", color=COLORS["good"], alpha=0.9)
    axes[0].set_title("Brier Score Before vs After Calibration")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)

    bars3 = axes[1].bar(x - width / 2, ece_before, width, label="Before", color=COLORS["bad"], alpha=0.85)
    bars4 = axes[1].bar(x + width / 2, ece_after, width, label="After", color=COLORS["good"], alpha=0.9)
    axes[1].set_title("ECE Before vs After Calibration")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False)

    for bars in (bars1, bars2, bars3, bars4):
        for b in bars:
            axes[0 if bars in (bars1, bars2) else 1].text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.002,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.savefig(ASSET_DIR / "calibration_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = save_summary_json()
    build_split_chart(summary)
    build_training_curves()
    build_model_comparison_chart()
    build_calibration_chart()
    print(f"Assets saved to {ASSET_DIR}")


if __name__ == "__main__":
    main()
