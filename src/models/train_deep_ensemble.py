import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROCESSED_DIR = Path(r"D:\CHB-MIT-Data\processed")
DEFAULT_SPLIT_FILE = Path(r"D:\CHB-MIT-Data\splits\patient_split_balanced.json")
DEFAULT_BASE_OUT_DIR = Path(r"E:\CHB-MIT\runs\deep_ensemble_seq6_stride2")
DEFAULT_REFERENCE_RUN_DIR = Path(r"E:\CHB-MIT\runs\cnn_lstm_seq6_stride2_v1")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    total = max(1, total)
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def build_train_command(args, seed: int, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_cnn_lstm.py"),
        "--processed-dir",
        str(args.processed_dir),
        "--split-file",
        str(args.split_file),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(seed),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--seq-len",
        str(args.seq_len),
        "--seq-stride",
        str(args.seq_stride),
        "--embedding-dim",
        str(args.embedding_dim),
        "--hidden-size",
        str(args.hidden_size),
        "--lstm-layers",
        str(args.lstm_layers),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
    ]
    if args.balanced_sampling:
        cmd.append("--balanced-sampling")
    if args.wandb:
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-run-name",
                f"{args.wandb_run_prefix}-seed{seed}",
            ]
        )
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_mode:
            cmd.extend(["--wandb-mode", args.wandb_mode])
    return cmd


def estimate_seed_seconds(args) -> float:
    metrics_path = args.reference_run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
            if history:
                return float(sum(item["elapsed_sec"] for item in history) / len(history))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return float(args.estimate_minutes_per_seed) * 60.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple CNN+LSTM seeds for a Deep Ensemble")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--base-out-dir", type=Path, default=DEFAULT_BASE_OUT_DIR)
    parser.add_argument("--reference-run-dir", type=Path, default=DEFAULT_REFERENCE_RUN_DIR)
    parser.add_argument("--seeds", type=str, default="42,1337,2025")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--estimate-minutes-per-seed", type=float, default=65.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="CHB-MIT")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-run-prefix", type=str, default="deep-ensemble")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.base_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base output dir: {args.base_out_dir}")
    print(f"Seeds: {seeds}")
    print(f"Sequence config: seq_len={args.seq_len}, seq_stride={args.seq_stride}")
    print(f"Training config: lr={args.lr}, dropout={args.dropout}, batch_size={args.batch_size}")
    seed_estimate_sec = estimate_seed_seconds(args)
    print(
        "Estimated per-seed time: "
        f"{format_seconds(seed_estimate_sec)} "
        f"(reference: {args.reference_run_dir if args.reference_run_dir.exists() else 'fallback estimate'})"
    )

    total_start = time.time()
    completed = []
    skipped = []

    for idx, seed in enumerate(seeds, start=1):
        done_count = len(completed) + len(skipped)
        avg_completed_sec = (
            sum(item["elapsed_sec"] for item in completed) / len(completed)
            if completed
            else seed_estimate_sec
        )
        remaining_runs = len(seeds) - done_count
        total_eta_sec = avg_completed_sec * remaining_runs

        run_dir = args.base_out_dir / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best_model.pt"
        if args.skip_existing and metrics_path.exists() and checkpoint_path.exists():
            print(f"\n[{idx}/{len(seeds)}] Seed {seed} already exists in {run_dir}; skipping.")
            skipped.append(str(run_dir))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(args, seed, run_dir)
        print(
            f"\nDeep Ensemble progress {progress_bar(done_count, len(seeds))} "
            f"{done_count}/{len(seeds)} done | total ETA {format_seconds(total_eta_sec)}"
        )
        print(f"[{idx}/{len(seeds)}] Training seed {seed}")
        print(f"Seed output dir: {run_dir}")
        print(f"Estimated seed ETA: {format_seconds(avg_completed_sec)}")
        print("Command:")
        print(" ".join(cmd))
        start = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        completed.append({"seed": seed, "run_dir": str(run_dir), "elapsed_sec": round(elapsed, 2)})
        done_count = len(completed) + len(skipped)
        avg_completed_sec = sum(item["elapsed_sec"] for item in completed) / len(completed)
        remaining_runs = len(seeds) - done_count
        total_eta_sec = avg_completed_sec * remaining_runs
        print(
            f"Finished seed {seed} in {format_seconds(elapsed)} | "
            f"overall {progress_bar(done_count, len(seeds))} "
            f"{done_count}/{len(seeds)} done | remaining {format_seconds(total_eta_sec)}"
        )

    total_elapsed = time.time() - total_start
    print("\nDeep Ensemble training summary")
    print(f"  Total elapsed: {format_seconds(total_elapsed)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Skipped:   {len(skipped)}")
    for item in completed:
        print(f"  Seed {item['seed']}: {item['run_dir']} ({format_seconds(item['elapsed_sec'])})")
    for item in skipped:
        print(f"  Skipped existing: {item}")


if __name__ == "__main__":
    main()
