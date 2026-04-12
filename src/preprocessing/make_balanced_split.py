import argparse
import json
from itertools import combinations
from pathlib import Path
# GenAI is only used as an auxiliary tool to improve code efficiency and optimize bugs.


DEFAULT_INPUT_SPLIT = Path(r"D:\CHB-MIT-Data\splits\patient_split.json")
DEFAULT_OUTPUT_SPLIT = Path(r"D:\CHB-MIT-Data\splits\patient_split_balanced.json")
SAME_PATIENT_GROUPS = [("chb01", "chb21")]


def load_stats(split_file: Path) -> dict:
    payload = json.loads(split_file.read_text())
    return payload["patient_stats"]


def build_units(patient_stats: dict) -> list[tuple[str, list[str]]]:
    grouped = set()
    units = []
    for pair in SAME_PATIENT_GROUPS:
        if all(pid in patient_stats for pid in pair):
            name = "__".join(pair)
            units.append((name, list(pair)))
            grouped.update(pair)
    for pid in sorted(patient_stats):
        if pid not in grouped:
            units.append((pid, [pid]))
    return units


def split_stats(patient_ids: list[str], patient_stats: dict) -> tuple[int, int, int, float]:
    windows = sum(patient_stats[pid]["n_windows"] for pid in patient_ids)
    preictal = sum(patient_stats[pid]["n_preictal"] for pid in patient_ids)
    interictal = sum(patient_stats[pid]["n_interictal"] for pid in patient_ids)
    ratio = preictal / max(1, windows)
    return windows, preictal, interictal, ratio


def expand_units(unit_names: tuple[str, ...], unit_lookup: dict[str, list[str]]) -> list[str]:
    patient_ids = []
    for unit_name in unit_names:
        patient_ids.extend(unit_lookup[unit_name])
    return sorted(patient_ids)


def search_best_split(patient_stats: dict, n_train: int, n_val: int, n_test: int):
    units = build_units(patient_stats)
    unit_lookup = dict(units)
    unit_names = [name for name, _ in units]

    total_windows, total_preictal, _, overall_ratio = split_stats(list(patient_stats), patient_stats)
    target_train_prop = n_train / (n_train + n_val + n_test)
    target_val_prop = n_val / (n_train + n_val + n_test)
    target_test_prop = n_test / (n_train + n_val + n_test)

    best = None

    for val_units in combinations(unit_names, 3):
        val_set = set(val_units)
        remaining = [name for name in unit_names if name not in val_set]

        for test_units in combinations(remaining, 3):
            test_set = set(test_units)
            train_units = tuple(name for name in remaining if name not in test_set)

            train_patients = expand_units(train_units, unit_lookup)
            val_patients = expand_units(val_units, unit_lookup)
            test_patients = expand_units(test_units, unit_lookup)

            if len(train_patients) != n_train or len(val_patients) != n_val or len(test_patients) != n_test:
                continue

            train_windows, train_pre, _, train_ratio = split_stats(train_patients, patient_stats)
            val_windows, val_pre, _, val_ratio = split_stats(val_patients, patient_stats)
            test_windows, test_pre, _, test_ratio = split_stats(test_patients, patient_stats)

            score = (
                20 * (train_ratio - overall_ratio) ** 2
                + 25 * (val_ratio - overall_ratio) ** 2
                + 25 * (test_ratio - overall_ratio) ** 2
                + 1 * ((train_windows / total_windows) - target_train_prop) ** 2
                + 2 * ((val_windows / total_windows) - target_val_prop) ** 2
                + 2 * ((test_windows / total_windows) - target_test_prop) ** 2
                + 1 * ((train_pre / total_preictal) - target_train_prop) ** 2
                + 2 * ((val_pre / total_preictal) - target_val_prop) ** 2
                + 2 * ((test_pre / total_preictal) - target_test_prop) ** 2
            )

            candidate = {
                "score": score,
                "train": train_patients,
                "val": val_patients,
                "test": test_patients,
                "stats": {
                    "overall": {
                        "windows": total_windows,
                        "preictal": total_preictal,
                        "ratio": overall_ratio,
                    },
                    "train": {
                        "windows": train_windows,
                        "preictal": train_pre,
                        "ratio": train_ratio,
                    },
                    "val": {
                        "windows": val_windows,
                        "preictal": val_pre,
                        "ratio": val_ratio,
                    },
                    "test": {
                        "windows": test_windows,
                        "preictal": test_pre,
                        "ratio": test_ratio,
                    },
                },
            }

            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("No valid balanced split found.")

    return best


def main():
    parser = argparse.ArgumentParser(description="Create a balanced patient-level split for CHB-MIT")
    parser.add_argument("--input-split", type=Path, default=DEFAULT_INPUT_SPLIT)
    parser.add_argument("--output-split", type=Path, default=DEFAULT_OUTPUT_SPLIT)
    parser.add_argument("--n-train", type=int, default=18)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=3)
    args = parser.parse_args()

    patient_stats = load_stats(args.input_split)
    best = search_best_split(patient_stats, args.n_train, args.n_val, args.n_test)

    payload = {
        "source_split_file": str(args.input_split),
        "method": "balanced_bruteforce_patient_level_split",
        "same_patient_groups": SAME_PATIENT_GROUPS,
        "score": best["score"],
        "split": {
            "train": best["train"],
            "val": best["val"],
            "test": best["test"],
            "excluded": [],
        },
        "patient_stats": patient_stats,
        "summary": best["stats"],
    }

    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.write_text(json.dumps(payload, indent=2))

    print(f"Saved balanced split to {args.output_split}")
    for name in ("train", "val", "test"):
        stats = best["stats"][name]
        print(
            f"{name}: patients={payload['split'][name]} | "
            f"windows={stats['windows']} preictal={stats['preictal']} ratio={stats['ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
