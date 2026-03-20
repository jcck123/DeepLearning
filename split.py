#!/usr/bin/env python3
"""
Patient-level Train/Val/Test Split
===================================
按病人分组，不能让同一病人的数据出现在不同集中（防止数据泄露）。

策略:
  - 癫痫少的病人 → 训练集（学习更稳定）
  - 中等的 → 验证集（调参）
  - 多的 → 测试集（评估更可靠）
  - chb01 和 chb21 是同一个人，必须放在一起

默认分配: 18 训练 / 3 验证 / 3 测试

Usage:
  python split.py --processed-dir /Volumes/T9/data/processed
  python split.py --processed-dir /Volumes/T9/data/processed --out-dir /Volumes/T9/data/splits
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

SAME_PATIENT = [('chb01', 'chb21')]


def load_patient_stats(processed_dir):
    """读取每个病人的预处理统计信息。"""
    stats = {}
    root = Path(processed_dir)

    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith("chb"):
            continue

        meta_path = pdir / "meta.json"
        labels_path = pdir / "labels.npy"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stats[pdir.name] = {
                'n_windows': meta['n_windows'],
                'n_preictal': meta['n_preictal'],
                'n_interictal': meta['n_interictal'],
                'preictal_ratio': meta['preictal_ratio'],
            }
        elif labels_path.exists():
            labels = np.load(labels_path)
            n_pre = int(np.sum(labels == 1))
            n_inter = int(np.sum(labels == 0))
            stats[pdir.name] = {
                'n_windows': len(labels),
                'n_preictal': n_pre,
                'n_interictal': n_inter,
                'preictal_ratio': round(n_pre / max(1, len(labels)), 4),
            }

    return stats


def create_split(stats, n_train=18, n_val=3, n_test=3, seed=42):
    """
    按癫痫数量排序后分组。
    chb01 和 chb21 必须在同一个 split。
    """
    rng = np.random.RandomState(seed)

    # 排除没有 preictal 数据的病人
    valid = {pid: s for pid, s in stats.items() if s['n_preictal'] > 0}
    excluded = [pid for pid in stats if pid not in valid]

    if excluded:
        print(f"  Excluded (no preictal): {excluded}")

    # 按 preictal 数量升序排列
    sorted_pids = sorted(valid.keys(), key=lambda p: valid[p]['n_preictal'])

    # 调整数量
    total_needed = n_train + n_val + n_test
    if len(sorted_pids) < total_needed:
        print(f"  Warning: only {len(sorted_pids)} valid patients, need {total_needed}")
        ratio_t = n_train / total_needed
        ratio_v = n_val / total_needed
        n_train = max(1, int(len(sorted_pids) * ratio_t))
        n_val = max(1, int(len(sorted_pids) * ratio_v))
        n_test = len(sorted_pids) - n_train - n_val

    # 分三组: 少→训练, 中→验证, 多→测试
    train_pids = sorted_pids[:n_train]
    val_pids = sorted_pids[n_train:n_train + n_val]
    test_pids = sorted_pids[n_train + n_val:n_train + n_val + n_test]

    # 组内随机打乱
    rng.shuffle(train_pids)
    rng.shuffle(val_pids)
    rng.shuffle(test_pids)

    split = {
        'train': sorted(train_pids),
        'val': sorted(val_pids),
        'test': sorted(test_pids),
        'excluded': sorted(excluded),
    }

    # 强制 chb01 和 chb21 在同一组
    for p1, p2 in SAME_PATIENT:
        if p1 not in valid or p2 not in valid:
            continue

        # 找到两个人分别在哪个组
        p1_set = None
        p2_set = None
        for set_name in ['train', 'val', 'test']:
            if p1 in split[set_name]:
                p1_set = set_name
            if p2 in split[set_name]:
                p2_set = set_name

        # 如果不在同一组，把 p2 移到 p1 的组
        if p1_set and p2_set and p1_set != p2_set:
            split[p2_set].remove(p2)
            split[p1_set].append(p2)
            split[p1_set] = sorted(split[p1_set])
            print(f"  Moved {p2} to {p1_set} (same patient as {p1})")

    return split


def print_split(split, stats):
    """打印分组详情。"""
    print(f"\n{'=' * 60}")
    print("PATIENT SPLIT")
    print(f"{'=' * 60}")

    for set_name in ['train', 'val', 'test']:
        pids = split[set_name]
        total_win = sum(stats.get(p, {}).get('n_windows', 0) for p in pids)
        total_pre = sum(stats.get(p, {}).get('n_preictal', 0) for p in pids)
        total_inter = sum(stats.get(p, {}).get('n_interictal', 0) for p in pids)

        print(f"\n  {set_name.upper()} ({len(pids)} patients):")
        print(f"    Patients: {pids}")
        print(f"    Windows:  {total_win} "
              f"(preictal={total_pre}, interictal={total_inter})")
        if total_win > 0:
            print(f"    Preictal ratio: {100*total_pre/total_win:.1f}%")

        # 逐病人详情
        for pid in pids:
            s = stats.get(pid, {})
            print(f"      {pid}: {s.get('n_windows', 0)} win, "
                  f"pre={s.get('n_preictal', 0)}, "
                  f"inter={s.get('n_interictal', 0)}")

    if split.get('excluded'):
        print(f"\n  EXCLUDED: {split['excluded']}")

    # 检查 chb01/chb21
    for p1, p2 in SAME_PATIENT:
        for set_name in ['train', 'val', 'test']:
            has_p1 = p1 in split[set_name]
            has_p2 = p2 in split[set_name]
            if has_p1 and has_p2:
                print(f"\n  ✓ {p1} and {p2} are both in {set_name} (same patient)")
            elif has_p1 or has_p2:
                print(f"\n  ✗ WARNING: {p1} and {p2} are in DIFFERENT splits!")


def main():
    parser = argparse.ArgumentParser(description="Patient-level split")
    parser.add_argument("--processed-dir", required=True,
                        help="预处理数据目录")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录 (默认: <processed-dir>/../splits)")
    parser.add_argument("--n-train", type=int, default=18)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir) if args.out_dir else processed_dir.parent / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading patient statistics...")
    stats = load_patient_stats(processed_dir)
    print(f"  Found {len(stats)} patients with processed data")

    print("\nCreating split...")
    split = create_split(stats, args.n_train, args.n_val, args.n_test, args.seed)

    print_split(split, stats)

    # 保存
    output = {
        'seed': args.seed,
        'split': split,
        'patient_stats': stats,
        'same_patient_constraint': SAME_PATIENT,
    }

    out_path = out_dir / "patient_split.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()