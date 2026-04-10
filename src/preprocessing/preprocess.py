
import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import mne
import warnings
mne.set_log_level("ERROR")


STANDARD_18 = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]

DEFAULTS = {
    'bandpass_low': 0.5,
    'bandpass_high': 50.0,
    'notch_freq': 60.0,       # US power line
    'preictal_sec': 1800,     # 30 min before seizure
    'postictal_sec': 300,     # 5 min after seizure (excluded)
    'window_sec': 30,         # 30s windows
    'stride_sec': 10,         # 10s stride (overlapping)
    'sfreq': 256,
}



def parse_summary(path):
    seizures = defaultdict(list)
    if not os.path.exists(path):
        return dict(seizures)

    with open(path, 'r', errors='replace') as f:
        text = f.read()

    current_file = None
    start = None

    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'File\s*Name:\s*(\S+)', line)
        if m:
            current_file = m.group(1)
            start = None
            continue
        m = re.match(r'Seizure\s*\d*\s*Start\s*Time:\s*(\d+)\s*seconds', line)
        if m and current_file:
            start = int(m.group(1))
            continue
        m = re.match(r'Seizure\s*\d*\s*End\s*Time:\s*(\d+)\s*seconds', line)
        if m and current_file and start is not None:
            seizures[current_file].append((start, int(m.group(1))))
            start = None

    return dict(seizures)


def match_channels(raw_ch_names):
    norm_to_raw = {}
    for ch in raw_ch_names:
        clean = ch.strip()
        norm = clean.replace(' ', '').upper()

        parts = norm.split('-')
        if len(parts) == 3 and parts[-1].isdigit():
            norm_base = parts[0] + '-' + parts[1]
        else:
            norm_base = norm

        if norm_base not in norm_to_raw:
            norm_to_raw[norm_base] = clean

    mapping = {}
    for std_ch in STANDARD_18:
        norm = std_ch.replace(' ', '').upper()
        if norm in norm_to_raw:
            mapping[std_ch] = norm_to_raw[norm]

    if len(mapping) < 18:
        return None

    return mapping


def load_and_filter(filepath, ch_mapping, cfg):

    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

    pick_names = [ch_mapping[std] for std in STANDARD_18]
    raw.pick_channels(pick_names)
    raw.reorder_channels(pick_names)

    sfreq = raw.info['sfreq']

    data = raw.get_data()
    del raw
    import gc; gc.collect()

    data = data.astype(np.float32)

    sos = butter(4, [cfg['bandpass_low'], cfg['bandpass_high']],
                 btype='band', fs=sfreq, output='sos')
    data = sosfiltfilt(sos, data, axis=1).astype(np.float32)

    b, a = iirnotch(cfg['notch_freq'], Q=30, fs=sfreq)
    data = filtfilt(b, a, data, axis=1).astype(np.float32)

    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std

    return data, sfreq


def create_labels(n_samples, sfreq, seizures, cfg):

    labels = np.zeros(n_samples, dtype=np.int8)

    for sz_start, sz_end in seizures:
        s0 = int(sz_start * sfreq)
        s1 = int(sz_end * sfreq)

        labels[max(0, s0):min(n_samples, s1)] = -1

        post_end = int((sz_end + cfg['postictal_sec']) * sfreq)
        labels[min(n_samples, s1):min(n_samples, post_end)] = -1

        pre_start = int((sz_start - cfg['preictal_sec']) * sfreq)
        pre_start = max(0, pre_start)
        mask = labels[pre_start:s0] == 0
        labels[pre_start:s0] = np.where(mask, 1, labels[pre_start:s0])

    return labels


def segment_windows(data, labels, sfreq, cfg):

    win_samples = int(cfg['window_sec'] * sfreq)
    stride_samples = int(cfg['stride_sec'] * sfreq)
    n_samples = data.shape[1]

    windows = []
    wlabels = []

    for start in range(0, n_samples - win_samples + 1, stride_samples):
        end = start + win_samples
        seg_labels = labels[start:end]

        if np.any(seg_labels == -1):
            continue
        label = 1 if np.mean(seg_labels == 1) > 0.5 else 0

        windows.append(data[:, start:end])
        wlabels.append(label)

    if not windows:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    return (np.array(windows, dtype=np.float32),
            np.array(wlabels, dtype=np.int64))



def process_patient(pid, raw_dir, out_dir, excluded_files, cfg):
    import gc

    pdir = Path(raw_dir) / pid
    odir = Path(out_dir) / pid
    odir.mkdir(parents=True, exist_ok=True)

    if (odir / "windows.npy").exists() and (odir / "labels.npy").exists():
        print(f"  [SKIP] {pid}: already processed")
        return pid, 0, 0, 0

    summaries = [s for s in pdir.rglob("*summary*")
                 if not s.name.startswith('._')]
    seizure_map = parse_summary(str(summaries[0])) if summaries else {}

    edfs = sorted(f for f in pdir.rglob("*.edf")
                  if not f.name.startswith('._'))

    # First pass: process each file and save to temp files
    files_ok = 0
    files_skip = 0
    total_pre = 0
    total_inter = 0
    temp_files = []  # (tmp_w_path, tmp_l_path, n_windows)
    win_samples = int(cfg['window_sec'] * cfg['sfreq'])

    for edf in edfs:
        rel = f"{pid}/{edf.name}"
        if rel in excluded_files:
            files_skip += 1
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
            ch_mapping = match_channels(raw.ch_names)
            del raw

            if ch_mapping is None:
                files_skip += 1
                continue

            data, sfreq = load_and_filter(str(edf), ch_mapping, cfg)

            file_seizures = seizure_map.get(edf.name, [])
            labels = create_labels(data.shape[1], sfreq, file_seizures, cfg)

            windows, wlabels = segment_windows(data, labels, sfreq, cfg)

            del data, labels
            gc.collect()

            if len(windows) > 0:
                tmp_w = odir / f"_tmp_{edf.stem}_w.npy"
                tmp_l = odir / f"_tmp_{edf.stem}_l.npy"
                np.save(tmp_w, windows)
                np.save(tmp_l, wlabels)

                n_w = len(wlabels)
                n_pre = int(np.sum(wlabels == 1))
                n_inter = int(np.sum(wlabels == 0))
                total_pre += n_pre
                total_inter += n_inter
                temp_files.append((tmp_w, tmp_l, n_w))
                files_ok += 1

                if file_seizures:
                    print(f"    {edf.name}: {n_w} win "
                          f"(pre={n_pre}, inter={n_inter}, "
                          f"sz={len(file_seizures)})")

                del windows, wlabels
            else:
                files_ok += 1

            gc.collect()

        except Exception as e:
            print(f"    [ERR] {edf.name}: {str(e)[:100]}")
            files_skip += 1
            gc.collect()

    if temp_files:
        total_n = sum(n for _, _, n in temp_files)

        out_w = np.lib.format.open_memmap(
            str(odir / "windows.npy"), mode='w+',
            dtype=np.float32, shape=(total_n, 18, win_samples))
        out_l = np.lib.format.open_memmap(
            str(odir / "labels.npy"), mode='w+',
            dtype=np.int64, shape=(total_n,))

        offset = 0
        for tmp_w, tmp_l, n_w in temp_files:
            w = np.load(tmp_w)
            l = np.load(tmp_l)
            out_w[offset:offset + n_w] = w
            out_l[offset:offset + n_w] = l
            offset += n_w
            del w, l
            gc.collect()

            tmp_w.unlink()
            tmp_l.unlink()

        del out_w, out_l
        gc.collect()

        print(f"  [{pid}] {total_n} windows (pre={total_pre}, "
              f"inter={total_inter}, ratio={total_pre/max(1,total_n):.3f}), "
              f"{files_ok} files, {files_skip} skipped")

        meta = {
            'patient_id': pid,
            'n_windows': total_n,
            'n_preictal': total_pre,
            'n_interictal': total_inter,
            'preictal_ratio': round(total_pre / max(1, total_n), 4),
            'files_processed': files_ok,
            'files_skipped': files_skip,
            'config': cfg,
        }
        with open(odir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        return pid, total_n, total_pre, total_inter
    else:
        print(f"  [{pid}] No windows produced! ({files_ok} files ok, {files_skip} skipped)")
        return pid, 0, 0, 0



def main():
    parser = argparse.ArgumentParser(description="Preprocess CHB-MIT EEG data")
    parser.add_argument("--raw-dir", required=True,
                        help="Raw data directory (e.g. /Volumes/T9/data/raw)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory (e.g. /Volumes/T9/data/processed)")
    parser.add_argument("--exclusion-list", default=None,
                        help="Exclusion list JSON (default: <raw-dir>/../reports/exclusion_list.json)")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Only process specified patients")
    parser.add_argument("--preictal-sec", type=int, default=1800,
                        help="Preictal window seconds (default: 1800=30min)")
    parser.add_argument("--window-sec", type=int, default=30,
                        help="Window length in seconds (default: 30)")
    parser.add_argument("--stride-sec", type=int, default=10,
                        help="Window stride in seconds (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing of existing results")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg['preictal_sec'] = args.preictal_sec
    cfg['window_sec'] = args.window_sec
    cfg['stride_sec'] = args.stride_sec

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excl_path = args.exclusion_list
    if excl_path is None:
        excl_path = raw_dir.parent / "reports" / "exclusion_list.json"
    excluded_files = set()
    if Path(excl_path).exists():
        with open(excl_path) as f:
            excl = json.load(f)
        excluded_files = set(excl.get('excluded_files', []))
        print(f"Loaded exclusion list: {len(excluded_files)} files excluded")
    else:
        print(f"Warning: no exclusion list at {excl_path}")

    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = sorted(
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and d.name.startswith("chb")
        )

    print("=" * 60)
    print(f"PREPROCESS: {len(patient_ids)} patients")
    print(f"  Raw:    {raw_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Window: {cfg['window_sec']}s, stride: {cfg['stride_sec']}s")
    print(f"  Preictal: {cfg['preictal_sec']}s ({cfg['preictal_sec']//60}min)")
    print(f"  Filter: {cfg['bandpass_low']}-{cfg['bandpass_high']}Hz, "
          f"notch={cfg['notch_freq']}Hz")
    print("=" * 60)

    if args.force:
        for pid in patient_ids:
            odir = out_dir / pid
            for f in ['windows.npy', 'labels.npy', 'meta.json']:
                p = odir / f
                if p.exists():
                    p.unlink()

    results = []
    for pid in patient_ids:
        print(f"\n  Processing {pid}...")
        result = process_patient(pid, raw_dir, out_dir, excluded_files, cfg)
        results.append(result)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    total_win = 0
    total_pre = 0
    total_inter = 0
    for pid, n_win, n_pre, n_inter in results:
        total_win += n_win
        total_pre += n_pre
        total_inter += n_inter
        if n_win > 0:
            bar_pre = "█" * min(n_pre // 10, 30)
            print(f"  {pid}: {n_win:6d} windows, "
                  f"pre={n_pre:5d}, inter={n_inter:6d} {bar_pre}")

    print(f"\nTOTAL: {total_win} windows")
    print(f"  Preictal:    {total_pre} ({100*total_pre/max(1,total_win):.1f}%)")
    print(f"  Interictal:  {total_inter} ({100*total_inter/max(1,total_win):.1f}%)")
    print(f"  Imbalance:   1:{total_inter//max(1,total_pre)}")

    summary = {
        'total_windows': total_win,
        'total_preictal': total_pre,
        'total_interictal': total_inter,
        'imbalance_ratio': round(total_inter / max(1, total_pre), 1),
        'config': cfg,
        'patients': {pid: {'windows': w, 'preictal': p, 'interictal': i}
                     for pid, w, p, i in results},
    }
    with open(out_dir / "preprocessing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  -> {out_dir / 'preprocessing_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()