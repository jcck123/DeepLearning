#!/usr/bin/env python3
"""

Usage:
  python data_pipeline.py download                         # 下载全部24个case
  python data_pipeline.py download --patients chb01 chb02  # 下载指定case
  python data_pipeline.py inspect                          # 检查所有文件
  python data_pipeline.py clean                            # 生成排除列表
  python data_pipeline.py all                              # 全部执行
"""
import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# Config
CHBMIT_URL = "https://physionet.org/files/chbmit/1.0.0/"
ALL_CASES = [f"chb{i:02d}" for i in range(1, 25)]

STANDARD_18 = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]
STANDARD_18_NORM = {ch.replace(' ', '').upper() for ch in STANDARD_18}

SAME_PATIENT = [('chb01', 'chb21')]
EXPECTED_SFREQ = 256
DATA_DIR = "data/raw"
REPORT_DIR = "data/reports"


# ── Step 1: Download ────────────────────────────────────────────────

def _list_files_from_index(index_url):
    """从PhysioNet目录索引页解析出文件名列表。"""
    import urllib.request
    import ssl
    from html.parser import HTMLParser

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.files = []
        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                for k, v in attrs:
                    if k == 'href' and v and not v.startswith('?') and not v.startswith('/'):
                        self.files.append(v)

    req = urllib.request.Request(index_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        html = resp.read().decode('utf-8', errors='replace')

    parser = LinkParser()
    parser.feed(html)
    return parser.files


def _download_file(url, dest_path, max_retries=3):
    """下载单个文件，失败自动重试，返回 (filename, success, detail)。"""
    import urllib.request
    import ssl
    import time

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.name, True, "skip"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=600, context=ctx) as resp:
                with open(dest, 'wb') as f:
                    while True:
                        chunk = resp.read(1024 * 512)
                        if not chunk:
                            break
                        f.write(chunk)
            mb = dest.stat().st_size / 1e6
            return dest.name, True, f"{mb:.1f}MB"
        except Exception as e:
            if dest.exists():
                dest.unlink()
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            return dest.name, False, str(e)[:80]


def download(data_dir=DATA_DIR, patients=None, workers=4):
    """
    从PhysioNet并发下载。先收集所有病人的文件列表，然后统一并发下载。
    比逐个病人下载快很多。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    patients = patients or ALL_CASES

    print("=" * 60)
    print(f"DOWNLOAD: {len(patients)} cases -> {out.resolve()}")
    print(f"  Workers: {workers}")
    print("=" * 60)

    # Phase 1: 收集所有需要下载的文件
    all_tasks = []  # (url, dest_path, patient_id)
    for pid in patients:
        pdir = out / pid

        if pdir.exists() and list(pdir.glob("*.edf")):
            n = len(list(pdir.glob("*.edf")))
            print(f"  [SKIP] {pid}: already has {n} .edf files")
            continue

        pdir.mkdir(parents=True, exist_ok=True)
        base_url = f"{CHBMIT_URL}{pid}/"

        try:
            files = _list_files_from_index(base_url)
            targets = [f for f in files
                       if f.endswith('.edf') or 'summary' in f.lower()]
            for fname in targets:
                all_tasks.append((base_url + fname, pdir / fname, pid))
            print(f"  {pid}: {len(targets)} files queued")
        except Exception as e:
            print(f"  [ERR] {pid}: cannot list files: {e}")

    if not all_tasks:
        print("  Nothing to download.")
        return

    total = len(all_tasks)
    # 预估：跳过已存在的
    to_download = sum(1 for _, dest, _ in all_tasks
                      if not (dest.exists() and dest.stat().st_size > 0))
    print(f"\n  Total: {total} files ({to_download} to download, "
          f"{total - to_download} already exist)")
    print(f"  Starting {workers}-thread download...\n")

    # Phase 2: 并发下载全部
    done = 0
    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_file, url, dest): (pid, dest.name)
            for url, dest, pid in all_tasks
        }
        for future in as_completed(futures):
            pid, fname = futures[future]
            name, success, detail = future.result()
            done += 1
            if success:
                ok += 1
                if detail != "skip":
                    print(f"  [{done}/{total}] {pid}/{name} ({detail})")
            else:
                failed += 1
                print(f"  [{done}/{total}] ERR {pid}/{name}: {detail}")

    # 统计
    print(f"\nDownload complete: {ok} ok, {failed} failed, {total - ok - failed} skipped")
    for pid in patients:
        pdir = out / pid
        if pdir.exists():
            n = len(list(pdir.glob("*.edf")))
            if n > 0:
                print(f"  {pid}: {n} EDF files")
    print()


# ── Step 2: Inspect ─────────────────────────────────────────────────

def parse_summary(path):
    """
    解析summary文件。
    兼容 "Seizure Start Time" / "Seizure 1 Start Time" / "Seizure1 Start Time"
    返回 {filename: [(start_s, end_s), ...]}
    """
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


def inspect_edf(filepath):
    """
    检查单个EDF文件，返回元数据dict。
    用mne读取（对CHB-MIT兼容性最好）。
    """
    info = {
        'filename': os.path.basename(filepath),
        'valid': False,
        'error': None,
    }

    try:
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

        ch_names = [ch.strip() for ch in raw.ch_names]
        sfreq = raw.info['sfreq']
        duration = raw.n_times / sfreq

        # 标准化通道名，匹配标准18通道
        ch_norm = {ch.replace(' ', '').replace('-', '-').upper() for ch in ch_names}
        present = ch_norm & STANDARD_18_NORM
        missing = STANDARD_18_NORM - ch_norm

        # 识别非EEG通道
        non_eeg = [ch for ch in ch_names
                   if ch.strip() in ('-', '--', '.', '')
                   or ch.strip().upper() in ('ECG', 'EKG', 'VNS')]

        info.update({
            'valid': True,
            'n_channels': len(ch_names),
            'channel_names': ch_names,
            'sfreq': sfreq,
            'duration_sec': round(duration, 1),
            'duration_h': round(duration / 3600, 2),
            'n_std18': len(present),
            'missing_std': sorted(missing),
            'has_all_18': len(missing) == 0,
            'non_eeg_channels': non_eeg,
            'size_mb': round(os.path.getsize(filepath) / 1e6, 1),
        })
    except Exception as e:
        info['error'] = str(e)[:200]

    return info


def inspect(data_dir=DATA_DIR):
    """检查所有case的所有EDF文件，打印报告。"""
    root = Path(data_dir)
    pdirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("chb"))

    if not pdirs:
        print(f"ERROR: no patient dirs in {data_dir}. Run download first.")
        return {}

    print("=" * 60)
    print(f"INSPECT: {len(pdirs)} cases in {root.resolve()}")
    print("=" * 60)

    reports = {}

    for pdir in pdirs:
        pid = pdir.name

        # 找summary文件（过滤macOS ._隐藏文件）
        summaries = [s for s in pdir.rglob("*summary*") if not s.name.startswith('._')]
        seizure_map = parse_summary(str(summaries[0])) if summaries else {}

        # 找所有edf（去重，过滤macOS ._隐藏文件）
        edfs_all = sorted(pdir.rglob("*.edf"))
        seen_names = set()
        edfs = []
        for e in edfs_all:
            # 跳过 macOS 元数据文件 (._xxx.edf) 和 .seizures 文件
            if e.name.startswith('._') or '.seizures' in e.name:
                continue
            if e.name not in seen_names:
                seen_names.add(e.name)
                edfs.append(e)

        files_info = []
        total_h = 0.0

        for edf in edfs:
            fi = inspect_edf(str(edf))
            fi['seizures'] = seizure_map.get(edf.name, [])
            fi['n_seizures'] = len(fi['seizures'])
            fi['file_issues'] = []

            if not fi['valid']:
                fi['file_issues'].append(f"CORRUPT: {fi['error']}")
            else:
                total_h += fi['duration_h']

                if fi['sfreq'] != EXPECTED_SFREQ:
                    fi['file_issues'].append(
                        f"SFREQ={fi['sfreq']}Hz (expect {EXPECTED_SFREQ})")

                if fi['n_std18'] < 14:
                    fi['file_issues'].append(
                        f"Only {fi['n_std18']}/18 standard channels")

                if fi['duration_sec'] < 60:
                    fi['file_issues'].append(
                        f"Too short: {fi['duration_sec']}s")

            files_info.append(fi)

        total_sz = sum(len(v) for v in seizure_map.values())
        n_valid = sum(1 for f in files_info if f['valid'] and not f['file_issues'])
        n_bad = sum(1 for f in files_info if f['file_issues'])

        report = {
            'patient_id': pid,
            'n_files': len(edfs),
            'n_valid': n_valid,
            'n_bad': n_bad,
            'total_seizures': total_sz,
            'total_hours': round(total_h, 1),
            'has_summary': bool(summaries),
            'files': files_info,
        }
        reports[pid] = report

        # 打印状态
        flag = "!!" if n_bad > 0 else "OK"
        print(f"  [{flag}] {pid}: {n_valid}/{len(edfs)} ok, "
              f"{total_sz} seizures, {total_h:.1f}h"
              + (f", {n_bad} problem files" if n_bad else ""))
        for fi in files_info:
            for iss in fi['file_issues']:
                print(f"       {fi['filename']}: {iss}")

    # 癫痫分布
    print(f"\n{'─' * 60}")
    print("SEIZURE DISTRIBUTION:")
    for pid in sorted(reports, key=lambda p: reports[p]['total_seizures'], reverse=True):
        r = reports[pid]
        bar = "█" * min(r['total_seizures'], 30)
        print(f"  {pid}: {r['total_seizures']:3d} {bar}")

    total_f = sum(r['n_files'] for r in reports.values())
    total_ok = sum(r['n_valid'] for r in reports.values())
    total_sz = sum(r['total_seizures'] for r in reports.values())
    total_h = sum(r['total_hours'] for r in reports.values())
    print(f"\nTOTAL: {len(reports)} cases, {total_ok}/{total_f} files ok, "
          f"{total_sz} seizures, {total_h:.0f}h recording")

    return reports


# ── Step 3: Clean ───────────────────────────────────────────────────

def clean(data_dir=DATA_DIR, report_dir=REPORT_DIR, reports=None):
    """分析检查结果，生成排除列表。"""
    if reports is None:
        reports = inspect(data_dir)
    if not reports:
        return

    print(f"\n{'=' * 60}")
    print("CLEAN: generating exclusion list")
    print("=" * 60)

    excluded_files = []
    patient_stats = {}
    sz_lost = 0

    for pid, rep in reports.items():
        good = 0
        for fi in rep['files']:
            if fi['file_issues']:
                excluded_files.append({
                    'patient': pid,
                    'filename': fi['filename'],
                    'reasons': fi['file_issues'],
                    'seizures_lost': fi['n_seizures'],
                })
                sz_lost += fi['n_seizures']
            elif fi['valid']:
                good += 1

        patient_stats[pid] = {
            'usable_files': good,
            'total_seizures': rep['total_seizures'],
            'total_hours': rep['total_hours'],
        }

    excluded_patients = [pid for pid, s in patient_stats.items()
                         if s['usable_files'] == 0]

    # 构造输出
    result = {
        'generated': datetime.now().isoformat(),
        'same_patient_pairs': SAME_PATIENT,
        'excluded_patients': excluded_patients,
        'excluded_files': [f"{e['patient']}/{e['filename']}" for e in excluded_files],
        'excluded_details': excluded_files,
        'patient_stats': patient_stats,
        'summary': {
            'total_cases': len(reports),
            'usable_cases': len(reports) - len(excluded_patients),
            'total_files': sum(r['n_files'] for r in reports.values()),
            'excluded_file_count': len(excluded_files),
            'seizures_lost': sz_lost,
            'total_seizures': sum(r['total_seizures'] for r in reports.values()),
        },
    }

    # print
    s = result['summary']
    print(f"  Cases:    {s['usable_cases']}/{s['total_cases']} usable")
    print(f"  Files:    {s['total_files'] - s['excluded_file_count']}/{s['total_files']} usable")
    print(f"  Seizures: {s['total_seizures']} total, {s['seizures_lost']} in excluded files")
    if excluded_patients:
        print(f"  Excluded patients: {excluded_patients}")
    if excluded_files:
        print(f"\n  Excluded files ({len(excluded_files)}):")
        for e in excluded_files:
            reasons = "; ".join(e['reasons'])
            sz_tag = f" [LOSES {e['seizures_lost']} sz!]" if e['seizures_lost'] else ""
            print(f"    {e['patient']}/{e['filename']}: {reasons}{sz_tag}")
    print(f"\n  REMINDER: chb01 == chb21 (同一病人，必须在同一个split!)")

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "exclusion_list.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  -> {out / 'exclusion_list.json'}")

    slim = {}
    for pid, rep in reports.items():
        slim[pid] = {
            'patient_id': pid,
            'n_files': rep['n_files'],
            'n_valid': rep['n_valid'],
            'total_seizures': rep['total_seizures'],
            'total_hours': rep['total_hours'],
            'files': [{
                'filename': fi['filename'],
                'valid': fi['valid'],
                'sfreq': fi.get('sfreq'),
                'duration_sec': fi.get('duration_sec'),
                'n_channels': fi.get('n_channels'),
                'n_std18': fi.get('n_std18'),
                'has_all_18': fi.get('has_all_18'),
                'n_seizures': fi.get('n_seizures', 0),
                'seizures': fi.get('seizures', []),
                'non_eeg_channels': fi.get('non_eeg_channels', []),
                'file_issues': fi.get('file_issues', []),
                'size_mb': fi.get('size_mb'),
            } for fi in rep['files']],
        }

    with open(out / "inspection_report.json", 'w') as f:
        json.dump(slim, f, indent=2, default=str)
    print(f"  -> {out / 'inspection_report.json'}")

    return result




def main():
    parser = argparse.ArgumentParser(description="CHB-MIT data pipeline")
    parser.add_argument("step", choices=["download", "inspect", "clean", "all"])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="并行下载线程数 (default: 4)")
    args = parser.parse_args()

    if args.step in ("download", "all"):
        download(args.data_dir, args.patients, workers=args.workers)

    reports = None
    if args.step in ("inspect", "all"):
        reports = inspect(args.data_dir)

    if args.step in ("clean", "all"):
        clean(args.data_dir, args.report_dir, reports)

    print("\nDone.")


if __name__ == "__main__":
    main()