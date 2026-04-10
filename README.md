# EEG Seizure Prediction — CHB-MIT

## Setup

```bash
pip install mne pyedflib pyyaml scipy torch torchvision scikit-learn matplotlib
```

## Step 1: Download Data (~42 GB)

```bash
pip install awscli
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ /Volumes/T9/data/raw/  # Change to your download path
```

## Step 2: Inspect & Clean

```bash
python data_pipeline.py inspect --data-dir /Volumes/T9/data/raw  # Your download path
python data_pipeline.py clean --data-dir /Volumes/T9/data/raw  # Your download path
```

Results: 24 patients, 683/686 files usable, 198 seizures, 983h recording. 3 files excluded (chb12_27/28/29, non-standard montage, 13 seizures lost).

## Step 3: Preprocess

```bash
python preprocess.py \
  --raw-dir /Volumes/T9/data/raw \   # Your download path
  --out-dir /Volumes/T9/data/processed \  # Your download path
  --stride-sec 30
```

## Step 4: Split

```bash
python split.py --processed-dir /Volumes/T9/data/processed
```

## Step 5: Train (TODO)



## Step 6: Evaluate (TODO)


---

## Preprocessing Results

Window: 30s, stride: 30s (no overlap), filter: 0.5–50 Hz + 60 Hz notch, 18 standard bipolar channels.

| Patient | Windows | Preictal | Interictal | Ratio |
|---------|---------|----------|------------|-------|
| chb01 | 4,772 | 324 | 4,448 | 6.8% |
| chb02 | 4,196 | 124 | 4,072 | 3.0% |
| chb03 | 4,472 | 287 | 4,185 | 6.4% |
| chb04 | 18,663 | 230 | 18,433 | 1.2% |
| chb05 | 4,607 | 228 | 4,379 | 4.9% |
| chb06 | 7,897 | 498 | 7,399 | 6.3% |
| chb07 | 8,002 | 179 | 7,823 | 2.2% |
| chb08 | 2,315 | 298 | 2,017 | 12.9% |
| chb09 | 8,088 | 237 | 7,851 | 2.9% |
| chb10 | 5,911 | 402 | 5,509 | 6.8% |
| chb11 | 4,114 | 116 | 3,998 | 2.8% |
| chb12 | 2,226 | 484 | 1,742 | 21.7% |
| chb13 | 3,817 | 369 | 3,448 | 9.7% |
| chb14 | 3,026 | 413 | 2,613 | 13.6% |
| chb15 | 4,530 | 577 | 3,953 | 12.7% |
| chb16 | 2,168 | 323 | 1,845 | 14.9% |
| chb17 | 2,477 | 178 | 2,299 | 7.2% |
| chb18 | 4,214 | 270 | 3,944 | 6.4% |
| chb19 | 3,557 | 128 | 3,429 | 3.6% |
| chb20 | 3,214 | 284 | 2,930 | 8.8% |
| chb21 | 3,888 | 220 | 3,668 | 5.7% |
| chb22 | 3,685 | 161 | 3,524 | 4.4% |
| chb23 | 3,093 | 314 | 2,779 | 10.2% |
| chb24 | 2,398 | 629 | 1,769 | 26.2% |
| **Total** | **115,330** | **7,273 (6.3%)** | **108,057 (93.7%)** | **1:14** |

## Seizure Distribution

```
chb12:  40 ██████████████████████████████
chb15:  20 ████████████████████
chb24:  16 ████████████████
chb13:  12 ████████████
chb06:  10 ██████████
chb16:  10 ██████████
chb14:   8 ████████
chb20:   8 ████████
chb01:   7 ███████
chb03:   7 ███████
chb10:   7 ███████
chb23:   7 ███████
chb18:   6 ██████
chb05:   5 █████
chb08:   5 █████
chb04:   4 ████
chb09:   4 ████
chb21:   4 ████
chb02:   3 ███
chb07:   3 ███
chb11:   3 ███
chb17:   3 ███
chb19:   3 ███
chb22:   3 ███
```