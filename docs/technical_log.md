# CHB-MIT Seizure Prediction Technical Log

## Purpose

This document is a living technical record for the CHB-MIT seizure prediction project.
It tracks:

- what we set out to build,
- which codebases and data pipelines were used,
- what problems we encountered,
- how we verified the root causes,
- what fixes were implemented,
- and what experimental results we obtained.

This file should be updated whenever we change preprocessing, splitting, model design, training strategy, calibration, uncertainty estimation, or evaluation.

## Project Goal

The target task is **seizure prediction**, not seizure detection.

Current task definition:

- Input: a 30-second EEG window with 18 bipolar channels.
- Label: `1 = preictal`, `0 = interictal`.
- Current preictal definition: the 30 minutes before seizure onset.
- Current exclusion rule: ictal and 5-minute postictal regions are excluded.

The model should ultimately output:

- a probability that seizure onset will occur in the target future window,
- and later an uncertainty estimate for that probability.

## Codebases Used

### 1. Original lightweight repository

Local repo: [E:\CHB-MIT](/E:/CHB-MIT)  
Original upstream repo: [tothemoon10080/CHB-MIT-data-preprocessing-and-prediction](https://github.com/tothemoon10080/CHB-MIT-data-preprocessing-and-prediction)

This repo was cloned locally and pushed to the user's fork:

- fork: [MLWhizKid/CHB-MIT](https://github.com/MLWhizKid/CHB-MIT)

Initial review showed that this repo was not suitable as the main prediction pipeline because it implemented seizure-period detection rather than future seizure prediction.

Key issues found:

- It trained a `DecisionTreeClassifier` instead of a deep model.
- It labeled windows as positive only when the current timestamp fell inside the seizure interval.
- It applied `SMOTE` before train/test split, creating data leakage.
- It treated channels from the same temporal window as separate samples, creating another leakage risk.

Relevant files:

- [train.py](/E:/CHB-MIT/train.py)
- [src/data/loaddata.py](/E:/CHB-MIT/src/data/loaddata.py)
- [src/data/extractFeture.py](/E:/CHB-MIT/src/data/extractFeture.py)

### 2. Prediction-oriented preprocessing repository

Reference repo cloned locally for inspection:

- local path: [E:\CHB-MIT\DeepLearning_ref](/E:/CHB-MIT/DeepLearning_ref)
- source: [jcck123/DeepLearning](https://github.com/jcck123/DeepLearning)

This repo contains the preprocessing and split logic that is much closer to the intended task.

Important files:

- [data_pipeline.py](/E:/CHB-MIT/DeepLearning_ref/data_pipeline.py)
- [preprocess.py](/E:/CHB-MIT/DeepLearning_ref/preprocess.py)
- [split.py](/E:/CHB-MIT/DeepLearning_ref/split.py)

Note:

- `train.py` and `test.py` in this repo were empty, so the actual deep learning training pipeline was implemented in this project repository instead.

## Environment and Storage Layout

### Environment

A dedicated Conda environment was created:

- environment name: `chb-mit-gpu`
- Python: `3.11`
- GPU stack: `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, `torchaudio 2.6.0+cu124`
- hardware verified: NVIDIA RTX 4080 Laptop GPU

Environment helper files added to this repo:

- [requirements.txt](/E:/CHB-MIT/requirements.txt)
- [setup_gpu_env.ps1](/E:/CHB-MIT/setup_gpu_env.ps1)
- [.gitignore](/E:/CHB-MIT/.gitignore)

### Storage paths

Raw and processed data were intentionally split across drives:

- raw dataset: `E:\CHB-MIT-Data\raw`
- processed windows: `D:\CHB-MIT-Data\processed`
- reports: `D:\CHB-MIT-Data\reports`
- splits: `D:\CHB-MIT-Data\splits`

This layout was chosen because the raw CHB-MIT download is about 42.6 GB, and processed windows can also occupy tens of GB even with a 30-second stride.

## Dataset Download and Preprocessing

### Raw download

The full CHB-MIT dataset was downloaded from PhysioNet using terminal commands.

### Cleaning and inspection

The pipeline in [data_pipeline.py](/E:/CHB-MIT/DeepLearning_ref/data_pipeline.py) was used to:

- inspect all patient folders,
- verify usable EDF files,
- identify excluded files,
- and enforce the `chb01/chb21` same-patient constraint.

### Preprocessing configuration

The current preprocessing came from [preprocess.py](/E:/CHB-MIT/DeepLearning_ref/preprocess.py) with the following settings:

- channels: 18 standard bipolar channels
- bandpass: `0.5-50 Hz`
- notch: `60 Hz`
- sampling rate: `256 Hz`
- window size: `30 seconds`
- stride: `30 seconds`
- preictal window: `30 minutes`
- postictal exclusion: `5 minutes`

This produced:

- total windows: `115,330`
- total preictal windows: `7,273`
- total interictal windows: `108,057`
- overall positive ratio: about `6.31%`

Example tensor shape for one patient:

- `windows.npy`: `(N, 18, 7680)`
- `labels.npy`: `(N,)`

where `7680 = 30 sec * 256 Hz`.

## Initial Deep Learning Training Pipeline

The first real deep learning trainer for this project was implemented in:

- [train_cnn.py](/E:/CHB-MIT/train_cnn.py)

Key properties of this trainer:

- model: `1D CNN`
- input: memmapped EEG windows from `windows.npy`
- output: one scalar logit per window
- probability output: `sigmoid(logit)`
- metrics: AUROC, AUPRC, Brier score, precision, recall, F1
- progress reporting: tqdm progress bars, ETA, average loss, learning rate, GPU memory

The initial model architecture was:

- multiple `Conv1d + BatchNorm + ReLU + MaxPool` blocks,
- adaptive average pooling,
- fully connected head,
- sigmoid probability at inference time.

## First Training Attempt: Problem Discovery

### What happened

The first training attempt used the original split generated by:

- [DeepLearning_ref/split.py](/E:/CHB-MIT/DeepLearning_ref/split.py)

Observed behavior:

- training loss decreased steadily,
- validation loss increased sharply,
- AUROC stayed close to random,
- AUPRC was only slightly above the random baseline,
- threshold values became unstable.

Example logged results from the original split run:

| Epoch | Train Loss | Val Loss | Val AUPRC | Val AUROC | Val F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0156 | 4.0823 | 0.1224 | 0.5026 | 0.2112 |
| 3 | 0.7511 | 12.3057 | 0.1419 | 0.5336 | 0.2096 |
| 7 | 0.6204 | 12.1278 | 0.1772 | 0.5767 | 0.2269 |

### Root cause analysis

The main issue was not a vague "model weakness". It was a concrete bug in the split strategy.

In the original split code:

- [DeepLearning_ref/split.py:78](/E:/CHB-MIT/DeepLearning_ref/split.py#L78) sorts patients by preictal count
- [DeepLearning_ref/split.py:91](/E:/CHB-MIT/DeepLearning_ref/split.py#L91) to [DeepLearning_ref/split.py:94](/E:/CHB-MIT/DeepLearning_ref/split.py#L94) then assigns:
  - patients with fewer positives to train,
  - medium positives to validation,
  - higher positives to test

This created a severe label distribution shift:

### Old split distribution

| Split | Windows | Preictal | Positive Ratio |
|---|---:|---:|---:|
| Train | 89,342 | 4,270 | 0.0478 |
| Val | 11,163 | 1,299 | 0.1164 |
| Test | 14,825 | 1,704 | 0.1149 |

Interpretation:

- the model trained on a world where positives were only `4.78%`,
- then was validated on a world where positives were about `11.6%`,
- so the training and validation distributions did not match.

This was identified as the primary confirmed cause of the poor and unstable validation behavior.

## Fixes Implemented

### 1. Balanced patient-level split

A new split generator was added:

- [make_balanced_split.py](/E:/CHB-MIT/make_balanced_split.py)

This script:

- preserves the `chb01/chb21` same-patient constraint,
- searches over valid patient-level splits,
- optimizes for matched positive ratios across train/val/test,
- and also keeps overall window volumes reasonably balanced.

Generated output:

- [patient_split_balanced.json](/D:/CHB-MIT-Data/splits/patient_split_balanced.json)

### New split distribution

| Split | Windows | Preictal | Positive Ratio |
|---|---:|---:|---:|
| Train | 86,596 | 5,445 | 0.0629 |
| Val | 14,220 | 904 | 0.0636 |
| Test | 14,514 | 924 | 0.0637 |

This is much closer to the overall dataset positive ratio of about `0.0631`.

### 2. Balanced sampling during training

The training loader in [train_cnn.py:328](/E:/CHB-MIT/train_cnn.py#L328) to [train_cnn.py:392](/E:/CHB-MIT/train_cnn.py#L392) was updated to support:

- `WeightedRandomSampler`
- balanced batch construction

This means the model now sees positives much more regularly during training.

### 3. More stable optimizer setup

The training defaults were also changed:

- learning rate lowered from `1e-3` to `3e-4`
- balanced sampling exposed via CLI
- memmap loading retained to avoid RAM overflow

## Results After Fixes

The repaired run was executed with:

- balanced split
- balanced sampling
- lower learning rate

Run artifacts:

- [E:\CHB-MIT\runs\cnn1d_balanced_v1](/E:/CHB-MIT/runs/cnn1d_balanced_v1)

Best recorded validation checkpoint:

- best epoch: `1`
- best threshold: `0.9302`

Final metrics from that best checkpoint:

### Validation

- AUROC: `0.7347`
- AUPRC: `0.1746`
- F1: `0.2895`
- Precision: `0.2535`
- Recall: `0.3374`

### Test

- AUROC: `0.7009`
- AUPRC: `0.1177`
- F1: `0.0698`
- Precision: `0.1505`
- Recall: `0.0455`

## Interpretation of Current Results

### Architecture vs prediction target

It is important to separate **label semantics** from **model architecture**.

The current model already performs **future-risk prediction** at the window level, even though it is only a `1D CNN`.

Reason:

- the current label is positive when the present 30-second window lies inside the 30-minute preictal period before seizure onset,
- so the model is learning whether seizure onset is expected within the upcoming 30 minutes,
- not whether seizure activity is happening right now.

Therefore:

- the current output is a **window-level future seizure risk score**,
- not just an "instantaneous ictal-state probability",
- and not yet a patient-level long-horizon probability trajectory.

Adding an LSTM would not change the forecast horizon by itself.
The horizon is determined by the label definition.

What LSTM would change is the **amount of historical temporal context** the model can use.

### Why LSTM is not required for future prediction

The current `1D CNN` sees one current 30-second window and predicts the probability that this window belongs to the preictal state.

This is equivalent to asking:

- "based on the current 30-second EEG segment, how likely is seizure onset within the next 30 minutes?"

So:

- **future prediction** is already happening,
- **LSTM** is an optional architecture upgrade for richer temporal modeling,
- it is not the mechanism that turns detection into prediction.

### What improved

Compared with the original split run:

- validation AUROC improved from roughly random (`~0.50-0.58`) to `0.7347`
- validation AUPRC improved relative to the new random baseline
- the model now clearly has nontrivial ranking ability

This means the model is now learning something real, rather than mostly reacting to split distortion.

### What is still weak

The current model is still not strong enough in thresholded classification:

- test F1 is low,
- test recall is low,
- the chosen threshold is very high (`0.9302`),
- and the raw probabilities are not yet calibrated.

So the model can rank some windows better than chance, but it is not yet a polished clinical-style probability predictor.

## Metric Notes

### Why AUPRC must be interpreted relative to the positive rate

For imbalanced data, the random baseline for AUPRC is approximately the positive class ratio.

Under the new balanced split:

- validation positive rate is about `0.0636`
- validation AUPRC is `0.1746`

So the model is not "barely above random" anymore.
It is about `2.7x` the random baseline on validation.

### Why AUROC and AUPRC tell different stories

- AUROC measures how well the model ranks positives above negatives overall.
- AUPRC is stricter for imbalanced problems and better reflects positive-class usefulness.

For seizure prediction, AUPRC is especially important because positives are rare.

## Commands Used for the Repaired Run

### Create balanced split

```powershell
conda activate chb-mit-gpu

python E:\CHB-MIT\make_balanced_split.py `
  --input-split D:\CHB-MIT-Data\splits\patient_split.json `
  --output-split D:\CHB-MIT-Data\splits\patient_split_balanced.json
```

### Train repaired 1D CNN

```powershell
conda activate chb-mit-gpu

python E:\CHB-MIT\train_cnn.py `
  --processed-dir "D:\CHB-MIT-Data\processed" `
  --split-file "D:\CHB-MIT-Data\splits\patient_split_balanced.json" `
  --out-dir "E:\CHB-MIT\runs\cnn1d_balanced_v1" `
  --epochs 8 `
  --batch-size 32 `
  --num-workers 0 `
  --balanced-sampling
```

## CNN + LSTM Upgrade

### Why add LSTM

The plain `1D CNN` baseline only sees one 30-second window at a time.

The new `CNN + LSTM` trainer upgrades this by feeding the model a **sequence of consecutive windows** and asking it to predict the label of the **last window** in that sequence.

With the current default:

- `seq_len = 10`
- `window_sec = 30`
- `stride = 30`

the model can use the previous `10 * 30 = 300 seconds = 5 minutes` of EEG context before making the probability prediction for the current window.

This means:

- the **forecast horizon** is still determined by the label definition,
- but the model now has a longer temporal history available when deciding whether the current window belongs to the preictal state.

### Implemented trainer

The new sequence trainer was added in:

- [train_cnn_lstm.py](/E:/CHB-MIT/train_cnn_lstm.py)

Key design choices:

- shared `CNN` encoder for each 30-second window
- `LSTM` over consecutive window embeddings
- prediction taken from the last time step
- memmapped loading, so windows are not loaded fully into RAM
- optional balanced sampling
- tqdm progress bars, ETA, GPU memory display
- optional `wandb` logging

### Important implementation note

The current processed dataset does **not** store explicit per-window file-boundary metadata.

So the first `CNN + LSTM` version builds sequences from consecutive saved windows **within each patient**.
This is still a valid and useful first sequence model, but it means sequence boundaries are based on processed ordering rather than an explicit EDF-boundary table.

This should be documented clearly in any report.

### W&B integration

`wandb` support has been added directly to [train_cnn_lstm.py](/E:/CHB-MIT/train_cnn_lstm.py).

Supported arguments include:

- `--wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-run-name`
- `--wandb-tags`
- `--wandb-mode`
- `--wandb-log-artifact`

Environment status:

- `wandb` is installed in `chb-mit-gpu`
- user still needs to run `wandb login` once before online sync
- current default project name in the trainer: `CHB-MIT`

Logged metric groups:

- `train/*`: epoch-level training metrics
- `val/*`: epoch-level validation metrics
- `best/*`: running best validation checkpoint metrics
- `final_test/*`: final test metrics logged once after training

Important evaluation design choice:

- `train`, `val`, and `best` are tracked across epochs as line charts
- `test` is **not** tracked every epoch to avoid test-set overfitting during model development
- only the final test metrics are logged once at the end

### Smoke-test verification

The new script was verified with a small local smoke test:

```powershell
conda activate chb-mit-gpu

python E:\CHB-MIT\train_cnn_lstm.py `
  --processed-dir "D:\CHB-MIT-Data\processed" `
  --split-file "D:\CHB-MIT-Data\splits\patient_split_balanced.json" `
  --out-dir "E:\CHB-MIT\runs\cnn_lstm_smoke" `
  --epochs 1 `
  --batch-size 2 `
  --num-workers 0 `
  --seq-len 10 `
  --balanced-sampling `
  --max-train-batches 2 `
  --max-val-batches 1 `
  --max-test-batches 1 `
  --wandb `
  --wandb-mode offline `
  --wandb-run-name "smoke-cnn-lstm"
```

Smoke-test result:

- script runs end to end
- sequence loading works
- `CNN + LSTM` forward pass works on GPU
- checkpoint saving works
- offline `wandb` logging works

The smoke-test metrics themselves are **not meaningful**, because the batch limits were too small and some validation/test subsets had no positive samples.

### Why one run used 8 epochs and not 12

There are now two different contexts:

- the repaired `1D CNN` diagnostic run used `8` epochs only to validate that the split fix actually worked
- the formal default training budget remains `12` epochs

So:

- `8 epochs` was a **fast debugging run**
- `12 epochs` is the **intended full training default**

The new `CNN + LSTM` script therefore defaults back to `12` epochs.

### Recommended command for the first real CNN + LSTM run

```powershell
conda activate chb-mit-gpu

wandb login

python E:\CHB-MIT\train_cnn_lstm.py `
  --processed-dir "D:\CHB-MIT-Data\processed" `
  --split-file "D:\CHB-MIT-Data\splits\patient_split_balanced.json" `
  --out-dir "E:\CHB-MIT\runs\cnn_lstm_v1" `
  --epochs 12 `
  --batch-size 4 `
  --num-workers 0 `
  --seq-len 10 `
  --seq-stride 1 `
  --balanced-sampling `
  --wandb `
  --wandb-project "CHB-MIT" `
  --wandb-run-name "cnn-lstm-seq10-v1"
```

### First full CNN + LSTM result

The first full `CNN + LSTM` run on the balanced split produced:

- best epoch: `1`
- early stopping epoch: `5`
- validation AUROC: `0.7460`
- validation AUPRC: `0.1359`
- test AUROC: `0.7480`
- test AUPRC: `0.1233`
- test F1: `0.1720`

Interpretation:

- the model reached its best generalization very early
- later epochs improved training metrics but degraded validation metrics
- this indicates early overfitting rather than a broken training loop

Compared with the repaired `1D CNN` baseline:

- validation AUROC improved
- validation AUPRC decreased
- test AUROC improved
- test AUPRC improved slightly
- test F1 improved substantially

So the first `CNN + LSTM` result is not uniformly better than `1D CNN`, but it is a meaningful upgrade candidate, especially on held-out test ranking and F1.

### W&B logging note

The `wandb` integration now logs:

- `train/*`
- `val/*`
- `best/*`
- `final_test/*`

and the final-test logging step was adjusted to remain monotonic, so W&B does not drop the final test metrics.

### Second tuned CNN + LSTM run

A second `CNN + LSTM` run was then launched with:

- `seq_len=6`
- `seq_stride=2`
- `lr=1e-4`
- `dropout=0.3`
- no `balanced-sampling`, using the original class distribution plus `pos_weight`

This run produced:

- best epoch: `2`
- validation AUROC: `0.7040`
- validation AUPRC: `0.1511`
- validation F1: `0.2668`
- test AUROC: `0.7972`
- test AUPRC: `0.1892`
- test F1: `0.1895`

Interpretation:

- validation AUROC decreased relative to the first `CNN + LSTM` run
- validation AUPRC and validation F1 improved
- held-out test AUROC improved substantially
- held-out test AUPRC improved substantially
- held-out test F1 also improved

This makes the second tuned `CNN + LSTM` run the strongest held-out deep-learning result so far.

## Probability Calibration

### What calibration means

Probability calibration does **not** change the ordering ability of the model.
It changes whether the numeric probability values are trustworthy.

Example:

- before calibration, a model may output `0.80` for samples that are actually positive only `0.20` of the time
- after calibration, a predicted `0.80` should correspond much more closely to an observed `80%` event frequency

This is different from uncertainty estimation:

- calibration asks: "is the predicted probability numerically reliable?"
- uncertainty asks: "how unsure is the model about this prediction?"

So calibration is **not** the same thing as `MC Dropout`.

### Temperature scaling vs Platt scaling

Temperature scaling requires **logits**.
It rescales the logits by a single scalar temperature before sigmoid.

Platt scaling fits a logistic regression mapping from model scores to calibrated probabilities.
When only probabilities are available, clipped probabilities can first be converted to logit-like scores and then calibrated with Platt scaling.

For the current `cnn_lstm_v2` run:

- only probabilities were available in the saved CSVs
- therefore the calibration script automatically used **Platt scaling**

### Implemented calibration script

Calibration was implemented in:

- [calibrate_predictions.py](/E:/CHB-MIT/calibrate_predictions.py)

The script:

- reads `val_predictions.csv` and `test_predictions.csv`
- fits the calibrator only on validation outputs
- applies calibration to validation and test probabilities
- saves calibrated CSVs
- saves a metrics JSON
- saves a reliability diagram

### Calibration result for cnn_lstm_v2

Command used:

```powershell
conda activate chb-mit-gpu

python E:\CHB-MIT\calibrate_predictions.py `
  --run-dir "E:\CHB-MIT\runs\cnn_lstm_v2" `
  --method auto
```

Output directory:

- [calibration_auto](/E:/CHB-MIT/runs/cnn_lstm_v2/calibration_auto)

Observed result:

- method used: `platt`
- AUROC unchanged
- AUPRC unchanged
- Brier improved strongly
- ECE improved strongly

Detailed numbers:

Original:

- validation Brier: `0.0979`
- validation ECE: `0.1011`
- test Brier: `0.0731`
- test ECE: `0.0707`

Calibrated:

- validation Brier: `0.0582`
- validation ECE: `0.0072`
- test Brier: `0.0584`
- test ECE: `0.0133`

Interpretation:

- model ranking stayed the same
- the probabilities became much more trustworthy numerically
- this is useful for later thresholding and uncertainty work

### Future support for temperature scaling

The `CNN + LSTM` trainer now saves a `logit` column in future prediction CSVs.
So future runs can use **temperature scaling** directly instead of falling back to Platt scaling.

### Calibration result for cnn_lstm_seq6_stride2_v1

Command used:

```powershell
conda activate chb-mit-gpu

python E:\CHB-MIT\calibrate_predictions.py `
  --run-dir "E:\CHB-MIT\runs\cnn_lstm_seq6_stride2_v1" `
  --method auto
```

Observed result:

- method used: `temperature`
- AUROC unchanged
- AUPRC unchanged
- Brier improved
- ECE improved strongly

Detailed numbers:

Original:

- validation Brier: `0.0589`
- validation ECE: `0.0411`
- test Brier: `0.0602`
- test ECE: `0.0497`

Calibrated:

- validation Brier: `0.0570`
- validation ECE: `0.0032`
- test Brier: `0.0565`
- test ECE: `0.0133`

Interpretation:

- ranking quality stayed the same
- probability reliability improved again
- this makes the tuned `CNN + LSTM` result both stronger on held-out test ranking and better calibrated numerically

## Current Status

Current best confirmed state:

- preprocessing: completed
- full dataset download: completed
- balanced patient split: completed
- first working deep model: completed
- probability output: completed
- root-cause repair of split issue: completed
- `CNN + LSTM` training script: completed
- `wandb` integration in `CNN + LSTM` trainer: completed
- first full `CNN + LSTM` result: completed
- probability calibration script: completed
- first calibrated `CNN + LSTM` result: completed
- tuned `CNN + LSTM` run with reduced overlap: completed
- calibrated tuned `CNN + LSTM` result: completed
- uncertainty estimation: not yet done

## Recommended Next Steps

1. Treat `cnn_lstm_seq6_stride2_v1` as the current main deep-learning baseline.
2. Compare calibrated `1D CNN` and calibrated tuned `CNN + LSTM`.
3. Decide whether further architecture tuning is still worth the cost.
4. If probability quality is stable, add uncertainty estimation.
5. Only after that consider `MC Dropout` or ensemble-based UQ.

## Presentation Script

### Honest but strong speaking version

The following short script is recommended for project reporting:

"We first audited the available repositories and confirmed that the earliest codebase behaved more like seizure detection than strict future seizure prediction. We then adopted a prediction-oriented preprocessing pipeline on the full CHB-MIT dataset, using 18 bipolar channels, 30-second windows, and a preictal definition of the 30 minutes before seizure onset.

In the first round of deep learning experiments, we found that the original patient split introduced a serious distribution mismatch between train and validation. So before tuning the model itself, we repaired the split and rebuilt the training pipeline around a balanced patient-level protocol.

After that fix, a 1D CNN baseline already moved clearly beyond random behavior. We then upgraded to CNN+LSTM to use longer temporal context, and after reducing sequence overlap and switching to a more realistic training distribution, the tuned CNN+LSTM achieved the strongest held-out test performance so far, with test AUROC 0.797 and test AUPRC 0.189.

Finally, we calibrated the tuned model with temperature scaling. This did not change ranking performance, but it significantly improved probability reliability: the calibrated test ECE dropped to 0.0133. So our current best result is not just a model that ranks risk better, but one that also outputs more trustworthy probabilities."

### Short closing line

"Our current baseline is therefore a calibrated CNN+LSTM under a strict patient-level split, which gives us a credible foundation for the next stage: uncertainty estimation."

## Presentation Deliverables

PPT deliverables created for this project:

- [CHB-MIT_Project_Report.pptx](/E:/CHB-MIT/deliverables/chbmit_ppt/CHB-MIT_Project_Report.pptx)
- [chbmit_report.js](/E:/CHB-MIT/deliverables/chbmit_ppt/chbmit_report.js)
- [summary.json](/E:/CHB-MIT/deliverables/chbmit_ppt/assets/summary.json)
- [montage.png](/E:/CHB-MIT/deliverables/chbmit_ppt/montage.png)

English interim-report deliverables:

- [CHB-MIT_Project_Report_EN.pptx](/E:/CHB-MIT/deliverables/chbmit_ppt/CHB-MIT_Project_Report_EN.pptx)
- [chbmit_report_en.js](/E:/CHB-MIT/deliverables/chbmit_ppt/chbmit_report_en.js)
- [presentation_script_en.md](/E:/CHB-MIT/deliverables/chbmit_ppt/presentation_script_en.md)
- [montage_en.png](/E:/CHB-MIT/deliverables/chbmit_ppt/montage_en.png)

Recommended framing for the English deck:

- It is an interim project update rather than the final project close-out.
- The strongest current result is the calibrated tuned `CNN + LSTM`.
- Uncertainty quantification is still pending, so the project should be described as strong-but-not-complete.

## UQ Next Step

Current UQ implementation status:

- an `MC Dropout` evaluation script has been added at [uq_mcdropout.py](/E:/CHB-MIT/uq_mcdropout.py)
- it reloads the current best `CNN + LSTM` checkpoint without retraining
- it keeps `BatchNorm` in evaluation mode and activates only dropout layers during stochastic inference
- it exports:
  - mean probability,
  - probability standard deviation,
  - predictive entropy,
  - expected entropy,
  - mutual information
- it also evaluates a first selective-prediction protocol by:
  - choosing the classification threshold on validation,
  - choosing uncertainty cutoffs on validation for several target coverages,
  - applying the same uncertainty cutoffs to test

Expected next deliverables after the first full run:

- `uq_metrics.json`
- `val_uq.csv`
- `test_uq.csv`
- `uq_summary.png`

Interpretation target:

- if uncertainty is meaningful, incorrect predictions should have higher uncertainty on average
- if selective prediction is useful, metrics on the retained high-confidence subset should improve as coverage decreases

### First Full MC Dropout Run

Run:

- checkpoint: [best_model.pt](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/best_model.pt)
- UQ output dir: [uq_mcdropout_mc20](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/uq_mcdropout_mc20)
- command setting: `mc_samples=20`, uncertainty score = `mutual_information`

Headline results:

- MC val AUROC: `0.7158`
- MC val AUPRC: `0.1612`
- MC test AUROC: `0.7854`
- MC test AUPRC: `0.1825`
- test error-detection AUROC: `0.7810`
- test error-detection AUPRC: `0.3019`

Interpretation:

- `MC Dropout` does not improve ranking compared with the deterministic tuned model; that is not its main purpose.
- The important signal is that test mean uncertainty is higher on incorrect predictions than on correct predictions:
  - correct: `0.00120`
  - incorrect: `0.00211`
- error-detection `AUROC=0.7810` means uncertainty is meaningfully aligned with model mistakes.

Selective prediction observation:

- under the current single-threshold setup, lowering coverage improves `Brier` and slightly improves retained `AUROC`
- however retained `AUPRC` and `F1` do not improve monotonically
- likely reason:
  - the most uncertain windows also contain a non-trivial share of hard positive examples
  - a fixed classification threshold is being reused while class balance changes as coverage shrinks

Current conclusion:

- the first UQ step is successful enough to keep
- the model now shows useful error-aware uncertainty
- the selective-prediction policy still needs refinement before it becomes a strong final reporting result

### Uncertainty-Score Comparison with Threshold Refit

Run:

- compare-all UQ output dir: [uq_mcdropout_mc20_compare](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/uq_mcdropout_mc20_compare)
- comparison plot: [uq_comparison.png](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/uq_mcdropout_mc20_compare/uq_comparison.png)
- command setting: `mc_samples=20`, uncertainty score = `all`

What was changed in this run:

- three uncertainty scores were compared:
  - `std_probs`
  - `predictive_entropy`
  - `mutual_information`
- for each target coverage, the classification threshold was re-fitted on the retained validation subset instead of being kept fixed

Test-set error-detection comparison:

- `predictive_entropy`: `AUROC=0.8616`, `AUPRC=0.4842`
- `std_probs`: `AUROC=0.8423`, `AUPRC=0.4433`
- `mutual_information`: `AUROC=0.7810`, `AUPRC=0.3019`

Interpretation:

- all three uncertainty scores are informative
- `predictive_entropy` is currently the strongest score for identifying future model errors
- `std_probs` is also strong and easier to explain
- `mutual_information` is weaker in this project than the other two scores, although it still outperforms chance

Selective-prediction result after threshold refit:

- re-fitting the classification threshold on the retained validation subset did not produce a better retained test `AUPRC` than full coverage
- for all three uncertainty scores, the best retained test `AUPRC` still occurred at `coverage=1.0`

Current interpretation:

- the uncertainty estimates are useful for error awareness
- they are not yet translating into a clearly better abstention policy for the main prediction metric
- therefore the current best next step is not more threshold tuning, but moving to the next UQ stage after documenting this result clearly

Working recommendation:

- use `predictive_entropy` as the primary uncertainty score for reporting
- keep `std_probs` as a secondary sanity-check score
- treat `mutual_information` as a supporting analysis rather than the main deployment score

### Three-State Triage Policy

Policy design:

- keep the underlying predictor as a binary `CNN + LSTM`
- do not retrain a separate 3-class model
- add a decision layer on top of `MC Dropout` outputs:
  - `Alert`: `mean_probability >= p_alert` and `predictive_entropy <= u_th`
  - `Review`: `mean_probability >= p_alert` and `predictive_entropy > u_th`
  - `No Alert`: `mean_probability < p_alert`

Implementation note:

- the probability threshold is kept at the calibrated `MC Dropout` decision threshold
- only the uncertainty threshold is tuned on validation
- this avoids the failure mode where selective prediction simply drops large numbers of samples

Run:

- output dir: [uq_mcdropout_mc20_triage](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/uq_mcdropout_mc20_triage)
- triage figure: [triage_policy.png](/E:/CHB-MIT/runs/cnn_lstm_seq6_stride2_v1/uq_mcdropout_mc20_triage/triage_policy.png)
- settings:
  - uncertainty score = `predictive_entropy`
  - `max_review_rate = 0.05`
  - `min_alert_recall_fraction = 0.5`

Validation selection result:

- selected uncertainty threshold: `0.5109`
- validation alert precision / recall / F1: `0.2221 / 0.3348 / 0.2670`
- validation review rate: `0.0051`

Test result:

- test alert precision / recall / F1: `0.2295 / 0.1822 / 0.2031`
- test alert / review / no-alert rates: `0.0505 / 0.0000 / 0.9495`

Interpretation:

- the triage policy is technically correct and reproducible
- however, on held-out test data it collapses back to the baseline decision rule:
  - no windows are routed to `Review`
  - `Alert` metrics match the base `MC Dropout` classifier
- this means the current `predictive_entropy` threshold does not create a useful deployment split across unseen patients

Current decision:

- keep the triage implementation as an analysis artifact
- do not claim this triage rule as a final performance-improving result
- the stronger current UQ result remains `error detection`, not `abstention`

## Update Policy

This document should be updated whenever any of the following changes happen:

- preprocessing changes,
- split changes,
- architecture changes,
- loss or sampling changes,
- calibration is added,
- uncertainty is added,
- new results replace current best results.
