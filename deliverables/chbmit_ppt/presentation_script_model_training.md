# Model + Training Talk Plan

## Goal

This script is for the **90-second model/training segment only**.
The target is not to say everything.
The target is to show:

- the architecture was chosen for a reason,
- the training pipeline was engineered carefully,
- and the final model was selected through evidence rather than guesswork.

## 90-second timing plan

### 0-10 s: open with the task

Suggested line:

"My part focused on the model and training pipeline. The task is not seizure detection, but future seizure-risk prediction from EEG windows."

### 10-40 s: Slide 1, model rationale

Suggested line:

"Our final model is a CNN plus LSTM. Each 30-second EEG window is first encoded by a 1D CNN to capture local waveform and spectral patterns. Then an LSTM integrates six consecutive windows, so the model uses about three minutes of temporal context before predicting the probability that the current window belongs to the preictal state, meaning seizure risk in the upcoming target horizon."

What to emphasize:

- why CNN: local EEG morphology
- why LSTM: temporal evolution across windows
- why probability output: useful risk score, not just a hard label

### 40-75 s: Slide 2, training strategy

Suggested line:

"On the training side, the biggest issue was not only model choice. We first fixed the patient split because the original train and validation sets had mismatched positive ratios, which made evaluation unstable. Then we trained on the real class distribution with `pos_weight`, used early stopping and learning-rate scheduling, and compared several candidates rather than assuming a larger model would be better."

### 75-90 s: results and decision

Suggested line:

"The strongest final model was the tuned and calibrated CNN plus LSTM. On held-out test data it achieved AUROC 0.797 and AUPRC 0.189, clearly outperforming the CNN baseline. We also tested a Deep Ensemble, but it did not improve the final trade-off, so we kept the calibrated CNN plus LSTM as the final training/model choice."

## Short Q&A backup lines

### Why not just use a CNN?

"The CNN baseline was useful, but adding the LSTM improved temporal modeling across consecutive windows and gave the strongest held-out result."

### Why not keep the Deep Ensemble?

"We tested it, but on held-out data it did not beat the tuned single model and it also did not produce a useful review class, so it was not the best final choice."

### Why talk about calibration?

"Because this project outputs a probability, not just a class label. Calibration makes that probability numerically more trustworthy."

## Delivery advice

- Do not read the slide text word for word.
- Point once to the architecture flow on Slide 1.
- Point once to the green row on Slide 2.
- End with the model-selection decision, because examiners reward justified decisions.
