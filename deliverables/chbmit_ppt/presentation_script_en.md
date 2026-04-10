# CHB-MIT Interim Team Update

We first audited the available repositories and confirmed that the earliest codebase behaved more like seizure detection than strict future seizure prediction.

We then adopted a prediction-oriented preprocessing pipeline on the full CHB-MIT dataset, using 18 bipolar channels, 30-second windows, and a preictal definition of the 30 minutes before seizure onset.

In the first round of deep learning experiments, we found that the original patient split created a serious distribution mismatch between train and validation, so we repaired the split before tuning the model itself.

Under the repaired patient-level setting, a 1D CNN baseline already moved beyond random behavior. We then upgraded to CNN+LSTM and reduced sequence overlap, which produced the strongest held-out test result so far: AUROC 0.797, AUPRC 0.189, and F1 0.189.

Finally, we calibrated the tuned CNN+LSTM with temperature scaling. Ranking stayed unchanged, but test ECE dropped from 0.0497 to 0.0133, which means the output probabilities are now much more trustworthy.

On top of that, we added a first uncertainty quantification stage with MC Dropout. Predictive entropy is currently the strongest uncertainty score, and it aligns well with future model errors, which means the model now has a useful error-awareness layer.

At the same time, the current selective three-state triage rule is not yet robust on unseen patients, so the correct framing is that we now have a strong calibrated predictor plus first-stage UQ, while Deep Ensemble remains the next major milestone.
