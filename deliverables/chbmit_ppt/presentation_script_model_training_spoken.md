# Model + Training Spoken Script

## 90-second spoken version aligned to the 2-slide deck

### Slide 1

Hi, I was mainly responsible for the model and training pipeline.

For the model design, our input is six consecutive 30-second EEG windows, with 18 bipolar channels, and the forecasting target is the next 30 minutes. We chose a CNN plus LSTM because the two parts do different jobs. The 1D CNN learns local EEG morphology from each 30-second window, while the LSTM models temporal progression across the six-window sequence, which gives the model about three minutes of context.

The head outputs a probability, not just a hard label. So instead of making a purely deterministic yes-or-no decision, the model produces a future-risk probability that we can later calibrate and interpret.

So our final design decision was to use the tuned CNN plus LSTM as the main model, because it captures temporal evolution better than a pure CNN while still being trainable on our GPU and dataset size.

### Slide 2

For training, the first thing we fixed was the patient split, because we found a train-validation ratio mismatch. After that, we trained on the real class distribution using `pos_weight` instead of synthetic balancing.

In the training loop, we used early stopping, learning-rate scheduling, and memmap loading, which helped make the pipeline stable and practical on our hardware.

We then compared four stages on held-out patients: a 1D CNN baseline, a CNN plus LSTM with longer context, a tuned CNN plus LSTM, and a Deep Ensemble. The tuned CNN plus LSTM gave the best overall result, with test AUROC 0.797, AUPRC 0.189, and F1 0.189.

For probabilistic output, we also calibrated the final model, and the test ECE dropped from 0.0497 to 0.0133. So the final model we selected was the tuned and calibrated CNN plus LSTM, because it gave the strongest balance of ranking quality, classification performance, and probability reliability.

## Shorter emergency version

Hi, I worked on the model and training pipeline. Our final model is a CNN plus LSTM: the CNN learns local EEG patterns from each 30-second window, and the LSTM adds temporal context across six windows. On the training side, we first fixed the patient split, then trained with the real class distribution, early stopping, and learning-rate scheduling. We compared several candidates, including Deep Ensemble, and finally selected the tuned and calibrated CNN plus LSTM because it gave the strongest held-out performance and the most reliable probabilities.

## Likely Q&A

### 1. Why did you choose CNN + LSTM instead of only CNN?

The CNN baseline was useful, but it only sees one short window at a time. The LSTM lets the model use temporal evolution across consecutive windows, which is more appropriate for seizure prediction because preictal changes can develop gradually rather than appearing in a single isolated segment.

### 2. Why did you not keep the Deep Ensemble as the final model?

We implemented and tested it, but on held-out data it did not improve the final overall trade-off. In particular, it did not outperform the tuned single CNN + LSTM on the main metrics we cared about, and it also did not produce a strong enough three-state decision policy. So we kept the simpler model that generalized better.

### 3. Why is the output a probability instead of a hard label?

Because the task is risk prediction. A probability is more informative than a yes-or-no label, and it also allows calibration and uncertainty analysis later. That makes the system more useful than a purely deterministic classifier.

### 4. What was the biggest training problem you had to solve?

The biggest problem was not just overfitting. It was that the original patient split created a mismatch between the positive ratio in train and validation. That made evaluation unstable. Once we fixed the split, the results became much more meaningful.

### 5. Why did you use `pos_weight` instead of just oversampling everything?

Because we wanted training to stay closer to the real class distribution. `pos_weight` handles class imbalance at the loss level without changing the underlying data distribution too aggressively, which gave us a more stable and more realistic training setup.

### 6. Why do you say calibration matters?

Because a model can rank samples well but still output unreliable probabilities. Calibration helps make the predicted probability numerically trustworthy, so if the model says a sample is high risk, that score is more meaningful.

### 7. What result are you most confident in?

The result I am most confident in is the tuned and calibrated CNN plus LSTM under the balanced patient-level split, because that is the strongest held-out result and also the most defensible training setup.

### 8. If you had more time, what would you improve next?

I would first strengthen uncertainty quantification and then revisit multi-model strategies only if they improve held-out performance in a meaningful way. I would prioritize reliability and generalization over simply making the architecture bigger.

## Delivery tips

- Speak slightly slower than you think you need to.
- Do not read every sentence on the slide.
- Point once to the model flow on Slide 1 and once to the green row on Slide 2.
- End with the model-selection decision, because that makes your part sound technically mature.
