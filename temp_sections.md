## 11. UNKNOWN-POSE REJECTION



Standard classifiers always choose one of their known classes, so an explicit rejection rule is required.



- Accept a prediction only if **both** hold: `top1_score ≥ min_confidence` **and** `(top1 − top2) ≥ min_margin`.

- **Tune thresholds on out-of-fold validation predictions**, not on the test set. Provide a sweep that reports the *accuracy-versus-coverage* trade-off (how many frames are accepted vs. how accurate the accepted ones are). Choose the smallest threshold reaching a target accepted-precision (default 0.95) with a minimum coverage; store the chosen values in the model artifact.

- If the user recorded **evaluation-only negative samples** (poses outside the trained classes, stored with a reserved internal label and never used to fit the classifier), also report the **rejection rate on negatives** and false-reject rate on valid poses.

- Make clear in code comments and docs that Random Forest "probabilities" are vote fractions and SVM probabilities are Platt-scaled approximations; **neither is a guaranteed calibrated probability of correctness.** Optionally support `CalibratedClassifierCV` when enough data exists, and document the trade-off.

- Show the confidence in the UI as "model score", not "probability of being correct".



---



## 12. SMOOTHING AND STABILITY



- **Temporal voting:** majority vote across the last `vote_window` accepted frames; require at least `min_votes` to publish a label.

- **Hysteresis:** switch the displayed label only when the new label has held the majority for `min_votes` frames *and* the previous label's votes have fallen below the new one. Prevents rapid flicker.

- **State debouncing:** require `no_person_debounce_frames` consecutive misses before showing `No Person Detected`, so a single dropped detection does not flash the UI.

- **Optional landmark smoothing** (`none | ema | one_euro`): off by default. Document that more smoothing reduces jitter but adds latency; expose the parameters and test that the filter does not diverge.

- **Do not smooth during training-data capture**; samples must reflect the raw validated features.

- Unit-test: flicker suppression, switching latency bound, and state debouncing.



---



## 13. LIVE INFERENCE STATE MACHINE



Implement `runtime/state.py` as an explicit, testable state machine evaluated every frame, in this priority order:



1. Camera failure → `CAMERA_UNAVAILABLE`

2. No landmarks (after debounce) → `NO_PERSON`

3. Quality gate failed → `LOW_QUALITY` (with the specific reasons)

4. No model loaded → `NO_MODEL` (still show skeleton and quality)

5. Prediction rejected → `UNKNOWN_POSE`

6. Otherwise → `CONFIDENT` with the smoothed label and score



Never run the classifier on frames that fail the quality gate. Never show a pose label together with `NO_PERSON` or `LOW_QUALITY`.



`runtime/perf.py`: report rolling FPS (exponential moving average), and per-stage latency (capture, detection, feature extraction, classification, drawing). Log a summary every `log_interval_s` seconds and flag when FPS falls below `target_fps`.



---


