# Future Work

These features are **not implemented** but have design hooks in the codebase.

## Level 3 Features

### Movement Recognition
- Sliding window of feature vectors feeding a sequence model
- Design hook: `SequenceClassifier` protocol and `FeatureWindow` buffer
- Example: detecting "waving" or "sit-to-stand" transitions

### Multi-Person Detection
- Per-person pose results with a simple tracker
- Separate smoothers per person
- Design hook: `num_poses` config parameter, deterministic person selection

### Advanced Settings Panel
- Full configuration UI for all TOML parameters
- Design hook: `settings_view.py` with config binding

### Multi-Classifier Comparison Dashboard
- Side-by-side comparison of different model types and hyperparameters
- Design hook: estimator factory pattern in `ml/models.py`

### Alternative Classifiers
- Gradient boosting, small neural network
- Design hook: estimator factory pattern, same `predict_proba` interface

### Alternative Front-Ends
- PySide6, web interface
- Design hook: UI-agnostic `controller.py` separates business logic from presentation
