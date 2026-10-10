# Architecture

> Detailed architecture documentation will be added as modules are implemented.

## Pipeline

```
FrameSource ──► PoseDetector ──► QualityGate ──► FeatureExtractor ──► PoseClassifier
 (webcam/file)   (MediaPipe)    (validate)       (normalize+angles)    (RF / SVM)
                                                                          │
                                                                          ▼
                 UI Overlay ◄── StateMachine ◄── Smoother ◄── RejectionRule
```

## Key Design Decisions

### Separation of concerns
- **Pose estimation** (MediaPipe) is separate from **pose classification** (scikit-learn)
- UI code contains no ML logic; ML code contains no UI code
- Camera code contains no classification code

### Dependency inversion
Core modules depend on abstract interfaces (`Protocol`) for camera, pose detector, and classifier.

### Feature schema as single source of truth
One module (`features/schema.py`) defines feature names, order, and version. All consumers import from it.

### Threading model
- Capture/inference worker thread pushes results into a `Queue(maxsize=1)`
- UI thread (Tk main loop) polls via `after()` at ~30ms
- Training runs in a background thread with progress reporting

## Module Map

| Package | Responsibility |
|---------|---------------|
| `camera/` | Frame capture abstraction and implementations |
| `pose/` | Pose detection, landmark constants, quality validation |
| `features/` | Feature schema, normalization, angles, relational features |
| `data/` | Dataset management, labels, validation, session-aware splits |
| `ml/` | Classifiers, training, evaluation, model artifacts |
| `runtime/` | Pipeline orchestration, smoothing, state machine, performance |
| `ui/` | Tkinter application, views, overlay drawing |
