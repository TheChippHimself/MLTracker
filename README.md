# PoseSense

**Real-time human pose recognition and custom pose trainer** using MediaPipe and scikit-learn.

PoseSense captures live video from your webcam, detects body landmarks with MediaPipe's pretrained PoseLandmarker, converts them into a normalized feature vector (coordinates + joint angles + relational features), and classifies your pose using a custom scikit-learn classifier that **you train on your own data**.

## Features

- 🎥 Real-time webcam pose tracking at ≥15 FPS on CPU
- 🦴 33-landmark skeleton overlay with quality indicators
- 🏋️ Custom pose training: create labels, collect samples, train, and deploy
- 📊 Proper ML evaluation: session-aware splits, per-class metrics, confusion matrix
- 🎯 Unknown-pose rejection with tuned thresholds
- 🔒 Privacy-first: all processing is local, no video stored by default

## Quick Start

### Prerequisites

- Python 3.12 (MediaPipe requires Python 3.9–3.12)
- A webcam (for live mode; not required for training/evaluation)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd posesense

# Create a virtual environment with Python 3.12
uv venv --python 3.12
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv sync

# Download the MediaPipe model
python -m posesense download-model --variant lite

# Verify the environment
python -m posesense check
```

### Usage

```bash
# Run live pose recognition (OpenCV HUD)
python -m posesense live

# Launch the full GUI
python -m posesense gui

# Collect training data
python -m posesense collect --label "T-Pose" --burst 100

# Train a classifier
python -m posesense train --model rf --activate

# View evaluation results
python -m posesense evaluate --run latest

# Dataset operations
python -m posesense dataset stats
```

## Target Pose Classes

| Pose | Description |
|------|-------------|
| Standing | Neutral upright, arms relaxed |
| Sitting | Hips and knees bent, seated |
| Arms Raised | Both arms above head |
| T-Pose | Arms extended sideways at shoulder height |
| Arms Crossed | Forearms crossing the torso |
| Hands on Hips | Both hands near the hips |
| One Arm Raised | Exactly one arm lifted |
| Squatting | Knees and hips deeply bent |

Custom labels can be created through the GUI.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

```
FrameSource → PoseDetector → QualityGate → FeatureExtractor → PoseClassifier
 (webcam)      (MediaPipe)    (validate)    (normalize+angles)   (RF/SVM)
                                                                    │
              UI Overlay ← StateMachine ← Smoother ← RejectionRule ┘
```

## Requirements

- Python 3.12
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.9.0
- scikit-learn ≥ 1.4.0
- NumPy, pandas, matplotlib, joblib

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Dataset Guide](docs/DATASET_GUIDE.md)
- [Evaluation](docs/EVALUATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Future Work](docs/FUTURE_WORK.md)

## Limitations

- Single-person detection only
- Static poses only (no movement/sequence recognition)
- `Sitting` and `Squatting` depend on lower-body visibility and camera angle
- MediaPipe's visibility scores are hints, not guarantees
- Classifier scores are NOT calibrated probabilities

## License

[MIT](LICENSE)

to run, 
.venv\Scripts\activate
python -m posesense osc --ip 127.0.0.1 --port 39539
