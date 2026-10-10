<USER_REQUEST>
# PoseSense — Master Implementation Prompt for the AI Coding Agent

> Paste this entire document into your IDE's AI agent (Cursor, Claude Code, GitHub Copilot Agent, Windsurf, etc.), or save it in the repository root as `PROMPT.md` and tell the agent: *"Read PROMPT.md fully before doing anything, then follow it."*

---

## 0. ROLE, CONTEXT AND OPERATING MODE

You are a **senior machine learning engineer and software architect** with deep experience in computer vision, real-time Python applications, and production-quality ML tooling. You are building **PoseSense**, a real-time human pose recognition and custom pose trainer, as a complete, well-tested, well-documented project.

### 0.1 How you must work

1. **Read this entire prompt before writing any code.** Then produce a short implementation plan (maximum one page) that restates the goal, lists the phases, and flags any ambiguity or risk you see. Wait for my confirmation only if you identify a *blocking* ambiguity; otherwise proceed.
2. **Work phase by phase** (Section 21). Do not start a phase until the previous phase's acceptance criteria are met and verified. At the end of each phase, stop and give a report in the format defined in Section 25.
3. **Prefer the simplest correct design.** Do not over-engineer. Do not add features outside the current phase. Do not introduce dependencies that are not listed in Section 4 without justifying them in writing.
4. **Verify, don't assume.** MediaPipe, OpenCV, and scikit-learn APIs change between versions. Before using any API, check the installed version and the official documentation. Never invent function names, arguments, or file URLs. If you are unsure, say so and verify.
5. **Every module must be testable without a webcam.** Hardware access is isolated behind interfaces so that all core logic can be tested with fake or synthetic inputs.
6. **Be honest about results.** Never fabricate metrics, never hard-code accuracy numbers, and never describe a feature as complete if it has not been run and tested. If something fails, report it plainly with the error output and your proposed fix.
7. **Ask at most one clarifying question at a time**, and only when the answer materially changes the design.

### 0.2 Definition of quality

Your output is judged against a professional standard: readable, typed, documented, tested, reproducible, and safe. The finished project must be something a university examiner, a hiring manager, or a teammate can clone, install, run, and understand in under 15 minutes.

---

## 1. PROJECT MISSION AND SUCCESS CRITERIA

### 1.1 Mission

Build **PoseSense**, a desktop application and Python package that:

1. Captures live video from a webcam.
2. Detects 33 body landmarks per frame using a **pretrained MediaPipe Pose Landmarker** model.
3. Converts landmarks into a **normalized, versioned feature vector** (coordinates plus joint angles plus relational features).
4. Classifies the person's pose with a **custom scikit-learn classifier trained on the user's own collected data**.
5. Displays the skeleton, the predicted pose, a confidence indicator, camera/detection status, and performance metrics in real time.
6. Lets the user **create labels, collect samples, retrain, evaluate, and deploy a new model without editing source code**.

### 1.2 Key conceptual separation (must be reflected in the architecture)

- **Pose estimation** (where are the body landmarks?) is handled by MediaPipe's pretrained model. We do **not** train this.
- **Pose classification** (which pose is this?) is handled by **our own** classifier trained on landmark-derived features. This is the machine learning contribution of the project.

### 1.3 Target pose classes (initial eight static poses)

| # | Label | Description |
|---|-------|-------------|
| 1 | `Standing` | Neutral upright position, arms relaxed |
| 2 | `Sitting` | Hips and knees bent, seated |
| 3 | `Arms Raised` | Both arms above the head |
| 4 | `T-Pose` | Both arms extended sideways at shoulder height |
| 5 | `Arms Crossed` | Forearms crossing the torso |
| 6 | `Hands on Hips` | Both hands near the hips |
| 7 | `One Arm Raised` | Exactly one arm lifted (left or right) |
| 8 | `Squatting` | Knees and hips deeply bent |

### 1.4 Runtime states that are NOT training classes

These are decisions made by the application, not labels in the dataset:

| State | Source |
|-------|--------|
| `Camera Unavailable` | Camera layer |
| `No Person Detected` | Pose detector returned no landmarks |
| `Low Quality / Partial Body` | Landmark validation failed (missing or low-visibility required landmarks) |
| `Unknown Pose` | Classifier evidence is below the validated rejection threshold |
| `No Model Loaded` | No trained model is available |

### 1.5 Measurable success criteria

The project is successful only if **all** of the following are demonstrably true:

- [ ] The application runs end to end on a standard laptop CPU (no GPU) at **≥ 15 FPS** at 640×480 with the default model settings (report the actual measured value and hardware).
- [ ] A user can create a new label, capture samples, train, and see live predictions using only the GUI.
- [ ] Evaluation is performed on **held-out recording sessions**, never on random adjacent frames, and the report includes accuracy, per-class precision/recall/F1, macro-F1, and a confusion matrix.
- [ ] The system shows `No Person Detected`, `Low Quality / Partial Body`, and `Unknown Pose` correctly in tests and in live use.
- [ ] Camera failures are handled without crashing and the camera is always released on exit.
- [ ] No video or images are stored unless the user explicitly enables it.
- [ ] Core (non-UI) test coverage is **≥ 80%** and the full test suite passes without a webcam.
- [ ] Documentation allows a new user to install and run the project from scratch.

---

## 2. SCOPE

### 2.1 In scope (Levels 1 and 2)

**Level 1 — MVP**

1. Real-time webcam tracking with clear camera/detection status.
2. Skeleton overlay with optional debug info (landmark index, coordinates, visibility).
3. Pose classification with predicted label displayed.
4. Custom training-data collection with per-class sample counts, undo, delete, and clear.
5. Classifier training (Random Forest first, SVM as a comparison).
6. Prediction confidence and `Unknown Pose` rejection.

**Level 2 — Proper ML project features**

7. Dataset manager (view, count, relabel, delete, export, import).
8. Automatic landmark normalization.
9. Joint-angle and relational features.
10. Proper train/validation/test evaluation split by recording session.
11. Evaluation report (accuracy, precision, recall, F1, confusion matrix).
12. Model saving and loading with metadata and compatibility checks.
13. Prediction smoothing (temporal voting and hysteresis).
14. Detection-quality check.
15. Performance monitor (FPS, per-stage latency).
16. Optional session logging.

### 2.2 Out of scope for implementation (design hooks only)

Level 3 features (movement/sequence recognition, multi-person detection, advanced settings panel, multi-classifier comparison dashboard) must **not be implemented**. Leave clean extension points and document them (Section 22).

---

## 3. NON-NEGOTIABLE ENGINEERING PRINCIPLES

1. **Separation of concerns.** UI code contains no ML logic. ML code contains no UI code. Camera code contains no classification code.
2. **Dependency inversion.** Core modules depend on abstract interfaces (`Protocol` or ABC) for camera, pose detector, and classifier.
3. **Single source of truth for the feature schema.** One module defines feature names, order, and version. The dataset, trainer, and inference engine all import from it. Never duplicate feature ordering.
4. **Reproducibility.** Fixed random seeds, pinned dependency versions, deterministic splits, and metadata saved with every model.
5. **Fail loudly and safely.** Invalid inputs raise typed exceptions inside the core; the application layer catches them and shows a clear message. No bare `except:`. No silently swallowed errors.
6. **No hidden global state.** Configuration is passed explicitly. Singletons are avoided.
7. **No `print` for diagnostics.** Use the `logging` module with structured, leveled messages.
8. **Atomic file writes.** Datasets, models, and configuration are written to a temporary file and then renamed, so a crash cannot corrupt existing data.
9. **Privacy by default.** Process locally, store numbers not pictures, and make any recording opt-in and visible.
10. **Resource hygiene.** Every resource (camera, MediaPipe landmarker, file handles, threads) is released through context managers and `finally` blocks.

---

## 4. TECHNOLOGY STACK

### 4.1 Required

| Technology | Purpose |
|------------|---------|
| **Python 3.11 or 3.12** | Main language. Verify the Python versions currently supported by the MediaPipe wheel before locking this. `tomllib` (3.11+) is used for configuration. |
| **MediaPipe (Tasks API — `PoseLandmarker`)** | Pretrained 33-landmark pose estimation. **Use the Tasks API, not the legacy `mp.solutions.pose` API.** |
| **OpenCV (`opencv-python`)** | Camera capture, frame conversion, drawing |
| **NumPy** | Numerical computation |
| **Pandas** | Dataset handling and CSV I/O |
| **scikit-learn** | Classifiers, pipelines, metrics, model selection |
| **Joblib** | Model serialization |
| **Matplotlib** | Saving evaluation figures (confusion matrix, charts) |
| **Tkinter** (standard library) | GUI. Keep it behind a thin presentation layer so it can later be replaced by PySide6. |
| **pytest, pytest-cov** | Testing |
| **ruff** | Linting and formatting |
| **mypy** | Static type checking |
| **pre-commit** | Enforce quality checks before commits |

### 4.2 Dependency management

- Use `pyproject.toml` (PEP 621) as the single project definition. Provide a **pinned** `requirements.lock` or equivalent generated from a tested environment.
- Provide a virtual-environment setup guide for Windows, macOS, and Linux.
- Record the exact versions of Python, MediaPipe, OpenCV, NumPy, pandas, scikit-learn, and joblib in every saved model's metadata.

### 4.3 MediaPipe model file

- The `PoseLandmarker` requires a `.task` model bundle (lite, full, or heavy variants exist).
- Create `scripts/download_model.py` that downloads the model into `models/mediapipe/`. Check the **official MediaPipe Pose Landmarker documentation** for the current model URLs and **verify the URL before hard-coding it**. The documented pattern is of the form `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_<variant>/float16/latest/pose_landmarker_<variant>.task`.
- The script must: skip the download if the file exists, check the file is non-empty after download, and print a clear error with manual-download instructions if the network is unavailable.
- **Never commit the `.task` file to Git.** Add it to `.gitignore`.
- At application start, verify the model file exists (typed `ModelFileNotFoundError` with fix instructions if missing).
- Default variant: `lite` (best CPU speed). Make the variant configurable (`lite | full | heavy`).

### 4.4 MediaPipe running mode

- Implement the detector behind a `PoseDetector` interface.
- **Default implementation:** `RunningMode.VIDEO` using `detect_for_video(image, timestamp_ms)` called from the processing thread. Timestamps **must be strictly monotonically increasing integers in milliseconds**; generate them from a monotonic clock (`time.monotonic_ns() // 1_000_000`) and enforce `ts > last_ts` (increment by 1 if equal).
- **Optional second implementation:** `RunningMode.LIVE_STREAM` with a result callback, for experimentation. Document that live-stream mode is asynchronous, requires correct timestamps, and may drop frames when it cannot keep up.
- **Also provide `RunningMode.IMAGE`** support for offline processing of still images and for unit/integration tests.
- Convert frames from OpenCV **BGR to RGB** before creating `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`. Add a test asserting the conversion occurs.

---

## 5. REPOSITORY LAYOUT

Create exactly this structure (add files only when needed by a phase):

```
posesense/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.lock
├── CHANGELOG.md
├── config/
│   └── default.toml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   ├── DATASET_GUIDE.md
│   ├── EVALUATION.md
│   ├── TROUBLESHOOTING.md
│   └── FUTURE_WORK.md
├── scripts/
│   ├── download_model.py
│   └── check_environment.py
├── src/
│   └── posesense/
│       ├── __init__.py
│       ├── __main__.py              # `python -m posesense` entry
│       ├── cli.py                   # command-line interface
│       ├── config.py                # typed configuration loading/validation
│       ├── exceptions.py            # all custom exception types
│       ├── logging_setup.py
│       ├── types.py                 # shared dataclasses and enums
│       ├── camera/
│       │   ├── __init__.py
│       │   ├── base.py              # FrameSource protocol
│       │   ├── webcam.py            # OpenCV webcam implementation
│       │   ├── video_file.py        # file/folder source for tests & offline use
│       │   └── discovery.py         # enumerate available cameras
│       ├── pose/
│       │   ├── __init__.py
│       │   ├── base.py              # PoseDetector protocol
│       │   ├── mediapipe_detector.py
│       │   ├── landmarks.py         # landmark indices, names, skeleton connections
│       │   └── quality.py           # landmark validation and quality scoring
│       ├── features/
│       │   ├── __init__.py
│       │   ├── schema.py            # feature names, order, version (single source of truth)
│       │   ├── normalize.py         # centering, scaling, optional mirroring
│       │   ├── angles.py            # joint-angle math
│       │   ├── relational.py        # wrist/shoulder/hip relational features
│       │   └── extractor.py         # landmarks -> validated feature vector
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py           # DatasetManager (CRUD, stats, import/export)
│       │   ├── labels.py            # label registry
│       │   ├── validation.py        # dataset health checks
│       │   └── splits.py            # session-aware splitting
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── models.py            # estimator factories (RF, SVM)
│       │   ├── trainer.py           # training orchestration
│       │   ├── evaluator.py         # metrics, confusion matrix, reports
│       │   ├── rejection.py         # unknown-pose rejection + threshold tuning
│       │   ├── artifact.py          # save/load model with metadata & compatibility checks
│       │   └── inference.py         # PoseClassifier (features -> Prediction)
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── pipeline.py          # frame -> result orchestration
│       │   ├── smoothing.py         # temporal voting, hysteresis, optional landmark filter
│       │   ├── state.py             # application state machine
│       │   ├── perf.py              # FPS and latency monitor
│       │   └── session_log.py       # optional prediction logging
│       └── ui/
│           ├── __init__.py
│           ├── app.py               # Tk main window
│           ├── controller.py        # UI-agnostic controller (testable)
│           ├── views/
│           │   ├── live_view.py
│           │   ├── collect_view.py
│           │   ├── train_view.py
│           │   ├── evaluate_view.py
│           │   └── settings_view.py
│           ├── overlay.py           # drawing skeleton/HUD on frames
│           └── workers.py           # background threads (capture, training)
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── synthetic_poses.py       # synthetic landmark generators
│   ├── unit/
│   ├── integration/
│   └── hardware/                    # tests requiring a webcam (skipped by default)
├── data/                            # git-ignored; user data lives here
│   ├── datasets/
│   ├── logs/
│   └── labels.json
├── models/                          # git-ignored
│   └── mediapipe/
└── artifacts/                       # git-ignored; training runs and reports
    └── runs/
```

---

## 6. ARCHITECTURE AND DATA CONTRACTS

### 6.1 Pipeline

```
FrameSource ──► PoseDetector ──► QualityGate ──► FeatureExtractor ──► PoseClassifier
 (webcam/file)   (MediaPipe)    (validate)       (normalize+angles)    (RF / SVM)
                                                                          │
                                                                          ▼
                 UI Overlay ◄── StateMachine ◄── Smoother ◄── RejectionRule
```

### 6.2 Shared types (`types.py`)

Define immutable (`frozen=True`, `slots=True`) dataclasses and enums:

```python
class AppState(Enum):
    CAMERA_UNAVAILABLE = "Camera Unavailable"
    NO_PERSON = "No Person Detected"
    LOW_QUALITY = "Low Quality / Partial Body"
    NO_MODEL = "No Model Loaded"
    UNKNOWN_POSE = "Unknown Pose"
    CONFIDENT = "Confident"

@dataclass(frozen=True, slots=True)
class Landmark:
    x: float; y: float; z: float
    visibility: float
    presence: float

@dataclass(frozen=True, slots=True)
class PoseResult:
    landmarks: tuple[Landmark, ...]        # 33 image-space (normalized) landmarks
    world_landmarks: tuple[Landmark, ...] | None   # 33 world-space landmarks if available
    timestamp_ms: int
    frame_width: int
    frame_height: int

@dataclass(frozen=True, slots=True)
class QualityReport:
    ok: bool
    score: float                           # min visibility across required landmarks
    missing: tuple[int, ...]               # landmark indices that failed
    reasons: tuple[str, ...]               # human-readable reasons

@dataclass(frozen=True, slots=True)
class Prediction:
    label: str | None
    confidence: float | None               # classifier score, NOT a calibrated probability
    margin: float | None                   # top-1 minus top-2 score
    all_scores: Mapping[str, float]
    accepted: bool                         # passed rejection rule
```

Use type hints everywhere; the project must pass `mypy --strict` for the `features`, `data`, `ml`, and `runtime` packages.

### 6.3 Threading model (for the GUI)

- **Capture/inference worker thread**: reads frames, runs detection, features, classification, smoothing. Pushes the *latest* result into a `queue.Queue(maxsize=1)` using a "drop oldest" policy so the UI never lags behind.
- **UI thread (Tk main loop)**: polls the queue via `after()` at ~30 ms and draws. **Never call Tk methods from worker threads.**
- **Training worker thread**: runs training in the background, reporting progress through a thread-safe queue; the UI stays responsive.
- Provide a `threading.Event` for clean shutdown and join all threads on exit with a timeout.

---

## 7. MODULE SPECIFICATIONS

### 7.1 `config.py` and `config/default.toml`

All tunable values live in `config/default.toml` and are loaded into typed, validated dataclasses. Invalid values raise `ConfigError` with the offending key. Include at minimum:

```toml
[camera]
index = 0
width = 640
height = 480
backend = "auto"            # auto | dshow | msmf | v4l2 | avfoundation
mirror_display = true       # display only; features always use the un-mirrored frame
reconnect_attempts = 3
reconnect_backoff_s = 1.0
black_frame_threshold = 3.0 # mean pixel intensity below this counts as "black"
black_frame_max_consecutive = 30

[detector]
model_variant = "lite"      # lite | full | heavy
model_path = "models/mediapipe/pose_landmarker_lite.task"
running_mode = "video"      # video | live_stream | image
num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

[quality]
min_visibility = 0.5
min_presence = 0.5
edge_margin = 0.02          # landmarks closer than this to the frame edge count as clipped
min_torso_length_norm = 0.08
required_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

[features]
schema_version = 1
coordinate_source = "image" # image | world
use_z = false
center = "mid_hip"
scale = "torso_length"      # torso_length | shoulder_width
rotate_to_torso = false
mirror_augmentation = false # training only

[smoothing]
landmark_filter = "none"    # none | ema | one_euro
ema_alpha = 0.5
vote_window = 7
min_votes = 4
no_person_debounce_frames = 5

[rejection]
min_confidence = 0.60       # placeholder; tuned and stored per model
min_margin = 0.15           # placeholder; tuned and stored per model

[training]
random_seed = 42
default_model = "random_forest"
test_session_fraction = 0.2
min_samples_per_class = 30
imbalance_warning_ratio = 3.0

[performance]
target_fps = 15
log_interval_s = 5

[privacy]
record_video = false
log_predictions = false
```

### 7.2 `camera/`

**`FrameSource` protocol:** `open()`, `read() -> Frame`, `release()`, context-manager support, and an `is_opened` property.

**`WebcamSource`:**
- Open via `cv2.VideoCapture(index, backend)`; always check `isOpened()` and raise `CameraOpenError` otherwise.
- Apply requested width/height and **read back the actual values** (cameras may ignore the request); log both.
- `read()` must raise `FrameReadError` when `ret` is false or the frame is `None`/empty.
- **Black-frame detection:** if mean intensity stays below the threshold for N consecutive frames, raise `BlackFrameWarning` (a non-fatal condition surfaced as a UI message).
- **Controlled recovery:** on repeated read failures, release and reopen with exponential backoff up to `reconnect_attempts`; if still failing, transition the app to `CAMERA_UNAVAILABLE` and stop the loop cleanly instead of crashing.
- `release()` must be idempotent and safe to call from `finally`/`atexit`.
- Handle OS permission denial: when open fails, include platform-specific guidance (Windows camera privacy settings, macOS System Settings → Privacy & Security → Camera, Linux device permissions/`video` group).

**`discovery.py`:** probe indices 0–N (default N=5) quickly and return those that open and return a frame; always release each probe. Must not hang the UI (run in a worker thread).

**`VideoFileSource` / image-folder source:** same protocol, used for deterministic offline tests and for the optional "collect from recorded video" feature.

### 7.3 `pose/`

**`landmarks.py`** — constants for the 33-landmark MediaPipe topology. At minimum:

| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | nose | 23 | left_hip |
| 11 | left_shoulder | 24 | right_hip |
| 12 | right_shoulder | 25 | left_knee |
| 13 | left_elbow | 26 | right_knee |
| 14 | right_elbow | 27 | left_ankle |
| 15 | left_wrist | 28 | right_ankle |
| 16 | right_wrist | | |

Provide the full 33-name list and the skeleton connection list for drawing. Use an `IntEnum` (e.g., `LM.LEFT_SHOULDER`) so the rest of the code never uses magic numbers.

**`MediaPipeDetector`:**
- Create the landmarker **once** and reuse it (never re-initialize per frame).
- Return `None` (not an exception) when no person is detected; this is a normal condition, not an error.
- If `num_poses > 1` is ever configured, select deterministically (e.g., largest bounding box) and document that multi-person is out of scope.
- Convert MediaPipe results to the project's `PoseResult`.
- Expose `close()`; implement the context-manager protocol.
- Wrap API/argument errors in `DetectorInitError` with the installed MediaPipe version in the message.

**`quality.py` — Landmark validation (the "quality gate"):**
A `PoseResult` is accepted only if **all** hold:
1. Each required landmark has `visibility ≥ min_visibility` and `presence ≥ min_presence`.
2. Each required landmark lies inside the frame (with `edge_margin` tolerance), i.e., the body is not clipped.
3. All coordinates are finite (no NaN/Inf).
4. Torso length (mid-shoulder to mid-hip) in normalized units is ≥ `min_torso_length_norm` (the person is not too far away).
5. Shoulder width and torso length are non-degenerate (non-zero).

Return a `QualityReport` with human-readable reasons, e.g., "Left knee not visible", "Move back — your feet are outside the frame", "Move closer". Visibility is a *hint*, not a guarantee of correctness; document this limitation.

### 7.4 `features/` (the heart of the ML design)

**`schema.py` — single source of truth.**
Define `SCHEMA_VERSION` and the ordered list of feature names. The extractor must produce a vector whose order is *exactly* this list. Provide `validate_vector(vec)` which checks length, dtype, and finiteness, and raises `FeatureSchemaError` otherwise. Any change to features requires bumping `SCHEMA_VERSION`.

**Feature groups (schema version 1):**

*Group A — Normalized coordinates (13 landmarks × 2 or 3 axes)*
Landmarks: nose, both shoulders, both elbows, both wrists, both hips, both knees, both ankles. Features `x`, `y` (and `z` only if `use_z = true`, because MediaPipe's relative depth is noisier).

*Group B — Joint angles (8 angles, in degrees, range 0–180)*
| Feature | Vertex joint | Rays to |
|---------|--------------|---------|
| `angle_left_elbow` | left elbow | left shoulder, left wrist |
| `angle_right_elbow` | right elbow | right shoulder, right wrist |
| `angle_left_shoulder` | left shoulder | left hip, left elbow |
| `angle_right_shoulder` | right shoulder | right hip, right elbow |
| `angle_left_hip` | left hip | left shoulder, left knee |
| `angle_right_hip` | right hip | right shoulder, right knee |
| `angle_left_knee` | left knee | left hip, left ankle |
| `angle_right_knee` | right knee | right hip, right ankle |

*Group C — Relational features (all expressed in torso-length units, so they are scale-invariant)*
- Vertical offset of each wrist relative to its shoulder (`wrist_y − shoulder_y`) — separates raised vs. lowered arms.
- Vertical offset of each wrist relative to the nose.
- Horizontal distance of each wrist from the body midline.
- Distance from each wrist to the *opposite* shoulder (separates `Arms Crossed`).
- Distance from each wrist to its *same-side* hip (separates `Hands on Hips`).
- Wrist-to-wrist distance.
- Vertical offset between hip and knee on each side (separates `Sitting` vs. `Standing`).
- Vertical offset between knee and ankle on each side.
- Shoulder-line tilt angle and torso inclination angle (from vertical).
- Left/right arm-raised indicator values expressed as *continuous* numbers, not hard booleans.

**`normalize.py`:**
1. **Aspect-ratio correction first.** MediaPipe's image landmarks are normalized to [0,1] per axis, so x and y have different physical scales. Convert to a common metric (multiply x by frame width and y by frame height, or equivalent) **before** computing distances or angles. Add a unit test proving angles are unchanged when only the frame aspect ratio changes.
2. **Center** on the mid-hip point (configurable).
3. **Scale** by torso length (mid-shoulder to mid-hip) by default; shoulder width as an alternative. Guard against division by near-zero (raise `FeatureError`; never produce Inf/NaN).
4. **Optional rotation to torso axis** (`rotate_to_torso`, default off, because body tilt can be informative).
5. **Coordinate source** is configurable: `image` (default) or `world` (MediaPipe world landmarks, hip-centered, metric). Provide a small experiment script comparing both on your dataset.
6. **Mirroring** helper `mirror_features(vec)` that swaps left/right features and negates horizontal terms; used only for optional training-time augmentation (applied **after** the train/test split and **only** to the training partition).

**Critical rule:** features are always computed from the **un-mirrored** camera frame. The mirrored image is purely a *display* convenience. Add a test guarding this to prevent left/right reversal bugs.

**`angles.py`:**
```python
def joint_angle(a, b, c) -> float:
    """Angle in degrees at vertex b formed by points a-b-c. Returns NaN-safe result
    or raises FeatureError for zero-length segments."""
```
Use `np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)` before `arccos`. Include unit tests for 0°, 90°, 180°, and degenerate cases.

**`extractor.py`:**
`FeatureExtractor.extract(pose_result, quality_report) -> FeatureVector` — refuses to run on a failed quality report, returns a validated NumPy `float32` vector plus the schema version and feature names.

### 7.5 `data/`

**`labels.py` — Label registry** (`data/labels.json`)
- Create, rename, delete labels without touching source code.
- Labels are stored with a stable internal ID and a display name, so renaming does not corrupt old samples.
- Validation: non-empty, unique (case-insensitive), max length, no commas/newlines/control characters, reserved names forbidden (`Unknown Pose`, `No Person Detected`, etc.).
- Renaming updates the dataset atomically.

**`dataset.py` — `DatasetManager`**

CSV schema (long-term stable):

| Column | Meaning |
|--------|---------|
| `sample_id` | UUID4 |
| `session_id` | Identifies one continuous recording session (new ID per capture session) |
| `subject_id` | Optional pseudonymous person identifier (for generalization testing) |
| `timestamp_utc` | ISO-8601 |
| `label_id` | Stable label ID |
| `label` | Display label (denormalized for readability) |
| `schema_version` | Feature schema version |
| `quality_score` | Minimum required-landmark visibility at capture time |
| `f_<name>` … | One column per feature, in schema order |

Required operations (each unit-tested):
- `add_sample(label, features, session_id, quality)` — rejects invalid vectors and unreliable landmarks.
- `undo_last()` — remove the most recent sample (multi-level undo within a session).
- `delete_samples(ids)`; `delete_class(label)`; `clear(confirm=True)`.
- `rename_label(old, new)`.
- `stats()` → per-class counts, per-session counts, per-subject counts, imbalance ratio.
- `export_csv(path)` / `import_csv(path, mode="merge"|"replace")` with **strict validation**: required columns, numeric types, finite values, known schema version, non-empty labels. Report row-level errors rather than failing silently. Reject files whose `schema_version` differs unless a migration exists.
- **Automatic backup** before any destructive operation (`data/datasets/backups/`).
- **Atomic saves**; handle `PermissionError`/`OSError` by raising `DatasetIOError` with the path and cause.
- **Duplicate guard:** warn when a sample's feature vector (rounded) matches an existing one.

**Capture modes** (implemented in the controller, using `DatasetManager`):
1. *Single capture* (button/hotkey).
2. *Burst capture*: after a configurable countdown (default 3 s), capture N samples at a fixed interval (default one every 100 ms) **only when quality passes**; stop early and warn if quality drops. The user can cancel at any time.
3. Display live: current class, samples this session, total per class, and the current quality status.

**`validation.py` — Dataset health report**
Before training, run checks and return a report of **errors** (block training) and **warnings** (allow with notice):
- Error: fewer than 2 classes; any class with < 10 samples; NaN/Inf values; schema mismatch.
- Warning: any class below `min_samples_per_class`; imbalance ratio above threshold; fewer than 3 recording sessions; exact-duplicate vectors; single subject only (limited generalization); suspiciously low intra-class variance.

**`splits.py` — Session-aware splitting (critical)**
- **Never split randomly by frame.** Adjacent frames from one session are nearly identical; random splitting leaks information and gives falsely perfect results.
- Split by `session_id` (and optionally `subject_id`) using `GroupShuffleSplit`; for cross-validation use `GroupKFold` or `LeaveOneGroupOut`.
- Ensure **every class appears in both train and test** partitions; if impossible, raise `InsufficientSessionsError` explaining how many sessions per class are needed.
- If the dataset has only one session, **do not silently fall back to random splitting.** Offer a clearly labeled *contiguous-block* split and attach a prominent warning that the resulting numbers are optimistic.
- Persist the exact split (list of session IDs per partition) in the training run for reproducibility.

### 7.6 `ml/`

**`models.py`** — estimator factories returning scikit-learn `Pipeline` objects:
- **Random Forest** (default): `RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)`.
- **SVM**: `Pipeline([StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced")])`.
- Optional hyperparameter search via `GridSearchCV` with a **group-aware** CV (`GroupKFold`) and `scoring="f1_macro"`. The test partition is never used for tuning.

**`trainer.py`** — orchestration, in this order:
1. Load dataset → run health checks.
2. Session-aware split into train / test (and CV folds on train).
3. (Optional) mirror augmentation on train only.
4. Fit model(s); with group-aware CV compute out-of-fold predictions for threshold tuning.
5. Tune the rejection thresholds on **out-of-fold validation predictions** (Section 11).
6. Evaluate **once** on the held-out test sessions.
7. Save the run into `artifacts/runs/<UTC timestamp>/` containing: `model.joblib`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `per_class_metrics.csv`, `split.json`, `config_snapshot.toml`, and `feature_importance.csv` (Random Forest only).
8. Update `models/latest.joblib` **only** if the user confirms or an explicit `--activate` flag is given.

**`evaluator.py`** — compute and save: accuracy, balanced accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1/support, confusion matrix (raw and row-normalized), the **most-confused class pairs**, and inference latency per sample. Render figures with Matplotlib using a non-interactive backend (`Agg`) so it works in a worker thread. Include an honest summary of limitations in the report.

**`artifact.py`** — model file format (a single `joblib` file containing a dictionary):
```
{
  "estimator": <fitted sklearn Pipeline>,
  "label_names": [...],
  "feature_names": [...],
  "schema_version": 1,
  "thresholds": {"min_confidence": ..., "min_margin": ...},
  "metrics_summary": {...},
  "training_info": {
      "created_utc": ..., "dataset_hash": ..., "n_samples": ...,
      "split_sessions": {...}, "random_seed": ...,
      "versions": {"python":..., "sklearn":..., "numpy":..., "mediapipe":..., "opencv":...}
  }
}
```
On load: verify `schema_version` and `feature_names` equal the current schema (**refuse** with a clear message if they differ and offer retraining); warn if scikit-learn's major/minor version differs from training; handle corrupted or missing files with `ModelLoadError`. **Security note:** `joblib`/pickle files can execute code on load, so only load model files you created or trust; state this in the docs and never load models from untrusted sources.

**`inference.py` — `PoseClassifier`**
`predict(features) -> Prediction`: validate the vector; compute class scores via `predict_proba`; compute top-1 and top-2; apply the rejection rule; return the structured `Prediction`. Thread-safe, no side effects, and loaded once.

---

## 8. FEATURE ENGINEERING — DESIGN RATIONALE (document in `docs/ARCHITECTURE.md`)

Explain, with reasoning, why the design uses:
- **Normalization** to reduce sensitivity to where the person stands and how far they are from the camera (without claiming it removes all perspective or camera-angle effects).
- **Joint angles** because they are position- and scale-invariant and help separate poses with similar overall silhouettes (e.g., `T-Pose` vs. `Arms Raised`).
- **Relational features** because poses like `Arms Crossed` and `Hands on Hips` are defined by *relationships* between limbs rather than by absolute positions.
- **Continuous values instead of hard rules**, so the classifier learns the boundaries from data.

Document the **assumed camera setup** (camera roughly at chest-to-waist height, person facing the camera, full body visible for lower-body poses) and the known limitation that `Sitting` vs. `Standing` and `Squatting` depend on lower-body visibility and camera angle.

---

## 9. DATA COLLECTION GUIDELINES (document in `docs/DATASET_GUIDE.md` and show hints in the GUI)

- Target **50–150 usable samples per pose** as a *starting experiment*, then use the evaluation to decide whether more are needed. This is a starting point, not a guaranteed sufficient amount.
- Collect **at least 3 separate sessions** per pose (ideally on different occasions), so a held-out session split is meaningful.
- Vary: distance from the camera, small body rotations (±15–20°), lighting conditions, clothing, and background.
- Include **more than one person** if the model must generalize to others; otherwise state clearly that it is a personal model.
- Record "in-between" and transitional poses as *separate evaluation-only negatives* (see Section 11) rather than labeling them as a trained class.
- Check the live per-class count display and keep classes reasonably balanced.
- Review and delete mislabeled samples before training.

---

## 10. TRAINING AND EVALUATION PROTOCOL

1. **Splits:** train (used for fitting and CV), held-out **test sessions** (touched exactly once at the end).
2. **Model selection:** compare Random Forest and SVM using group-aware cross-validation macro-F1 on the training partition only. Report both; choose by CV score, then report the final test result once.
3. **Primary metric:** macro-F1 (robust to class imbalance), alongside accuracy and balanced accuracy.
4. **Per-class analysis:** surface the weakest class and the top-3 most-confused pairs, with a plain-language suggestion (e.g., "`Arms Raised` is often predicted as `T-Pose` — collect more examples with the arms clearly above the head and review wrist-vs-shoulder features").
5. **Sanity checks:** warn when test accuracy is suspiciously perfect (possible leakage), when training accuracy greatly exceeds test accuracy (overfitting), or when one class dominates predictions.
6. **Baselines:** report a majority-class baseline and (optionally) a simple rule-based baseline so improvements are meaningful.
7. **Reproducibility:** identical seed and data must produce identical metrics; add a test.

---

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

## 14. USER INTERFACE SPECIFICATION

Build the UI in **two stages**:

- **Stage A (Phase 1–4):** an OpenCV window with a HUD and keyboard shortcuts, for fast iteration.
- **Stage B (Phase 5):** a Tkinter application using the same controller.

### 14.1 Main window layout (Stage B)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Status bar:  ● Camera Connected | Detection: OK | Model: v3 (RF) | FPS: 22 │
├──────────────────────────────────────┬───────────────────────────────┤
│                                      │  [ Live | Collect | Train |   │
│        VIDEO PANEL                   │    Evaluate | Settings ]      │
│        (skeleton overlay,            │                               │
│         predicted pose,              │   (active tab content)        │
│         quality hints)               │                               │
│                                      │                               │
└──────────────────────────────────────┴───────────────────────────────┘
```

### 14.2 Tabs

- **Live:** large predicted-pose label, model-score bar, state (`Confident / Unknown Pose / No Person / Low Quality`), reasons for low quality ("Move back — feet not visible"), FPS and latency.
- **Collect:** label dropdown with add/rename/delete; single capture and burst capture with countdown; live quality indicator; per-class sample table with count bars and imbalance warnings; buttons for *Undo last*, *Delete selected*, *Clear class*, *Export*, *Import*; session ID display.
- **Train:** choose classifier (RF / SVM / compare both), see dataset health warnings, start training in the background with progress, show a result summary, and offer **Activate model** only after reviewing results. Display whether the active model is *stale* (dataset changed since training).
- **Evaluate:** metrics table, confusion matrix image, per-class results, most-confused pairs, and a list of past runs.
- **Settings:** camera selector and resolution, mirror display toggle, model variant, thresholds, smoothing window, debug overlay toggles, and privacy toggles (`record_video` is off and clearly labeled). Persist settings to a user config file; validate on load.

### 14.3 UI requirements

- Never block the UI thread; long operations run in workers with cancel support.
- Display clear, actionable error dialogs (cause + how to fix), never raw tracebacks. Full tracebacks go to the log file.
- Confirm destructive actions (clear dataset, delete class, replace on import).
- Show a persistent, visible indicator whenever any recording or logging is active.
- Keyboard shortcuts (documented): capture, undo, toggle debug, quit.
- Accessibility: readable font sizes, sufficient contrast, status conveyed with text as well as color.

---

## 15. ERROR HANDLING MATRIX

Define a typed exception hierarchy in `exceptions.py` (base `PoseSenseError`). Implement and **test** handling for every case below:

| Condition | Exception / state | Required behavior |
|-----------|------------------|-------------------|
| Camera does not open (wrong index, in use, disconnected) | `CameraOpenError` | Show message, allow choosing another camera |
| Camera permission denied | `CameraOpenError` (+ guidance) | Show OS-specific permission instructions |
| Frame cannot be read | `FrameReadError` | Skip frame, attempt controlled recovery, then `CAMERA_UNAVAILABLE` |
| Black screen | `BlackFrameWarning` | Show warning, do not process frames |
| Camera still occupied after exit | — | Guaranteed release in `finally` and on shutdown; test with a fake camera |
| MediaPipe import fails | `DetectorInitError` | Explain how to fix the environment; run `scripts/check_environment.py` |
| Model `.task` file missing | `ModelFileNotFoundError` | Tell the user to run `scripts/download_model.py` |
| Unsupported API/argument (version mismatch) | `DetectorInitError` | Include installed vs. expected version |
| Invalid BGR/RGB conversion | — | Prevented by a single conversion function with a test |
| Invalid / non-increasing timestamp | — | Monotonic timestamp generator with a unit test |
| Live-stream produces no callbacks | `DetectorTimeoutError` | Watchdog warning after a timeout |
| Detection crash on a frame | `DetectionError` | Log, skip frame, continue; escalate after N consecutive failures |
| No body detected | state `NO_PERSON` | Display guidance |
| Partial body / clipped limbs | state `LOW_QUALITY` | Display specific reasons |
| Landmark jitter | — | Optional smoothing |
| Left/right reversal | — | Mirror only for display; test guards this |
| Multiple people | — | Single-person mode; deterministic selection; documented limitation |
| NaN/Inf or wrong-length features | `FeatureSchemaError` | Reject the sample/frame; never predict |
| Feature-order mismatch (old model vs. new schema) | `ModelLoadError` | Refuse to load; offer retraining |
| Dataset save failure (path/permission) | `DatasetIOError` | Report path and cause; keep in-memory data intact |
| Malformed CSV import | `DatasetImportError` | Row-level error report; nothing partially imported |
| Corrupted/incompatible model file | `ModelLoadError` | Clear message; offer retraining |
| Stale model after new data | — | UI badge "Model out of date" |
| Training with too few samples/classes/sessions | `InsufficientDataError` / `InsufficientSessionsError` | Explain exactly what is missing |
| Low FPS | — | Warning plus suggestions (lower resolution, `lite` variant) |
| UI freeze | — | Prevented by worker threads; verified manually |

---

## 16. PRIVACY AND SECURITY

- All processing is **local**. No network calls are made at runtime (only the explicit model-download script touches the network).
- **No video or images are saved by default.** Only numeric landmark-derived features are stored.
- `record_video` and `log_predictions` default to `false`, are clearly visible in the UI, and require explicit opt-in.
- Subject identifiers are optional and pseudonymous; document that datasets of body landmarks can still be personal data and should be handled accordingly.
- Warn about loading untrusted `joblib` files (they can execute arbitrary code).
- Do not log raw landmark coordinates at INFO level; keep them at DEBUG.
- Keep `data/`, `models/`, and `artifacts/` out of version control.

---

## 17. LOGGING AND SESSION LOGS

- Configure `logging` with rotating file handlers under `data/logs/` and a concise console format.
- Log: startup configuration (sanitized), library versions, camera properties actually applied, state transitions, model load/activate events, training start/finish with metrics, and recoverable errors with context.
- **Optional session log** (`log_predictions = true`): CSV of timestamp, state, label, score, FPS. Write with buffered flushing and handle write failures without crashing the app. Never include images.

---

## 18. TESTING STRATEGY

### 18.1 Principles
- The whole suite must pass **without a webcam, display, or network**.
- Use **synthetic landmark generators** (`tests/fixtures/synthetic_poses.py`) that build a plausible 33-landmark skeleton for each target pose (and noisy variants) from joint angles. These power feature, classifier, and pipeline tests.
- Use a `FakeCamera` (scripted frames and failures) and a `FakeDetector` (scripted `PoseResult`s) for runtime tests.

### 18.2 Required tests (minimum)

**Unit**
- Joint-angle math (0°, 90°, 180°, collinear, zero-length).
- Normalization invariance: translation invariance, uniform-scale invariance, aspect-ratio correction.
- Feature schema: order, length, finiteness, version; failure on NaN/Inf.
- Mirror-features swap correctness; "features use the un-mirrored frame" guard.
- Quality gate: each failure reason triggers independently.
- Label registry: validation, rename propagation, reserved names.
- DatasetManager: add, undo, delete, clear, rename, stats, atomic save, backup, import validation (malformed files), duplicate warning.
- Session-aware splitter: no session appears in both partitions; every class represented; correct failure messages; one-session fallback is flagged.
- Rejection rule: boundary conditions for confidence and margin; threshold tuning returns monotonic coverage.
- Smoother: flicker suppression, hysteresis, debounce.
- Artifact: save/load round trip; schema mismatch refused; corrupted file handled; version warning.
- Timestamp generator: strictly increasing.
- Config: invalid values rejected with the key name.

**Integration**
- Synthetic dataset → train → evaluate → save → load → predict; assert high accuracy on clean synthetic data and correct `Unknown Pose` on an out-of-distribution skeleton.
- Full pipeline with `FakeCamera` + `FakeDetector`: states `NO_PERSON → LOW_QUALITY → CONFIDENT` occur in the expected order and no label is shown outside `CONFIDENT`.
- Camera failure recovery: scripted failures lead to a clean `CAMERA_UNAVAILABLE` and the camera is released.
- Reproducibility: same seed ⇒ same metrics.

**Hardware (opt-in, `@pytest.mark.hardware`)**
- Webcam open/read/release; real MediaPipe on a bundled sample image (skipped when the model file is absent).

### 18.3 Coverage and CI
- ≥ 80% coverage on non-UI code; report it.
- Add a GitHub Actions workflow running `ruff`, `mypy`, and `pytest` on Linux and Windows for the supported Python versions (hardware tests excluded).

---

## 19. CODE QUALITY STANDARDS

- Full type hints; `mypy --strict` clean on core packages.
- Google- or NumPy-style docstrings on every public class and function, including units (degrees, normalized units, milliseconds).
- `ruff` for linting and formatting (line length 100); `pre-commit` hooks for ruff, mypy, trailing whitespace, and large-file checks.
- No magic numbers: constants are named, documented, and configurable where tunable.
- Functions ≤ ~40 lines where reasonable; modules have a single responsibility.
- Use `pathlib` for all paths; never assume the current working directory.
- Use `dataclasses`, `enum`, and `Protocol` idiomatically.
- Meaningful commit-sized changes: at the end of each phase, provide a suggested conventional-commit message (`feat:`, `fix:`, `test:`, `docs:`).

---

## 20. DOCUMENTATION DELIVERABLES

- **README.md:** project summary, screenshot placeholders, features, quick start (≤ 10 commands), requirements, hardware notes, supported OS, limitations, license.
- **docs/ARCHITECTURE.md:** diagrams (Mermaid), module responsibilities, data contracts, threading model, feature rationale, design decisions and trade-offs.
- **docs/USER_GUIDE.md:** step-by-step workflow: install → download model → collect → train → evaluate → activate → use live; keyboard shortcuts; screenshots.
- **docs/DATASET_GUIDE.md:** how to collect good data, session discipline, CSV schema, import/export, the leakage warning.
- **docs/EVALUATION.md:** protocol, metric definitions, how to read a confusion matrix, honest limitations, and a template for reporting results (real measured values only).
- **docs/TROUBLESHOOTING.md:** one entry per row of the error matrix (symptom → cause → fix).
- **docs/FUTURE_WORK.md:** Level 3 roadmap and design hooks.
- **CHANGELOG.md:** updated at each phase.
- **CLI help:** every command documented with `--help` and examples.

### 20.1 Command-line interface (`cli.py`)

```
python -m posesense live                 # run live recognition (OpenCV HUD)
python -m posesense gui                  # launch the full GUI
python -m posesense collect --label "T-Pose" --burst 100
python -m posesense dataset stats|export|import|validate
python -m posesense train --model rf|svm|compare [--activate]
python -m posesense evaluate --run <run_id>
python -m posesense check                # environment and camera diagnostics
python -m posesense download-model --variant lite
```

---

## 21. PHASED ROADMAP WITH ACCEPTANCE CRITERIA

Stop at the end of each phase and report (Section 25). Do not continue until the criteria are satisfied and verified by actually running the code and tests.

### Phase 0 — Foundation
**Deliver:** repository scaffold, `pyproject.toml`, pinned dependencies, config loading/validation, logging, exception hierarchy, shared types, pre-commit, CI skeleton, `scripts/check_environment.py`, `scripts/download_model.py`.
**Accept when:** a fresh clone installs with documented commands; `check` reports Python/library versions and model-file status; lint, type check, and the (initial) tests pass.

### Phase 1 — Working webcam and skeleton
**Deliver:** `WebcamSource` (with error handling and recovery), `MediaPipeDetector`, landmark constants, skeleton overlay, OpenCV HUD showing camera status, detection status, FPS, and optional debug info. **No classifier yet.**
**Accept when:** the live window shows the skeleton; unplugging/covering the camera produces a clean message instead of a crash; the camera is released on every exit path; BGR→RGB conversion and monotonic timestamps are tested; measured FPS is reported.

### Phase 2 — Features and dataset collection
**Deliver:** quality gate, feature schema, normalization, angles, relational features, extractor, label registry, `DatasetManager`, single and burst capture (HUD or CLI), dataset stats and export/import, dataset health checks.
**Accept when:** captured rows contain valid, schema-ordered features; every dataset operation (including undo, delete, rename, import validation, backup) is covered by tests; invariance tests for translation, scale, and aspect ratio pass.

### Phase 3 — First classifier
**Deliver:** session-aware splitting, Random Forest and SVM training, evaluator with reports and confusion matrices, model artifact with metadata and compatibility checks, run folders.
**Accept when:** training on a synthetic dataset and on a small real dataset works end to end; split leakage tests pass; the artifact round trip and mismatch refusal are tested; the report contains per-class metrics and the most-confused pairs; the one-session warning appears when applicable.

### Phase 4 — Live recognition
**Deliver:** `PoseClassifier`, rejection rule with threshold tuning, smoothing, state machine, performance monitor, live predicted label and confidence on the HUD.
**Accept when:** the live app displays `No Person Detected`, `Low Quality / Partial Body` (with reasons), `Unknown Pose`, and confident labels correctly; pipeline tests with fakes pass; flicker is measurably reduced (report label-switch counts with and without smoothing).

### Phase 5 — Robustness, evaluation, and GUI
**Deliver:** the Tkinter application (Live, Collect, Train, Evaluate, Settings), background workers, stale-model badge, settings persistence, optional session logging, full error-matrix coverage, finalized documentation.
**Accept when:** a new user can complete the full workflow using only the GUI; every row of the error matrix has a test or a documented manual verification; the UI never freezes during training; clean shutdown leaves no threads or camera handles alive.

### Phase 6 — Release polish
**Deliver:** coverage report (≥ 80% core), final benchmark (FPS, latency per stage, hardware description), final evaluation on held-out sessions with the report template filled using **real measured values**, README with demo instructions, version tag `v1.0.0`, and a short demo script/checklist for presentation.
**Accept when:** the Definition of Done (Section 23) is fully checked.

---

## 22. FUTURE WORK — EXTENSION POINTS ONLY (do NOT implement)

Create interfaces, comments, and `docs/FUTURE_WORK.md` entries for:
- **Movement recognition** (waving, sit-to-stand): a sliding window of feature vectors feeding a sequence model; keep a `SequenceClassifier` protocol and a `FeatureWindow` buffer design.
- **Multi-person detection and tracking:** per-person pose results, simple tracker, separate smoothers.
- **Advanced settings panel and dataset/model comparison dashboards.**
- **Alternative classifiers** (gradient boosting, small neural network) via the existing estimator factory.
- **Alternative front-ends** (PySide6, web) via the UI-agnostic controller.

---

## 23. DEFINITION OF DONE (final checklist)

- [ ] All six phases accepted with evidence (command outputs, test results).
- [ ] `ruff`, `mypy --strict` (core), and `pytest` pass; coverage ≥ 80% on non-UI code.
- [ ] Fresh-clone install and run verified using only the README.
- [ ] Eight pose classes trainable; custom labels creatable from the GUI.
- [ ] Held-out-session evaluation report generated with real values; limitations documented.
- [ ] `No Person Detected`, `Low Quality / Partial Body`, `Unknown Pose`, and `Camera Unavailable` demonstrated.
- [ ] Camera and threads release cleanly on normal exit, errors, and window close.
- [ ] No video/images stored by default; privacy settings visible.
- [ ] Model artifacts carry metadata and refuse incompatible schemas.
- [ ] Documentation complete and consistent with the code.
- [ ] No TODOs, dead code, commented-out blocks, or placeholder metrics remain.

---

## 24. PROHIBITED BEHAVIORS

- Do **not** use the legacy `mp.solutions.pose` API.
- Do **not** split data randomly by frame for evaluation.
- Do **not** tune thresholds or hyperparameters on the test set.
- Do **not** fabricate, estimate, or hard-code accuracy, FPS, or any other measurement.
- Do **not** feed mirrored-frame landmarks into feature extraction.
- Do **not** run classification on frames that fail the quality gate.
- Do **not** store video or images by default.
- Do **not** use bare `except:` or silently swallow exceptions.
- Do **not** call Tk methods from non-UI threads.
- Do **not** commit model files, datasets, logs, or credentials.
- Do **not** add dependencies, features, or abstractions beyond this specification without written justification.
- Do **not** claim a library API exists without verifying it against the installed version or the official documentation.
- Do **not** implement Level 3 features.

---

## 25. REPORTING FORMAT (end of every phase)

At the end of each phase, reply with exactly these sections:

1. **Summary** — what was built, in 3–6 sentences.
2. **Files created/changed** — list with one-line purposes.
3. **How to run and verify** — exact commands.
4. **Test results** — real output summary (passed/failed/skipped, coverage).
5. **Measured results** — real numbers only (FPS, latency, metrics), with the hardware used; write "not applicable" when none exist.
6. **Acceptance criteria** — checklist with ✅/❌ and evidence for each item.
7. **Known issues and risks** — honest list.
8. **Decisions made** — any deviation from this prompt and why.
9. **Suggested commit message(s).**
10. **Next phase** — a one-paragraph preview, then wait for my "proceed".

---

## 26. KICKOFF AND FOLLOW-UP PROMPTS

### 26.1 First action (do this now)

1. Read this entire document.
2. Reply with a one-page implementation plan: restated goal, phase list, top five technical risks (for example, MediaPipe version compatibility, webcam backend differences across operating systems, session-leakage in evaluation, lower-body visibility for `Sitting`/`Squatting`, and Tkinter threading), and any blocking questions.
3. Begin **Phase 0** immediately unless a blocking question exists.

### 26.2 Short prompts for me to use between phases

- **Proceed:** *"Phase N is accepted. Proceed to Phase N+1 following PROMPT.md, and use the Section 25 report format."*
- **Fix:** *"Phase N has this problem: <paste error or behavior>. Diagnose the root cause first, explain it, fix it with a regression test, and re-run the full test suite."*
- **Audit:** *"Audit the current code against Sections 3, 15, 16, and 24 of PROMPT.md. List every violation with file and line, then fix them."*
- **Tune:** *"Using the latest evaluation run, identify the most-confused class pairs and propose data-collection or feature changes. Do not change the test split."*
- **Release:** *"Execute Phase 6, verify the Definition of Done item by item, and produce the final report."*

---

*End of master prompt. Follow it faithfully, verify everything you claim, and prioritize correctness, honesty, and clarity over speed.*


so basically it is this program. double check if all of these were implemented
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-09T21:04:41+08:00.

The user's current state is as follows:
Active Document: c:\Users\milla\Desktop\this is how boredom gets you\src\posesense\camera\webcam.py (LANGUAGE_PYTHON)
Cursor is on line: 1
Other open documents:
- c:\Users\milla\Desktop\this is how boredom gets you\src\posesense\__init__.py (LANGUAGE_PYTHON)
- c:\Users\milla\Desktop\this is how boredom gets you\src\posesense\camera\__init__.py (LANGUAGE_PYTHON)
- c:\Users\milla\Desktop\this is how boredom gets you\config\default.toml (LANGUAGE_TOML)
- c:\Users\milla\Desktop\this is how boredom gets you\CHANGELOG.md (LANGUAGE_MARKDOWN)
- c:\Users\milla\Desktop\this is how boredom gets you\tests\unit\test_download_model.py (LANGUAGE_PYTHON)
</ADDITIONAL_METADATA>