"""Shared immutable data types and enumerations for PoseSense.

All types are frozen dataclasses (``slots=True``) to prevent accidental
mutation and to reduce memory footprint.  Enums define the finite set of
application states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, unique


@unique
class AppState(Enum):
    """High-level application state shown to the user.

    The state machine in ``runtime/state.py`` evaluates these in priority
    order (CAMERA_UNAVAILABLE is highest priority).
    """

    CAMERA_UNAVAILABLE = "Camera Unavailable"
    NO_PERSON = "No Person Detected"
    LOW_QUALITY = "Low Quality / Partial Body"
    NO_MODEL = "No Model Loaded"
    UNKNOWN_POSE = "Unknown Pose"
    CONFIDENT = "Confident"


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single body landmark as reported by MediaPipe.

    Coordinates are normalised to [0, 1] relative to the frame dimensions
    for image-space landmarks, or in metres for world-space landmarks.

    Attributes:
        x: Horizontal position (normalised or metric).
        y: Vertical position (normalised or metric).
        z: Depth estimate (normalised or metric).  Noisier than x/y.
        visibility: Likelihood [0, 1] that the landmark is visible (not
            occluded).  This is a *hint*, not a guarantee.
        presence: Likelihood [0, 1] that the landmark exists within the
            frame boundaries.
    """

    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Output of the pose detector for one detected person.

    Attributes:
        landmarks: 33 image-space (normalised) landmarks.
        world_landmarks: 33 world-space landmarks if available, else None.
        timestamp_ms: Monotonic timestamp in milliseconds when the frame
            was captured.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.
    """

    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] | None
    timestamp_ms: int
    frame_width: int
    frame_height: int


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Result of the landmark quality gate.

    Attributes:
        ok: True if all quality checks passed.
        score: Minimum visibility across required landmarks (0–1).
        missing: Indices of landmarks that failed validation.
        reasons: Human-readable explanations for each failure.
    """

    ok: bool
    score: float
    missing: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Output of the pose classifier for a single feature vector.

    Attributes:
        label: Predicted class label, or None if rejected / no model.
        confidence: Top-1 classifier score (vote fraction for RF, Platt-
            scaled for SVM).  This is NOT a calibrated probability.
        margin: Difference between top-1 and top-2 scores, or None.
        all_scores: Mapping of class label to score.
        accepted: True if the prediction passed the rejection rule.
    """

    label: str | None
    confidence: float | None
    margin: float | None
    all_scores: Mapping[str, float]
    accepted: bool


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Complete result for one processed frame, consumed by the UI.

    Attributes:
        state: Current application state.
        pose_result: Raw detector output, or None.
        quality_report: Quality gate output, or None.
        prediction: Classifier output, or None.
        smoothed_label: Label after temporal smoothing, or None.
        fps: Current frames-per-second estimate.
        latencies_ms: Per-stage latency in milliseconds.
    """

    state: AppState
    pose_result: PoseResult | None
    quality_report: QualityReport | None
    prediction: Prediction | None
    smoothed_label: str | None
    fps: float
    latencies_ms: Mapping[str, float]
