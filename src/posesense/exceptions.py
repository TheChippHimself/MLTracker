"""Custom exception hierarchy for PoseSense.

All exceptions inherit from PoseSenseError so callers can catch the base
class for any project-level error.  Each exception carries enough context
for the UI layer to present an actionable message.
"""

from __future__ import annotations


class PoseSenseError(Exception):
    """Base exception for all PoseSense errors."""


# ── Camera ───────────────────────────────────────────────────────────────


class CameraError(PoseSenseError):
    """Base for camera-related errors."""


class CameraOpenError(CameraError):
    """Camera could not be opened (wrong index, in use, permission denied)."""

    def __init__(self, index: int, message: str, guidance: str = "") -> None:
        self.index = index
        self.guidance = guidance
        full = f"Cannot open camera {index}: {message}"
        if guidance:
            full += f"\n  → {guidance}"
        super().__init__(full)


class FrameReadError(CameraError):
    """A frame could not be read from the camera."""


class BlackFrameWarning(CameraError):
    """Camera is returning near-black frames (lens cap? driver issue?)."""


# ── Detector ─────────────────────────────────────────────────────────────


class DetectorError(PoseSenseError):
    """Base for pose-detector errors."""


class DetectorInitError(DetectorError):
    """The pose detector could not be initialized."""

    def __init__(self, message: str, installed_version: str = "unknown") -> None:
        self.installed_version = installed_version
        super().__init__(f"{message} (installed mediapipe version: {installed_version})")


class DetectorTimeoutError(DetectorError):
    """Live-stream detector produced no callbacks within the timeout."""


class DetectionError(DetectorError):
    """Detection failed on a specific frame."""


class ModelFileNotFoundError(DetectorError):
    """The MediaPipe .task model file is missing."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"Model file not found: {path}\n"
            "  → Run: python -m posesense download-model --variant lite\n"
            "  → Or download manually from:\n"
            "    https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task"
        )


# ── Features ─────────────────────────────────────────────────────────────


class FeatureError(PoseSenseError):
    """Base for feature-extraction errors."""


class FeatureSchemaError(FeatureError):
    """Feature vector has wrong length, dtype, or contains NaN/Inf."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Feature schema violation: {reason}")


# ── Data / Dataset ───────────────────────────────────────────────────────


class DataError(PoseSenseError):
    """Base for dataset errors."""


class DatasetIOError(DataError):
    """Dataset could not be read or written."""

    def __init__(self, path: str, cause: str) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Dataset I/O error at {path}: {cause}")


class DatasetImportError(DataError):
    """An imported CSV is malformed."""


class LabelError(DataError):
    """Invalid label operation (duplicate, reserved name, etc.)."""


# ── ML / Training ────────────────────────────────────────────────────────


class TrainingError(PoseSenseError):
    """Base for training errors."""


class InsufficientDataError(TrainingError):
    """Not enough data to train (too few samples, classes, etc.)."""


class InsufficientSessionsError(TrainingError):
    """Not enough recording sessions to create a valid train/test split."""


class ModelLoadError(PoseSenseError):
    """A saved model artifact could not be loaded."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot load model from {path}: {reason}")


# ── Config ───────────────────────────────────────────────────────────────


class ConfigError(PoseSenseError):
    """Configuration value is invalid."""

    def __init__(self, key: str, value: object, reason: str) -> None:
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid config [{key}] = {value!r}: {reason}")
