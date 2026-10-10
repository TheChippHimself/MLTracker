"""Typed configuration loading and validation for PoseSense.

All tunable values live in ``config/default.toml`` and are loaded into
validated, typed dataclasses.  Invalid values raise :class:`ConfigError`
with the offending key.

Usage::

    cfg = load_config()                 # loads default.toml
    cfg = load_config(Path("my.toml"))  # loads a custom file
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from posesense.exceptions import ConfigError

logger = logging.getLogger("posesense.config")

_DEFAULT_CONFIG_PATH = Path("config") / "default.toml"


# ── Section dataclasses ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera capture settings."""

    index: int = 0
    width: int = 640
    height: int = 480
    backend: str = "auto"
    mirror_display: bool = True
    reconnect_attempts: int = 3
    reconnect_backoff_s: float = 1.0
    black_frame_threshold: float = 3.0
    black_frame_max_consecutive: int = 30


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """MediaPipe PoseLandmarker settings."""

    model_variant: str = "lite"
    model_path: str = "models/mediapipe/pose_landmarker_lite.task"
    running_mode: str = "video"
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Landmark quality gate settings."""

    min_visibility: float = 0.5
    min_presence: float = 0.5
    edge_margin: float = 0.02
    min_torso_length_norm: float = 0.08
    required_landmarks: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    """Feature extraction settings."""

    schema_version: int = 1
    coordinate_source: str = "image"
    use_z: bool = False
    center: str = "mid_hip"
    scale: str = "torso_length"
    rotate_to_torso: bool = False
    mirror_augmentation: bool = False


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Prediction smoothing settings."""

    landmark_filter: str = "none"
    ema_alpha: float = 0.5
    vote_window: int = 7
    min_votes: int = 4
    no_person_debounce_frames: int = 5


@dataclass(frozen=True, slots=True)
class RejectionConfig:
    """Unknown-pose rejection settings.

    These are placeholders that get tuned and stored per-model after
    training.
    """

    min_confidence: float = 0.60
    min_margin: float = 0.15


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training and evaluation settings."""

    random_seed: int = 42
    default_model: str = "random_forest"
    test_session_fraction: float = 0.2
    min_samples_per_class: int = 30
    imbalance_warning_ratio: float = 3.0


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    """Performance monitoring settings."""

    target_fps: int = 15
    log_interval_s: int = 5


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Privacy-related flags.  Both default to ``False``."""

    record_video: bool = False
    log_predictions: bool = False


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Inference settings."""

    active_run_id: str = "latest"


@dataclass(frozen=True, slots=True)
class PoseSenseConfig:
    """Top-level configuration container."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ── Validation helpers ───────────────────────────────────────────────────

_VALID_BACKENDS = {"auto", "dshow", "msmf", "v4l2", "avfoundation"}
_VALID_VARIANTS = {"lite", "full", "heavy"}
_VALID_RUNNING_MODES = {"video", "live_stream", "image"}
_VALID_COORDINATE_SOURCES = {"image", "world"}
_VALID_CENTERS = {"mid_hip"}
_VALID_SCALES = {"torso_length", "shoulder_width"}
_VALID_FILTERS = {"none", "ema", "one_euro"}
_VALID_MODELS = {"random_forest", "svm"}


def _validate(cfg: PoseSenseConfig) -> None:
    """Raise :class:`ConfigError` for any invalid value."""
    c = cfg.camera
    if c.index < 0:
        raise ConfigError("camera.index", c.index, "must be ≥ 0")
    if c.width < 1 or c.height < 1:
        raise ConfigError("camera.width/height", f"{c.width}×{c.height}", "must be ≥ 1")
    if c.backend not in _VALID_BACKENDS:
        raise ConfigError("camera.backend", c.backend, f"must be one of {_VALID_BACKENDS}")
    if c.reconnect_attempts < 0:
        raise ConfigError(
            "camera.reconnect_attempts", c.reconnect_attempts, "must be ≥ 0"
        )
    if c.reconnect_backoff_s <= 0:
        raise ConfigError(
            "camera.reconnect_backoff_s", c.reconnect_backoff_s, "must be > 0"
        )

    d = cfg.detector
    if d.model_variant not in _VALID_VARIANTS:
        raise ConfigError("detector.model_variant", d.model_variant, f"must be one of {_VALID_VARIANTS}")
    if d.running_mode not in _VALID_RUNNING_MODES:
        raise ConfigError("detector.running_mode", d.running_mode, f"must be one of {_VALID_RUNNING_MODES}")
    if d.num_poses < 1:
        raise ConfigError("detector.num_poses", d.num_poses, "must be ≥ 1")
    for name in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence"):
        val = getattr(d, name)
        if not 0.0 <= val <= 1.0:
            raise ConfigError(f"detector.{name}", val, "must be in [0, 1]")

    q = cfg.quality
    for name in ("min_visibility", "min_presence"):
        val = getattr(q, name)
        if not 0.0 <= val <= 1.0:
            raise ConfigError(f"quality.{name}", val, "must be in [0, 1]")
    if q.edge_margin < 0.0 or q.edge_margin >= 0.5:
        raise ConfigError("quality.edge_margin", q.edge_margin, "must be in [0, 0.5)")
    if q.min_torso_length_norm <= 0.0:
        raise ConfigError(
            "quality.min_torso_length_norm", q.min_torso_length_norm, "must be > 0"
        )
    for idx in q.required_landmarks:
        if not 0 <= idx <= 32:
            raise ConfigError(
                "quality.required_landmarks", idx, "landmark index must be in [0, 32]"
            )

    f = cfg.features
    if f.schema_version < 1:
        raise ConfigError("features.schema_version", f.schema_version, "must be ≥ 1")
    if f.coordinate_source not in _VALID_COORDINATE_SOURCES:
        raise ConfigError(
            "features.coordinate_source",
            f.coordinate_source,
            f"must be one of {_VALID_COORDINATE_SOURCES}",
        )
    if f.center not in _VALID_CENTERS:
        raise ConfigError("features.center", f.center, f"must be one of {_VALID_CENTERS}")
    if f.scale not in _VALID_SCALES:
        raise ConfigError("features.scale", f.scale, f"must be one of {_VALID_SCALES}")

    s = cfg.smoothing
    if s.landmark_filter not in _VALID_FILTERS:
        raise ConfigError(
            "smoothing.landmark_filter", s.landmark_filter, f"must be one of {_VALID_FILTERS}"
        )
    if not 0.0 < s.ema_alpha <= 1.0:
        raise ConfigError("smoothing.ema_alpha", s.ema_alpha, "must be in (0, 1]")
    if s.vote_window < 1:
        raise ConfigError("smoothing.vote_window", s.vote_window, "must be ≥ 1")
    if s.min_votes < 1 or s.min_votes > s.vote_window:
        raise ConfigError(
            "smoothing.min_votes", s.min_votes, "must be in [1, vote_window]"
        )
    if s.no_person_debounce_frames < 0:
        raise ConfigError(
            "smoothing.no_person_debounce_frames",
            s.no_person_debounce_frames,
            "must be ≥ 0",
        )

    r = cfg.rejection
    if not 0.0 <= r.min_confidence <= 1.0:
        raise ConfigError("rejection.min_confidence", r.min_confidence, "must be in [0, 1]")
    if r.min_margin < 0.0:
        raise ConfigError("rejection.min_margin", r.min_margin, "must be ≥ 0")

    t = cfg.training
    if t.random_seed < 0:
        raise ConfigError("training.random_seed", t.random_seed, "must be ≥ 0")
    if t.default_model not in _VALID_MODELS:
        raise ConfigError(
            "training.default_model", t.default_model, f"must be one of {_VALID_MODELS}"
        )
    if not 0.0 < t.test_session_fraction < 1.0:
        raise ConfigError(
            "training.test_session_fraction",
            t.test_session_fraction,
            "must be in (0, 1)",
        )
    if t.min_samples_per_class < 1:
        raise ConfigError(
            "training.min_samples_per_class", t.min_samples_per_class, "must be ≥ 1"
        )

    p = cfg.performance
    if p.target_fps < 1:
        raise ConfigError("performance.target_fps", p.target_fps, "must be ≥ 1")


# ── Loading ──────────────────────────────────────────────────────────────


def _build_section(cls: type, raw: dict[str, object], section: str) -> object:  # type: ignore[type-arg]
    """Construct a section dataclass from raw TOML dict, raising ConfigError
    for unknown keys."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        raise ConfigError(
            section,
            sorted(unknown),
            f"unknown keys; valid keys are {sorted(valid_fields)}",
        )
    # Convert list to tuple for required_landmarks
    coerced: dict[str, object] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            coerced[k] = tuple(v)
        else:
            coerced[k] = v
    return cls(**coerced)  # type: ignore[arg-type]


def load_config(path: Path | None = None) -> PoseSenseConfig:
    """Load and validate configuration from a TOML file.

    Args:
        path: Path to a TOML file.  Defaults to ``config/default.toml``.

    Returns:
        A validated :class:`PoseSenseConfig`.

    Raises:
        ConfigError: If any value is invalid.
        FileNotFoundError: If the config file does not exist.
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file %s not found; using defaults", path)
        cfg = PoseSenseConfig()
        _validate(cfg)
        return cfg

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    sections: dict[str, object] = {}
    section_map: dict[str, type] = {  # type: ignore[type-arg]
        "camera": CameraConfig,
        "detector": DetectorConfig,
        "quality": QualityConfig,
        "features": FeaturesConfig,
        "smoothing": SmoothingConfig,
        "rejection": RejectionConfig,
        "training": TrainingConfig,
        "performance": PerformanceConfig,
        "privacy": PrivacyConfig,
        "inference": InferenceConfig,
    }

    for section_name, cls in section_map.items():
        if section_name in raw:
            sections[section_name] = _build_section(cls, raw[section_name], section_name)  # type: ignore[arg-type]

    cfg = PoseSenseConfig(**sections)  # type: ignore[arg-type]
    _validate(cfg)
    logger.info("Configuration loaded from %s", path)
    return cfg


def update_active_model(run_id: str, path: Path | None = None) -> None:
    """Update the config file to set the active model run_id."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
        
    if not path.exists():
        logger.warning("Config file %s not found, cannot update", path)
        return
        
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Simple TOML updater for the specific [inference] block
    in_inference = False
    replaced = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith("["):
            if in_inference and not replaced:
                new_lines.append(f"active_run_id = \"{run_id}\"\n\n")
                replaced = True
            in_inference = (line.strip() == "[inference]")
        elif in_inference and line.strip().startswith("active_run_id"):
            line = f"active_run_id = \"{run_id}\"\n"
            replaced = True
            
        new_lines.append(line)
        
    if in_inference and not replaced:
        new_lines.append(f"active_run_id = \"{run_id}\"\n")
        replaced = True
        
    if not replaced:
        # Append [inference] section
        if not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append("\n[inference]\n")
        new_lines.append(f"active_run_id = \"{run_id}\"\n")
        
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    logger.info("Updated active model to %s in %s", run_id, path)
