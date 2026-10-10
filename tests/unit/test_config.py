"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from posesense.config import (
    CameraConfig,
    DetectorConfig,
    PoseSenseConfig,
    QualityConfig,
    load_config,
)
from posesense.exceptions import ConfigError


class TestLoadConfig:
    """Test TOML loading and defaults."""

    def test_load_valid_config(self, tmp_config: Path) -> None:
        cfg = load_config(tmp_config)
        assert isinstance(cfg, PoseSenseConfig)
        assert cfg.camera.width == 640
        assert cfg.camera.height == 480
        assert cfg.detector.model_variant == "lite"
        assert cfg.quality.min_visibility == 0.5
        assert cfg.features.schema_version == 1
        assert cfg.privacy.record_video is False

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.toml"
        cfg = load_config(missing)
        assert isinstance(cfg, PoseSenseConfig)
        assert cfg.camera.index == 0

    def test_partial_config(self, tmp_path: Path) -> None:
        """Only a camera section; other sections use defaults."""
        partial = tmp_path / "partial.toml"
        partial.write_text('[camera]\nindex = 2\nwidth = 320\nheight = 240\n', encoding="utf-8")
        cfg = load_config(partial)
        assert cfg.camera.index == 2
        assert cfg.camera.width == 320
        assert cfg.detector.model_variant == "lite"  # default

    def test_unknown_key_raises_config_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text('[camera]\nbogus_key = 42\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_config(bad)


class TestValidation:
    """Test that invalid values produce ConfigError with the key name."""

    def test_negative_camera_index(self) -> None:
        cfg = PoseSenseConfig(camera=CameraConfig(index=-1))
        with pytest.raises(ConfigError, match="camera.index"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_backend(self) -> None:
        cfg = PoseSenseConfig(camera=CameraConfig(backend="bogus"))
        with pytest.raises(ConfigError, match="camera.backend"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_model_variant(self) -> None:
        cfg = PoseSenseConfig(detector=DetectorConfig(model_variant="mega"))
        with pytest.raises(ConfigError, match="detector.model_variant"):
            from posesense.config import _validate
            _validate(cfg)

    def test_confidence_out_of_range(self) -> None:
        cfg = PoseSenseConfig(detector=DetectorConfig(min_pose_detection_confidence=1.5))
        with pytest.raises(ConfigError, match="detector.min_pose_detection_confidence"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_running_mode(self) -> None:
        cfg = PoseSenseConfig(detector=DetectorConfig(running_mode="batch"))
        with pytest.raises(ConfigError, match="detector.running_mode"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_landmark_index(self) -> None:
        cfg = PoseSenseConfig(quality=QualityConfig(required_landmarks=(99,)))
        with pytest.raises(ConfigError, match="quality.required_landmarks"):
            from posesense.config import _validate
            _validate(cfg)

    def test_negative_edge_margin(self) -> None:
        cfg = PoseSenseConfig(quality=QualityConfig(edge_margin=-0.1))
        with pytest.raises(ConfigError, match="quality.edge_margin"):
            from posesense.config import _validate
            _validate(cfg)

    def test_edge_margin_too_large(self) -> None:
        cfg = PoseSenseConfig(quality=QualityConfig(edge_margin=0.5))
        with pytest.raises(ConfigError, match="quality.edge_margin"):
            from posesense.config import _validate
            _validate(cfg)

    def test_zero_num_poses(self) -> None:
        cfg = PoseSenseConfig(detector=DetectorConfig(num_poses=0))
        with pytest.raises(ConfigError, match="detector.num_poses"):
            from posesense.config import _validate
            _validate(cfg)

    def test_vote_window_less_than_min_votes(self) -> None:
        from posesense.config import SmoothingConfig
        cfg = PoseSenseConfig(smoothing=SmoothingConfig(vote_window=3, min_votes=5))
        with pytest.raises(ConfigError, match="smoothing.min_votes"):
            from posesense.config import _validate
            _validate(cfg)

    def test_negative_reconnect_attempts(self) -> None:
        cfg = PoseSenseConfig(camera=CameraConfig(reconnect_attempts=-1))
        with pytest.raises(ConfigError, match="camera.reconnect_attempts"):
            from posesense.config import _validate
            _validate(cfg)

    def test_zero_reconnect_backoff(self) -> None:
        cfg = PoseSenseConfig(camera=CameraConfig(reconnect_backoff_s=0.0))
        with pytest.raises(ConfigError, match="camera.reconnect_backoff_s"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_schema_version(self) -> None:
        from posesense.config import FeaturesConfig
        cfg = PoseSenseConfig(features=FeaturesConfig(schema_version=0))
        with pytest.raises(ConfigError, match="features.schema_version"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_coordinate_source(self) -> None:
        from posesense.config import FeaturesConfig
        cfg = PoseSenseConfig(features=FeaturesConfig(coordinate_source="mixed"))
        with pytest.raises(ConfigError, match="features.coordinate_source"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_ema_alpha(self) -> None:
        from posesense.config import SmoothingConfig
        cfg = PoseSenseConfig(smoothing=SmoothingConfig(ema_alpha=0.0))
        with pytest.raises(ConfigError, match="smoothing.ema_alpha"):
            from posesense.config import _validate
            _validate(cfg)

    def test_rejection_confidence_out_of_range(self) -> None:
        from posesense.config import RejectionConfig
        cfg = PoseSenseConfig(rejection=RejectionConfig(min_confidence=1.5))
        with pytest.raises(ConfigError, match="rejection.min_confidence"):
            from posesense.config import _validate
            _validate(cfg)

    def test_negative_rejection_margin(self) -> None:
        from posesense.config import RejectionConfig
        cfg = PoseSenseConfig(rejection=RejectionConfig(min_margin=-0.1))
        with pytest.raises(ConfigError, match="rejection.min_margin"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_training_model(self) -> None:
        from posesense.config import TrainingConfig
        cfg = PoseSenseConfig(training=TrainingConfig(default_model="xgboost"))
        with pytest.raises(ConfigError, match="training.default_model"):
            from posesense.config import _validate
            _validate(cfg)

    def test_invalid_test_session_fraction(self) -> None:
        from posesense.config import TrainingConfig
        cfg = PoseSenseConfig(training=TrainingConfig(test_session_fraction=0.0))
        with pytest.raises(ConfigError, match="training.test_session_fraction"):
            from posesense.config import _validate
            _validate(cfg)

    def test_negative_target_fps(self) -> None:
        from posesense.config import PerformanceConfig
        cfg = PoseSenseConfig(performance=PerformanceConfig(target_fps=0))
        with pytest.raises(ConfigError, match="performance.target_fps"):
            from posesense.config import _validate
            _validate(cfg)
