"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from posesense.exceptions import (
    BlackFrameWarning,
    CameraOpenError,
    ConfigError,
    DatasetImportError,
    DatasetIOError,
    DetectorInitError,
    FeatureSchemaError,
    FrameReadError,
    InsufficientDataError,
    InsufficientSessionsError,
    LabelError,
    ModelFileNotFoundError,
    ModelLoadError,
    PoseSenseError,
)


class TestExceptionHierarchy:
    """Verify the exception inheritance tree."""

    def test_all_inherit_from_base(self) -> None:
        for exc_class in [
            CameraOpenError, FrameReadError, BlackFrameWarning,
            DetectorInitError, ModelFileNotFoundError,
            FeatureSchemaError,
            DatasetIOError, DatasetImportError, LabelError,
            InsufficientDataError, InsufficientSessionsError,
            ModelLoadError, ConfigError,
        ]:
            assert issubclass(exc_class, PoseSenseError)

    def test_camera_open_error_has_guidance(self) -> None:
        exc = CameraOpenError(index=0, message="not found", guidance="Check USB")
        assert "camera 0" in str(exc)
        assert "Check USB" in str(exc)
        assert exc.index == 0
        assert exc.guidance == "Check USB"

    def test_camera_open_error_without_guidance(self) -> None:
        exc = CameraOpenError(index=1, message="busy")
        assert "camera 1" in str(exc)
        assert exc.guidance == ""

    def test_config_error_carries_key_and_value(self) -> None:
        exc = ConfigError(key="camera.index", value=-1, reason="must be ≥ 0")
        assert "camera.index" in str(exc)
        assert "-1" in str(exc)
        assert exc.key == "camera.index"
        assert exc.value == -1

    def test_model_file_not_found_has_instructions(self) -> None:
        exc = ModelFileNotFoundError(path="models/missing.task")
        assert "download" in str(exc).lower()
        assert exc.path == "models/missing.task"

    def test_detector_init_error_has_version(self) -> None:
        exc = DetectorInitError("bad arg", installed_version="0.10.14")
        assert "0.10.14" in str(exc)
        assert exc.installed_version == "0.10.14"

    def test_feature_schema_error(self) -> None:
        exc = FeatureSchemaError("wrong length: expected 34, got 33")
        assert "wrong length" in str(exc)

    def test_dataset_io_error(self) -> None:
        exc = DatasetIOError(path="/tmp/data.csv", cause="Permission denied")
        assert "/tmp/data.csv" in str(exc)
        assert "Permission denied" in str(exc)

    def test_model_load_error(self) -> None:
        exc = ModelLoadError(path="model.joblib", reason="schema mismatch")
        assert "model.joblib" in str(exc)
        assert "schema mismatch" in str(exc)

    def test_base_exception_catches_all(self) -> None:
        with pytest.raises(PoseSenseError):
            raise ConfigError("key", "val", "bad")

        with pytest.raises(PoseSenseError):
            raise CameraOpenError(0, "fail")

        with pytest.raises(PoseSenseError):
            raise InsufficientSessionsError("need more sessions")
