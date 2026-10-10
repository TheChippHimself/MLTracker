"""Tests for the MediaPipe detector wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from posesense.config import DetectorConfig
from posesense.exceptions import DetectionError, ModelFileNotFoundError
from posesense.pose.mediapipe_detector import MediaPipeDetector


class TestMediaPipeDetector:
    """Test MediaPipe integration."""

    @pytest.fixture()
    def config(self, tmp_path) -> DetectorConfig:  # type: ignore[no-untyped-def]
        # Create a dummy model file
        model_file = tmp_path / "dummy.task"
        model_file.write_bytes(b"dummy")
        return DetectorConfig(
            model_variant="lite",
            model_path=str(model_file),
            running_mode="video",
        )

    def test_missing_model_raises_error(self) -> None:
        cfg = DetectorConfig(model_path="nonexistent.task")
        with pytest.raises(ModelFileNotFoundError):
            MediaPipeDetector(cfg)

    @patch("mediapipe.tasks.python.vision.PoseLandmarker")
    def test_init_success(self, mock_pl, config: DetectorConfig) -> None:  # type: ignore[no-untyped-def]
        # Should initialize without throwing
        detector = MediaPipeDetector(config)
        assert detector._landmarker is not None
        detector.close()

    @patch("mediapipe.tasks.python.vision.PoseLandmarker")
    @patch("cv2.cvtColor")
    def test_process_frame_rgb_conversion(self, mock_cv2_cvtcolor, mock_pl, config: DetectorConfig) -> None:  # type: ignore[no-untyped-def]
        mock_landmarker = MagicMock()
        mock_pl.create_from_options.return_value = mock_landmarker
        # Make the mock cvtColor return a dummy numpy array
        mock_cv2_cvtcolor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        # Mock empty result
        mock_result = MagicMock()
        mock_result.pose_landmarks = []
        mock_landmarker.detect_for_video.return_value = mock_result

        with MediaPipeDetector(config) as detector:
            frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            res = detector.process_frame(frame_bgr, 1000)

            assert res is None
            # Verify BGR to RGB conversion was called
            mock_cv2_cvtcolor.assert_called_once_with(frame_bgr, cv2.COLOR_BGR2RGB)

    @patch("mediapipe.tasks.python.vision.PoseLandmarker")
    def test_monotonic_timestamps(self, mock_pl, config: DetectorConfig) -> None:  # type: ignore[no-untyped-def]
        mock_landmarker = MagicMock()
        mock_pl.create_from_options.return_value = mock_landmarker
        mock_result = MagicMock()
        mock_result.pose_landmarks = []
        mock_landmarker.detect_for_video.return_value = mock_result

        with MediaPipeDetector(config) as detector:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)

            # Send timestamp 1000
            detector.process_frame(frame, 1000)
            args, _ = mock_landmarker.detect_for_video.call_args
            assert args[1] == 1000

            # Send timestamp 900 (backwards). Should be coerced to 1001.
            detector.process_frame(frame, 900)
            args, _ = mock_landmarker.detect_for_video.call_args
            assert args[1] == 1001

    @patch("mediapipe.tasks.python.vision.PoseLandmarker")
    def test_detection_error(self, mock_pl, config: DetectorConfig) -> None:  # type: ignore[no-untyped-def]
        mock_landmarker = MagicMock()
        mock_pl.create_from_options.return_value = mock_landmarker
        mock_landmarker.detect_for_video.side_effect = RuntimeError("Backend crash")

        with MediaPipeDetector(config) as detector:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            with pytest.raises(DetectionError, match="Backend crash"):
                detector.process_frame(frame, 100)
