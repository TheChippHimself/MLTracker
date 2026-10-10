"""Tests for camera implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from posesense.camera.webcam import WebcamSource
from posesense.config import CameraConfig
from posesense.exceptions import BlackFrameWarning, CameraOpenError, FrameReadError


class TestWebcamSource:
    """Test the OpenCV webcam source wrapper."""

    @pytest.fixture()
    def config(self) -> CameraConfig:
        return CameraConfig(
            index=99,
            width=320,
            height=240,
            reconnect_attempts=2,
            reconnect_backoff_s=0.01,  # Fast for tests
            black_frame_threshold=5.0,
            black_frame_max_consecutive=3,
        )

    def test_open_success(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 320.0

        with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
            cam = WebcamSource(config)
            cam.open()

            mock_vc.assert_called_once_with(99, cv2.CAP_ANY)
            mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 320)
            assert cam.is_opened

    def test_open_failure(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = WebcamSource(config)
            with pytest.raises(CameraOpenError, match="Cannot open camera 99"):
                cam.open()
            assert not cam.is_opened

    def test_read_success(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.ones((240, 320, 3), dtype=np.uint8) * 100
        mock_cap.read.return_value = (True, fake_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap), WebcamSource(config) as cam:
            frame = cam.read()
            assert frame.shape == (240, 320, 3)

    def test_read_black_frame_warning(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Completely black frame
        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, black_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap), WebcamSource(config) as cam:
            cam.read()
            cam.read()
            with pytest.raises(BlackFrameWarning):
                cam.read()  # 3rd consecutive triggers warning

    def test_read_failure_and_recovery(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.ones((240, 320, 3), dtype=np.uint8) * 100

        # Fail once, then succeed
        mock_cap.read.side_effect = [(False, None), (True, fake_frame)]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("time.sleep") as mock_sleep:
                with WebcamSource(config) as cam:
                    frame = cam.read()
                    assert frame is not None
                    mock_sleep.assert_called_once_with(0.01)

    def test_read_failure_exhausts_retries(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Always fail
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap), patch("time.sleep"):
            with WebcamSource(config) as cam:
                with pytest.raises(FrameReadError, match="failed to return a frame"):
                    cam.read()

    def test_release_is_idempotent(self, config: CameraConfig) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = WebcamSource(config)
            cam.open()
            cam.release()
            cam.release()  # Should not crash
            assert mock_cap.release.call_count == 1
