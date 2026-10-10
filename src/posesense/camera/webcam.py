"""Webcam implementation of FrameSource."""

from __future__ import annotations

import logging
import sys
import time
import types

import cv2
import numpy as np

from posesense.camera.base import FrameSource
from posesense.config import CameraConfig
from posesense.exceptions import BlackFrameWarning, CameraOpenError, FrameReadError

logger = logging.getLogger(__name__)


def _get_platform_guidance() -> str:
    """Return OS-specific guidance for camera permission errors."""
    if sys.platform == "win32":
        return "Windows: Check Settings -> Privacy & security -> Camera."
    if sys.platform == "darwin":
        return "macOS: Check System Settings -> Privacy & Security -> Camera."
    if sys.platform.startswith("linux"):
        return "Linux: Ensure you are in the 'video' group and device permissions are correct."
    return "Check your OS privacy settings for the camera."


class WebcamSource(FrameSource):
    """OpenCV webcam source with error recovery and black-frame detection."""

    def __init__(self, config: CameraConfig) -> None:
        """Initialize the webcam source.

        Args:
            config: The camera configuration settings.
        """
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._consecutive_black_frames = 0
        self._failed_reads = 0

    @property
    def is_opened(self) -> bool:
        """Return True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def _apply_backend(self, index: int, backend_str: str) -> int:
        """Convert config backend string to cv2 API preference."""
        backend_map = {
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "v4l2": cv2.CAP_V4L2,
            "avfoundation": cv2.CAP_AVFOUNDATION,
        }
        return backend_map.get(backend_str.lower(), cv2.CAP_ANY)

    def open(self) -> None:
        """Open the camera, applying requested resolution.

        Raises:
            CameraOpenError: If the camera cannot be opened.
        """
        if self.is_opened:
            return

        api_pref = self._apply_backend(self.config.index, self.config.backend)
        logger.debug("Opening camera %d with backend %s", self.config.index, self.config.backend)

        self._cap = cv2.VideoCapture(self.config.index, api_pref)

        if not self._cap.isOpened():
            self.release()
            guidance = _get_platform_guidance()
            raise CameraOpenError(
                index=self.config.index,
                message="cv2.VideoCapture isOpened() returned False.",
                guidance=guidance,
            )

        # Request resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        # Read back actual resolution (cameras may ignore the request)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened. Requested %dx%d, actual %dx%d",
            self.config.index,
            self.config.width,
            self.config.height,
            actual_w,
            actual_h,
        )

        self._consecutive_black_frames = 0
        self._failed_reads = 0

    def read(self) -> np.ndarray:
        """Read a frame from the camera, handling failures and backoff.

        Returns:
            An OpenCV BGR frame (numpy array).

        Raises:
            FrameReadError: If a frame could not be read after recovery attempts.
            BlackFrameWarning: If too many consecutive black frames are detected.
        """
        if not self.is_opened:
            self.open()

        # The loop handles reconnection backoff on failure
        while True:
            assert self._cap is not None
            ret, frame = self._cap.read()

            if ret and frame is not None and frame.size > 0:
                self._failed_reads = 0
                return self._check_black_frame(frame)

            self._failed_reads += 1
            if self._failed_reads > self.config.reconnect_attempts:
                logger.error("Camera %d failed to read after %d attempts.", self.config.index, self._failed_reads)
                self.release()
                raise FrameReadError(f"Camera {self.config.index} failed to return a frame.")

            logger.warning(
                "Camera %d read failed. Attempt %d/%d. Waiting %.1f seconds.",
                self.config.index,
                self._failed_reads,
                self.config.reconnect_attempts,
                self.config.reconnect_backoff_s,
            )
            self.release()
            time.sleep(self.config.reconnect_backoff_s)

            try:
                self.open()
            except CameraOpenError as e:
                # If we can't even reopen, fast fail
                raise FrameReadError(f"Camera {self.config.index} dropped and could not be reopened.") from e

    def _check_black_frame(self, frame: np.ndarray) -> np.ndarray:
        """Check if the frame is near-black and raise a warning if persistent."""
        mean_intensity = np.mean(frame)
        if mean_intensity < self.config.black_frame_threshold:
            self._consecutive_black_frames += 1
            if self._consecutive_black_frames >= self.config.black_frame_max_consecutive:
                raise BlackFrameWarning("Camera returning black frames.")
        else:
            self._consecutive_black_frames = 0

        return frame

    def release(self) -> None:
        """Release the camera resources cleanly."""
        if self._cap is not None:
            if self._cap.isOpened():
                self._cap.release()
                logger.debug("Camera %d released.", self.config.index)
            self._cap = None

    def __enter__(self) -> WebcamSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.release()
