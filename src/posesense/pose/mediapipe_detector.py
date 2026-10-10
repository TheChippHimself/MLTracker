"""MediaPipe PoseLandmarker implementation."""

from __future__ import annotations

import logging
import types
from typing import Any

import cv2
import numpy as np

from posesense.config import DetectorConfig
from posesense.exceptions import DetectionError, DetectorInitError, ModelFileNotFoundError
from posesense.pose.base import PoseDetector
from posesense.types import Landmark, PoseResult

logger = logging.getLogger(__name__)


def _mediapipe_version() -> str:
    """Safely get the installed MediaPipe version."""
    try:
        import mediapipe as mp
        return getattr(mp, "__version__", "unknown")
    except ImportError:
        return "not installed"


class MediaPipeDetector(PoseDetector):
    """Pose detector using MediaPipe Tasks API PoseLandmarker."""

    def __init__(self, config: DetectorConfig) -> None:
        """Initialize the MediaPipe detector.

        Args:
            config: Detection configuration including model path and confidence.

        Raises:
            ModelFileNotFoundError: If the .task file does not exist.
            DetectorInitError: If MediaPipe fails to initialize.
        """
        self.config = config
        self._landmarker = None
        self._last_timestamp_ms = -1

        import pathlib

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = pathlib.Path(config.model_path)
        if not model_path.exists():
            raise ModelFileNotFoundError(str(model_path))

        try:
            # Map running mode string to enum
            rm_map = {
                "video": vision.RunningMode.VIDEO,
                "image": vision.RunningMode.IMAGE,
                "live_stream": vision.RunningMode.LIVE_STREAM,
            }
            running_mode = rm_map.get(config.running_mode, vision.RunningMode.VIDEO)

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_poses=config.num_poses,
                min_pose_detection_confidence=config.min_pose_detection_confidence,
                min_pose_presence_confidence=config.min_pose_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp_image = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
            logger.info("MediaPipe PoseLandmarker initialized (%s mode)", config.running_mode)

        except Exception as e:
            version = _mediapipe_version()
            raise DetectorInitError(f"Failed to initialize MediaPipe: {e}", version) from e

    def process_frame(self, frame_bgr: np.ndarray[Any, Any], timestamp_ms: int) -> PoseResult | None:
        """Process a frame and return the detected pose, or None.

        Args:
            frame_bgr: OpenCV BGR image array.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            PoseResult if a person is detected, or None.

        Raises:
            DetectionError: If MediaPipe processing crashes.
        """
        if self._landmarker is None:
            raise DetectionError("Detector is not initialized.")

        # Enforce monotonically increasing timestamps (required by MediaPipe video mode)
        if self.config.running_mode == "video":
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

        try:
            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = self._mp_image(image_format=self._mp_format, data=frame_rgb)

            if self.config.running_mode == "video":
                result = self._landmarker.detect_for_video(mp_img, timestamp_ms)
            else:
                result = self._landmarker.detect(mp_img)
        except Exception as e:
            raise DetectionError(f"MediaPipe detection failed: {e}") from e

        if not result.pose_landmarks:
            return None

        # Process only the first detected person (multi-person out of scope)
        norm_landmarks = result.pose_landmarks[0]

        # World landmarks are optional; might be empty if the model doesn't support them
        world_landmarks = None
        if result.pose_world_landmarks:
            w_lms = result.pose_world_landmarks[0]
            world_landmarks = tuple(
                Landmark(
                    x=lm.x, y=lm.y, z=lm.z,
                    visibility=lm.visibility if getattr(lm, "has_visibility", True) and lm.visibility is not None else 1.0,
                    presence=lm.presence if getattr(lm, "has_presence", True) and lm.presence is not None else 1.0,
                ) for lm in w_lms
            )

        landmarks = tuple(
            Landmark(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if getattr(lm, "has_visibility", True) and lm.visibility is not None else 1.0,
                presence=lm.presence if getattr(lm, "has_presence", True) and lm.presence is not None else 1.0,
            ) for lm in norm_landmarks
        )

        h, w = frame_bgr.shape[:2]

        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            timestamp_ms=timestamp_ms,
            frame_width=w,
            frame_height=h,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as e:
                logger.warning("Error closing MediaPipe detector: %s", e)
            finally:
                self._landmarker = None
                logger.debug("MediaPipe detector closed.")

    def __enter__(self) -> MediaPipeDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()
