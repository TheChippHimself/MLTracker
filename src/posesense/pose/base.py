"""Base protocols for pose detection."""

from __future__ import annotations

import types
from typing import Any, Protocol

import numpy as np

from posesense.types import PoseResult


class PoseDetector(Protocol):
    """Protocol for a pose detection model (e.g., MediaPipe)."""

    def process_frame(self, frame_bgr: np.ndarray[Any, Any], timestamp_ms: int) -> PoseResult | None:
        """Process a single frame and return the detected pose.

        Args:
            frame_bgr: OpenCV BGR image array.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            PoseResult if a person is detected, or None if no person is found.

        Raises:
            DetectorError: If processing fails.
        """
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...

    def __enter__(self) -> PoseDetector:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        ...
