"""Base protocols for camera and frame sources."""

from __future__ import annotations

import types
from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    """Protocol for any source of video frames (webcam, file, etc.)."""

    @property
    def is_opened(self) -> bool:
        """Return True if the source is open and ready to read."""
        ...

    def open(self) -> None:
        """Open the frame source.

        Raises:
            CameraOpenError: If the source cannot be opened.
        """
        ...

    def read(self) -> np.ndarray:
        """Read the next frame.

        Returns:
            An OpenCV BGR frame (numpy array).

        Raises:
            FrameReadError: If a frame could not be read.
            BlackFrameWarning: If the frame is too dark.
        """
        ...

    def release(self) -> None:
        """Release the frame source. Safe to call multiple times."""
        ...

    def __enter__(self) -> FrameSource:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        ...
