"""Drawing functions for the skeleton and HUD overlay."""

from __future__ import annotations

import cv2
import numpy as np

from posesense.pose.landmarks import SKELETON_CONNECTIONS
from posesense.types import AppState, PoseResult

# Colours (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_BLUE = (255, 0, 0)


def draw_skeleton(frame: np.ndarray, pose_result: PoseResult, debug: bool = False) -> None:
    """Draw the 33-landmark skeleton on the frame.

    Args:
        frame: The OpenCV BGR frame (modified in-place).
        pose_result: The detected pose landmarks.
        debug: If True, draw landmark indices.
    """
    h, w = frame.shape[:2]
    pts = []

    # Pre-calculate pixel coordinates
    for lm in pose_result.landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts.append((px, py))

    # Draw connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        lm1 = pose_result.landmarks[start_idx]
        lm2 = pose_result.landmarks[end_idx]

        # Only draw if both are reasonably visible and present
        if (lm1.visibility > 0.1 and lm1.presence > 0.1 and
            lm2.visibility > 0.1 and lm2.presence > 0.1):

            p1 = pts[start_idx]
            p2 = pts[end_idx]
            cv2.line(frame, p1, p2, COLOR_WHITE, 2)

    # Draw landmarks
    for i, lm in enumerate(pose_result.landmarks):
        if lm.visibility > 0.1 and lm.presence > 0.1:
            px, py = pts[i]
            # Use red for face, blue for arms, green for legs to make it readable
            color = COLOR_WHITE
            if i <= 10:
                color = COLOR_RED
            elif 11 <= i <= 22:
                color = COLOR_BLUE
            elif 23 <= i <= 32:
                color = COLOR_GREEN

            cv2.circle(frame, (px, py), 4, color, -1)

            if debug:
                cv2.putText(frame, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)


def draw_hud(
    frame: np.ndarray,
    state: AppState,
    fps: float,
    label: str | None = None,
    confidence: float | None = None,
    reasons: tuple[str, ...] | None = None,
) -> None:
    """Draw the heads-up display on the frame.

    Args:
        frame: The OpenCV BGR frame (modified in-place).
        state: Current application state.
        fps: Current frames-per-second.
        label: Predicted pose label.
        confidence: Prediction confidence score (0-1).
        reasons: Reasons for LOW_QUALITY state.
    """
    h, w = frame.shape[:2]

    # FPS counter (top right)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    # State banner (top left)
    state_color = COLOR_GRAY
    if state == AppState.CONFIDENT:
        state_color = COLOR_GREEN
    elif state in (AppState.CAMERA_UNAVAILABLE, AppState.LOW_QUALITY):
        state_color = COLOR_RED
    elif state in (AppState.NO_PERSON, AppState.UNKNOWN_POSE):
        state_color = COLOR_YELLOW
    elif state == AppState.NO_MODEL:
        state_color = COLOR_BLUE

    cv2.putText(frame, f"State: {state.value}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # Prediction / Label (bottom left)
    if state in (AppState.CONFIDENT, AppState.LOW_QUALITY) and label:
        conf_str = f" ({confidence*100:.0f}%)" if confidence is not None else ""
        pred_text = f"Pose: {label}{conf_str}"
        text_color = COLOR_GREEN if state == AppState.CONFIDENT else COLOR_RED
        cv2.putText(frame, pred_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    elif state == AppState.UNKNOWN_POSE:
        cv2.putText(frame, "Pose: ???", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_YELLOW, 2)

    # Reasons (bottom right, moving up)
    if state == AppState.LOW_QUALITY and reasons:
        y_offset = h - 30
        for reason in reversed(reasons):
            cv2.putText(frame, reason, (w - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
            y_offset -= 20
