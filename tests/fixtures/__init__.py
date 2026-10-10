"""Synthetic landmark generators for testing.

These functions build plausible 33-landmark skeletons for each target pose
so that feature extraction, classification, and pipeline tests work without
a webcam or real images.
"""

from __future__ import annotations

from posesense.types import Landmark


def _lm(x: float, y: float, z: float = 0.0, vis: float = 0.99, pres: float = 0.99) -> Landmark:
    """Shorthand for creating a Landmark with defaults for visibility and presence."""
    return Landmark(x=x, y=y, z=z, visibility=vis, presence=pres)


def make_standing_landmarks() -> tuple[Landmark, ...]:
    """Generate a plausible standing pose (neutral upright, arms down).

    Coordinate system: image-normalised [0, 1] with origin at top-left.
    x increases right, y increases down.
    """
    # Build a dictionary with indices for clarity, then fill gaps with (0.5, 0.5)
    lms: dict[int, Landmark] = {
        0: _lm(0.50, 0.10),          # nose
        1: _lm(0.49, 0.09),          # left eye inner
        2: _lm(0.48, 0.09),          # left eye
        3: _lm(0.47, 0.09),          # left eye outer
        4: _lm(0.51, 0.09),          # right eye inner
        5: _lm(0.52, 0.09),          # right eye
        6: _lm(0.53, 0.09),          # right eye outer
        7: _lm(0.46, 0.10),          # left ear
        8: _lm(0.54, 0.10),          # right ear
        9: _lm(0.49, 0.12),          # mouth left
        10: _lm(0.51, 0.12),         # mouth right
        11: _lm(0.42, 0.25),         # left shoulder
        12: _lm(0.58, 0.25),         # right shoulder
        13: _lm(0.40, 0.40),         # left elbow
        14: _lm(0.60, 0.40),         # right elbow
        15: _lm(0.39, 0.55),         # left wrist
        16: _lm(0.61, 0.55),         # right wrist
        17: _lm(0.38, 0.56),         # left pinky
        18: _lm(0.62, 0.56),         # right pinky
        19: _lm(0.38, 0.57),         # left index
        20: _lm(0.62, 0.57),         # right index
        21: _lm(0.39, 0.56),         # left thumb
        22: _lm(0.61, 0.56),         # right thumb
        23: _lm(0.45, 0.50),         # left hip
        24: _lm(0.55, 0.50),         # right hip
        25: _lm(0.44, 0.68),         # left knee
        26: _lm(0.56, 0.68),         # right knee
        27: _lm(0.43, 0.85),         # left ankle
        28: _lm(0.57, 0.85),         # right ankle
        29: _lm(0.42, 0.87),         # left heel
        30: _lm(0.58, 0.87),         # right heel
        31: _lm(0.43, 0.89),         # left foot index
        32: _lm(0.57, 0.89),         # right foot index
    }
    return tuple(lms[i] for i in range(33))


def make_tpose_landmarks() -> tuple[Landmark, ...]:
    """T-Pose: arms extended sideways at shoulder height."""
    lms = list(make_standing_landmarks())
    # Move elbows and wrists to shoulder height, extended out
    lms[13] = _lm(0.25, 0.25)   # left elbow
    lms[14] = _lm(0.75, 0.25)   # right elbow
    lms[15] = _lm(0.10, 0.25)   # left wrist
    lms[16] = _lm(0.90, 0.25)   # right wrist
    lms[17] = _lm(0.09, 0.25)   # left pinky
    lms[18] = _lm(0.91, 0.25)   # right pinky
    lms[19] = _lm(0.09, 0.26)   # left index
    lms[20] = _lm(0.91, 0.26)   # right index
    lms[21] = _lm(0.10, 0.24)   # left thumb
    lms[22] = _lm(0.90, 0.24)   # right thumb
    return tuple(lms)


def make_arms_raised_landmarks() -> tuple[Landmark, ...]:
    """Both arms above head."""
    lms = list(make_standing_landmarks())
    lms[13] = _lm(0.40, 0.15)   # left elbow
    lms[14] = _lm(0.60, 0.15)   # right elbow
    lms[15] = _lm(0.42, 0.04)   # left wrist
    lms[16] = _lm(0.58, 0.04)   # right wrist
    lms[17] = _lm(0.42, 0.03)
    lms[18] = _lm(0.58, 0.03)
    lms[19] = _lm(0.43, 0.03)
    lms[20] = _lm(0.57, 0.03)
    lms[21] = _lm(0.41, 0.04)
    lms[22] = _lm(0.59, 0.04)
    return tuple(lms)


def make_low_visibility_landmarks() -> tuple[Landmark, ...]:
    """Landmarks with low visibility on key joints — should fail quality gate."""
    lms = list(make_standing_landmarks())
    # Set left knee and left ankle to very low visibility
    lms[25] = Landmark(x=0.44, y=0.68, z=0.0, visibility=0.1, presence=0.1)
    lms[27] = Landmark(x=0.43, y=0.85, z=0.0, visibility=0.1, presence=0.1)
    return tuple(lms)


def make_partial_landmarks() -> tuple[Landmark, ...]:
    """Landmarks near frame edge — should fail edge-margin check."""
    lms = list(make_standing_landmarks())
    # Move ankles outside frame
    lms[27] = _lm(0.43, 0.995)   # left ankle near bottom edge
    lms[28] = _lm(0.57, 0.995)   # right ankle near bottom edge
    return tuple(lms)
