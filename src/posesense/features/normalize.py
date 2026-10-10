"""Normalization for landmarks: aspect ratio, translation, scaling, mirroring."""

import math
from posesense.config import FeaturesConfig
from posesense.pose.landmarks import LM
from posesense.types import Landmark, PoseResult


def aspect_ratio_correction(lms: tuple[Landmark, ...], width: int, height: int) -> list[Landmark]:
    """Convert normalized [0,1] coordinates back to pixel-proportional scale for distance math."""
    corrected = []
    for lm in lms:
        from dataclasses import replace
        # We multiply by width and height so x,y are in the same physical unit scale.
        corrected.append(replace(lm, x=lm.x * width, y=lm.y * height))
    return corrected


def get_center(lms: list[Landmark], config: FeaturesConfig) -> tuple[float, float]:
    """Calculate the origin point for normalization."""
    if config.center == "mid_hip":
        hx = (lms[LM.LEFT_HIP].x + lms[LM.RIGHT_HIP].x) / 2.0
        hy = (lms[LM.LEFT_HIP].y + lms[LM.RIGHT_HIP].y) / 2.0
        return hx, hy
    return 0.0, 0.0


def get_scale(lms: list[Landmark], config: FeaturesConfig) -> float:
    """Calculate the scaling factor (e.g. torso length)."""
    if config.scale == "torso_length":
        hx = (lms[LM.LEFT_HIP].x + lms[LM.RIGHT_HIP].x) / 2.0
        hy = (lms[LM.LEFT_HIP].y + lms[LM.RIGHT_HIP].y) / 2.0
        sx = (lms[LM.LEFT_SHOULDER].x + lms[LM.RIGHT_SHOULDER].x) / 2.0
        sy = (lms[LM.LEFT_SHOULDER].y + lms[LM.RIGHT_SHOULDER].y) / 2.0
        dist = math.hypot(sx - hx, sy - hy)
        return dist if dist > 1e-6 else 1.0
    elif config.scale == "shoulder_width":
        sx1, sy1 = lms[LM.LEFT_SHOULDER].x, lms[LM.LEFT_SHOULDER].y
        sx2, sy2 = lms[LM.RIGHT_SHOULDER].x, lms[LM.RIGHT_SHOULDER].y
        dist = math.hypot(sx1 - sx2, sy1 - sy2)
        return dist if dist > 1e-6 else 1.0
    return 1.0


def mirror_features(features: dict[str, float]) -> dict[str, float]:
    """Swap left and right features and negate horizontal components.
    Used for training augmentation. (To be implemented fully later)
    """
    # Not needed for Phase 2 capture, applied during Phase 3 training.
    return features
