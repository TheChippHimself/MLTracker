"""Angle calculation logic for pose features."""

from __future__ import annotations

import math

from posesense.types import Landmark


def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Calculate the interior angle (in degrees) formed by points A-B-C.

    The angle is at vertex B.
    Returns a value between 0.0 and 180.0.

    Args:
        a: First endpoint.
        b: Vertex point.
        c: Second endpoint.

    Returns:
        The angle in degrees (0.0 to 180.0).
    """
    # Using atan2 to compute the angle
    ang = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    )

    # Normalize to 0-360
    ang = ang + 360 if ang < 0 else ang

    # We want the interior angle (0-180)
    if ang > 180:
        ang = 360 - ang

    return ang
