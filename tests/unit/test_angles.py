"""Tests for angle calculations."""

from __future__ import annotations

import math

from posesense.features.angles import calculate_angle
from posesense.types import Landmark


def test_calculate_angle_90_degrees() -> None:
    # A (0, 1), B (0, 0), C (1, 0) -> Angle at B is 90
    a = Landmark(0.0, 1.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(1.0, 0.0, 0.0, 1.0, 1.0)

    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 90.0, abs_tol=1e-5)


def test_calculate_angle_180_degrees() -> None:
    # A (-1, 0), B (0, 0), C (1, 0) -> Angle at B is 180
    a = Landmark(-1.0, 0.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(1.0, 0.0, 0.0, 1.0, 1.0)

    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 180.0, abs_tol=1e-5)


def test_calculate_angle_45_degrees() -> None:
    # A (0, 1), B (0, 0), C (1, 1) -> Angle at B is 45
    a = Landmark(0.0, 1.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(1.0, 1.0, 0.0, 1.0, 1.0)

    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 45.0, abs_tol=1e-5)


def test_calculate_angle_interior() -> None:
    # Ensure it returns the interior angle (<= 180) regardless of order
    a = Landmark(0.0, 1.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(1.0, 0.0, 0.0, 1.0, 1.0)

    angle1 = calculate_angle(a, b, c)
    angle2 = calculate_angle(c, b, a)

    assert math.isclose(angle1, 90.0, abs_tol=1e-5)
    assert math.isclose(angle2, 90.0, abs_tol=1e-5)
