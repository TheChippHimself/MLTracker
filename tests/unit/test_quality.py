"""Tests for the QualityGate."""

from __future__ import annotations

import pytest
from tests.fixtures import (
    make_low_visibility_landmarks,
    make_partial_landmarks,
    make_standing_landmarks,
)

from posesense.config import QualityConfig
from posesense.pose.quality import QualityGate
from posesense.types import Landmark, PoseResult


@pytest.fixture()
def config() -> QualityConfig:
    return QualityConfig(
        min_visibility=0.5,
        min_presence=0.5,
        edge_margin=0.02,
        min_torso_length_norm=0.05,
        # Require nose(0), l_shoulder(11), r_shoulder(12), l_hip(23), r_hip(24)
        required_landmarks=(0, 11, 12, 23, 24),
    )


class TestQualityGate:
    """Test validation of poses against quality rules."""

    def test_passing_pose(self, config: QualityConfig) -> None:
        gate = QualityGate(config)
        lms = make_standing_landmarks()
        result = PoseResult(landmarks=lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)

        report = gate.evaluate(result)
        assert report.ok
        assert report.score == 1.0
        assert len(report.reasons) == 0

    def test_missing_required_landmark(self, config: QualityConfig) -> None:
        gate = QualityGate(config)
        # Low visibility on required landmarks
        lms = make_low_visibility_landmarks()
        # We also need to manually hide a required one to ensure it fails
        lms_list = list(lms)
        lms_list[0] = Landmark(x=0.5, y=0.1, z=0.0, visibility=0.1, presence=0.1) # NOSE
        result = PoseResult(landmarks=tuple(lms_list), world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)

        report = gate.evaluate(result)
        assert not report.ok
        assert report.score < 1.0
        assert 0 in report.missing
        assert any("NOSE" in r for r in report.reasons)

    def test_edge_margin_violation(self, config: QualityConfig) -> None:
        gate = QualityGate(config)
        lms = make_partial_landmarks()
        result = PoseResult(landmarks=lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)

        report = gate.evaluate(result)
        assert not report.ok
        assert any("too close to edge" in r for r in report.reasons)

    def test_torso_too_small(self, config: QualityConfig) -> None:
        # Require a huge torso to force failure
        cfg = QualityConfig(min_torso_length_norm=0.9)
        gate = QualityGate(cfg)
        lms = make_standing_landmarks()
        result = PoseResult(landmarks=lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)

        report = gate.evaluate(result)
        assert not report.ok
        assert any("too far" in r for r in report.reasons)
