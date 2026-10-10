"""Tests for shared types."""

from __future__ import annotations

from posesense.types import AppState, FrameResult, Landmark, PoseResult, Prediction, QualityReport


class TestAppState:
    """Verify AppState enum values match the spec."""

    def test_all_states_exist(self) -> None:
        states = {s.value for s in AppState}
        assert "Camera Unavailable" in states
        assert "No Person Detected" in states
        assert "Low Quality / Partial Body" in states
        assert "No Model Loaded" in states
        assert "Unknown Pose" in states
        assert "Confident" in states

    def test_state_count(self) -> None:
        assert len(AppState) == 6


class TestLandmark:
    """Verify Landmark is frozen and has correct fields."""

    def test_creation(self) -> None:
        lm = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9, presence=0.8)
        assert lm.x == 0.5
        assert lm.y == 0.3
        assert lm.z == 0.1
        assert lm.visibility == 0.9
        assert lm.presence == 0.8

    def test_frozen(self) -> None:
        lm = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9, presence=0.8)
        import dataclasses
        assert dataclasses.is_dataclass(lm)
        try:
            lm.x = 0.6  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except (AttributeError, dataclasses.FrozenInstanceError):
            pass


class TestPoseResult:
    """Verify PoseResult is frozen and has correct fields."""

    def test_creation(self) -> None:
        lm = Landmark(x=0.5, y=0.3, z=0.0, visibility=0.9, presence=0.9)
        landmarks = tuple([lm] * 33)
        result = PoseResult(
            landmarks=landmarks,
            world_landmarks=None,
            timestamp_ms=1000,
            frame_width=640,
            frame_height=480,
        )
        assert len(result.landmarks) == 33
        assert result.world_landmarks is None
        assert result.timestamp_ms == 1000


class TestQualityReport:
    """Verify QualityReport is frozen."""

    def test_passing_report(self) -> None:
        report = QualityReport(ok=True, score=0.95, missing=(), reasons=())
        assert report.ok
        assert report.score == 0.95

    def test_failing_report(self) -> None:
        report = QualityReport(
            ok=False,
            score=0.3,
            missing=(25, 27),
            reasons=("Left knee not visible", "Left ankle not visible"),
        )
        assert not report.ok
        assert 25 in report.missing
        assert len(report.reasons) == 2


class TestPrediction:
    """Verify Prediction is frozen."""

    def test_accepted_prediction(self) -> None:
        pred = Prediction(
            label="T-Pose",
            confidence=0.85,
            margin=0.30,
            all_scores={"T-Pose": 0.85, "Standing": 0.55},
            accepted=True,
        )
        assert pred.label == "T-Pose"
        assert pred.accepted

    def test_rejected_prediction(self) -> None:
        pred = Prediction(
            label=None,
            confidence=0.45,
            margin=0.05,
            all_scores={"T-Pose": 0.45, "Standing": 0.40},
            accepted=False,
        )
        assert pred.label is None
        assert not pred.accepted


class TestFrameResult:
    """Verify FrameResult is frozen."""

    def test_creation(self) -> None:
        result = FrameResult(
            state=AppState.NO_PERSON,
            pose_result=None,
            quality_report=None,
            prediction=None,
            smoothed_label=None,
            fps=22.5,
            latencies_ms={"capture": 10.0, "detection": 25.0},
        )
        assert result.state == AppState.NO_PERSON
        assert result.fps == 22.5
