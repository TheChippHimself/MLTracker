"""Tests for the live inference state machine."""

from posesense.runtime.state import FrameContext, InferenceStateMachine
from posesense.types import AppState


def test_state_machine_priority() -> None:
    sm = InferenceStateMachine(debounce_frames=1, vote_window=1, min_votes=1)
    
    # 1. Camera failure
    ctx = FrameContext(camera_failed=True, has_landmarks=True, has_model=True, raw_predicted_label="A")
    out = sm.evaluate(ctx)
    assert out.state == AppState.CAMERA_UNAVAILABLE
    
    # 2. No landmarks
    ctx = FrameContext(camera_failed=False, has_landmarks=False, has_model=True)
    out = sm.evaluate(ctx)
    assert out.state == AppState.NO_PERSON
    
    # 3. Quality failure
    ctx = FrameContext(camera_failed=False, has_landmarks=True, quality_reasons=["missing nose"], has_model=True)
    out = sm.evaluate(ctx)
    assert out.state == AppState.LOW_QUALITY
    
    # 4. No model
    ctx = FrameContext(camera_failed=False, has_landmarks=True, has_model=False)
    out = sm.evaluate(ctx)
    assert out.state == AppState.NO_MODEL
    
    # 5. Rejected prediction (None label)
    ctx = FrameContext(camera_failed=False, has_landmarks=True, has_model=True, raw_predicted_label=None)
    out = sm.evaluate(ctx)
    assert out.state == AppState.UNKNOWN_POSE
    
    # 6. Confident
    ctx = FrameContext(camera_failed=False, has_landmarks=True, has_model=True, raw_predicted_label="Sitting", raw_predicted_score=0.9)
    out = sm.evaluate(ctx)
    assert out.state == AppState.CONFIDENT
    assert out.smoothed_label == "Sitting"
    assert out.smoothed_score == 0.9
