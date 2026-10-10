"""Live inference state machine."""

from dataclasses import dataclass
from typing import Optional

from posesense.types import AppState
from posesense.runtime.smoothing import LabelSmoother, StateDebouncer


@dataclass
class FrameContext:
    """Inputs required to evaluate the state for a single frame."""
    camera_failed: bool = False
    has_landmarks: bool = False
    quality_reasons: Optional[list[str]] = None
    has_model: bool = False
    raw_predicted_label: Optional[str] = None
    raw_predicted_score: float = 0.0


@dataclass
class StateOutput:
    """The evaluated state for the UI."""
    state: AppState
    smoothed_label: Optional[str] = None
    smoothed_score: float = 0.0
    quality_reasons: Optional[list[str]] = None


class InferenceStateMachine:
    """Evaluates the application state for each frame in priority order."""

    def __init__(self, debounce_frames: int = 3, vote_window: int = 5, min_votes: int = 3) -> None:
        self.person_debouncer = StateDebouncer(debounce_frames=debounce_frames)
        self.smoother = LabelSmoother(vote_window=vote_window, min_votes=min_votes)
        
        # We also want to smooth the score of the winning label slightly, 
        # or just pass through the latest score of the winning label.
        self._last_score = 0.0

    def evaluate(self, ctx: FrameContext) -> StateOutput:
        """Evaluate the state machine based on the frame context."""
        
        # 1. Camera failure
        if ctx.camera_failed:
            self._reset_smoothers()
            return StateOutput(AppState.CAMERA_UNAVAILABLE)
            
        # 2. No landmarks (after debounce)
        # Note: condition is "no landmarks", so we pass `not ctx.has_landmarks` to debouncer
        no_person_debounced = self.person_debouncer.update(not ctx.has_landmarks)
        if no_person_debounced:
            self.smoother.update(None) # Feed None to smoother to clear it eventually
            return StateOutput(AppState.NO_PERSON)
            
        # If we have landmarks, but debouncer hasn't cleared yet, we might temporarily 
        # still be in NO_PERSON, but since debouncer clears instantly on False (in our impl),
        # we proceed.
        
        # 3. Quality gate failed
        is_low_quality = bool(ctx.quality_reasons)
            
        # 4. No model loaded
        if not ctx.has_model:
            self.smoother.update(None)
            return StateOutput(AppState.LOW_QUALITY if is_low_quality else AppState.NO_MODEL, quality_reasons=ctx.quality_reasons)
            
        # 5/6. Classification
        smoothed_label = self.smoother.update(ctx.raw_predicted_label)
        
        if ctx.raw_predicted_label == smoothed_label and smoothed_label is not None:
            # Only update score if the raw label matches the smoothed one
            self._last_score = ctx.raw_predicted_score
            
        if smoothed_label is None:
            state = AppState.LOW_QUALITY if is_low_quality else AppState.UNKNOWN_POSE
            return StateOutput(state, quality_reasons=ctx.quality_reasons)
            
        state = AppState.LOW_QUALITY if is_low_quality else AppState.CONFIDENT
        return StateOutput(state, smoothed_label=smoothed_label, smoothed_score=self._last_score, quality_reasons=ctx.quality_reasons)

    def _reset_smoothers(self) -> None:
        """Reset internal temporal state."""
        self.person_debouncer._consecutive_hits = 0
        self.person_debouncer._state = False
        self.smoother._history.clear()
        self.smoother._current_label = None
        self.smoother._current_votes = 0
        self._last_score = 0.0
