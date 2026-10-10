"""Tests for label smoothing and hysteresis."""

from posesense.runtime.smoothing import LabelSmoother, StateDebouncer


def test_label_smoother_voting() -> None:
    smoother = LabelSmoother(vote_window=5, min_votes=3)
    
    # Needs 3 votes to become confident
    assert smoother.update("A") is None
    assert smoother.update("A") is None
    assert smoother.update("A") == "A"
    
def test_label_smoother_hysteresis() -> None:
    smoother = LabelSmoother(vote_window=5, min_votes=3)
    smoother.update("A")
    smoother.update("A")
    smoother.update("A")
    assert smoother._current_label == "A"
    
    # Try to switch to B, but it shouldn't switch until B strictly beats A and hits min_votes
    smoother.update("B")
    smoother.update("B")
    # Window: A, A, A, B, B (counts: A=3, B=2) -> should still be A
    assert smoother._current_label == "A"
    
    smoother.update("B")
    # Window: A, A, B, B, B (counts: B=3, A=2) -> should switch to B
    assert smoother.update("B") == "B"

def test_state_debouncer() -> None:
    debouncer = StateDebouncer(debounce_frames=3)
    
    assert debouncer.update(True) is False
    assert debouncer.update(True) is False
    assert debouncer.update(True) is True
    
    # Resets immediately on False
    assert debouncer.update(False) is False
    assert debouncer.update(True) is False
