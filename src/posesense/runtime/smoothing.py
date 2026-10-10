"""Smoothing and stability filters for live inference."""

import collections
from typing import Optional


class LabelSmoother:
    """Applies temporal voting and hysteresis to predicted labels."""

    def __init__(self, vote_window: int = 5, min_votes: int = 3) -> None:
        if min_votes > vote_window:
            raise ValueError("min_votes cannot be greater than vote_window.")
            
        self.vote_window = vote_window
        self.min_votes = min_votes
        
        self._history: collections.deque[Optional[str]] = collections.deque(maxlen=vote_window)
        self._current_label: Optional[str] = None
        self._current_votes: int = 0

    def update(self, new_label: Optional[str]) -> Optional[str]:
        """Update the smoother with a new label and return the smoothed label."""
        self._history.append(new_label)
        
        # Count votes for all labels in history
        counts: collections.Counter[Optional[str]] = collections.Counter(self._history)
        
        # Find the label with the most votes
        top_label, top_votes = counts.most_common(1)[0]
        
        # Hysteresis logic
        if self._current_label is not None:
            # How many votes does the current label still have?
            current_label_votes = counts.get(self._current_label, 0)
            
            # Switch only if the new top label has strictly more votes than current,
            # AND the new top label meets min_votes
            if top_label != self._current_label and top_votes > current_label_votes and top_votes >= self.min_votes:
                self._current_label = top_label
                self._current_votes = top_votes
            else:
                # Keep current label if new one isn't dominant enough
                # (but if it falls below some threshold entirely, we might drop it.
                # However, hysteresis usually implies keeping the old state until firmly replaced, 
                # unless its votes drop to 0)
                if current_label_votes == 0:
                    # Current label completely disappeared, fallback to top_label if it meets min_votes
                    if top_votes >= self.min_votes:
                        self._current_label = top_label
                        self._current_votes = top_votes
                    else:
                        self._current_label = None
                        self._current_votes = 0
        else:
            # We don't have a current label, adopt top_label if it meets min_votes
            if top_votes >= self.min_votes:
                self._current_label = top_label
                self._current_votes = top_votes
                
        return self._current_label


class StateDebouncer:
    """Debounces boolean state transitions (e.g. no person detected)."""
    
    def __init__(self, debounce_frames: int = 3) -> None:
        self.debounce_frames = debounce_frames
        self._consecutive_hits = 0
        self._state = False
        
    def update(self, condition: bool) -> bool:
        """Update the condition and return the debounced state."""
        if condition:
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.debounce_frames:
                self._state = True
        else:
            self._consecutive_hits = 0
            self._state = False
            
        return self._state
