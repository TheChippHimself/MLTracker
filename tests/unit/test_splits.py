"""Tests for dataset splitting."""

import pandas as pd
import pytest

from posesense.model.splits import create_train_test_split


def test_create_train_test_split() -> None:
    # Create synthetic dataframe
    df = pd.DataFrame({
        "session_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "val": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    
    train_df, test_df = create_train_test_split(df, test_size=0.25, random_state=42)
    
    # Check no leakage of sessions
    train_sessions = set(train_df["session_id"].unique())
    test_sessions = set(test_df["session_id"].unique())
    
    assert train_sessions.isdisjoint(test_sessions)
    assert len(train_df) + len(test_df) == len(df)
    
def test_create_train_test_split_missing_session() -> None:
    df = pd.DataFrame({"label": ["A"], "val": [1]})
    with pytest.raises(ValueError, match="must contain 'session_id'"):
        create_train_test_split(df)
