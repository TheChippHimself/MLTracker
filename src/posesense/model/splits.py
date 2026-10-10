"""Data splitting utilities ensuring session-aware separation."""

from typing import Any
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sets without leaking sessions.
    
    All rows with the same session_id will be in either train or test, never both.
    """
    if "session_id" not in df.columns:
        raise ValueError("DataFrame must contain 'session_id' column for group splitting.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Use session_id as the group
    train_idx, test_idx = next(gss.split(df, groups=df["session_id"]))
    
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
