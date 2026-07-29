"""Dataset loading utilities for the Loan Approval Prediction project."""

from pathlib import Path

import pandas as pd

from src.config import TARGET_COLUMN, TRAIN_DATA_PATH


def validate_dataset(data: pd.DataFrame, target_column: str) -> None:
    """Validate that a loaded dataset is usable for supervised training."""
    if data.empty:
        raise ValueError("Training dataset is empty.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )


def load_training_data(file_path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load the training dataset from disk and validate its basic structure."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Training dataset was not found at: {file_path}"
        )

    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Training dataset is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Training dataset could not be parsed as CSV: {file_path}"
        ) from exc

    validate_dataset(data, TARGET_COLUMN)
    return data
