"""Preprocessing utilities for the Loan Approval Prediction project."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    columns_to_drop: Sequence[str] = COLUMNS_TO_DROP,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into model features and target values."""
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    existing_drop_columns = [
        column for column in columns_to_drop if column in data.columns
    ]
    features = data.drop(columns=[target_column, *existing_drop_columns])
    target = data[target_column]

    return features, target


def _validate_feature_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Validate that all configured feature columns are present."""
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required feature columns: "
            f"{missing_columns}"
        )


def _create_numeric_pipeline() -> Pipeline:
    """Create preprocessing steps for numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _create_categorical_pipeline() -> Pipeline:
    """Create preprocessing steps for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def create_preprocessing_pipeline(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Create a reusable sklearn preprocessing transformer."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", _create_numeric_pipeline(), list(numeric_features)),
            (
                "categorical",
                _create_categorical_pipeline(),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor.set_output(transform="pandas")


def fit_transform_features(
    features: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Fit the preprocessor on features and return transformed features."""
    required_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    _validate_feature_columns(features, required_columns)

    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()

    transformed_features = preprocessor.fit_transform(features)

    return transformed_features, preprocessor


def transform_features(
    features: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """Transform features with an already fitted preprocessor."""
    required_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    _validate_feature_columns(features, required_columns)

    return preprocessor.transform(features)


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Return feature names produced by a fitted preprocessing transformer."""
    return list(preprocessor.get_feature_names_out())


def preprocess_train_test(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    ColumnTransformer,
]:
    """Fit on training features and transform training and test features."""
    x_train, y_train = split_features_target(train_data)
    x_test, y_test = split_features_target(test_data)

    x_train_processed, preprocessor = fit_transform_features(x_train)
    x_test_processed = transform_features(x_test, preprocessor)

    return (
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        preprocessor,
    )
