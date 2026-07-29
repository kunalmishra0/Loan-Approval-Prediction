"""Prediction utilities for the Loan Approval Prediction project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


APPROVAL_LABELS = {
    "0": "Loan Rejected",
    "1": "Loan Approved",
    "N": "Loan Rejected",
    "Y": "Loan Approved",
}


def load_pipeline(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the trained preprocessing-classifier pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model pipeline was not found at: {model_path}"
        )

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded model artifact is not an sklearn Pipeline.")

    logger.info("Loaded model pipeline from %s", model_path)
    return pipeline


def load_metadata(
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """Load model metadata if it exists."""
    if not metadata_path.exists():
        logger.warning("Model metadata was not found at %s", metadata_path)
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    logger.info("Loaded model metadata from %s", metadata_path)
    return metadata


def _to_dataframe(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Convert supported input formats into a pandas DataFrame."""
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    raise TypeError("Input data must be a dictionary or pandas DataFrame.")


def validate_input(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Validate prediction input and return features in model schema order."""
    data = _to_dataframe(input_data)
    if data.empty:
        raise ValueError("Prediction input is empty.")

    required_features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    missing_features = [
        feature for feature in required_features if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Prediction input is missing required features: "
            f"{missing_features}"
        )

    return data[required_features]


def decode_prediction(
    prediction: Any,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Convert an encoded model output into a business-friendly label."""
    raw_prediction = str(prediction)
    inverse_mapping = {}

    if metadata:
        inverse_mapping = metadata.get("inverse_target_mapping", {})

    original_label = inverse_mapping.get(raw_prediction, raw_prediction)
    return APPROVAL_LABELS.get(str(original_label), str(original_label))


def predict_proba(
    input_data: dict[str, Any] | pd.DataFrame,
    pipeline: Pipeline | None = None,
) -> pd.DataFrame | None:
    """Return class probabilities when supported by the trained pipeline."""
    pipeline = pipeline or load_pipeline()
    data = validate_input(input_data)

    if not hasattr(pipeline, "predict_proba"):
        logger.warning("The trained pipeline does not support predict_proba.")
        return None

    probabilities = pipeline.predict_proba(data)
    classes = [str(class_label) for class_label in pipeline.classes_]
    probability_columns = [
        f"probability_class_{class_label}" for class_label in classes
    ]

    return pd.DataFrame(probabilities, columns=probability_columns)


def _get_approval_probability(probability_df: pd.DataFrame) -> pd.Series:
    """Return the probability of the approved class from model output."""
    for column in ("probability_class_1", "probability_class_Y"):
        if column in probability_df:
            return probability_df[column]

    raise ValueError(
        "The trained model does not expose a recognizable approval class."
    )


def predict(
    input_data: dict[str, Any] | pd.DataFrame,
    pipeline: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate loan approval predictions for new applicant records."""
    pipeline = pipeline or load_pipeline()
    metadata = metadata if metadata is not None else load_metadata()
    data = validate_input(input_data)

    raw_predictions = pipeline.predict(data)
    probability_df = predict_proba(data, pipeline)

    results = pd.DataFrame(
        {
            "raw_prediction": raw_predictions,
            "prediction": [
                decode_prediction(prediction, metadata)
                for prediction in raw_predictions
            ],
        }
    )

    if probability_df is not None:
        results = pd.concat([results, probability_df], axis=1)
        results["probability"] = _get_approval_probability(probability_df)
        results["confidence"] = probability_df.max(axis=1)

    return results


def main() -> None:
    """Run a minimal CLI prediction demo with one sample applicant."""
    sample_application = {
        "ApplicantIncome": 6500,
        "CoapplicantIncome": 1800.0,
        "LoanAmount": 180.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }

    pipeline = load_pipeline()
    metadata = load_metadata()
    result = predict(sample_application, pipeline, metadata).iloc[0]

    probability = result.get("probability", "Not available")
    confidence = result.get("confidence", "Not available")

    print(f"Predicted class: {result['prediction']}")
    print(f"Probability: {probability}")
    print(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()


