"""Model training pipeline for the Loan Approval Prediction project."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    OUTPUT_DIRS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_training_data
from src.preprocessing import (
    create_preprocessing_pipeline,
    get_transformed_feature_names,
    split_features_target,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CV_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    """Load the validated training dataset."""
    try:
        return load_training_data()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset loading failed: %s", exc)
        raise


def prepare_target(target: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Encode the target variable for model training."""
    if target.isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains nulls.")

    if target.nunique() < 2:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain at least two classes."
        )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    target_mapping = {
        str(label): int(index)
        for index, label in enumerate(label_encoder.classes_)
    }

    return pd.Series(encoded_target, index=target.index), target_mapping


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into training and testing sets."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )


def build_models() -> dict[str, BaseEstimator]:
    """Create candidate classifiers for model comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.05,
            max_depth=3,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }


def build_training_pipeline(model: BaseEstimator) -> Pipeline:
    """Combine preprocessing and classifier into one sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessing_pipeline()),
            ("classifier", model),
        ]
    )


def train_single_model(
    model_name: str,
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train one model as a complete preprocessing-classifier pipeline."""
    pipeline = build_training_pipeline(model)
    pipeline.fit(x_train, y_train)
    logger.info("Trained model: %s", model_name)

    return pipeline


def _get_positive_class_scores(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
) -> Any:
    """Return positive-class scores for ROC AUC calculation."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]

    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(x_test)

    return pipeline.predict(x_test)


def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained pipeline on the holdout test set."""
    predictions = pipeline.predict(x_test)
    positive_class_scores = _get_positive_class_scores(pipeline, x_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, positive_class_scores),
    }


def _create_cv_strategy() -> StratifiedKFold:
    """Create a reproducible stratified cross-validation strategy."""
    return StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def _create_scoring_metrics() -> dict[str, str | Any]:
    """Create model selection metrics for cross-validation."""
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }


def cross_validate_model(
    model_name: str,
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float | str]:
    """Evaluate one model using stratified cross-validation."""
    pipeline = build_training_pipeline(model)
    scores = cross_validate(
        estimator=pipeline,
        X=x_train,
        y=y_train,
        cv=_create_cv_strategy(),
        scoring=_create_scoring_metrics(),
    )

    logger.info("Cross-validated model: %s", model_name)

    return {
        "Algorithm": model_name,
        "Accuracy": float(scores["test_accuracy"].mean()),
        "Precision": float(scores["test_precision"].mean()),
        "Recall": float(scores["test_recall"].mean()),
        "F1 Score": float(scores["test_f1"].mean()),
        "ROC AUC": float(scores["test_roc_auc"].mean()),
    }


def compare_models(
    models: dict[str, BaseEstimator],
    x_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[str, Pipeline, pd.DataFrame]:
    """Compare models with CV and fit the selected model on training data."""
    if isinstance(y_train, pd.DataFrame) and isinstance(x_test, pd.Series):
        y_train = x_test
        logger.info("Received legacy compare_models call signature.")

    if y_train is None:
        raise ValueError("Training target is required for model comparison.")

    if x_test is not None or y_test is not None:
        logger.info("Ignoring test data during cross-validated model selection.")

    results: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        try:
            metrics = cross_validate_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
            )
        except Exception as exc:
            logger.exception(
                "Cross-validation failed for %s: %s",
                model_name,
                exc,
            )
            continue

        results.append(metrics)

    if not results:
        raise RuntimeError("All candidate model validation runs failed.")

    comparison_df = pd.DataFrame(results).sort_values(
        by=["F1 Score", "ROC AUC"],
        ascending=False,
    )

    best_algorithm = str(comparison_df.iloc[0]["Algorithm"])
    best_pipeline = train_single_model(
        best_algorithm,
        models[best_algorithm],
        x_train,
        y_train,
    )

    return best_algorithm, best_pipeline, comparison_df.reset_index(drop=True)


def save_model(pipeline: Pipeline, model_path: Path = MODEL_PATH) -> None:
    """Persist the complete preprocessing-classifier pipeline."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved best model pipeline to %s", model_path)


def save_metadata(
    metadata: dict[str, Any],
    metadata_path: Path = MODEL_METADATA_PATH,
) -> None:
    """Persist model training metadata as JSON."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Saved model metadata to %s", metadata_path)


def save_confusion_matrix(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> Path:
    """Save a confusion matrix image for the final selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pipeline.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(values_format="d")
    display.figure_.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(display.figure_)
    logger.info("Saved confusion matrix to %s", output_path)

    return output_path


def _create_output_directories() -> None:
    """Create configured output directories if they do not exist."""
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)


def _build_metadata(
    best_algorithm: str,
    best_pipeline: Pipeline,
    comparison_df: pd.DataFrame,
    final_test_metrics: dict[str, float],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    target_mapping: dict[str, int],
    confusion_matrix_path: Path,
) -> dict[str, Any]:
    """Build serializable metadata for the selected model."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    inverse_target_mapping = {
        str(encoded_value): label
        for label, encoded_value in target_mapping.items()
    }

    return {
        "best_algorithm": best_algorithm,
        "accuracy": float(final_test_metrics["Accuracy"]),
        "precision": float(final_test_metrics["Precision"]),
        "recall": float(final_test_metrics["Recall"]),
        "f1": float(final_test_metrics["F1 Score"]),
        "roc_auc": float(final_test_metrics["ROC AUC"]),
        "number_of_training_samples": int(x_train.shape[0]),
        "number_of_testing_samples": int(x_test.shape[0]),
        "number_of_input_features": int(x_train.shape[1]),
        "transformed_feature_names": get_transformed_feature_names(
            preprocessor
        ),
        "target_mapping": target_mapping,
        "inverse_target_mapping": inverse_target_mapping,
        "confusion_matrix_path": str(confusion_matrix_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "cross_validation_comparison": comparison_df.to_dict(
            orient="records"
        ),
    }


def main() -> None:
    """Run the complete model training workflow."""
    _create_output_directories()
    data = load_dataset()
    features, target = split_features_target(data)
    encoded_target, target_mapping = prepare_target(target)
    x_train, x_test, y_train, y_test = split_data(features, encoded_target)

    models = build_models()
    best_algorithm, best_pipeline, comparison_df = compare_models(
        models=models,
        x_train=x_train,
        y_train=y_train,
    )
    final_test_metrics = evaluate_model(best_pipeline, x_test, y_test)
    confusion_matrix_path = save_confusion_matrix(
        best_pipeline,
        x_test,
        y_test,
    )

    metadata = _build_metadata(
        best_algorithm=best_algorithm,
        best_pipeline=best_pipeline,
        comparison_df=comparison_df,
        final_test_metrics=final_test_metrics,
        x_train=x_train,
        x_test=x_test,
        target_mapping=target_mapping,
        confusion_matrix_path=confusion_matrix_path,
    )

    save_model(best_pipeline)
    save_metadata(metadata)
    logger.info("Best model selected: %s", best_algorithm)


if __name__ == "__main__":
    main()


