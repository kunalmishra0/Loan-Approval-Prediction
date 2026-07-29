"""Project-wide configuration for the Loan Approval Prediction system."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Root directory of the project."""

DATA_DIR = PROJECT_ROOT / "data"
"""Directory containing all dataset files."""

RAW_DATA_DIR = DATA_DIR / "raw"
"""Directory containing original Kaggle dataset files."""

PROCESSED_DATA_DIR = DATA_DIR / "processed"
"""Directory containing cleaned or transformed dataset files."""

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
"""Path to the raw training dataset."""

TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
"""Path to the raw test dataset."""

PROCESSED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "processed_train.csv"
"""Path to the optional processed training dataset."""

MODELS_DIR = PROJECT_ROOT / "models"
"""Directory where trained model artifacts are stored."""

MODEL_PATH = MODELS_DIR / "loan_model.pkl"
"""Path where the serialized trained model pipeline is saved."""

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
"""Path where model metrics and training metadata are saved."""

REPORTS_DIR = PROJECT_ROOT / "reports"
"""Directory for evaluation summaries and project reports."""

FIGURES_DIR = REPORTS_DIR / "figures"
"""Directory for plots generated during EDA and model evaluation."""

RANDOM_SEED = 42
"""Random seed used for reproducible splits and model training."""

TEST_SIZE = 0.2
"""Proportion of training data reserved for validation."""

TARGET_COLUMN = "Loan_Status"
"""Prediction target column in the Kaggle loan training dataset."""

COLUMNS_TO_DROP = ["Loan_ID"]
"""Identifier or non-predictive columns removed before model training."""

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
"""Numeric input features used by the preprocessing pipeline."""

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
"""Categorical input features used by the preprocessing pipeline."""

OUTPUT_DIRS = [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]
"""Project output directories that may need to be created by scripts."""
