# Design Decisions

## Pipeline

`Pipeline` combines preprocessing and the classifier into one fitted object. This prevents training and inference from drifting apart and allows deployment code to call `pipeline.predict()` directly.

## ColumnTransformer

`ColumnTransformer` applies different transformations to numeric and categorical columns while preserving a single sklearn-compatible interface. It avoids manual column-by-column preprocessing and keeps feature routing explicit.

## Logistic Regression

Logistic Regression is an interpretable, efficient baseline for binary classification. It provides a useful linear benchmark and produces probability estimates that are often easier to inspect than those from tree ensembles.

## Decision Tree

A Decision Tree captures nonlinear thresholds and interactions without requiring feature scaling. It is easy to explain visually, although it can overfit without additional constraints.

## Random Forest

Random Forest aggregates many randomized trees, reducing the variance of a single tree and providing a strong nonlinear baseline. Its `n_jobs=-1` configuration uses available CPU cores during fitting.

## XGBoost

XGBoost is a powerful gradient-boosted tree method for tabular data. It provides a competitive nonlinear comparison and can model interactions that a linear baseline may miss.

## StandardScaler

`StandardScaler` standardizes numerical features after median imputation. This is important for Logistic Regression because differently scaled variables can affect optimization and coefficient comparability. Scaling is harmless inside the shared pipeline for tree candidates.

## OneHotEncoder

`OneHotEncoder` represents categorical values without imposing a false ordinal relationship. `handle_unknown="ignore"` makes inference more robust when a category appears that was not observed during training.

## Cross Validation

Stratified cross-validation compares models across several training folds while preserving class proportions. It reduces dependence on one arbitrary validation split and keeps the final holdout set available for an unbiased final estimate.

## Streamlit

Streamlit provides a lightweight interactive interface for demonstrating the model without building a separate frontend stack. The UI delegates prediction to `src.predict`, so it remains thin and does not own ML logic.

## joblib

joblib efficiently serializes sklearn pipelines containing NumPy arrays and fitted estimators. The saved artifact includes preprocessing and classification together, which makes deployment simple. Joblib files must only be loaded from trusted sources because they use pickle-based serialization.
