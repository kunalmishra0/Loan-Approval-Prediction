# Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── docs/
│   ├── architecture.md
│   ├── design_decisions.md
│   ├── future_scope.md
│   └── project_structure.md
├── models/
│   ├── loan_model.pkl
│   └── model_metadata.json
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── reports/
│   └── figures/
│       └── confusion_matrix.png
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
└── requirements.txt
```

## Folders

### `app/`

Contains the user-facing Streamlit application. It handles layout, form controls, sidebar metadata, result presentation, and user-facing errors. It does not train models or duplicate preprocessing.

### `data/`

Stores project datasets. `raw/` contains source CSV files obtained from the Kaggle dataset. A future project version may add `processed/` for explicitly versioned analytical outputs, although the production model pipeline currently performs transformations in memory.

### `docs/`

Contains architecture, design rationale, repository structure, and future-scope documentation for maintainers, reviewers, and portfolio readers.

### `models/`

Contains the serialized complete sklearn pipeline and human-readable metadata. The pipeline includes both preprocessing and the selected classifier.

### `notebooks/`

Contains exploratory analysis. The EDA notebook explains data quality, target balance, distributions, relationships, correlations, outliers, and modeling motivation.

### `reports/`

Stores generated evaluation outputs such as the final confusion matrix.

### `src/`

Contains the application and ML backend modules. The modules are intentionally separated by responsibility to make the workflow reusable and testable.

## Files

### `src/config.py`

Defines project-root-relative paths, target and feature names, random seed, test size, model artifact locations, and output directories.

### `src/data_loader.py`

Loads `train.csv`, checks file availability, catches common CSV errors, and validates that the dataset is non-empty and contains the target column.

### `src/preprocessing.py`

Defines feature-target separation and a `ColumnTransformer` containing numeric imputation plus scaling and categorical imputation plus one-hot encoding. It also exposes feature-name recovery utilities.

### `src/train.py`

Runs the training workflow: target encoding, stratified splitting, candidate-model construction, cross-validation, model selection, final holdout evaluation, confusion-matrix generation, pipeline serialization, and metadata serialization.

### `src/predict.py`

Loads the saved pipeline and metadata, validates inference schema, calls the pipeline directly, returns probabilities and confidence, and converts encoded outputs into business-friendly labels.

### `app/streamlit_app.py`

Provides the interactive web UI and delegates inference to `src.predict`.

### `notebooks/01_exploratory_data_analysis.ipynb`

Documents the exploratory analysis that motivated imputation, encoding, scaling, and model selection.

### `models/loan_model.pkl`

The saved preprocessing-plus-classifier pipeline used for inference. It is intentionally tracked while small enough to keep a fresh clone runnable.

### `models/model_metadata.json`

Stores the selected algorithm, final holdout metrics, cross-validation comparison, feature names, target mappings, artifact path, timestamp, and runtime versions.

### `reports/figures/confusion_matrix.png`

Visualizes final holdout classification errors for the selected model.

### `requirements.txt`

Pins the direct project dependencies for Python 3.11.

### `.python-version`

Declares Python 3.11 for tools that support pyenv-style version files.

### `.gitignore`

Excludes virtual environments, caches, logs, IDE metadata, operating-system files, and local secrets.

### `LICENSE`

Provides the MIT license for the project source and documentation. Dataset usage remains subject to its original Kaggle terms.

### `README.md`

Provides the project overview, setup instructions, usage commands, architecture summary, metrics, limitations, and portfolio context.
