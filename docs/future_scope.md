# Future Scope

## Short Term

- Add unit and integration tests for data loading, schema validation, preprocessing, prediction, and label decoding.
- Keep the executed EDA notebook outputs synchronized with the current dataset and model artifacts.
- Add dataset schema validation and stricter input value validation.
- Report cross-validation standard deviations alongside mean metrics.
- Add a model reload control or artifact-version-aware Streamlit cache.
- Add dataset attribution, screenshots, and a complete usage walkthrough to the README.

## Medium Term

- Evaluate Model Calibration with reliability curves, Brier score, and calibrated classifiers.
- Add subgroup **Fairness** analysis for gender, education, marital status, and property area.
- Add CI/CD for linting, tests, notebook execution, and artifact checks.
- Add Docker packaging for consistent local and hosted execution.
- Expose the inference layer through FastAPI for programmatic clients.
- Add threshold analysis so business stakeholders can study approval-recall tradeoffs.

## Production Improvements

- Deploy the service through a controlled Cloud Deployment environment.
- Store models in a versioned model registry rather than relying on local pickle files.
- Add model and data **Monitoring**, including drift, missingness, latency, error rates, and prediction distribution.
- Add structured audit logs, access controls, artifact checksums, and dependency vulnerability scanning.
- Add rollback support and explicit model/data compatibility checks.
- Establish governance for privacy, retention, human review, adverse-action explanations, and regulatory compliance.
- Add load testing, health checks, readiness checks, and observability dashboards.

## Research Improvements

- Use Explainable AI techniques such as SHAP for local and global feature explanations.
- Compare calibrated linear, tree, and gradient-boosted models under repeated cross-validation.
- Investigate class-weighting and decision-threshold optimization based on the cost of false approvals and false rejections.
- Study fairness-aware learning and constrained optimization.
- Evaluate temporal validation when newer loan records become available.
- Explore feature engineering such as total income, loan-to-income ratio, and repayment burden, with strict leakage review.
- Assess whether a larger and more representative dataset changes model conclusions.

