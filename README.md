# Credit Score Prediction — VS Code Project

## Setup (Windows)

Open this folder in VS Code, then open Terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate
```

## Dataset

Place `synthetic_credit_card_customer_behavior_dataset.csv` beside `main.py`.

Or run with a full path:
```bash
python main.py --data "C:/Users/YourName/Downloads/your_dataset.csv"
```

## Run

Normal run:
```bash
python main.py
```

Faster run:
```bash
python main.py --quick
```

## Generated outputs

Everything is saved in `outputs/`:
- `cross_validation_benchmark.csv`
- `final_test_metrics.csv`
- `feature_formula_audit.csv`
- `target_correlations.csv`
- `permutation_importance.csv`
- `subgroup_error_analysis.csv` (if applicable)
- `test_predictions.csv`
- `model_card.json`
- `poster_results_summary.txt`
- `credit_score_prediction_pipeline.joblib`
- `poster_assets/` containing 300-DPI charts

## Poster structure

1. Problem statement and objective
2. Dataset overview and data-quality audit
3. Exploratory data analysis
4. ML methodology and validation strategy
5. Cross-validation model benchmark
6. Final test metrics (MAE, RMSE, R²)
7. Actual-vs-predicted and residual diagnostics
8. Permutation feature importance
9. Subgroup error diagnostics
10. Limitations and future scope

## Scientific limitation

The dataset is synthetic. This is an industry-oriented academic prototype, not a validated lending model. Strong results on synthetic data do not establish performance on real customers. Verify feature formulas and prediction-time availability before making claims.
