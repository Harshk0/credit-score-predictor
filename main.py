import argparse, json, warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent background thread crash on Windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET = "Credit_Score"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic_credit_card_customer_behavior_dataset.csv")
    p.add_argument("--output", default="outputs")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def formula_audit(df, out):
    rows = []

    def check(feature, formula, actual, predicted, valid=None):
        if valid is None:
            valid = actual.notna() & predicted.notna()
        if valid.sum():
            diff = np.abs(actual[valid].astype(float) - predicted[valid].astype(float))
            rows.append([feature, formula, bool(diff.max() <= 1e-5), float(diff.max())])

    spend = ["Online_Shopping_Spending", "Grocery_Spending", "Fuel_Spending",
             "Dining_Spending", "Travel_Spending", "Entertainment_Spending",
             "Utility_Bill_Spending"]
    if all(c in df for c in spend + ["Monthly_Spending"]):
        check("Monthly_Spending", "Sum of spending categories",
              df["Monthly_Spending"], df[spend].sum(axis=1))

    if all(c in df for c in ["Credit_Utilization", "Outstanding_Balance", "Credit_Limit"]):
        v = df["Credit_Limit"].notna() & (df["Credit_Limit"] != 0) & df["Credit_Utilization"].notna()
        for m in [1, 100]:
            check("Credit_Utilization", f"Outstanding_Balance / Credit_Limit * {m}",
                  df["Credit_Utilization"],
                  df["Outstanding_Balance"] / df["Credit_Limit"] * m, v)

    if all(c in df for c in ["Payment_Ratio", "Payment_Amount", "Statement_Balance"]):
        v = df["Statement_Balance"].notna() & (df["Statement_Balance"] != 0) & df["Payment_Ratio"].notna()
        for m in [1, 100]:
            check("Payment_Ratio", f"Payment_Amount / Statement_Balance * {m}",
                  df["Payment_Ratio"],
                  df["Payment_Amount"] / df["Statement_Balance"] * m, v)

    pd.DataFrame(rows, columns=["Feature", "Possible_Formula", "Exact_Match", "Max_Difference"]).to_csv(
        out / "feature_formula_audit.csv", index=False)


def make_preprocessor(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])


def main():
    args = parse_args()
    out = Path(args.output)
    plots = out / "poster_assets"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}. Put the CSV beside main.py or use --data.")

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    df = df.dropna(axis=1, how="all").copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)

    # Subsample to keep the trained model file well under GitHub's 100MB file size limit
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows after cleaning:", int(df.duplicated().sum()))

    formula_audit(df, out)

    # EDA: target distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=40, kde=True, ax=ax)
    ax.set(title="Credit Score Distribution", xlabel="Credit Score", ylabel="Customer Count")
    savefig(fig, plots / "01_target_distribution.png")

    # EDA: numerical correlation
    numeric = df.select_dtypes(include=np.number)
    if TARGET in numeric and numeric.shape[1] > 1:
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(13, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numerical Feature Correlation Matrix")
        savefig(fig, plots / "02_correlation_heatmap.png")
        tc = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        tc.to_csv(out / "target_correlations.csv", header=["correlation"])
        fig, ax = plt.subplots(figsize=(9, 6))
        tc.head(15).sort_values().plot(kind="barh", ax=ax)
        ax.set(title="Top Numerical Correlations with Credit Score", xlabel="Pearson Correlation")
        savefig(fig, plots / "03_target_correlations.png")

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_values": df.nunique(),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })
    profile.to_csv(out / "dataset_profile.csv")

    excluded = [c for c in ["Credit_Score", "Customer_ID", "Payment_Ratio", "Credit_Utilization"]
                if c in df.columns]
    excluded += [c for c in df.columns if c.lower().startswith("unnamed")]
    X = df.drop(columns=excluded, errors="ignore")
    y = df[TARGET]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE
    )
    prep = make_preprocessor(X_train)
    folds = 2 if args.quick else 3
    trees = 35 if args.quick else 70
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Dummy Baseline": DummyRegressor(),
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                               random_state=42, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=2,
                                           random_state=42, n_jobs=1)
    }

    scores = []
    for name, estimator in models.items():
        print("Benchmarking:", name)
        pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
        r = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring={"mae": "neg_mean_absolute_error",
                                    "rmse": "neg_root_mean_squared_error", "r2": "r2"})
        scores.append([name, -r["test_mae"].mean(), -r["test_rmse"].mean(),
                       r["test_rmse"].std(), r["test_r2"].mean()])

    benchmark = pd.DataFrame(scores, columns=["Model", "CV_MAE", "CV_RMSE", "CV_RMSE_STD", "CV_R2"])
    benchmark = benchmark.sort_values("CV_RMSE")
    benchmark.to_csv(out / "cross_validation_benchmark.csv", index=False)
    print("\nCross-validation benchmark:\n", benchmark.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=benchmark, x="Model", y="CV_RMSE", ax=ax)
    ax.set(title="Model Benchmarking — Cross-Validated RMSE", ylabel="Mean CV RMSE")
    ax.tick_params(axis="x", rotation=15)
    savefig(fig, plots / "04_model_benchmark.png")

    tune_pipe = Pipeline([
        ("preprocessor", prep),
        ("model", ExtraTreesRegressor(random_state=42, n_jobs=1))
    ])
    params = {
        "model__n_estimators": [35, 60, 90] if args.quick else [70, 120, 180],
        "model__max_depth": [None, 15, 25],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt"]
    }
    search = RandomizedSearchCV(
        tune_pipe, params, n_iter=3 if args.quick else 6,
        scoring="neg_root_mean_squared_error", cv=cv,
        random_state=42, n_jobs=-1, verbose=1
    )
    print("\nTuning Extra Trees...")
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    medae = median_absolute_error(y_test, pred)
    metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse),
               "R2": float(r2), "Median_Absolute_Error": float(medae),
               "Tuned_CV_RMSE": float(-search.best_score_)}
    pd.DataFrame([metrics]).to_csv(out / "final_test_metrics.csv", index=False)
    print("\nHELD-OUT TEST METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    predictions = pd.DataFrame({
        "Actual_Credit_Score": np.asarray(y_test),
        "Predicted_Credit_Score": pred,
        "Residual": np.asarray(y_test) - pred,
        "Absolute_Error": np.abs(np.asarray(y_test) - pred)
    })
    predictions.to_csv(out / "test_predictions.csv", index=False)

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=pred, alpha=.4, s=20, ax=ax)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", label="Ideal prediction")
    ax.set(title="Actual vs Predicted Credit Scores", xlabel="Actual Score", ylabel="Predicted Score")
    ax.legend()
    savefig(fig, plots / "05_actual_vs_predicted.png")

    # Residual diagnostics
    residual = np.asarray(y_test) - pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(residual, bins=35, kde=True, ax=axes[0])
    axes[0].axvline(0, linestyle="--")
    axes[0].set(title="Residual Distribution", xlabel="Actual - Predicted")
    sns.scatterplot(x=pred, y=residual, alpha=.4, s=18, ax=axes[1])
    axes[1].axhline(0, linestyle="--")
    axes[1].set(title="Residuals vs Predicted", xlabel="Predicted Score", ylabel="Residual")
    savefig(fig, plots / "06_residual_diagnostics.png")

    # Permutation importance
    print("Calculating permutation importance...")
    perm = permutation_importance(model, X_test, y_test, scoring="neg_root_mean_squared_error",
                                  n_repeats=2 if args.quick else 4, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"Feature": X_test.columns,
                        "Importance_Mean": perm.importances_mean,
                        "Importance_STD": perm.importances_std}).sort_values("Importance_Mean", ascending=False)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    top = imp.head(15).sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Feature"], top["Importance_Mean"], xerr=top["Importance_STD"])
    ax.set(title="Top Features — Permutation Importance",
           xlabel="Increase in RMSE when feature is shuffled", ylabel="Feature")
    savefig(fig, plots / "07_permutation_importance.png")

    # Subgroup diagnostics
    test_rows = df.iloc[test_idx].reset_index(drop=True).copy()
    test_rows["Actual"] = np.asarray(y_test)
    test_rows["Predicted"] = pred
    test_rows["Absolute_Error"] = np.abs(test_rows["Actual"] - test_rows["Predicted"])
    if "Age" in test_rows:
        test_rows["Age_Group"] = pd.cut(test_rows["Age"], [0, 25, 35, 45, 55, 65, np.inf],
                                        labels=["<25", "25-35", "36-45", "46-55", "56-65", "65+"])
    subgroup_tables = []
    for col in [c for c in ["Gender", "Occupation", "Card_Type", "Age_Group"] if c in test_rows]:
        group = test_rows.groupby(col, dropna=False).agg(
            Customers=("Absolute_Error", "size"), MAE=("Absolute_Error", "mean")
        ).reset_index()
        group["Group_Feature"] = col
        subgroup_tables.append(group)
        plot_group = group[group["Customers"] >= 30]
        if not plot_group.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=plot_group, x=col, y="MAE", ax=ax)
            ax.set(title=f"Prediction Error by {col}", ylabel="MAE")
            ax.tick_params(axis="x", rotation=20)
            savefig(fig, plots / f"08_subgroup_{col.lower()}.png")
    if subgroup_tables:
        pd.concat(subgroup_tables, ignore_index=True).to_csv(out / "subgroup_error_analysis.csv", index=False)

    # Save pipeline and model card
    joblib.dump(model, out / "credit_score_prediction_pipeline.joblib")
    model_card = {
        "project": "Credit Score Prediction Using Machine Learning",
        "dataset": path.name,
        "dataset_type": "Synthetic",
        "target": TARGET,
        "problem_type": "Regression",
        "model": "Tuned Extra Trees Regressor",
        "features_used": X.columns.tolist(),
        "excluded_features": excluded,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "random_state": 42, "best_parameters": search.best_params_,
        "metrics": metrics,
        "limitations": [
            "Synthetic data does not establish real-world credit-risk validity.",
            "Feature-generation formulas and prediction-time availability require verification.",
            "Random holdout does not test future-time generalization.",
            "Subgroup diagnostics are not formal fairness certification.",
            "This is an academic prototype, not a production lending decision system."
        ],
        "created_at": datetime.now().isoformat()
    }
    with open(out / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    with open(out / "poster_results_summary.txt", "w", encoding="utf-8") as f:
        f.write("CREDIT SCORE PREDICTION USING MACHINE LEARNING\n")
        f.write(f"Records: {len(df):,}\nFeatures: {X.shape[1]}\n")
        f.write(f"Training rows: {len(X_train):,}\nTest rows: {len(X_test):,}\n")
        f.write("Model: Tuned Extra Trees Regressor\n")
        f.write(f"Best parameters: {search.best_params_}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nDataset is synthetic; results are an academic prototype only.\n")

    print(f"\nCompleted. Outputs: {out.resolve()}")
    print(f"Poster charts: {plots.resolve()}")


if __name__ == "__main__":
    main()