"""Train and compare baseline models using a leakage-conscious pipeline.

The UCI Cleveland dataset is small and historical. This is a learning exercise,
not a clinically validated model.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "heart_disease.csv"
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

def load_data():
    if not DATA_PATH.exists():
        from download_dataset import main
        main()
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing_cols = [c for c in FEATURES + ["num"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    # Convert '?' and other non-numeric values to NaN, then impute within pipeline.
    for col in FEATURES + ["num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["num"]).copy()
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = (df["num"] > 0).astype(int)
    return X, y

def make_pipeline(estimator):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        )
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison = {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "recall": "recall", "f1": "f1"},
            n_jobs=-1
        )
        comparison[name] = {
            "cv_roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 3),
            "cv_roc_auc_std": round(float(scores["test_roc_auc"].std()), 3),
            "cv_recall_mean": round(float(scores["test_recall"].mean()), 3),
            "cv_f1_mean": round(float(scores["test_f1"].mean()), 3)
        }

    best_name = max(comparison, key=lambda k: comparison[k]["cv_roc_auc_mean"])
    final_model = make_pipeline(candidates[best_name])
    final_model.fit(X_train, y_train)

    pred = final_model.predict(X_test)
    proba = final_model.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset": "UCI Heart Disease (id 45)",
        "rows_used": int(len(X)),
        "features": FEATURES,
        "target_definition": "1 if num > 0, otherwise 0",
        "best_model_by_training_cv_roc_auc": best_name,
        "cross_validation_comparison": comparison,
        "holdout_test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 3),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, zero_division=0, output_dict=True
            )
        },
        "limitations": [
            "Small, historical dataset with 303 records.",
            "Single train/test split and cross-validation are educational estimates.",
            "Not externally validated or clinically validated.",
            "Not suitable for diagnosis, triage, treatment, or patient care."
        ]
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURES, "model_name": best_name}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print("WARNING: educational research only; not a medical device or clinical risk tool.")

if __name__ == "__main__":
    main()
