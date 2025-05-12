import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from utils import load_and_preprocess


def main(args):
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(args.data)
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ----- Base Estimator -----
    base_rf = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        random_state=100,
        n_jobs=-1,
    )

    # ----- RandomizedSearchCV -----
    param_grid = {
        "max_depth": [None],
        "max_features": [8, 10, 12],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [8, 10, 12],
        "bootstrap": [True],
        "criterion": ["gini"],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
    randomizedSearchCV = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print("Starting RandomizedSearchCV...")
    t0 = time.perf_counter()
    randomizedSearchCV.fit(X_train, y_train)
    t1 = time.perf_counter()
    print(f"RandomizedSearchCV finished in {(t1 - t0):.1f} seconds")
    print("Best Parameters:", randomizedSearchCV.best_params_)

    best_rf = randomizedSearchCV.best_estimator_

    # ----- Hold-out Evaluation -----
    print("Evaluating on hold-out test set...")
    y_pred = best_rf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]),
    }
    print("Hold-out Metrics:", metrics)

    # ----- Save Artifacts -----
    print("Saving model and metrics...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_rf, "scaler": scaler}, args.out)

    # Convert np.ndarray to lists for JSON serialization
    cv_results_serializable = {
        k: v.tolist() if isinstance(v, (np.ndarray, pd.Series)) else v
        for k, v in randomizedSearchCV.cv_results_.items()
    }

    metrics_out_path = Path(args.out).with_suffix(".metrics.json")
    with open(metrics_out_path, "w") as fp:
        json.dump({"randomizedSearch_CV": cv_results_serializable, "holdout": metrics}, fp, indent=2)

    print(f"Model saved to {args.out}")
    print(f"Metrics saved to {metrics_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to financial_data.csv")
    parser.add_argument(
        "--out", default="models/best_rf.pkl", help="Path to save the model"
    )
    args = parser.parse_args()
    main(args)
