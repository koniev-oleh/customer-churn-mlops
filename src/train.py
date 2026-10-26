import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preparation import load_data, prepare_features

DATA_PATH = "data/processed/churn_v1.csv"
DATA_VERSION = "v1"
MODEL_PATH = "models/latest_model.joblib"
EXPERIMENT_NAME = "customer-churn-baseline"
REGISTERED_MODEL_NAME = "CustomerChurnClassifier"


def build_model(model_type: str):
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def main(model_type: str = "logistic_regression"):
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(model_type)

    with mlflow.start_run(run_name=f"{model_type}_{DATA_VERSION}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_version", DATA_VERSION)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        report_path = f"reports/metrics_{model_type}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"model_type": model_type, "data_version": DATA_VERSION, **metrics}, f, indent=2)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(model_type, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
    )
    args = parser.parse_args()
    main(args.model_type)