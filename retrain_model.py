import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from landmark_utils import normalize_dataframe_columns

TRAIN_PATH = "Dataset/data.csv"
VAL_PATH = "Dataset/validation.csv"
MODEL_OUT = "resources/model.pl"

LABELS = ["click", "release", "scroll_up", "scroll_down"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


def load(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df = normalize_dataframe_columns(df, label_col="label")
    df["label"] = df["label"].map(LABEL_TO_IDX)
    y = df["label"]
    X = df.drop(columns="label")
    return X, y


def main():
    X_train, y_train = load(TRAIN_PATH)
    X_val, y_val = load(VAL_PATH)

    candidates = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("clf", RandomForestClassifier(n_estimators=300, random_state=88)),
            ]
        ),
    }

    best_name, best_model, best_score = None, None, -1.0

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        print(
            f"{name:18s} | train CV acc: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}) "
            f"| validation acc: {val_score:.4f}"
        )
        if val_score > best_score:
            best_name, best_model, best_score = name, model, val_score

    print(f"\nBest model: {best_name} (validation acc: {best_score:.4f})")

    y_pred = best_model.predict(X_val)
    idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}
    print("\nPer-label validation accuracy:")
    for idx, label in idx_to_label.items():
        mask = y_val == idx
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == y_val[mask]).mean()
        print(f"  {label:12s}: {acc:.4f}  ({mask.sum()} samples)")

    joblib.dump(best_model, MODEL_OUT)
    print(f"\nSaved best model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
