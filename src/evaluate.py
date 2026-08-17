from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_model(model, X, y, split_name):
    """İkili sınıflandırma modelini temel metriklerle değerlendirir."""
    return evaluate_binary_predictions(
        model.predict_proba(X)[:, 1],
        y,
        split_name,
    )


def evaluate_binary_predictions(probabilities, y, split_name, threshold=0.50):
    """Olasılıkları belirlenen eşikle sınıfa çevirip ölçer."""
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": float(threshold),
        "row_count": len(y),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ) if y.nunique() == 2 else np.nan,
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ) if y.nunique() == 2 else np.nan,
        "tn": int(confusion_matrix(y, predictions, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y, predictions, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y, predictions, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y, predictions, labels=[0, 1])[1, 1]),
    }

    return metrics


def find_best_threshold(probabilities, y):
    """Threshold'u yalnızca validation setinde F1 maksimize ederek seçer."""
    best = {
        "threshold": 0.50,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        current = {
            "threshold": float(round(threshold, 2)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
        }

        if current["f1"] > best["f1"]:
            best = current

    return best


def save_metrics(metrics, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
