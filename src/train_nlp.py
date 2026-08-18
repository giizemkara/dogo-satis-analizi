from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REVIEW_PATH = PROCESSED_DIR / "return_nlp_summary.xlsx"
MODEL_PATH = MODELS_DIR / "return_nlp_model.joblib"
METRICS_PATH = PROCESSED_DIR / "return_nlp_model_metrics.csv"


LABEL_FIXES = {
    "Kalite / Renk / Beğeni": "Model / Renk / Beğeni",
}


def build_model():
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])


def load_manual_labels(path=REVIEW_PATH):
    data = pd.read_excel(path, sheet_name="review_queue")
    required = {"normalized_reason", "manual_category"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "review_queue içinde eksik kolonlar var: "
            + ", ".join(sorted(missing))
        )

    data["text"] = data["normalized_reason"].fillna("").astype(str).str.strip()
    data["manual_primary_category"] = (
        data["manual_category"]
        .fillna("")
        .astype(str)
        .str.split(" | ", regex=False)
        .str[0]
        .replace(LABEL_FIXES)
    )
    data = data[
        data["text"].ne("")
        & data["manual_primary_category"].ne("")
    ].copy()

    if data["manual_primary_category"].nunique() < 2:
        raise ValueError("NLP modeli için en az iki kategori gerekir.")

    return data


def train(path=REVIEW_PATH):
    data = load_manual_labels(path)
    X = data["text"]
    y = data["manual_primary_category"]

    n_splits = min(3, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError(
            "En az iki örneği olan kategorilerle cross-validation yapılamıyor."
        )

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )
    validation_model = build_model()
    predictions = cross_val_predict(
        validation_model,
        X,
        y,
        cv=cross_validator,
    )

    metrics = {
        "model": "tfidf_logistic",
        "evaluation": f"{n_splits}-fold stratified cross-validation",
        "row_count": len(data),
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "weighted_f1": f1_score(y, predictions, average="weighted"),
    }

    rule_accuracy = (
        data["primary_category"] == data["manual_primary_category"]
    ).mean()
    baseline_metrics = {
        "model": "rule_based_baseline",
        "evaluation": "manual labels with non-empty text",
        "row_count": len(data),
        "accuracy": rule_accuracy,
        "macro_f1": f1_score(
            y,
            data["primary_category"],
            labels=sorted(y.unique()),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            data["primary_category"],
            labels=sorted(y.unique()),
            average="weighted",
            zero_division=0,
        ),
    }

    final_model = build_model()
    final_model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "target": "manual_primary_category",
            "row_count": len(data),
            "label_fixes": LABEL_FIXES,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([baseline_metrics, metrics]).to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("NLP modeli eğitildi:")
    print(metrics)
    print(f"Kural baseline doğruluğu: {rule_accuracy:.3f}")
    print(f"Model dosyası: {MODEL_PATH}")
    print(f"Metrik dosyası: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train()
