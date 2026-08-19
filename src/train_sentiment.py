from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .return_nlp import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REVIEW_PATH = PROCESSED_DIR / "return_nlp_summary.xlsx"
MODEL_PATH = MODELS_DIR / "return_sentiment_model.joblib"
METRICS_PATH = PROCESSED_DIR / "return_sentiment_model_metrics.csv"

ALLOWED_LABELS = {
    "Nötr talep",
    "Memnuniyetsizlik",
    "Hayal kırıklığı",
    "Sert şikâyet",
    "Belirsiz",
}

LABEL_ALIASES = {
    "notr talep": "Nötr talep",
    "memnuniyetsizlik": "Memnuniyetsizlik",
    "hayal kirikligi": "Hayal kırıklığı",
    "sert sikayet": "Sert şikâyet",
    "belirsiz": "Belirsiz",
}


def build_model():
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
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
    required = {"normalized_reason", "manual_sentiment"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "review_queue içinde eksik kolonlar var: "
            + ", ".join(sorted(missing))
        )

    data["text"] = data["normalized_reason"].fillna("").astype(str).str.strip()
    # Excel'de "Nötr Talep", "nötr talep" veya aksansız yazım olabilir.
    # Eğitimden önce hepsini tek bir kanonik etikete çeviriyoruz.
    data["sentiment_key"] = data["manual_sentiment"].apply(normalize_text)
    data["sentiment"] = data["sentiment_key"].map(LABEL_ALIASES)
    data = data[
        data["text"].ne("")
        & data["sentiment"].isin(ALLOWED_LABELS - {"Belirsiz"})
    ].copy()

    if data["sentiment"].nunique() < 2:
        raise ValueError("Duygu modeli için en az iki etiketli sınıf gerekir.")

    return data


def train(path=REVIEW_PATH):
    data = load_manual_labels(path)
    X = data["text"]
    y = data["sentiment"]

    n_splits = min(3, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError(
            "Her duygu sınıfında en az iki etiketli kayıt olmalıdır."
        )

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )
    validation_model = build_model()
    predictions = cross_val_predict(validation_model, X, y, cv=cross_validator)

    metrics = {
        "model": "tfidf_logistic_sentiment",
        "evaluation": f"{n_splits}-fold stratified cross-validation",
        "row_count": len(data),
        "class_count": y.nunique(),
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "weighted_f1": f1_score(y, predictions, average="weighted"),
    }

    final_model = build_model()
    final_model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "target": "manual_sentiment",
            "labels": sorted(y.unique()),
            "row_count": len(data),
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Duygu/ton NLP modeli eğitildi:")
    print(metrics)
    print(f"Model dosyası: {MODEL_PATH}")
    print(f"Metrik dosyası: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
