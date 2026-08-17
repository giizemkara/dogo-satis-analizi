from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def predict_with_model(model_filename, feature_file):
    """Hazırlanmış feature dosyası için risk olasılığı üretir."""
    artifact = joblib.load(MODELS_DIR / model_filename)
    model = artifact["model"]

    features = pd.read_excel(feature_file)
    feature_columns = artifact["feature_columns"]

    X = features[feature_columns].copy()

    for column in ["tarih", "order_date"]:
        if column in X.columns:
            X = X.drop(columns=column)

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("BILINMIYOR")
            .astype("object")
        )

    probability = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "siparis_no": features["siparis_no"],
        f"{artifact['target']}_risk": probability,
    })

    return result
