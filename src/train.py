from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import (
    PROCESSED_DIR,
    load_training_data,
    make_temporal_split,
    prepare_xy,
)
from .evaluate import (
    evaluate_binary_model,
    evaluate_binary_predictions,
    find_best_threshold,
    save_metrics,
)


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(X):
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="BILINMIYOR",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        ),
    ])


def train_target(target, model_name, maturity_days=0):
    data = load_training_data()

    # İade için son 21 günlük kayıtların etiketi henüz olgunlaşmamış olabilir.
    if maturity_days > 0:
        latest_date = data["order_date"].max()
        cutoff_date = latest_date - pd.Timedelta(days=maturity_days)
        data = data[data["order_date"] <= cutoff_date].copy()

    X, y, feature_columns = prepare_xy(data, target)
    data_for_split = data.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_data, validation_data, test_data = make_temporal_split(
        data_for_split
    )

    train_indices = train_data.index
    validation_indices = validation_data.index
    test_indices = test_data.index

    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_validation = X.loc[validation_indices]
    y_validation = y.loc[validation_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    if y_train.nunique() < 2:
        raise ValueError(
            f"{target} için eğitim kümesinde iki sınıf bulunmuyor."
        )

    validation_model = build_pipeline(X_train)
    validation_model.fit(X_train, y_train)

    validation_probabilities = validation_model.predict_proba(X_validation)[:, 1]
    threshold_result = find_best_threshold(
        validation_probabilities,
        y_validation,
    )
    selected_threshold = threshold_result["threshold"]

    metrics = [
        evaluate_binary_predictions(
            validation_probabilities,
            y_validation,
            "validation",
            threshold=selected_threshold,
        ),
        evaluate_binary_predictions(
            validation_model.predict_proba(X_test)[:, 1],
            y_test,
            "test",
            threshold=selected_threshold,
        ),
    ]

    # Nihai model, modelleme dönemi içindeki bütün olgunlaşmış kayıtlarla eğitilir.
    final_model = build_pipeline(X)
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "target": target,
        "feature_columns": feature_columns,
        "maturity_days": maturity_days,
        "row_count": len(data),
        "positive_count": int(y.sum()),
        "decision_threshold": selected_threshold,
        "metrics": metrics,
    }

    model_path = MODELS_DIR / model_name
    joblib.dump(artifact, model_path)

    metrics_path = PROCESSED_DIR / model_name.replace(
        ".joblib",
        "_metrics.csv",
    )
    save_metrics(metrics, metrics_path)

    return {
        "target": target,
        "rows": len(data),
        "positive_count": int(y.sum()),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main():
    cancellation_result = train_target(
        target="is_cancelled",
        model_name="cancellation_model.joblib",
        maturity_days=0,
    )

    return_result = train_target(
        target="is_returned",
        model_name="return_model.joblib",
        maturity_days=21,
    )

    print("İptal modeli eğitildi:")
    print(cancellation_result)
    print()
    print("İade modeli eğitildi:")
    print(return_result)


if __name__ == "__main__":
    main()
