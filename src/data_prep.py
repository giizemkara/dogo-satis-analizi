from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_FEATURES_PATH = PROCESSED_DIR / "model_features_pre_order.xlsx"
ORDERS_PATH = PROCESSED_DIR / "merge_data.xlsx"


# Bu alanlar sipariş anında henüz bilinmediği için modele girmemelidir.
LEAKAGE_COLUMNS = {
    "siparis_no",
    "tarih",
    "order_date",
}


def load_training_data():
    """İşlenmiş feature dosyasını hedef etiketlerle birleştirir."""
    if not MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dosyası bulunamadı: {MODEL_FEATURES_PATH}"
        )

    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Sipariş dosyası bulunamadı: {ORDERS_PATH}"
        )

    features = pd.read_excel(MODEL_FEATURES_PATH)
    orders = pd.read_excel(
        ORDERS_PATH,
        sheet_name="orders"
    )

    features["siparis_no"] = features["siparis_no"].astype("string").str.strip()
    orders["siparis_no"] = orders["siparis_no"].astype("string").str.strip()

    label_columns = [
        "siparis_no",
        "status_norm",
        "is_cancelled",
        "is_returned",
        "has_return_request",
    ]

    labels = orders[label_columns].copy()

    data = features.merge(
        labels,
        on="siparis_no",
        how="inner",
        validate="one_to_one",
    )

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce",
    )

    if data["order_date"].isna().any():
        raise ValueError("order_date alanında parse edilemeyen kayıtlar var.")

    return data


def get_feature_columns(data):
    """Model girdisi olacak, hedef/leakage içermeyen kolonları döndürür."""
    target_columns = {
        "status_norm",
        "is_cancelled",
        "is_returned",
        "has_return_request",
    }

    excluded = LEAKAGE_COLUMNS | target_columns

    return [
        column
        for column in data.columns
        if column not in excluded
    ]


def make_temporal_split(data):
    """Veriyi kronolojik olarak train/validation/test kümelerine ayırır."""
    unique_dates = (
        data["order_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) < 10:
        raise ValueError("Zamansal bölme için yeterli farklı tarih yok.")

    train_cutoff = unique_dates.iloc[int(len(unique_dates) * 0.70)]
    test_cutoff = unique_dates.iloc[int(len(unique_dates) * 0.85)]

    train = data[data["order_date"] < train_cutoff].copy()
    validation = data[
        (data["order_date"] >= train_cutoff)
        & (data["order_date"] < test_cutoff)
    ].copy()
    test = data[data["order_date"] >= test_cutoff].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Zamansal train/validation/test bölmesi boş kaldı.")

    return train, validation, test


def prepare_xy(data, target):
    """Bir hedef için X ve y veri setlerini hazırlar."""
    feature_columns = get_feature_columns(data)
    X = data[feature_columns].copy()
    y = data[target].astype(int).copy()

    # Ham tarih kolonları model pipeline'ına sokulmaz.
    for column in ["tarih", "order_date"]:
        if column in X.columns:
            X = X.drop(columns=column)

    # PyArrow string tiplerinin sklearn ile sorun çıkarmaması için object'e çevir.
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        # Bazı kolonlarda metin ve tarih nesneleri birlikte bulunabiliyor.
        # OneHotEncoder'a gitmeden önce hepsini tutarlı biçimde metne çevir.
        X[column] = (
            X[column]
            .astype("string")
            .fillna("BILINMIYOR")
            .astype("object")
        )

    return X, y, feature_columns
