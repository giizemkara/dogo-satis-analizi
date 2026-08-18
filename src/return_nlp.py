import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dogo_iade_aciklamali.xlsx"
NLP_MODEL_PATH = PROJECT_ROOT / "models" / "return_nlp_model.joblib"

TR_MAP = str.maketrans({
    "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
})

CATEGORY_RULES = {
    "Beden / Kalıp Uyumsuzluğu": [
        "beden", "numara", "kalip", "kucuk", "buyuk", "dar", "bol",
        "ayak", "sik", "uyma", "uygun degil", "olmadi", "bicimsiz", "sert",
    ],
    "Kalite / Hasar": [
        "kalite", "kalitesiz", "hasar", "defolu", "yirtik", "sokuk",
        "kiril", "bozuk", "fermuar", "dakis", "dikis", "yara",
    ],
    "Model / Renk / Beğeni": [
        "model", "renk", "desen", "begeni", "begendim", "begenmedim",
        "sevmedim", "hoslanmadim", "durus", "hayal", "beklenti", "kaba",
    ],
    "Yanlış / Eksik Ürün": [
        "yanlis", "eksik", "farkli", "gelmedi", "degisik",
    ],
    "Kargo / Paketleme": [
        "kargo", "kutu", "paket", "ambalaj", "yirt", "dagil", "gecik", "yanlis",
    ],
    "Değişim Talebi": [
        "degisim", "degistir",
    ],
}

PRODUCT_TYPE_RULES = [
    ("Terlik", ["terlik"]),
    ("Babet", ["babet"]),
    ("Sandalet", ["sandalet"]),
    ("Sneakers", ["sneaker"]),
    ("Loafer", ["loafer"]),
    ("Bot", ["bot", "boot"]),
    ("Çizme", ["cizme"]),
    ("Çanta", ["canta", "bag", "tote", "omuz cantasi", "sirt cantasi"]),
    ("Cüzdan", ["cuzdan", "wallet"]),
    ("Bileklik", ["bileklik", "bracelet"]),
    ("Aksesuar", ["aksesuar", "kemer"]),
]


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).replace("_x000D_", " ").translate(TR_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold()

    text = re.sub(r"\bTR(?:[\s-]*\d){24}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+@\S+\b", " ", text)
    text = re.sub(r"\+?\d[\d\s().-]{8,}\d", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_reason(text):
    normalized = normalize_text(text)
    matched = []

    for category, keywords in CATEGORY_RULES.items():
        if any(
            re.search(rf"\b{re.escape(keyword)}\b", normalized)
            for keyword in keywords
        ):
            matched.append(category)

    if not matched:
        matched = ["Diğer / Belirsiz"]

    confidence = (
        "düşük"
        if matched == ["Diğer / Belirsiz"]
        else "orta"
        if len(matched) > 1
        else "yüksek"
    )
    return matched[0], " | ".join(matched), normalized, confidence


def classify_product_type(text):
    normalized = normalize_text(text)

    for product_type, keywords in PRODUCT_TYPE_RULES:
        if any(keyword in normalized for keyword in keywords):
            return product_type

    return "Diğer"


def _apply_trained_model(detail):
    """Varsa elle etiketlenmiş NLP modelini metin bulunan satırlara uygular."""
    detail = detail.copy()
    detail["model_category"] = pd.NA

    if not NLP_MODEL_PATH.exists():
        return detail, "rule_based"

    artifact = joblib.load(NLP_MODEL_PATH)
    model = artifact["model"]
    mask = detail["normalized_reason"].fillna("").astype(str).str.strip().ne("")

    if mask.any():
        detail.loc[mask, "model_category"] = model.predict(
            detail.loc[mask, "normalized_reason"]
        )

    detail["model_category"] = detail["model_category"].fillna(
        detail["primary_category"]
    )

    return detail, "tfidf_logistic"


def load_return_nlp(raw_path=None):
    raw_path = Path(raw_path) if raw_path else RAW_PATH
    detail = pd.read_excel(raw_path)
    detail.columns = [
        normalize_text(column).replace(" ", "_")
        for column in detail.columns
    ]

    detail["talep_tarihi"] = pd.to_datetime(detail["talep_tarihi"], errors="coerce")

    classified = detail["iade_nedeni"].apply(classify_reason)
    detail[[
        "primary_category",
        "detected_categories",
        "normalized_reason",
        "nlp_confidence",
    ]] = pd.DataFrame(
        classified.tolist(),
        index=detail.index,
    )
    detail["product_type"] = detail["urun_adi"].apply(classify_product_type)
    detail, nlp_method = _apply_trained_model(detail)
    category_column = (
        "model_category"
        if nlp_method == "tfidf_logistic"
        else "primary_category"
    )

    summary = (
        detail.groupby(category_column)
        .agg(
            return_lines=("siparis_no", "size"),
            unique_orders=("siparis_no", "nunique"),
        )
        .reset_index()
        .sort_values("return_lines", ascending=False)
        .rename(columns={category_column: "primary_category"})
    )
    summary["share_of_return_lines"] = summary["return_lines"] / summary["return_lines"].sum() * 100

    detail["month"] = detail["talep_tarihi"].dt.to_period("M").astype(str)
    monthly = (
        detail.groupby(["month", category_column])
        .size()
        .reset_index(name="return_lines")
        .rename(columns={category_column: "primary_category"})
    )

    product_summary = (
        detail.groupby("product_type")
        .agg(
            return_lines=("siparis_no", "size"),
            unique_orders=("siparis_no", "nunique"),
        )
        .reset_index()
        .sort_values("return_lines", ascending=False)
    )
    product_summary["share_of_return_lines"] = (
        product_summary["return_lines"] / product_summary["return_lines"].sum() * 100
    )

    product_reason = (
        detail.groupby(["product_type", category_column])
        .size()
        .reset_index(name="return_lines")
        .rename(columns={category_column: "primary_category"})
    )

    review_queue = detail[
        (detail["primary_category"] == "Diğer / Belirsiz")
        | detail["detected_categories"].str.contains(" | ", regex=False)
    ].copy()
    review_queue.insert(0, "review_id", range(1, len(review_queue) + 1))
    review_queue = review_queue[[
        "review_id",
        "normalized_reason",
        "product_type",
        "primary_category",
        "detected_categories",
        "nlp_confidence",
    ]]
    if "model_category" in detail.columns:
        review_queue["model_category"] = detail.loc[
            review_queue.index, "model_category"
        ].to_numpy()

    return (
        detail,
        summary,
        monthly,
        product_summary,
        product_reason,
        review_queue,
        nlp_method,
    )


def export_nlp_summary(raw_path=None, output_dir=None):
    (
        detail,
        summary,
        monthly,
        product_summary,
        product_reason,
        review_queue,
        nlp_method,
    ) = load_return_nlp(raw_path)
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "return_nlp_summary.xlsx"

    # Kullanıcının daha önce doldurduğu manual_category kolonunu koru.
    if output_path.exists():
        try:
            previous_queue = pd.read_excel(output_path, sheet_name="review_queue")
            if {"review_id", "manual_category"}.issubset(previous_queue.columns):
                same_queue = (
                    len(previous_queue) == len(review_queue)
                    and previous_queue["review_id"].reset_index(drop=True).equals(
                        review_queue["review_id"].reset_index(drop=True)
                    )
                )
                if same_queue:
                    review_queue = review_queue.merge(
                        previous_queue[["review_id", "manual_category"]],
                        on="review_id",
                        how="left",
                    )
                elif {"normalized_reason", "product_type"}.issubset(
                    previous_queue.columns
                ):
                    manual_labels = previous_queue[
                        ["normalized_reason", "product_type", "manual_category"]
                    ].drop_duplicates(
                        subset=["normalized_reason", "product_type"]
                    )
                    review_queue = review_queue.merge(
                        manual_labels,
                        on=["normalized_reason", "product_type"],
                        how="left",
                    )
                if "manual_category" in review_queue.columns:
                    empty_reason = (
                        review_queue["normalized_reason"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .eq("")
                    )
                    review_queue.loc[
                        empty_reason & review_queue["manual_category"].isna(),
                        "manual_category",
                    ] = "Diğer / Belirsiz"
        except Exception:
            # Eski dosya bozuksa veya açıkken okunamıyorsa yeni çıktı üretilebilir.
            pass

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="reason_summary", index=False)
        product_summary.to_excel(writer, sheet_name="product_summary", index=False)
        product_reason.to_excel(writer, sheet_name="product_reason", index=False)
        monthly.to_excel(writer, sheet_name="monthly_reason", index=False)
        review_queue.to_excel(writer, sheet_name="review_queue", index=False)

    print(f"NLP özeti kaydedildi: {output_path} ({nlp_method})")
    return output_path


if __name__ == "__main__":
    export_nlp_summary()
