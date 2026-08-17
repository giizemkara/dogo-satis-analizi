import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dogo_iade_aciklamali.xlsx"

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
        if any(keyword in normalized for keyword in keywords):
            matched.append(category)

    if not matched:
        matched = ["Diğer / Belirsiz"]

    return matched[0], " | ".join(matched), normalized


def classify_product_type(text):
    normalized = normalize_text(text)

    for product_type, keywords in PRODUCT_TYPE_RULES:
        if any(keyword in normalized for keyword in keywords):
            return product_type

    return "Diğer"


def load_return_nlp():
    detail = pd.read_excel(RAW_PATH)
    detail.columns = [
        normalize_text(column).replace(" ", "_")
        for column in detail.columns
    ]

    detail["talep_tarihi"] = pd.to_datetime(detail["talep_tarihi"], errors="coerce")

    classified = detail["iade_nedeni"].apply(classify_reason)
    detail[["primary_category", "detected_categories", "normalized_reason"]] = pd.DataFrame(
        classified.tolist(),
        index=detail.index,
    )
    detail["product_type"] = detail["urun_adi"].apply(classify_product_type)

    summary = (
        detail.groupby("primary_category")
        .agg(
            return_lines=("siparis_no", "size"),
            unique_orders=("siparis_no", "nunique"),
        )
        .reset_index()
        .sort_values("return_lines", ascending=False)
    )
    summary["share_of_return_lines"] = summary["return_lines"] / summary["return_lines"].sum() * 100

    detail["month"] = detail["talep_tarihi"].dt.to_period("M").astype(str)
    monthly = detail.groupby(["month", "primary_category"]).size().reset_index(name="return_lines")

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
        detail.groupby(["product_type", "primary_category"])
        .size()
        .reset_index(name="return_lines")
    )

    return detail, summary, monthly, product_summary, product_reason


def export_nlp_summary():
    detail, summary, monthly, product_summary, product_reason = load_return_nlp()
    output_path = PROJECT_ROOT / "data" / "processed" / "return_nlp_summary.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="reason_summary", index=False)
        product_summary.to_excel(writer, sheet_name="product_summary", index=False)
        product_reason.to_excel(writer, sheet_name="product_reason", index=False)
        monthly.to_excel(writer, sheet_name="monthly_reason", index=False)

    print(f"NLP özeti kaydedildi: {output_path}")
    return output_path


if __name__ == "__main__":
    export_nlp_summary()
