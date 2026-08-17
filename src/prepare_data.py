import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_column_name(name):
    mapping = str.maketrans({
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
    })
    text = str(name).translate(mapping)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def clean_dataframe(df):
    df = df.copy()
    columns = [clean_column_name(column) for column in df.columns]

    if len(columns) != len(set(columns)):
        raise ValueError("Kolon temizliği sonrası duplicate kolon oluştu.")

    df.columns = columns

    for column in df.columns:
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            df[column] = df[column].map(
                lambda value: (
                    value.replace("\xa0", " ").strip()
                    if isinstance(value, str)
                    else value
                )
            )
            df[column] = df[column].replace("", pd.NA)

    return df


def join_unique(series):
    values = []

    for value in series:
        if pd.notna(value):
            value = str(value).strip()
            if value and value not in values:
                values.append(value)

    return " | ".join(values) if values else pd.NA


def add_date_features(df, date_column, prefix):
    date_series = df[date_column]
    df[f"{prefix}_year"] = date_series.dt.year.astype("Int64")
    df[f"{prefix}_month"] = date_series.dt.month.astype("Int64")
    df[f"{prefix}_day"] = date_series.dt.day.astype("Int64")
    df[f"{prefix}_day_of_week"] = date_series.dt.dayofweek.astype("Int64")
    df[f"{prefix}_week_of_year"] = date_series.dt.isocalendar().week.astype("Int64")
    df[f"{prefix}_quarter"] = date_series.dt.quarter.astype("Int64")
    df[f"{prefix}_is_weekend"] = date_series.dt.dayofweek.isin([5, 6]).astype("int8")
    return df


def build_processed_files():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    teslim = clean_dataframe(pd.read_excel(RAW_DIR / "dogo_teslim_edilenler.xlsx"))
    iptal = clean_dataframe(pd.read_excel(RAW_DIR / "dogo_iptal.xlsx"))
    iade = clean_dataframe(pd.read_excel(RAW_DIR / "dogo_iade.xlsx"))
    iade_detail = clean_dataframe(pd.read_excel(RAW_DIR / "dogo_iade_aciklamali.xlsx"))

    teslim["source_dataset"] = "teslim"
    iptal["source_dataset"] = "iptal"
    iade["source_dataset"] = "iade"

    for dataframe in [teslim, iptal, iade, iade_detail]:
        dataframe["siparis_no"] = (
            dataframe["siparis_no"].astype("string").str.replace("\xa0", " ", regex=False).str.strip()
        )

    iade_detail["talep_tarihi"] = pd.to_datetime(iade_detail["talep_tarihi"], errors="coerce")

    for column in ["siparis_edilen_miktar", "iade_edilecek_miktar"]:
        if column in iade_detail.columns:
            iade_detail[column] = pd.to_numeric(iade_detail[column], errors="coerce")

    return_summary = (
        iade_detail.groupby("siparis_no", as_index=False)
        .agg(
            return_line_count=("siparis_no", "size"),
            return_reason_row_count=("iade_nedeni", lambda series: series.notna().sum()),
            return_reason=("iade_nedeni", join_unique),
            return_request_date=("talep_tarihi", "min"),
            return_qty_ordered=("siparis_edilen_miktar", lambda series: series.sum(min_count=1)),
            return_qty_requested=("iade_edilecek_miktar", lambda series: series.sum(min_count=1)),
        )
    )

    core = pd.concat([teslim, iptal, iade], ignore_index=True)

    if core["siparis_no"].duplicated().any():
        raise ValueError("Ana veride duplicate sipariş numarası bulundu.")

    core = core.merge(
        return_summary,
        on="siparis_no",
        how="left",
        validate="one_to_one",
    )

    unmatched = iade_detail[
        ~iade_detail["siparis_no"].isin(set(core["siparis_no"]))
    ].copy()

    for column in ["tarih", "fatura_tarihi", "return_request_date"]:
        core[column] = pd.to_datetime(core[column], errors="coerce")

    for column in [
        "id", "doviz_tutar", "tutar", "kdv", "kargo_toplami",
        "hizmet_bedeli", "gecen_sure_dk", "kur_fiyati",
        "return_line_count", "return_reason_row_count",
        "return_qty_ordered", "return_qty_requested",
    ]:
        if column in core.columns:
            core[column] = pd.to_numeric(core[column], errors="coerce")

    core["doviz_cinsi"] = core["doviz_cinsi"].astype("string").str.strip().str.upper()
    foreign_mask = core["doviz_cinsi"].isin(["USD", "EUR"])
    df = core.loc[~foreign_mask].copy()

    def normalize_status(value):
        if pd.isna(value):
            return pd.NA
        value = unicodedata.normalize("NFKD", str(value).strip())
        value = "".join(character for character in value if not unicodedata.combining(character))
        return value.casefold()

    df["status_norm"] = df["siparis_sureci"].map(normalize_status).astype("string")
    df["is_delivered"] = df["status_norm"].eq("teslim edildi").astype("int8")
    df["is_cancelled"] = df["status_norm"].eq("iptal edildi").astype("int8")
    df["is_returned"] = df["status_norm"].eq("iade edildi").astype("int8")
    df["has_return_request"] = df["return_line_count"].notna().astype("int8")

    df["order_date"] = df["tarih"].dt.normalize()
    df = add_date_features(df, "order_date", "order")
    df["order_hour"] = df["tarih"].dt.hour.astype("Int64")
    df["order_minute"] = df["tarih"].dt.minute.astype("Int64")

    df["order_month_sin"] = np.sin(2 * np.pi * (df["order_month"] - 1) / 12)
    df["order_month_cos"] = np.cos(2 * np.pi * (df["order_month"] - 1) / 12)
    df["order_day_of_week_sin"] = np.sin(2 * np.pi * df["order_day_of_week"] / 7)
    df["order_day_of_week_cos"] = np.cos(2 * np.pi * df["order_day_of_week"] / 7)
    df["order_hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
    df["order_hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)

    df["invoice_date"] = df["fatura_tarihi"].dt.normalize()
    df["is_invoiced"] = df["invoice_date"].notna().astype("int8")
    df = add_date_features(df, "invoice_date", "invoice")
    df["invoice_lag_days"] = (df["invoice_date"] - df["order_date"]).dt.days.astype("Int64")

    df = add_date_features(df, "return_request_date", "return_request")
    df["return_lag_days"] = (df["return_request_date"] - df["order_date"]).dt.days.astype("Int64")

    df["is_gift_voucher"] = df["hediye_ceki"].notna().astype("int8")
    df["is_offer"] = df["kampanya"].notna().astype("int8")
    df["amount_missing"] = df["tutar"].isna().astype("int8")

    amount = df["tutar"].fillna(0)
    df["realized_sales_amount"] = np.where(df["is_delivered"].eq(1), amount, 0)
    df["cancelled_order_amount"] = np.where(df["is_cancelled"].eq(1), amount, 0)
    df["returned_order_amount"] = np.where(df["is_returned"].eq(1), amount, 0)

    drop_columns = [
        "id", "uye_adi", "firma_uye", "cep_telefonu_uye", "uye_grup_kodu",
        "uye_grubu", "uye_ws_kodu", "firma_uye_adi_fatura", "e_posta_adresi",
        "vergi_tc_no", "vergi_dairesi", "semt_teslimat", "posta_kodu_teslimat",
        "cep_telefonu_teslimat", "adres_teslimat", "ad_teslimat", "firma_fatura",
        "ad_fatura", "semt_fatura", "cep_telefonu_fatura", "adres_fatura",
        "kargo_takip_no", "kargo_no", "kargo_no_iade", "fatura_numarasi",
        "irsaliye_numarasi", "platform_siparis_no", "genel_siparis_notu",
        "alt_odeme_tipi", "doviz_tutar", "sistem_kuru", "kur_fiyati",
    ]

    final_df = df.drop(columns=[column for column in drop_columns if column in df.columns]).copy()

    safe_model_columns = [
        "siparis_no", "tarih", "order_date", "order_year", "order_month",
        "order_day", "order_day_of_week", "order_week_of_year", "order_quarter",
        "order_is_weekend", "order_hour", "order_minute", "order_month_sin",
        "order_month_cos", "order_day_of_week_sin", "order_day_of_week_cos",
        "order_hour_sin", "order_hour_cos", "tutar", "kdv", "kargo_toplami",
        "hizmet_bedeli", "kargo", "amount_missing", "doviz_cinsi", "odeme_tipi",
        "banka", "kart", "pos", "platform", "kaynak", "araci", "il_teslimat",
        "ilce_teslimat", "ulke_teslimat", "hediye_ceki", "kampanya",
        "is_gift_voucher", "is_offer",
    ]
    safe_model_columns = [column for column in safe_model_columns if column in final_df.columns]
    model_features = final_df[safe_model_columns].copy()

    quality = pd.DataFrame({
        "metric": [
            "core_row_count", "core_unique_order_count", "excluded_usd_eur_count",
            "final_tl_row_count", "return_detail_unique_order_count",
            "unmatched_return_order_count", "missing_invoice_date_count",
            "missing_amount_count", "cancelled_count", "returned_count", "delivered_count",
        ],
        "value": [
            len(core), core["siparis_no"].nunique(), int(foreign_mask.sum()), len(final_df),
            iade_detail["siparis_no"].nunique(), unmatched["siparis_no"].nunique(),
            int(final_df["fatura_tarihi"].isna().sum()), int(final_df["tutar"].isna().sum()),
            int(final_df["is_cancelled"].sum()), int(final_df["is_returned"].sum()),
            int(final_df["is_delivered"].sum()),
        ],
    })

    with pd.ExcelWriter(PROCESSED_DIR / "merge_data.xlsx", engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="orders", index=False)
        quality.to_excel(writer, sheet_name="quality_report", index=False)

    model_features.to_excel(
        PROCESSED_DIR / "model_features_pre_order.xlsx",
        index=False,
    )

    if not unmatched.empty:
        columns = [
            "siparis_no", "talep_tarihi", "urun_adi", "alt_urun_adi",
            "iade_nedeni", "siparis_edilen_miktar", "iade_edilecek_miktar",
        ]
        columns = [column for column in columns if column in unmatched.columns]
        unmatched[columns].to_excel(
            PROCESSED_DIR / "unmatched_return_requests.xlsx",
            index=False,
        )

    return final_df


if __name__ == "__main__":
    result = build_processed_files()
    print(f"Veri hazırlama tamamlandı: {result.shape}")
