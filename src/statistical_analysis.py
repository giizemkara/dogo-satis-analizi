from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GROUP_COLUMNS = {
    "Şehir": "il_teslimat",
    "Ödeme tipi": "odeme_tipi",
    "Platform": "platform",
    "Kampanya durumu": "is_offer",
    "Banka": "banka",
}

TOP_CITY_COUNT = 5

DURATION_BINS = [-np.inf, 2 * 24 * 60, 7 * 24 * 60, 14 * 24 * 60, np.inf]
DURATION_LABELS = [
    "0–2 gün",
    "3–7 gün",
    "8–14 gün",
    "15+ gün",
]

TARGET_COLUMNS = {
    "İptal": "is_cancelled",
    "İade": "is_returned",
    "İade talebi": "has_return_request",
}

NUMERIC_COLUMNS = {
    "Geçen süre (dk)": "gecen_sure_dk",
    "Sipariş tutarı": "tutar",
    "Kargo toplamı": "kargo_toplami",
    "Hizmet bedeli": "hizmet_bedeli",
}


def _safe_text(series):
    return (
        series.astype("string")
        .fillna("Bilinmiyor")
        .str.strip()
        .replace("", "Bilinmiyor")
    )


def _prepare_group_dimension(orders, group_name, group_column):
    """İş anlamı olmayan grupları ayırıp karşılaştırılabilir kategoriler üretir."""
    data = orders.copy()

    if group_name == "Banka":
        payment = _safe_text(data["odeme_tipi"])
        data = data[payment.eq("Kredi Kartı")].copy()
        data = data[_safe_text(data[group_column]).ne("Bilinmiyor")].copy()

    elif group_name == "Platform":
        platform = _safe_text(data[group_column])
        data = data[~platform.str.contains("admin", case=False, na=False)].copy()

    data["group"] = _safe_text(data[group_column])

    if group_name == "Şehir":
        top_cities = data["group"].value_counts().nlargest(TOP_CITY_COUNT).index
        data["group"] = data["group"].where(
            data["group"].isin(top_cities),
            "Diğer şehirler",
        )
    elif group_name == "Kampanya durumu":
        data["group"] = data[group_column].map({1: "Kampanyalı", 0: "Kampanyasız"})
        data["group"] = data["group"].fillna("Bilinmiyor")
    elif group_name == "Platform":
        data["group"] = data["group"].replace({
            "mobile_site": "Mobil site",
            "Masaüstü Site": "Masaüstü site",
        })

    return data


def _wilson_interval(successes, total, confidence=0.95):
    if total == 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0, centre - margin), min(1, centre + margin)


def _benjamini_hochberg(p_values):
    """Birden fazla hipotez için Benjamini-Hochberg FDR düzeltmesi."""
    values = pd.to_numeric(pd.Series(p_values), errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return adjusted.to_numpy()

    ordered = values[valid].sort_values()
    count = len(ordered)
    running_min = 1.0
    for rank in range(count, 0, -1):
        index = ordered.index[rank - 1]
        running_min = min(running_min, ordered.loc[index] * count / rank)
        adjusted.loc[index] = min(running_min, 1.0)

    return adjusted.to_numpy()


def _cohens_d(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[~np.isnan(first)]
    second = second[~np.isnan(second)]
    if len(first) < 2 or len(second) < 2:
        return np.nan

    pooled_variance = (
        (len(first) - 1) * np.var(first, ddof=1)
        + (len(second) - 1) * np.var(second, ddof=1)
    ) / (len(first) + len(second) - 2)
    if pooled_variance <= 0:
        return 0.0
    return float((np.mean(first) - np.mean(second)) / np.sqrt(pooled_variance))


def group_behavior_stats(orders, group_name, group_column, min_group_size=20):
    """Grupların adet, oran, ortalama ve oran güven aralıklarını hesaplar."""
    data = _prepare_group_dimension(orders, group_name, group_column)
    counts = data["group"].value_counts()
    allowed = counts[counts >= min_group_size].index
    data = data[data["group"].isin(allowed)].copy()

    rows = []
    for group, subset in data.groupby("group", sort=False):
        total = len(subset)
        cancelled = int(subset["is_cancelled"].sum())
        returned = int(subset["is_returned"].sum())
        cancelled_low, cancelled_high = _wilson_interval(cancelled, total)
        returned_low, returned_high = _wilson_interval(returned, total)
        rows.append({
            "group": group,
            "order_count": total,
            "cancelled_count": cancelled,
            "cancelled_rate": cancelled / total,
            "cancelled_ci_low": cancelled_low,
            "cancelled_ci_high": cancelled_high,
            "returned_count": returned,
            "returned_rate": returned / total,
            "returned_ci_low": returned_low,
            "returned_ci_high": returned_high,
            "mean_elapsed_minutes": subset["gecen_sure_dk"].mean(),
            "median_order_amount": subset["tutar"].median(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("order_count", ascending=False)
        .reset_index(drop=True)
    )


def duration_behavior_stats(orders, band_count=4):
    """Geçen süreyi okunabilir operasyonel gün aralıklarına ayırır."""
    data = orders.copy()
    data["elapsed_minutes"] = pd.to_numeric(
        data["gecen_sure_dk"], errors="coerce"
    )
    data = data.dropna(subset=["elapsed_minutes"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["duration_band"] = pd.cut(
        data["elapsed_minutes"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        include_lowest=True,
    )

    result = (
        data.groupby("duration_band", observed=False)
        .agg(
            order_count=("is_cancelled", "size"),
            cancelled_count=("is_cancelled", "sum"),
            returned_count=("is_returned", "sum"),
            mean_elapsed_minutes=("elapsed_minutes", "mean"),
            min_elapsed_minutes=("elapsed_minutes", "min"),
            max_elapsed_minutes=("elapsed_minutes", "max"),
        )
        .reset_index()
    )
    result["cancelled_rate"] = result["cancelled_count"] / result["order_count"]
    result["returned_rate"] = result["returned_count"] / result["order_count"]
    return result


def chi_square_tests(orders, min_group_size=75):
    """Kategorik alanlar ile davranış hedefleri arasındaki ilişkiyi test eder."""
    rows = []
    for group_name, group_column in GROUP_COLUMNS.items():
        if group_column not in orders.columns:
            continue

        data = _prepare_group_dimension(orders, group_name, group_column)
        data = data[["group", *TARGET_COLUMNS.values()]].copy()
        allowed = data["group"].value_counts()
        allowed = allowed[allowed >= min_group_size].index
        data = data[data["group"].isin(allowed)]

        for target_name, target_column in TARGET_COLUMNS.items():
            table = pd.crosstab(data["group"], data[target_column])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi2, p_value, degrees_of_freedom, expected = stats.chi2_contingency(table)
            total = table.to_numpy().sum()
            minimum_dimension = min(table.shape) - 1
            cramers_v = (
                np.sqrt(chi2 / (total * minimum_dimension))
                if total and minimum_dimension > 0
                else np.nan
            )
            rows.append({
                "group_name": group_name,
                "group_column": group_column,
                "target": target_name,
                "target_column": target_column,
                "sample_size": int(total),
                "category_count": int(table.shape[0]),
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees_of_freedom),
                "p_value": float(p_value),
                "minimum_expected_count": float(np.min(expected)),
                "cramers_v": float(cramers_v),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_adjusted"] = _benjamini_hochberg(result["p_value"])
        result["assumption_ok"] = result["minimum_expected_count"] >= 5
        result["significant_05"] = (
            (result["p_value_adjusted"] < 0.05)
            & result["assumption_ok"]
        )
    return result


def welch_t_tests(orders, min_group_size=20):
    """Sayısal değişkenleri davranış hedeflerine göre Welch t-testiyle karşılaştırır."""
    rows = []
    for variable_name, variable_column in NUMERIC_COLUMNS.items():
        if variable_column not in orders.columns:
            continue
        for target_name, target_column in TARGET_COLUMNS.items():
            data = orders[[variable_column, target_column]].copy()
            data[variable_column] = pd.to_numeric(data[variable_column], errors="coerce")
            data = data.dropna()
            first = data.loc[data[target_column].eq(1), variable_column].to_numpy()
            second = data.loc[data[target_column].eq(0), variable_column].to_numpy()
            if len(first) < min_group_size or len(second) < min_group_size:
                continue

            test = stats.ttest_ind(first, second, equal_var=False)
            first_variance = np.var(first, ddof=1)
            second_variance = np.var(second, ddof=1)
            first_term = first_variance / len(first)
            second_term = second_variance / len(second)
            denominator = (
                first_term**2 / (len(first) - 1)
                + second_term**2 / (len(second) - 1)
            )
            degrees_of_freedom = (
                (first_term + second_term) ** 2 / denominator
                if denominator
                else np.nan
            )
            standard_error = np.sqrt(first_term + second_term)
            critical_value = stats.t.ppf(0.975, degrees_of_freedom)
            difference = np.mean(first) - np.mean(second)
            rows.append({
                "variable": variable_name,
                "variable_column": variable_column,
                "target": target_name,
                "positive_count": len(first),
                "negative_count": len(second),
                "positive_mean": np.mean(first),
                "negative_mean": np.mean(second),
                "mean_difference": difference,
                "difference_ci_low": difference - critical_value * standard_error,
                "difference_ci_high": difference + critical_value * standard_error,
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": float(degrees_of_freedom),
                "p_value": float(test.pvalue),
                "cohens_d": _cohens_d(first, second),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_adjusted"] = _benjamini_hochberg(result["p_value"])
        result["significant_05"] = result["p_value_adjusted"] < 0.05
    return result


def build_statistical_report(orders, min_group_size=20, test_min_group_size=75):
    """Dashboard ve Excel raporu için tüm istatistiksel çıktıları üretir."""
    groups = {
        name: group_behavior_stats(orders, name, column, min_group_size)
        for name, column in GROUP_COLUMNS.items()
        if column in orders.columns
    }
    return {
        "groups": groups,
        "duration": duration_behavior_stats(orders),
        "chi_square": chi_square_tests(orders, test_min_group_size),
        "welch_t_tests": welch_t_tests(orders, min_group_size),
        "metadata": pd.DataFrame([
            {"metric": "row_count", "value": len(orders)},
            {"metric": "min_group_size", "value": min_group_size},
            {"metric": "chi_square_min_group_size", "value": test_min_group_size},
            {"metric": "city_grouping", "value": "En çok sipariş alan 5 şehir + Diğer şehirler"},
            {"metric": "bank_scope", "value": "Yalnızca kredi kartı ve banka bilgisi bulunan siparişler"},
            {"metric": "platform_scope", "value": "Admin/operasyon kayıtları hariç"},
            {"metric": "duration_bands", "value": "0–2 gün; 3–7 gün; 8–14 gün; 15+ gün"},
            {"metric": "analysis_note", "value": "İlişki testi; nedensellik değildir."},
        ]),
    }


def statistical_report_to_excel(report):
    """Raporu dashboard download_button için Excel byte dizisine çevirir."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for group_name, frame in report["groups"].items():
            sheet_name = f"group_{group_name[:20]}"
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        report["duration"].to_excel(writer, sheet_name="duration_bands", index=False)
        report["chi_square"].to_excel(writer, sheet_name="chi_square", index=False)
        report["welch_t_tests"].to_excel(writer, sheet_name="welch_t_tests", index=False)
        report["metadata"].to_excel(writer, sheet_name="metadata", index=False)
    return output.getvalue()


def export_statistical_report(orders, output_path=None, min_group_size=20):
    report = build_statistical_report(orders, min_group_size=min_group_size)
    output_path = (
        Path(output_path)
        if output_path
        else PROCESSED_DIR / "statistical_report.xlsx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(statistical_report_to_excel(report))
    return output_path


if __name__ == "__main__":
    orders_path = PROCESSED_DIR / "merge_data.xlsx"
    orders = pd.read_excel(orders_path, sheet_name="orders")
    path = export_statistical_report(orders)
    print(f"İstatistiksel rapor kaydedildi: {path}")
