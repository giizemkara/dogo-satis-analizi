from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.forecast import backtest_daily, forecast_daily
from src.return_nlp import load_return_nlp


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="DOGO Satış ve İade Analizi",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_orders():
    data = pd.read_excel(PROCESSED_DIR / "merge_data.xlsx", sheet_name="orders")
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    return data


@st.cache_data
def load_nlp_data():
    return load_return_nlp()


@st.cache_data
def load_metrics(filename):
    path = PROCESSED_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


orders = load_orders()
(
    return_detail,
    return_summary,
    return_monthly,
    product_summary,
    product_reason,
) = load_nlp_data()

st.title("DOGO Satış ve İade Analizi")
st.caption("Satış trendi, gelecek dönem tahmini ve iade açıklamalarının NLP kategorileri")


# Genel özet
total_orders = len(orders)
total_sales = orders["realized_sales_amount"].sum()
cancel_count = int(orders["is_cancelled"].sum())
return_count = int(orders["is_returned"].sum())
return_share = return_count / total_orders * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam sipariş", f"{total_orders:,}".replace(",", "."))
col2.metric("Gerçekleşen satış", f"₺{total_sales:,.0f}".replace(",", "."))
col3.metric("İptal edilen", f"{cancel_count:,}".replace(",", "."))
col4.metric("İade oranı", f"{return_share:.1f}%")


# Gelecek tahmini
st.divider()
st.header("Gelecek dönem satış tahmini")

forecast_chart_col, forecast_control_col = st.columns([3, 1])

with forecast_control_col:
    forecast_metric = st.selectbox(
        "Tahmin metriği",
        ["Gerçekleşen satış (TL)", "Sipariş adedi"],
    )
    forecast_horizon = st.slider(
        "Tahmin süresi (gün)",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
    )

forecast_input = orders.copy()
value_column = "realized_sales_amount"

if forecast_metric == "Sipariş adedi":
    forecast_input["forecast_orders"] = 1
    value_column = "forecast_orders"

forecast_data = forecast_daily(
    forecast_input,
    value_column=value_column,
    horizon=forecast_horizon,
)

with forecast_chart_col:
    fig_forecast = px.line(
        forecast_data,
        x="date",
        y="value",
        color="kind",
        title=f"Geçmiş ve {forecast_horizon} günlük tahmin",
        labels={"date": "Tarih", "value": forecast_metric, "kind": ""},
        color_discrete_map={
            "Gerçekleşen": "#2563eb",
            "Tahmin": "#f97316",
        },
    )
    fig_forecast.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig_forecast, use_container_width=True)

forecast_values = forecast_data.loc[forecast_data["kind"] == "Tahmin", "value"]
forecast_total = forecast_values.sum()

with forecast_control_col:
    if forecast_metric == "Gerçekleşen satış (TL)":
        st.metric(
            f"Önümüzdeki {forecast_horizon} gün tahmini",
            f"₺{forecast_total:,.0f}".replace(",", "."),
        )
    else:
        st.metric(
            f"Önümüzdeki {forecast_horizon} gün tahmini",
            f"{forecast_total:,.0f}".replace(",", "."),
        )

backtest = backtest_daily(
    forecast_input,
    value_column=value_column,
    horizon=min(30, max(7, len(forecast_data[forecast_data["kind"] == "Gerçekleşen"]) // 4)),
)

if backtest["wape"] is not None:
    st.caption(
        f"Geçmiş holdout testi — {backtest['horizon']} gün WAPE: "
        f"{backtest['wape']:.1f}% | MAE: {backtest['mae']:,.0f}"
    )

st.info(
    "Tahmin, mevcut 181 günlük geçmişteki trend ve haftanın günü etkisine dayalı "
    "başlangıç modelidir. Yeni veri geldikçe tekrar çalıştırılmalıdır."
)


# NLP iade kategorileri
st.divider()
st.header("İade açıklamalarının NLP analizi")
st.write(
    "Açıklamalar kişisel ve ödeme bilgileri temizlenerek ana iade nedeni "
    "kategorilerine ayrılmıştır. Bu bölüm sipariş numarası değil, iade nedenlerini gösterir."
)

nlp_chart_col, nlp_month_col = st.columns(2)

with nlp_chart_col:
    category_chart = px.bar(
        return_summary.sort_values("return_lines"),
        x="return_lines",
        y="primary_category",
        orientation="h",
        text="return_lines",
        title="İade açıklamalarının kategori dağılımı",
        labels={
            "return_lines": "İade açıklaması sayısı",
            "primary_category": "Kategori",
        },
    )
    category_chart.update_layout(height=430)
    st.plotly_chart(category_chart, use_container_width=True)

with nlp_month_col:
    monthly_chart = px.bar(
        return_monthly.sort_values("month"),
        x="month",
        y="return_lines",
        color="primary_category",
        title="İade kategorilerinin aylık dağılımı",
        labels={
            "month": "Ay",
            "return_lines": "İade açıklaması sayısı",
            "primary_category": "Kategori",
        },
    )
    monthly_chart.update_layout(height=430, legend_title_text="Kategori")
    st.plotly_chart(monthly_chart, use_container_width=True)

summary_view = return_summary.copy()
summary_view["share_of_return_lines"] = summary_view["share_of_return_lines"].round(1)
summary_view = summary_view.rename(columns={
    "primary_category": "İade kategorisi",
    "return_lines": "Açıklama sayısı",
    "unique_orders": "Benzersiz sipariş sayısı",
    "share_of_return_lines": "İade açıklamalarındaki pay (%)",
})
st.dataframe(summary_view, hide_index=True, use_container_width=True)

product_chart_col, matrix_col = st.columns(2)

with product_chart_col:
    product_chart = px.bar(
        product_summary.sort_values("return_lines"),
        x="return_lines",
        y="product_type",
        orientation="h",
        text="return_lines",
        title="İade edilen ürün tipleri",
        labels={
            "return_lines": "İade satırı sayısı",
            "product_type": "Ürün tipi",
        },
    )
    product_chart.update_layout(height=430)
    st.plotly_chart(product_chart, use_container_width=True)

with matrix_col:
    matrix_data = product_reason.pivot_table(
        index="product_type",
        columns="primary_category",
        values="return_lines",
        aggfunc="sum",
        fill_value=0,
    )
    matrix_chart = px.imshow(
        matrix_data,
        text_auto=True,
        aspect="auto",
        title="Ürün tipi × iade nedeni",
        labels={
            "x": "İade nedeni",
            "y": "Ürün tipi",
            "color": "İade satırı",
        },
    )
    matrix_chart.update_layout(height=430)
    st.plotly_chart(matrix_chart, use_container_width=True)

st.warning(
    "Bu NLP sonucu kategorilerin iade açıklamaları içindeki payını gösterir. "
    "Ürün tipi bilgisi açıklamalı iade dosyasından çıkarılmıştır. Gerçek bir ürün "
    "iade riski olasılığı için aynı ürün tiplerinin toplam satış adedine de ihtiyaç vardır."
)


# Model durumu
st.divider()
st.header("Risk modellerinin durumu")

cancel_metrics = load_metrics("cancellation_model_metrics.csv")
return_metrics = load_metrics("return_model_metrics.csv")
model_left, model_right = st.columns(2)

with model_left:
    st.subheader("İptal modeli")
    if not cancel_metrics.empty:
        test = cancel_metrics[cancel_metrics["split"] == "test"].iloc[0]
        st.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
        st.write(f"Test PR-AUC: **{test['pr_auc']:.3f}**")
        st.write(f"Test recall: **{test['recall']:.3f}**")

with model_right:
    st.subheader("İade modeli")
    if not return_metrics.empty:
        test = return_metrics[return_metrics["split"] == "test"].iloc[0]
        st.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
        st.write(f"Test PR-AUC: **{test['pr_auc']:.3f}**")
        st.caption(
            "Mevcut iade etiketi ve veri kapsamıyla bu model henüz karar vermek için yeterli değil."
        )
