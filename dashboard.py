from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.return_nlp import load_return_nlp
from src.statistical_analysis import build_statistical_report, statistical_report_to_excel


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(page_title="DOGO Veri Analizi", page_icon="📊", layout="wide")


@st.cache_data
def load_orders():
    data = pd.read_excel(PROCESSED_DIR / "merge_data.xlsx", sheet_name="orders")
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    return data


@st.cache_data
def load_nlp_data():
    return load_return_nlp()


def percent(value):
    return f"{value:.1%}"


orders = load_orders()
(
    _return_detail,
    return_summary,
    _return_monthly,
    product_summary,
    product_reason,
    nlp_review_queue,
    nlp_method,
    sentiment_summary,
    sentiment_method,
) = load_nlp_data()
statistical_report = build_statistical_report(orders)

st.title("DOGO Sipariş ve İade İçgörüleri")
st.caption(
    "Bu dashboard gelecek değer tahmin etmez; mevcut veride gözlenen ilişkileri "
    "istatistiksel testlerle gösterir."
)


# Kısa özet
total_orders = len(orders)
delivered_orders = int(orders["is_delivered"].sum())
return_count = int(orders["is_returned"].sum())
return_rate = return_count / total_orders if total_orders else 0
total_sales = orders["realized_sales_amount"].sum()

summary_columns = st.columns(5)
summary_columns[0].metric("Toplam sipariş", f"{total_orders:,}".replace(",", "."))
summary_columns[1].metric("Teslim edilen", f"{delivered_orders:,}".replace(",", "."))
summary_columns[2].metric("Gerçekleşen satış", f"₺{total_sales:,.0f}".replace(",", "."))
summary_columns[3].metric("İade adedi", f"{return_count:,}".replace(",", "."))
summary_columns[4].metric("İade oranı", percent(return_rate))

with st.expander("Veri kapsamı ve kalite bilgisi"):
    quality_columns = st.columns(4)
    unmatched_path = PROCESSED_DIR / "unmatched_return_requests.xlsx"
    unmatched_count = 0
    if unmatched_path.exists():
        unmatched_data = pd.read_excel(unmatched_path)
        unmatched_count = unmatched_data["siparis_no"].nunique()
    quality_columns[0].metric(
        "Veri dönemi",
        f"{orders['order_date'].min():%d.%m.%Y} – {orders['order_date'].max():%d.%m.%Y}",
    )
    quality_columns[1].metric("Eksik sipariş tarihi", int(orders["order_date"].isna().sum()))
    quality_columns[2].metric("Eksik sipariş tutarı", int(orders["tutar"].isna().sum()))
    quality_columns[3].metric(
        "Eşleşmeyen iade siparişi",
        unmatched_count,
    )
    st.write(
        "USD/EUR kayıtları analiz dışında bırakıldı. Küçük gruplar testlerden "
        "çıkarıldı; böylece tek veya birkaç kayda dayalı sonuç üretilmedi."
    )


# İstatistiksel analiz
st.divider()
st.header("İstatistiksel davranış analizi")
st.write(
    "Oran, %95 güven aralığı, düzeltilmiş p-değeri ve etki büyüklüğü birlikte "
    "incelenir. İstatistiksel ilişki nedensellik anlamına gelmez."
)

stat_tabs = st.tabs([
    "Grupları karşılaştır",
    "Anlamlı ilişkiler",
    "Geçen süre",
    "Excel raporu",
])

with stat_tabs[0]:
    group_names = list(statistical_report["groups"].keys())
    selected_group = st.selectbox("Karşılaştırılacak boyut", group_names)
    selected_behavior = st.radio("Gösterilecek davranış", ["İptal", "İade"], horizontal=True)
    group_data = statistical_report["groups"][selected_group].copy()
    rate_column = "cancelled_rate" if selected_behavior == "İptal" else "returned_rate"
    count_column = "cancelled_count" if selected_behavior == "İptal" else "returned_count"
    chart_data = group_data.sort_values(rate_column, ascending=False).head(15)
    chart = px.bar(
        chart_data.sort_values(rate_column),
        x=rate_column,
        y="group",
        orientation="h",
        text=chart_data.sort_values(rate_column)[rate_column].map(percent),
        title=f"{selected_group} bazında {selected_behavior.lower()} oranı",
        labels={"group": selected_group, rate_column: "Oran"},
    )
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(height=430)
    st.plotly_chart(chart, use_container_width=True)

    highest_row = group_data.loc[group_data[rate_column].idxmax()]
    st.info(
        f"Bu boyutta en yüksek {selected_behavior.lower()} oranı "
        f"{highest_row['group']} grubunda: "
        f"{int(highest_row[count_column])} / {int(highest_row['order_count'])} "
        f"sipariş ({percent(highest_row[rate_column])})."
    )
    if selected_group == "Şehir":
        st.caption("Şehirler en çok sipariş alan 5 şehir ve Diğer şehirler olarak gruplanmıştır.")
    elif selected_group == "Banka":
        st.caption("Banka karşılaştırması yalnızca kredi kartı ve banka bilgisi bulunan siparişler içindir.")
    elif selected_group == "Platform":
        st.caption("Admin/operasyon kayıtları müşteri platformu karşılaştırmasına dahil edilmemiştir.")

    group_view = group_data[[
        "group", "order_count", count_column, rate_column,
    ]].copy()
    group_view[rate_column] = group_view[rate_column].map(percent)
    group_view = group_view.rename(columns={
        "group": selected_group,
        "order_count": "Sipariş adedi",
        count_column: f"{selected_behavior} adedi",
        rate_column: f"{selected_behavior} oranı",
    })
    st.dataframe(group_view, hide_index=True, use_container_width=True)

with stat_tabs[1]:
    chi = statistical_report["chi_square"]
    t_tests = statistical_report["welch_t_tests"]
    significant_chi = chi[chi["significant_05"]].copy()
    significant_t = t_tests[t_tests["significant_05"]].copy()
    st.subheader("Kategorik alanlar: ki-kare testi")
    st.caption(
        "Anlamlı sonuç: düzeltilmiş p<0,05 ve ki-kare varsayımı uygun. "
        "Cramer V ilişkinin gücünü gösterir."
    )
    if significant_chi.empty:
        st.info("Düzeltilmiş p<0,05 olan bir sonuç bulunamadı.")
    else:
        chi_view = significant_chi[[
            "group_name", "target", "p_value_adjusted", "cramers_v", "sample_size",
        ]].rename(columns={
            "group_name": "Boyut",
            "target": "Davranış",
            "p_value_adjusted": "Düzeltilmiş p",
            "cramers_v": "Cramer V",
            "sample_size": "Örneklem",
        })
        st.dataframe(chi_view, hide_index=True, use_container_width=True)

    st.subheader("Sayısal alanlar: Welch t-testi")
    st.caption("Cohen d, ortalamalar arasındaki farkın büyüklüğünü gösterir.")
    if significant_t.empty:
        st.info("Düzeltilmiş p<0,05 olan bir sonuç bulunamadı.")
    else:
        t_view = significant_t[[
            "variable", "target", "positive_mean", "negative_mean",
            "p_value_adjusted", "cohens_d",
        ]].rename(columns={
            "variable": "Değişken",
            "target": "Davranış",
            "positive_mean": "Davranış var ortalaması",
            "negative_mean": "Davranış yok ortalaması",
            "p_value_adjusted": "Düzeltilmiş p",
            "cohens_d": "Cohen d",
        })
        st.dataframe(t_view, hide_index=True, use_container_width=True)
    with st.expander("Tüm test sonuçlarını göster"):
        st.dataframe(chi, hide_index=True, use_container_width=True)
        st.dataframe(t_tests, hide_index=True, use_container_width=True)

with stat_tabs[2]:
    duration = statistical_report["duration"]
    st.write(
        "`gecen_sure_dk` işletme açısından okunabilir dört sabit gün aralığına ayrıldı. Bu bölüm nedensellik "
        "kurmaz; yalnızca gruplar arasındaki gözlenen farkı gösterir."
    )
    duration_chart = px.bar(
        duration,
        x="duration_band",
        y=["cancelled_rate", "returned_rate"],
        barmode="group",
        title="Geçen süre bantlarına göre oranlar",
        labels={"duration_band": "Süre bandı", "value": "Oran", "variable": "Davranış"},
    )
    duration_chart.update_yaxes(tickformat=".0%")
    duration_chart.update_layout(height=430)
    st.plotly_chart(duration_chart, use_container_width=True)
    highest_return = duration.loc[duration["returned_rate"].idxmax()]
    st.info(
        f"İade oranı en yüksek aralık {highest_return['duration_band']}: "
        f"{int(highest_return['returned_count'])} / "
        f"{int(highest_return['order_count'])} sipariş "
        f"({percent(highest_return['returned_rate'])})."
    )
    duration_view = duration.rename(columns={
        "duration_band": "Süre bandı",
        "order_count": "Sipariş adedi",
        "cancelled_count": "İptal adedi",
        "returned_count": "İade adedi",
        "min_elapsed_minutes": "Aralık başlangıcı (dk)",
        "max_elapsed_minutes": "Aralık sonu (dk)",
        "cancelled_rate": "İptal oranı",
        "returned_rate": "İade oranı",
    })
    duration_view["İptal oranı"] = duration_view["İptal oranı"].map(percent)
    duration_view["İade oranı"] = duration_view["İade oranı"].map(percent)
    st.dataframe(duration_view, hide_index=True, use_container_width=True)

with stat_tabs[3]:
    st.write(
        "Ham veriler yenilenip pipeline tekrar çalıştırıldığında rapor yeniden "
        "oluşturulur."
    )
    st.download_button(
        "İstatistiksel Excel raporunu indir",
        data=statistical_report_to_excel(statistical_report),
        file_name="statistical_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# NLP
st.divider()
st.header("İade açıklamalarının NLP analizi")
st.write(
    "Elle etiketlenmiş açıklamalarla eğitilen NLP modeli, iade nedenlerini ve "
    "ürün tiplerini özetler. Bu sonuçlar mevcut iade açıklamalarının dağılımıdır; "
    "ürünlerin gerçek iade oranı değildir."
)

nlp_columns = st.columns(2)
with nlp_columns[0]:
    category_chart = px.bar(
        return_summary.sort_values("return_lines"),
        x="return_lines",
        y="primary_category",
        orientation="h",
        text="return_lines",
        title="İade nedenleri",
        labels={"return_lines": "Açıklama adedi", "primary_category": "Neden"},
    )
    category_chart.update_layout(height=420)
    st.plotly_chart(category_chart, use_container_width=True)
with nlp_columns[1]:
    product_chart = px.bar(
        product_summary.sort_values("return_lines"),
        x="return_lines",
        y="product_type",
        orientation="h",
        text="return_lines",
        title="İade açıklamalarındaki ürün tipleri",
        labels={"return_lines": "Açıklama adedi", "product_type": "Ürün tipi"},
    )
    product_chart.update_layout(height=420)
    st.plotly_chart(product_chart, use_container_width=True)

st.caption(
    f"NLP yöntemi: {nlp_method}. Gözden geçirme kuyruğunda "
    f"{len(nlp_review_queue)} kayıt bulunuyor."
)
with st.expander("Ürün tipi × iade nedeni tablosu"):
    matrix_data = product_reason.pivot_table(
        index="product_type",
        columns="primary_category",
        values="return_lines",
        aggfunc="sum",
        fill_value=0,
    )
    st.dataframe(matrix_data, use_container_width=True)

if nlp_method == "tfidf_logistic":
    st.success(
        "Elle etiketli NLP modeli aktif. Model performansı yalnızca etiketli "
        "örneklerde ölçülmüştür; yeni dönemlerde tekrar doğrulanmalıdır."
    )

st.subheader("İkinci çıktı: müşteri açıklaması tonu")
st.write(
    "Bu bölüm iade nedeninden ayrı olarak müşterinin açıklamadaki iletişim tonunu "
    "gösterir. Gerçek psikolojik duygu durumu değil, yazılı metindeki memnuniyet "
    "ve şikâyet ifadesidir."
)
if sentiment_method == "tfidf_logistic" and not sentiment_summary.empty:
    sentiment_chart = px.bar(
        sentiment_summary.sort_values("return_lines"),
        x="return_lines",
        y="sentiment",
        orientation="h",
        text="return_lines",
        title="İade açıklamalarında müşteri tonu",
        labels={"return_lines": "Açıklama adedi", "sentiment": "Müşteri tonu"},
    )
    sentiment_chart.update_layout(height=360)
    st.plotly_chart(sentiment_chart, use_container_width=True)
else:
    st.info(
        "Duygu/ton modeli henüz eğitilmedi. Önce review_queue içindeki "
        "manual_sentiment sütununu etiketleyip train_sentiment.py dosyasını çalıştırın."
    )
