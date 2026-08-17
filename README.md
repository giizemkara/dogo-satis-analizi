# DOGO Satış, İade ve Tahmin Analizi

Bu proje DOGO sipariş verilerini kullanarak üç çıktıya odaklanır:

1. Satış ve sipariş trendlerini anlamak.
2. İade açıklamalarını NLP ile neden ve ürün tipi kategorilerine ayırmak.
3. İptal riskini ve gelecek dönem satışlarını başlangıç seviyesinde tahmin etmek.

## Proje çıktıları

| Çıktı | Açıklama |
|---|---|
| `merge_data.xlsx` | Temizlenmiş, yalnızca TL siparişlerden oluşan analiz datası |
| `model_features_pre_order.xlsx` | Sipariş anında bilinebilecek model girdileri |
| `cancellation_model.joblib` | İptal risk modeli |
| `return_model.joblib` | Deneysel iade modeli |
| `cancellation_model_metrics.csv` | İptal modeli validation/test metrikleri |
| `return_model_metrics.csv` | İade modeli validation/test metrikleri |
| `return_nlp_summary.xlsx` | İade nedeni, ürün tipi ve aylık NLP özetleri |

## Klasör ve Python dosyaları

### `dashboard.py`

Streamlit dashboard uygulamasıdır. Satış KPI'larını, günlük satış tahminini, NLP iade kategorilerini ve model metriklerini gösterir. Sipariş numarası listesi göstermez; amaç operasyonel detaydan çok yönetici seviyesinde içgörü sunmaktır.

### `src/prepare_data.py`

Ham Excel dosyalarını okur ve temizler:

- Teslim, iptal ve iade dosyalarını birleştirir.
- USD ve EUR kayıtlarını çıkarır.
- Tarih feature'ları üretir.
- İade açıklamalarını sipariş seviyesinde özetler.
- `merge_data.xlsx` ve `model_features_pre_order.xlsx` oluşturur.

### `src/data_prep.py`

Model eğitiminde kullanılacak işlenmiş dosyaları okur. Hedef kolonları ile model feature'larını ayırır ve zamana göre train/validation/test bölmesi yapar.

### `src/train.py`

İptal ve iade modellerini eğitir. Şu anda açıklanabilir bir Logistic Regression baseline kullanır. Sınıf dengesizliği için `class_weight="balanced"` aktiftir. Karar eşiği test setinde değil, validation setinde F1-score maksimize edilerek seçilir.

### `src/evaluate.py`

ROC-AUC, PR-AUC, precision, recall, F1 ve confusion matrix değerlerini hesaplar. Ayrıca validation threshold optimizasyonunu yapar.

### `src/predict.py`

Eğitilmiş `.joblib` modellerini yükleyerek feature dosyası üzerinde risk olasılığı üretir.

### `src/return_nlp.py`

İade açıklamalarını temizler ve şu iki seviyede sınıflandırır:

- İade nedeni: beden/kalıp, kalite/hasar, model/renk/beğeni, kargo/paketleme vb.
- Ürün tipi: terlik, sneakers, çanta, bot, sandalet, babet, loafer vb.

IBAN, e-posta ve telefon gibi kişisel/ödeme bilgileri analiz metninden çıkarılır.

### `src/forecast.py`

Günlük gerçekleşen satış veya sipariş adedi için trend ve haftanın günü etkisini kullanan basit bir gelecek tahmini üretir.

### `src/run_pipeline.py`

Veri hazırlama ve model eğitimini tek komutta çalıştırır.

## Çalıştırma

Önce Excel dosyalarının kapalı olduğundan emin ol.

Ham veriler değiştiğinde:

```powershell
cd C:\Users\gizem\DOGO_StajVerileri\dogo-satis-analizi
.\env\Scripts\python.exe -m src.run_pipeline
```

Dashboard'u açmak için:

```powershell
.\env\Scripts\python.exe -m streamlit run dashboard.py
```

Tarayıcı adresi:

```text
http://localhost:8501
```

8501 portu doluysa:

```powershell
.\env\Scripts\python.exe -m streamlit run dashboard.py --server.port 8502
```

## Mevcut bulgular

3133 TL sipariş analiz edilmiştir:

- Teslim: 2655
- İptal: 249
- İade: 229
- Ortalama sipariş tutarı: yaklaşık 3428 TL
- İptal oranı: yaklaşık %7,95
- İade oranı: yaklaşık %7,31
- Satış tahmini son 30 günlük holdout testinde WAPE: yaklaşık %31,2

İade açıklamalarında en yoğun kategori beden/kalıp uyumsuzluğudur. Ürün tipi dağılımında terlik ve sneakers öne çıkmaktadır.

## Model sonuçlarının yorumu

### İptal modeli

Validation setinde seçilen karar eşiği `0.95` olmuştur. Test sonuçları:

- ROC-AUC: `0.846`
- PR-AUC: `0.558`
- Precision: `0.391`
- Recall: `0.563`
- F1: `0.462`

Bu model başlangıç seviyesinde anlamlı bir sıralama gücüne sahiptir. Eşik yükseltilerek gereksiz risk alarmı azaltılmıştır.

### İade modeli

Test ROC-AUC `0.454` olduğu için şu an kullanılabilir seviyede değildir. Bunun temel sebepleri:

- İade talebi ve tamamlanmış iadenin aynı hedef olmaması.
- Açıklamalı iade dosyasının ana sipariş dosyasıyla tam eşleşmemesi.
- Son dönemlerde iade oranının değişmesi.
- Ürün/SKU bilgilerinin tüm siparişlerde bulunmaması.

Bu nedenle dashboard'da NLP iade kategorileri gösterilir; iade modeli henüz karar modeli olarak önerilmez.

## Sınırlılıklar ve sonraki adımlar

- Ürün tipi × iade nedeni analizi mevcut iade kayıtlarının dağılımını gösterir. Gerçek ürün iade oranı için aynı ürünlerin toplam satış adedi gerekir.
- Satış tahmini yalnızca yaklaşık 181 günlük veriye dayanır ve başlangıç modelidir.
- Satış tahmini için yaklaşık %31,2 WAPE, bu modelin yön gösterici bir baseline olduğunu; finansal bütçe tahmini olarak doğrudan kullanılmaması gerektiğini gösterir.
- Daha güçlü iade modeli için tüm siparişlere SKU/ürün/kategori/numara bilgisi bağlanmalıdır.
- Sonraki model adımı LightGBM veya CatBoost karşılaştırması ve zaman bazlı backtesting olabilir.
