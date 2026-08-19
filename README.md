# DOGO Sipariş ve İade İçgörüleri

Bu proje, DOGO sipariş verilerini geleceği kesin tahmin etmek için değil, mevcut davranışları ölçülebilir ve açıklanabilir biçimde incelemek için kullanır.

Projenin iki ana çıktısı vardır:

1. Şehir, ödeme tipi, platform, kampanya durumu, banka ve geçen süre gibi alanların sipariş davranışlarıyla ilişkisini istatistiksel testlerle göstermek.
2. İade açıklamalarını NLP ile iade nedeni ve ürün tipi kategorilerine ayırmak.

## Dashboard ne gösteriyor?

Dashboard artık tahmin grafiği veya kullanılabilirliği düşük risk modeli göstermiyor. Bunun yerine:

- Grup bazında iptal/iade oranlarını,
- %95 güven aralıklarını,
- Ki-kare testlerini ve Cramer V etki büyüklüğünü,
- Welch t-testlerini ve Cohen d etki büyüklüğünü,
- `gecen_sure_dk` bantlarına göre davranış oranlarını,
- NLP ile iade nedeni ve ürün tipi dağılımını,
- Veri kalitesi uyarılarını

gösterir.

İstatistiksel anlamlılık nedensellik değildir. Bir p-değerinin düşük olması, “bu alan davranışa kesin sebep oluyor” anlamına gelmez. Oran, güven aralığı ve etki büyüklüğü birlikte incelenmelidir.

## Python dosyaları

### `dashboard.py`

User dostu Streamlit arayüzüdür. Sipariş numarası gibi operasyonel detayları göstermez; karar vericinin görebileceği özet ve test sonuçlarını sunar.

### `src/prepare_data.py`

Ham Excel dosyalarını temizler, teslim/iptal/iade kayıtlarını birleştirir, USD/EUR kayıtlarını çıkarır, durum ve tarih alanlarını standardize eder ve `merge_data.xlsx` üretir.

### `src/statistical_analysis.py`

İstatistiksel analiz motorudur:

- Wilson %95 oran güven aralığı,
- Ki-kare testi,
- Benjamini-Hochberg çoklu test düzeltmesi,
- Cramer V,
- Welch t-testi,
- Cohen d,
- geçen süre bantları

hesaplar. Yeni `merge_data.xlsx` ile tekrar çalışabilir.

### `src/return_nlp.py`

İade açıklamalarını temizler. IBAN, e-posta ve telefon gibi bilgileri metinden çıkarır; iade nedeni ve ürün tipini sınıflandırır. Elle eğitilmiş NLP modeli varsa onu kullanır, metni boş satırlarda kural tabanına döner.

### `src/train_nlp.py`

`return_nlp_summary.xlsx` içindeki `manual_category` etiketleriyle TF-IDF + Logistic Regression NLP modelini eğitir.

### `src/train_sentiment.py`

`return_nlp_summary.xlsx` içindeki `manual_sentiment` etiketleriyle müşteri açıklaması tonunu sınıflandıran ikinci NLP modelini eğitir. Bu model iade nedeni modelinden ayrıdır; iade olasılığı tahmin etmez.

### `src/run_pipeline.py`

Veri temizleme, NLP özeti ve istatistiksel raporu tek komutta yeniler.

## Üretilen dosyalar

| Dosya | Amaç |
|---|---|
| `data/processed/merge_data.xlsx` | Temizlenmiş ana analiz verisi |
| `data/processed/quality_report` | Veri kalite ölçümleri |
| `data/processed/return_nlp_summary.xlsx` | NLP neden/ürün özetleri ve review queue |
| `data/processed/statistical_report.xlsx` | İstatistiksel testlerin Excel çıktısı |
| `models/return_nlp_model.joblib` | Elle etiketlenmiş NLP modeli |

Ham veriler `data/raw` altında tutulur ve GitHub'a gönderilmemelidir.

## Çalıştırma

Önce Excel dosyalarını kapat:

```powershell
.\env\Scripts\python.exe -m src.run_pipeline
```

Elle etiketlenmiş NLP modeli değiştiyse:

```powershell
.\env\Scripts\python.exe -m src.train_nlp
.\env\Scripts\python.exe -m src.return_nlp
```

Duygu/ton modelini kullanmak için önce `NLP_ETIKETLEME_REHBERI.md` dosyasındaki kurallara göre `review_queue` içindeki `manual_sentiment` sütununu doldurun. Ardından:

```powershell
.\env\Scripts\python.exe -m src.train_sentiment
.\env\Scripts\python.exe -m src.return_nlp
```

Dashboard:

```powershell
.\env\Scripts\python.exe -m streamlit run dashboard.py
```

İstatistiksel raporu dashboard olmadan üretmek için:

```powershell
.\env\Scripts\python.exe -m src.statistical_analysis
```

## Mevcut veri kalite sonucu

- 3.133 sipariş analiz edildi.
- Duplicate sipariş bulunmadı.
- Negatif tutar bulunmadı.
- Geçersiz sipariş durumu bulunmadı.
- Sipariş tarihlerinde eksik kayıt bulunmadı.
- 1 siparişte tutar eksik.
- 30 açıklamalı iade siparişi ana sipariş dosyasıyla eşleşmedi; bu kayıtlar silinmedi, veri kalite çıktısı olarak korundu.

## Mevcut istatistiksel bulgular

- Şehirler en çok sipariş alan 5 şehir ve “Diğer şehirler” olarak gruplanınca iptal ile ilişki görülüyor; Cramer V yaklaşık `0,15`.
- Ödeme tipi ile iptal davranışı arasında güçlü ilişki görülüyor; Cramer V yaklaşık `0,68`. Bu nedensellik kanıtı değildir; banka, kanal ve operasyon farkları da etkili olabilir.
- `gecen_sure_dk`, iade edilen ve edilmeyen gruplarda belirgin biçimde farklı; Cohen d yaklaşık `1,49`.
- `gecen_sure_dk` ile iptal arasında anlamlı fark bulunmadı.

Banka analizi yalnızca kredi kartı siparişlerinde, platform analizi ise admin/operasyon kayıtları çıkarıldıktan sonra yapılır. Süre analizi 0–2 gün, 3–7 gün, 8–14 gün ve 15+ gün aralıklarını kullanır.

`gecen_sure_dk` sipariş sonrasında oluşan bir alan olabilir. Bu nedenle ilişki analizinde kullanılabilir; sipariş anında bilinmiyorsa gelecekteki risk modeline özellik olarak eklenmemelidir.

## Sınırlılıklar

- Ürün tipi, açıklamalı iade dosyasının ürün adından çıkarılır; tüm satışların ürün paydası olmadığı için gerçek ürün iade oranı hesaplanmaz.
- 30 eşleşmeyen iade kaydı çözülmeden iade oranı analizinin kapsamı sınırlıdır.
- İstatistiksel testler ilişki gösterir, nedensellik göstermez.
- Yeni veri geldiğinde aynı kolon adları ve iş kuralları korunmalıdır.
