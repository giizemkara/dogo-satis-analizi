**DOGO Satış ve İade Verilerinin Analizi: İade Nedenleri, Satış Tahmini ve İptal Riski**

## 1. Problem

Amaç yalnızca kaç sipariş olduğunu göstermek değil; satış performansını, iade nedenlerini ve gelecek dönem beklentisini aynı akışta anlayabilmektir.

## 2. Veri

- Teslim, iptal ve iade Excel dosyaları birleştirildi.
- USD ve EUR kayıtları çıkarılarak analiz TL bazında yapıldı.
- 3133 TL sipariş analiz edildi.
- İade açıklamalı dosyada ürün adı, alt ürün ve barkod alanları kullanıldı.

## 3. Ana bulgu

İade açıklamalarının yaklaşık %63'ü beden/kalıp uyumsuzluğu ile ilişkilidir. Bu, ürün tipi ve beden bilgisinin iade analizinde en önemli geliştirme alanı olduğunu gösterir.

Terlik ve sneakers ürünleri iade kayıtlarında en yüksek yoğunluğa sahip ürün tipleridir. Ancak bu değerler henüz gerçek iade oranı değildir; toplam satış adedi paydası eklenmelidir.

## 4. Satış tahmini

Günlük gerçekleşen satışlar üzerinden trend ve haftanın günü etkisi kullanılarak 7–90 günlük tahmin oluşturuldu. Bu ilk tahmin modeli karar destek amaçlıdır; yeni veriler geldikçe yeniden çalıştırılmalıdır.

Son 30 gün holdout testiyle ölçülen satış tahmini WAPE değeri yaklaşık %31,2'dir. Bu nedenle sonuç bütçe taahhüdü değil, yön gösteren bir başlangıç tahminidir.

## 5. İptal modeli

İptal modeli siparişlerin iptal riskini sıralayabiliyor. Karar eşiği validation setinde optimize edildi.

Test sonuçları:

- ROC-AUC: 0.846
- Precision: 0.391
- Recall: 0.563

Bu sonuç, operasyon ekibinin öncelikli olarak incelemesi gereken siparişleri belirlemek için başlangıç seviyesinde kullanılabilir.

## 6. İade modeli hakkında dürüst sonuç

İade modeli henüz yeterli değil. Bunun nedeni yalnızca algoritma değildir; hedef tanımı ve veri eşleşmesi de sorunludur. İade talebi, tamamlanmış iade ve teslim edilmiş ama iade talebi açılmış siparişler ayrı ele alınmalıdır.

## 7. Önerilen sonraki adımlar

1. Tüm siparişlere ürün/SKU/kategori/beden bilgisini bağlamak.
2. Ürün tipi bazında toplam satış ve iade adedini hesaplamak.
3. Gerçek ürün iade oranını oluşturmak.
4. İade risk modelini bu yeni hedefle yeniden eğitmek.
5. Daha uzun tarih geçmişiyle satış tahminini geliştirmek.

## Müdüre söylenebilecek kısa kapanış

“İlk aşamada veriyi TL bazında temizleyip tek akışta birleştirdim. İade açıklamalarını NLP ile incelediğimde en büyük problemin beden ve kalıp uyumsuzluğu olduğunu gördüm. Satışlar için başlangıç tahmini, iptaller için ise eşik değeri optimize edilmiş bir risk modeli oluşturdum. İade riskini gerçek olasılığa çevirmek için bir sonraki adım tüm siparişlere ürün ve SKU bilgisini bağlamak.”
