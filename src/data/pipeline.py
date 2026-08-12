import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseDataHandler(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DataCleaner(BaseDataHandler):
    """Genel veriyi temizler ve formata sokar."""
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Tekrarlanan Sipariş No'ları temizle
        df = df.drop_duplicates(subset=['Sipariş No'], keep='last')
        
        # 2. Tarih formatını standartlaştır
        if 'Tarih' in df.columns:
            df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True, format='mixed', errors='coerce')
            
        fiyat_sutunlari = ['Tutar', 'Döviz Tutar', 'KDV', 'Kargo Toplamı', 'Hizmet Bedeli']
        
        def guvenli_float(x):
            if pd.isna(x) or str(x).strip() in ['', 'nan']:
                return np.nan
            
            x_str = str(x).strip()
            
            # Eğer veri içinde hem nokta hem virgül varsa (Örn: 2.800,00)
            if '.' in x_str and ',' in x_str:
                if x_str.rfind(',') > x_str.rfind('.'):
                    x_str = x_str.replace('.', '').replace(',', '.')
                else:
                    x_str = x_str.replace(',', '')
            # Sadece virgül varsa (Örn: 254,55)
            elif ',' in x_str:
                x_str = x_str.replace(',', '.')
                
            try:
                # Sadece gerçek sayıya çeviriyoruz. Kesme veya metne çevirme yok.
                return float(x_str)
            except ValueError:
                return np.nan

        for sutun in fiyat_sutunlari:
            if sutun in df.columns:
                df[sutun] = df[sutun].apply(guvenli_float)
        # 4. Gereksiz ve kişisel (PII) veri içeren sütunları sil
        # Kendi Excel'indeki silinmesini istediğin sütun adlarını bu listeye ekleyebilirsin
        silinecek_sutunlar = [
            'ID', 'Üye Adı', 'Firma (Üye)', 'Cep Telefonu (Üye)', 'Üye WS Kodu', 
            'Firma/Üye Adı (Fatura)', 'E-posta Adresi', 'Vergi / Tc No', 'Vergi Dairesi', 
            'Semt (Teslimat)', 'Posta Kodu (Teslimat)', 'Kargo Takip No', 'Kargo No', 
            'Sipariş Süreci', 'Ad(Teslimat)', 'Cep Telefonu (Teslimat)', 'Adres (Teslimat)', 
            'Fatura Tarihi', 'Fatura Numarası', 'İrsaliye Numarası', 'Platform Sipariş No', 
            'Firma(Fatura)', 'Ad(Fatura)', 'İl (Fatura)', 'İlçe (Fatura)', 'Semt (Fatura)', 
            'Ülke (Fatura)', 'Cep Telefonu (Fatura)', 'Adres (Fatura)', 'Üye Temsilci', 
            'Genel Sipariş Notu', 'Hopi BirdId','Hopi Paracık','Paracık Tutarı','Hopi Kampanya','Genel İade Açıklaması', 'İade Açıklaması',
        
        ]
        # Sadece dataframe içinde gerçekten var olan sütunları sil (hata almamak için)
        sutunlari_sil = [col for col in silinecek_sutunlar if col in df.columns]
        df = df.drop(columns=sutunlari_sil)
        
        return df

class FeatureEngineer(BaseDataHandler):
    """Hedef değişkeni (Target) ayarlar ve iş mantığını (Business Logic) uygular."""
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Süreç' in df.columns and 'İade Nedeni' in df.columns:
            # 1. Şart: Süreç zaten İade veya İptal mi?
            is_canceled_or_returned = df['Süreç'].isin(['İade', 'İptal'])
            
            # 2. Şart: Süreç Teslim görünse bile İade Nedeni dolu mu? (Boş veya NaN değilse)
            has_reason = df['İade Nedeni'].notna() & (df['İade Nedeni'].astype(str).str.strip() != '')
            
            # İki şarttan biri bile sağlanıyorsa 1, yoksa 0 yap
            df['is_returned'] = np.where(is_canceled_or_returned | has_reason, 1, 0)
            
        return df

class DataPipeline:
    def __init__(self, raw_path: Path, processed_path: Path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()

    def _kategorileri_cikar(self, urun_adi):
        """NLP/Regex ile ürün kategorilerini belirler."""
        if pd.isna(urun_adi):
             return "Bilinmiyor", "Bilinmiyor"
        
        urun_adi_lower = str(urun_adi).lower()
        kategori_sozlugu = {
            "Ayakkabı": ["sneakers", "günlük ayakkabı", "sandalet", "terlik", "bot", "topuklu", "outdoor", "loafer", "babet"],
            "Çanta": ["omuz çantası", "sırt çantası", "seyahat çantası", "el çantası", "tote bag", "çanta"],
            "Giyim": ["çorap", "t-shirt", "tshirt", "tişört", "giyim"],
            "Aksesuar": ["cüzdan", "pasaport", "fular", "bandana", "aksesuar"]
        }
        for ana_kategori, alt_kategoriler in kategori_sozlugu.items():
            for alt_kat in alt_kategoriler:
                if re.search(r'\b' + re.escape(alt_kat) + r'\b', urun_adi_lower):
                    return ana_kategori, alt_kat.title()
        return "Diğer", "Diğer"

    def _clean_aciklama_data(self, df_aciklama: pd.DataFrame) -> pd.DataFrame:
        """Sadece açıklamalı iade verisindeki özel kayma sorununu çözer."""
        df_aciklama.columns = df_aciklama.columns.str.strip()
        df_aciklama = df_aciklama.replace(to_replace=[r'_x000D_', r'\r', r'\n'], value=' ', regex=True)
        df_aciklama['Sipariş No'] = df_aciklama['Sipariş No'].astype(str).str.strip()
        
        istenmeyen_degerler = ['nan', '0', '', 'None']
        df_temiz = df_aciklama[~df_aciklama['Sipariş No'].isin(istenmeyen_degerler)].copy()
        
        # Kategorileri çıkar (Sadece temizlendikten sonra)
        df_temiz['Ana Kategori'], df_temiz['Alt Kategori'] = zip(*df_temiz['Ürün Adı'].apply(self._kategorileri_cikar))
        return df_temiz

    def load_and_merge(self) -> pd.DataFrame:
        logging.info("Ham veriler yükleniyor...")
        
        # Dosyaları Oku
        df_teslim = pd.read_excel(self.raw_path / "dogo_teslim_edilenler.xlsx")
        df_iptal = pd.read_excel(self.raw_path / "dogo_iptal.xlsx")
        df_iade = pd.read_excel(self.raw_path / "dogo_iade.xlsx")
        df_iade_aciklama = pd.read_excel(self.raw_path / "dogo_iade_aciklamali.xlsx")

        # 1. Açıklamalı Veriyi Kendi İçinde Temizle ve Kategorileri Çıkar
        df_aciklama_temiz = self._clean_aciklama_data(df_iade_aciklama)

        # 2. Ana Tabloyu Oluştur
        df_teslim['Süreç'] = 'Teslim'
        df_iptal['Süreç'] = 'İptal'
        df_iade['Süreç'] = 'İade'
        df_base = pd.concat([df_teslim, df_iptal, df_iade], ignore_index=True)
        df_base['Sipariş No'] = df_base['Sipariş No'].astype(str).str.strip() # Eşleşme için tipi sabitle

        # 3. Birleştirme (Merge) İşlemi
        logging.info("Veriler Sipariş No üzerinden birleştiriliyor...")
        # Süreç ve Talep Tarihi diğer dosyalarda da olabileceği için sadece benzersiz ve gerekli olanları alıyoruz
        sutunlar_to_merge = ['Sipariş No', 'İade Nedeni', 'Ana Kategori', 'Alt Kategori']
        
        # Eğer İade Açıklaması sütununun tam adı Excel'de farklıysa (Örn: 'İade Açıklama'), yukarıdaki listeyi ona göre güncellemelisin.
        df_merged = pd.merge(df_base, df_aciklama_temiz[sutunlar_to_merge], 
                             on='Sipariş No', 
                             how='left')
        
        # 4. Son Temizlik ve Özellik Çıkarımı
        df_merged = self.cleaner.process(df_merged)
        df_merged = self.feature_engineer.process(df_merged)
        
        return df_merged

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        output_file = self.processed_path / filename
        # decimal=',' parametresi sayıları Excel'in TR dil ayarına uygun kaydeder
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';', decimal=',')
        logging.info(f"Veri başarıyla kaydedildi: {output_file}")

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    pipeline = DataPipeline(raw_dir, processed_dir)
    merged_data = pipeline.load_and_merge()
    pipeline.save_processed_data(merged_data, "01_final_merged_data.csv")