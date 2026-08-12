import pandas as pd
import numpy as np
from pathlib import Path
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseFeatureTransformer(ABC):
    """Özellik dönüştürücü sınıflar için şablon arayüz (Interface)."""
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class TargetTransformer(BaseFeatureTransformer):
    """İş mantığına göre Çok Sınıflı (0, 1, 2) hedef değişkenini oluşturur."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def target_belirle(row):
            surec = str(row.get('Süreç', '')).strip()
            neden = str(row.get('İade Nedeni', '')).strip()
            
            if surec == 'İptal':
                return 2
            elif surec == 'İade' or (neden != 'nan' and neden != ''):
                return 1
            else:
                return 0
                
        df['target'] = df.apply(target_belirle, axis=1)
        logging.info("Hedef değişken (target) başarıyla oluşturuldu.")
        return df

class CategoryGrouper(BaseFeatureTransformer):
    """Kardinaliteyi düşürmek için kategorik verileri (İl, Platform, Alt Kategori) akıllıca gruplar."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Platform Gruplama
        def platform_birlestir(x):
            x_lower = str(x).lower()
            if 'admin' in x_lower:
                return 'Admin'
            elif any(kelime in x_lower for kelime in ['mobil', 'app', 'ios', 'android']):
                return 'Mobil'
            else:
                return 'Masaüstü'
        
        df['Platform_Gruplu'] = df['Platform'].apply(platform_birlestir)
        
        # 2. İl Gruplama (Top 10)
        if 'İl (Teslimat)' in df.columns:
            top_10_iller = df['İl (Teslimat)'].value_counts().nlargest(10).index
            df['İl_Gruplu'] = df['İl (Teslimat)'].where(df['İl (Teslimat)'].isin(top_10_iller), 'Diğer')
            
        # 3. Alt Kategori Gruplama (Top 10)
        if 'Alt Kategori' in df.columns:
            top_10_altkat = df['Alt Kategori'].value_counts().nlargest(10).index
            df['AltKat_Gruplu'] = df['Alt Kategori'].where(df['Alt Kategori'].isin(top_10_altkat), 'Diğer')
            
        logging.info("Kategorik veriler (İl, Platform, Alt Kategori) başarıyla gruplandı.")
        return df

class FeatureEncoder(BaseFeatureTransformer):
    """Makine öğrenmesi için gerekli özellikleri seçer ve One-Hot Encoding uygular."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        secilen_ozellikler = ['Platform_Gruplu', 'Kaynak', 'Ödeme Tipi', 'Ana Kategori', 'AltKat_Gruplu', 'İl_Gruplu']
        
        # Eksik verileri doldur
        X = df[secilen_ozellikler].fillna('Bilinmiyor')
        y = df['target']
        
        # Kategorik değişkenleri sayısallaştır
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Target'ı matrise geri ekle (Train/Test Split'i Model dosyasında yapacağız)
        df_model_ready = pd.concat([X_encoded, y], axis=1)
        
        logging.info(f"One-Hot Encoding tamamlandı. Matris boyutu: {df_model_ready.shape}")
        return df_model_ready

class FeaturePipeline:
    """Tüm özellik dönüştürücüleri sırasıyla çalıştıran ana boru hattı."""
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.transformers = [
            TargetTransformer(),
            CategoryGrouper(),
            FeatureEncoder()
        ]

    def run(self):
        logging.info(f"Veri okunuyor: {self.input_path}")
        # Temizlenmiş veriyi oku (Finansal tablo için virgül kullanmıştık)
        df = pd.read_csv(self.input_path, sep=';', decimal=',')
        
        # Tüm dönüştürücüleri sırayla uygula
        for transformer in self.transformers:
            df = transformer.transform(df)
            
        # Makine öğrenmesine gidecek veriyi standart formatta (virgül ayracı ve noktalı ondalık) kaydet
        df.to_csv(self.output_path, index=False)
        logging.info(f"Makine öğrenmesine hazır veri kaydedildi: {self.output_path}")

if __name__ == "__main__":
    # Yolların tanımlanması (Proje kök dizininden çalıştırıldığını varsayıyoruz)
    input_file = Path("data/processed/01_final_merged_data.csv")
    output_file = Path("data/processed/03_model_ready.csv")
    
    # Pipeline'ı Başlat
    pipeline = FeaturePipeline(input_file, output_file)
    pipeline.run()