import argparse
from pathlib import Path

import pandas as pd

from .prepare_data import build_processed_files
from .return_nlp import export_nlp_summary
from .statistical_analysis import export_statistical_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(raw_dir=None, processed_dir=None):
    print("1/3 - Ham veriler hazırlanıyor...")
    result = build_processed_files(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    print(f"Hazırlanan final veri boyutu: {result.shape}")
    print()

    print("2/3 - İade açıklamaları NLP ile sınıflandırılıyor...")
    export_nlp_summary(
        raw_path=(Path(raw_dir) / "dogo_iade_aciklamali.xlsx") if raw_dir else None,
        output_dir=processed_dir,
    )
    print()

    print("3/3 - İstatistiksel rapor oluşturuluyor...")
    processed_path = Path(processed_dir or PROJECT_ROOT / "data" / "processed")
    orders = pd.read_excel(processed_path / "merge_data.xlsx", sheet_name="orders")
    report_path = export_statistical_report(
        orders,
        output_path=processed_path / "statistical_report.xlsx",
    )
    print(f"İstatistiksel rapor: {report_path}")
    print()
    print("Pipeline tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOGO veri temizleme ve istatistik pipeline'ı")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Ham Excel dosyalarının bulunduğu klasör",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="İşlenmiş çıktıların yazılacağı klasör",
    )
    args = parser.parse_args()
    main(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
