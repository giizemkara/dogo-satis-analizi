from .prepare_data import build_processed_files
from .return_nlp import export_nlp_summary
from .train import main as train_models


def main():
    print("1/2 - Ham veriler hazırlanıyor...")
    result = build_processed_files()
    print(f"Hazırlanan final veri boyutu: {result.shape}")
    print()

    print("2/3 - İade açıklamaları NLP ile sınıflandırılıyor...")
    export_nlp_summary()
    print()

    print("3/3 - Modeller eğitiliyor...")
    train_models()
    print()
    print("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
