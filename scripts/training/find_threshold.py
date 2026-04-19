"""
Подбор оптимального threshold на val.csv.

Запуск:
    docker-compose run --rm threshold_finder
    # или: python -m scripts.training.find_threshold

Результат: output/threshold_results.txt
           Оптимальный threshold выводится в stdout.
"""

import os
import pandas as pd
from scripts.utils.predict_with_threshold import find_optimal_threshold

MODEL_PATH = "output/lang_detection_model.bin"
VAL_PATH   = "output/val.csv"
OUT_PATH   = "output/threshold_results.txt"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    for path in [MODEL_PATH, VAL_PATH]:
        if not os.path.exists(path):
            print(f"ОШИБКА: файл не найден: {path}")
            print("Сначала запустите: splitter → preprocessor → trainer.")
            return

    df = pd.read_csv(VAL_PATH, sep=";")
    val_data = list(zip(df["request_text"].astype(str), df["result"].astype(str)))
    print(f"Val примеров: {len(val_data)}")

    results, best = find_optimal_threshold(
        model_path=MODEL_PATH,
        test_data=val_data,
        thresholds=THRESHOLDS,
    )

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ ПОДБОРА THRESHOLD\n")
        f.write(f"Val: {VAL_PATH}  ({len(val_data)} примеров)\n\n")
        f.write(f"{'Threshold':>10}  {'Accuracy':>10}  {'Other%':>8}\n")
        f.write("-" * 35 + "\n")
        for t, r in sorted(results.items()):
            f.write(f"{t:>10.1f}  {r['accuracy']:>10.2%}  {r['other_ratio']:>8.2%}\n")
        f.write("\n")
        f.write(f"Оптимальный threshold: {best}\n")
        f.write(f"Accuracy:              {results[best]['accuracy']:.2%}\n")

    print(f"\nРезультаты сохранены: {OUT_PATH}")
    print(f"\nПрописать в detector.py:")
    print(f"    detector = LanguageDetector(..., threshold={best})")


if __name__ == "__main__":
    main()
