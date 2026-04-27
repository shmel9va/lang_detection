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
from scripts.utils.label_mapping import merge_label

MODEL_PATH  = "output/lang_detection_model.bin"
VAL_PATH    = "output/val.csv"
OUT_PATH    = "output/threshold_results.txt"
THRESHOLDS  = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Допустимое падение accuracy относительно baseline (threshold=0).
# Чем меньше tolerance, тем консервативнее порог (ближе к 0.3).
# Рекомендуемый диапазон: 0.01–0.03.
TOLERANCE   = 0.02


def _baseline_accuracy(val_data, model_path):
    """Accuracy без threshold (эквивалентно threshold=0)."""
    import fasttext
    from scripts.utils.label_mapping import merge_label as ml
    model = fasttext.load_model(model_path)
    correct = 0
    for text, true in val_data:
        text = text.replace("\n", " ").replace("\r", " ").strip()
        pred = ml(model.predict(text, k=1)[0][0].replace("__label__", ""))
        if pred == true:
            correct += 1
    return correct / len(val_data)


def _pick_threshold(results, baseline, tolerance):
    """
    Стратегия: максимальный threshold, при котором accuracy не падает
    ниже (baseline - tolerance). Высокий threshold = больше текстов
    попадёт в 'other' → лучше защита от неизвестных языков.
    """
    min_acc = baseline - tolerance
    candidates = [t for t, r in results.items() if r["accuracy"] >= min_acc]
    return max(candidates) if candidates else min(results.keys())


def main() -> None:
    for path in [MODEL_PATH, VAL_PATH]:
        if not os.path.exists(path):
            print(f"ОШИБКА: файл не найден: {path}")
            print("Сначала запустите: splitter → preprocessor → trainer.")
            return

    df = pd.read_csv(VAL_PATH, sep=";")
    val_data = list(zip(
        df["request_text"].astype(str),
        df["result"].apply(merge_label).astype(str),
    ))
    print(f"Val примеров: {len(val_data)}")

    print("Вычисляем baseline accuracy (без threshold)...")
    baseline = _baseline_accuracy(val_data, MODEL_PATH)
    print(f"Baseline accuracy: {baseline:.2%}")
    print(f"Tolerance: {TOLERANCE:.0%}  →  минимально допустимая accuracy: {baseline - TOLERANCE:.2%}")

    results, _ = find_optimal_threshold(
        model_path=MODEL_PATH,
        test_data=val_data,
        thresholds=THRESHOLDS,
    )

    best = _pick_threshold(results, baseline, TOLERANCE)

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ ПОДБОРА THRESHOLD\n")
        f.write(f"Val: {VAL_PATH}  ({len(val_data)} примеров)\n")
        f.write(f"Baseline (no threshold): {baseline:.2%}\n")
        f.write(f"Tolerance: {TOLERANCE:.0%}  → min acceptable: {baseline - TOLERANCE:.2%}\n\n")
        f.write(f"{'Threshold':>10}  {'Accuracy':>10}  {'Other%':>8}  {'':}\n")
        f.write("-" * 45 + "\n")
        for t, r in sorted(results.items()):
            marker = " <- CHOSEN" if t == best else ""
            f.write(f"{t:>10.1f}  {r['accuracy']:>10.2%}  {r['other_ratio']:>8.2%}{marker}\n")
        f.write(f"\nОптимальный threshold: {best}\n")
        f.write(f"Accuracy:              {results[best]['accuracy']:.2%}\n")
        f.write(f"Other ratio:           {results[best]['other_ratio']:.2%}\n")

    print(f"\nОптимальный threshold: {best}")
    print(f"Accuracy при нём:      {results[best]['accuracy']:.2%}")
    print(f"Other ratio:           {results[best]['other_ratio']:.2%}")
    print(f"\nРезультаты сохранены: {OUT_PATH}")
    print(f"\nПрописать в detector.py:")
    print(f"    detector = LanguageDetector(..., threshold={best})")


if __name__ == "__main__":
    main()
