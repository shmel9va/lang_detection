"""
Подбор оптимального threshold на val.csv (диагностический инструмент).

Метрика: macro F1. По умолчанию threshold=0 уже оптимален (other в обучении),
этот скрипт — для проверки и анализа.

Запуск:
    docker-compose run --rm threshold_finder

Результат: output/threshold_results.txt
"""

import os
import pandas as pd
from sklearn.metrics import f1_score
from scripts.utils.predict_with_threshold import find_optimal_threshold
from scripts.utils.label_mapping import merge_label

MODEL_PATH  = "output/lang_detection_model.bin"
VAL_PATH    = "output/val.csv"
OUT_PATH    = "output/threshold_results.txt"
THRESHOLDS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOLERANCE   = 0.005


def _baseline_metrics(val_data, model_path):
    import fasttext
    from scripts.data_processing.preprocess_text import normalize_for_detection, preprocess_text
    model = fasttext.load_model(model_path)
    y_true, y_pred = [], []
    for text, true in val_data:
        normalized = normalize_for_detection(text)
        processed = preprocess_text(normalized)
        prepared = processed.replace("\n", " ").replace("\r", " ").strip()
        if not prepared:
            continue
        pred = merge_label(model.predict(prepared, k=1)[0][0].replace("__label__", ""))
        y_true.append(true)
        y_pred.append(pred)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1


def _pick_threshold(results, tolerance):
    best_f1 = max(r["macro_f1"] for r in results.values())
    candidates = [t for t, r in results.items()
                  if r["macro_f1"] >= best_f1 - tolerance]
    return min(candidates)


def main() -> None:
    for path in [MODEL_PATH, VAL_PATH]:
        if not os.path.exists(path):
            print(f"ОШИБКА: файл не найден: {path}")
            print("Сначала запустите: splitter → trainer.")
            return

    df = pd.read_csv(VAL_PATH, sep=";")
    val_data = list(zip(
        df["request_text"].astype(str),
        df["result"].apply(merge_label).astype(str),
    ))
    print(f"Val примеров: {len(val_data)}")

    print("Вычисляем baseline (threshold=0)...")
    baseline_acc, baseline_f1 = _baseline_metrics(val_data, MODEL_PATH)
    print(f"Baseline accuracy:  {baseline_acc:.2%}")
    print(f"Baseline macro F1:  {baseline_f1:.2%}")

    results, _ = find_optimal_threshold(
        model_path=MODEL_PATH,
        test_data=val_data,
        thresholds=THRESHOLDS,
    )

    best = _pick_threshold(results, TOLERANCE)

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ ПОДБОРА THRESHOLD (метрика: macro F1)\n")
        f.write(f"Val: {VAL_PATH}  ({len(val_data)} примеров)\n")
        f.write(f"Baseline (t=0) accuracy: {baseline_acc:.2%}  macro F1: {baseline_f1:.2%}\n\n")
        f.write(f"{'Threshold':>10}  {'Accuracy':>10}  {'Macro F1':>10}  {'Other%':>8}  {'':}\n")
        f.write("-" * 55 + "\n")
        for t, r in sorted(results.items()):
            marker = " <- CHOSEN" if t == best else ""
            f.write(f"{t:>10.1f}  {r['accuracy']:>10.2%}  {r['macro_f1']:>10.2%}  {r['other_ratio']:>8.2%}{marker}\n")
        f.write(f"\nОптимальный threshold: {best}\n")
        f.write(f"Macro F1:              {results[best]['macro_f1']:.2%}\n")
        f.write(f"Accuracy:              {results[best]['accuracy']:.2%}\n")
        f.write(f"Other ratio:           {results[best]['other_ratio']:.2%}\n")

    print(f"\nОптимальный threshold: {best}")
    print(f"Macro F1 при нём:      {results[best]['macro_f1']:.2%}")
    print(f"Accuracy при нём:      {results[best]['accuracy']:.2%}")
    print(f"Other ratio:           {results[best]['other_ratio']:.2%}")
    print(f"\nРезультаты сохранены: {OUT_PATH}")


if __name__ == "__main__":
    main()
