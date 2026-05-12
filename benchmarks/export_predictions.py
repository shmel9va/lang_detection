"""
Выгрузка результатов предсказаний на реальных данных.
Сохраняет CSV: true_label, predicted_label, confidence, message

Запуск:
    docker-compose run --rm eval_real python benchmarks/export_predictions.py
    python benchmarks/export_predictions.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.detection.detector import LanguageDetector
from scripts.utils.label_mapping import merge_label
from benchmarks.eval_real_data import _read_real_data, VALID_LABELS


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "real_data.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output/predictions.csv"

    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        sys.exit(1)

    raw = _read_real_data(data_path)
    data = [(lbl, txt) for lbl, txt in raw if lbl in VALID_LABELS]
    print(f"Загружено: {len(data)} примеров")

    model_path = "output/lang_detection_model.bin"
    onnx_path = "output/distilbert_lang_detection.onnx"
    classifiers_dir = "output/sensitive_classifiers"

    detector = LanguageDetector(
        fasttext_model_path=model_path,
        sensitive_classifiers_dir=classifiers_dir,
        onnx_model_path=onnx_path if os.path.exists(onnx_path) else None,
        threshold=0.0,
        router_verbose=False,
    )

    rows = []
    total = len(data)
    t0_all = time.time()
    for i, (true_lbl, text) in enumerate(data):
        pred_lang, pred_conf = detector.detect(text)
        rows.append({
            "true_label": true_lbl,
            "predicted_label": pred_lang,
            "confidence": f"{pred_conf:.4f}",
            "correct": "1" if pred_lang == true_lbl else "0",
            "message": text.replace("\n", " ").replace("\r", " ")[:500],
        })
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0_all
            print(f"  {i + 1}/{total} ({elapsed:.1f}s)")

    elapsed = time.time() - t0_all
    print(f"Обработано: {total} за {elapsed:.1f}s ({total / elapsed:.0f} text/s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_label", "predicted_label", "confidence", "correct", "message"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(1 for r in rows if r["correct"] == "1")
    errors = total - correct
    print(f"Правильных: {correct}/{total} ({correct / total:.2%})")
    print(f"Ошибок: {errors}")
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
