"""
Сбор всех ошибок комбинированной модели на real_data.csv.
Выход: output/combined_errors.csv (text, true_label, pred_label)
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.detection.detector import LanguageDetector
from scripts.utils.label_mapping import merge_label


def main():
    rows = []
    with open("real_data.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            lbl = row[0].strip().strip('"')
            txt = row[1].strip().strip('"')
            if lbl and txt:
                rows.append((lbl, txt))

    valid_labels = {"hy", "uz", "en", "ur", "he", "sr", "ne", "ar", "am", "az",
                    "ka", "ro", "ru", "uk", "fr", "es", "tr", "hi", "kk", "fa", "pt", "other"}
    data = [(l, t) for l, t in rows if l in valid_labels]

    detector = LanguageDetector(
        fasttext_model_path="output/lang_detection_model.bin",
        sensitive_classifiers_dir="output/sensitive_classifiers",
        threshold=0.0,
        router_verbose=False,
    )

    errors = []
    for true_lbl, text in data:
        lang, conf = detector.detect(text)
        if lang != true_lbl:
            errors.append((true_lbl, lang, conf, text))

    out_path = "output/combined_errors.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["true_label", "pred_label", "confidence", "text"])
        for true_lbl, pred_lbl, conf, text in errors:
            writer.writerow([true_lbl, pred_lbl, f"{conf:.4f}", text])

    print(f"Ошибок: {len(errors)} / {len(data)}")
    print(f"Сохранено: {out_path}")

    from collections import Counter
    pairs = Counter(f"{t}->{p}" for t, p, _, _ in errors)
    print(f"\nТоп-20 направлений ошибок:")
    for pair, cnt in pairs.most_common(20):
        print(f"  {pair:<20} {cnt:>5}")


if __name__ == "__main__":
    main()
