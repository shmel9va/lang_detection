"""
Финальная оценка модели на test.csv.

Запускать ТОЛЬКО один раз, когда разработка полностью завершена.
test.csv не должен использоваться ни в каких промежуточных шагах.

Threshold читается из output/threshold_results.txt (последняя строка).
Переопределить можно переменной окружения THRESHOLD=0.6.

Запуск:
    docker-compose run --rm evaluator
"""

import os
import sys

import pandas as pd
from sklearn.metrics import classification_report

from scripts.detection.detector import LanguageDetector
from scripts.utils.label_mapping import merge_label

MODEL_PATH      = "output/lang_detection_model.bin"
CLASSIFIERS_DIR = "output/sensitive_classifiers"
TEST_PATH       = "output/test.csv"
THRESHOLD_FILE  = "output/threshold_results.txt"
OUT_PATH        = "output/final_evaluation.txt"


def _read_threshold() -> float:
    env = os.environ.get("THRESHOLD")
    if env:
        return float(env)
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith("Оптимальный threshold:"):
                    return float(line.split(":")[1].strip())
    print("ПРЕДУПРЕЖДЕНИЕ: threshold не найден, используется 0.5")
    return 0.5


def main() -> None:
    for path in [MODEL_PATH, TEST_PATH]:
        if not os.path.exists(path):
            print(f"ОШИБКА: файл не найден: {path}")
            sys.exit(1)

    threshold = _read_threshold()
    print("=" * 70)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST")
    print(f"  Модель:    {MODEL_PATH}")
    print(f"  Threshold: {threshold}")
    print(f"  Test:      {TEST_PATH}")
    print("=" * 70)

    detector = LanguageDetector(
        fasttext_model_path=MODEL_PATH,
        sensitive_classifiers_dir=CLASSIFIERS_DIR,
        threshold=threshold,
    )

    df = pd.read_csv(TEST_PATH, sep=";")
    print(f"Test примеров: {len(df)}\n")

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        text = str(row["request_text"])
        lang, _ = detector.detect(text)
        y_true.append(merge_label(str(row["result"])))
        y_pred.append(lang)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print(report)

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("ФИНАЛЬНАЯ ОЦЕНКА\n")
        f.write(f"Модель: {MODEL_PATH}  |  Threshold: {threshold}\n")
        f.write(f"Test:   {TEST_PATH}   ({len(df)} примеров)\n\n")
        f.write(report)

    print(f"Результаты сохранены: {OUT_PATH}")


if __name__ == "__main__":
    main()
