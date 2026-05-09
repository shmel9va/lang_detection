"""
Сравнение базовой и комбинированной моделей на test.csv.

Метрики:
  - Overall accuracy, macro F1
  - Per-class F1 comparison
  - Ошибки по чувствительным парам (для всех трёх моделей)
  - Скорость предсказания

Запуск:
    docker-compose run --rm comparer_solutions
"""

import os
import sys
import time

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from scripts.data_processing.preprocess_text import preprocess_text
from scripts.detection.detector import LanguageDetector
from scripts.utils.label_mapping import merge_label

SENSITIVE_PAIRS = [
    ("hy", "az"),
    ("ur", "hi"),
    ("ru", "uk"),
    ("he", "ar"),
    ("ar", "fa"),
]

BASELINE_MODEL = "output/baseline_model.bin"
COMBINED_MODEL = "output/lang_detection_model.bin"
CLASSIFIERS_DIR = "output/sensitive_classifiers"
THRESHOLD_FILE = "output/threshold_results.txt"
TEST_PATH = "output/test.csv"
REPORT_PATH = "output/solutions_comparison.txt"


def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, on_bad_lines="skip")
        except (UnicodeDecodeError, Exception):
            continue
    raise RuntimeError(f"Cannot read {path}")


def _read_threshold():
    env = os.environ.get("THRESHOLD")
    if env:
        return float(env)
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith("Оптимальный threshold:"):
                    return float(line.split(":")[1].strip())
    return 0.0


def _evaluate_baseline(model_path, df):
    model = fasttext.load_model(model_path)
    y_true, y_pred, times = [], [], []
    for _, row in df.iterrows():
        text = preprocess_text(str(row["request_text"]))
        text = text.replace("\n", " ").replace("\r", " ").strip()
        if not text:
            continue
        true = merge_label(str(row["result"]).strip())
        t0 = time.time()
        labels, _ = model.predict(text, k=1)
        times.append(time.time() - t0)
        pred = merge_label(labels[0].replace("__label__", ""))
        y_true.append(true)
        y_pred.append(pred)
    return y_true, y_pred, np.array(times) * 1000


def _evaluate_combined(model_path, classifiers_dir, threshold, df):
    detector = LanguageDetector(
        fasttext_model_path=model_path,
        sensitive_classifiers_dir=classifiers_dir,
        threshold=threshold,
        router_verbose=False,
    )
    y_true, y_pred, times = [], [], []
    for _, row in df.iterrows():
        text = str(row["request_text"])
        true = merge_label(str(row["result"]).strip())
        t0 = time.time()
        lang, _ = detector.detect(text)
        times.append(time.time() - t0)
        y_true.append(true)
        y_pred.append(lang)
    return y_true, y_pred, np.array(times) * 1000


def _pair_errors(y_true, y_pred, pairs):
    results = {}
    for lang1, lang2 in pairs:
        indices = [i for i, t in enumerate(y_true) if t in (lang1, lang2)]
        total = len(indices)
        errors = sum(1 for i in indices if y_pred[i] != y_true[i])
        cross_12 = sum(1 for i in indices if y_true[i] == lang1 and y_pred[i] == lang2)
        cross_21 = sum(1 for i in indices if y_true[i] == lang2 and y_pred[i] == lang1)
        results[(lang1, lang2)] = {
            "total": total,
            "errors": errors,
            f"{lang1}->{lang2}": cross_12,
            f"{lang2}->{lang1}": cross_21,
            "error_rate": errors / total if total else 0,
        }
    return results


def _per_class_f1(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    labels = sorted(set(y_true) | set(y_pred))
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {lbl: score for lbl, score in zip(labels, f1)}


def main():
    if not os.path.exists(TEST_PATH):
        print(f"test.csv не найден: {TEST_PATH}")
        sys.exit(1)

    df = _read_csv(TEST_PATH)
    print(f"Test: {len(df)} примеров\n")

    has_baseline = os.path.exists(BASELINE_MODEL)
    has_combined = os.path.exists(COMBINED_MODEL)

    b_true = b_pred = b_times = None
    c_true = c_pred = c_times = None
    c0_true = c0_pred = c0_times = None

    if has_baseline:
        print("=" * 70)
        print("БАЗОВАЯ МОДЕЛЬ (fastText, 21 класс)")
        print("=" * 70)
        b_true, b_pred, b_times = _evaluate_baseline(BASELINE_MODEL, df)
        print(f"  Accuracy:    {accuracy_score(b_true, b_pred):.4f}")
        print(f"  Macro F1:    {f1_score(b_true, b_pred, average='macro', zero_division=0):.4f}")
        print(f"  Скорость:    {b_times.mean():.3f} мс (среднее)")
        print(f"  P99:         {np.percentile(b_times, 99):.3f} мс")
    else:
        print("Базовая модель не найдена.")
        print("Запустите: docker-compose run --rm baseline_trainer\n")

    if has_combined:
        threshold = _read_threshold()
        print(f"\n{'=' * 70}")
        print(f"КОМБИНИРОВАННАЯ МОДЕЛЬ (threshold={threshold})")
        print("=" * 70)
        c_true, c_pred, c_times = _evaluate_combined(
            COMBINED_MODEL, CLASSIFIERS_DIR, threshold, df
        )
        print(f"  Accuracy:    {accuracy_score(c_true, c_pred):.4f}")
        print(f"  Macro F1:    {f1_score(c_true, c_pred, average='macro', zero_division=0):.4f}")
        print(f"  Скорость:    {c_times.mean():.3f} мс (среднее)")
        print(f"  P99:         {np.percentile(c_times, 99):.3f} мс")

        print(f"\n{'=' * 70}")
        print(f"КОМБИНИРОВАННАЯ МОДЕЛЬ (threshold=0, без отсечки)")
        print("=" * 70)
        c0_true, c0_pred, c0_times = _evaluate_combined(
            COMBINED_MODEL, CLASSIFIERS_DIR, 0.0, df
        )
        print(f"  Accuracy:    {accuracy_score(c0_true, c0_pred):.4f}")
        print(f"  Macro F1:    {f1_score(c0_true, c0_pred, average='macro', zero_division=0):.4f}")
        print(f"  Скорость:    {c0_times.mean():.3f} мс (среднее)")
        print(f"  P99:         {np.percentile(c0_times, 99):.3f} мс")
    else:
        print("Комбинированная модель не найдена.")
        print("Запустите полный pipeline обучения.\n")

    if not has_baseline and not has_combined:
        sys.exit(1)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'=' * 70}")
    out("СРАВНЕНИЕ")
    out("=" * 70)

    if has_baseline and has_combined:
        out(f"\n{'Метрика':<30} {'Базовая':>12} {'Комбинир.':>12} {'Комб.(t=0)':>12} {'Δ баз→t0':>10}")
        out("-" * 80)

        b_acc = accuracy_score(b_true, b_pred)
        c_acc = accuracy_score(c_true, c_pred)
        c0_acc = accuracy_score(c0_true, c0_pred)
        out(f"{'Accuracy':<30} {b_acc:>12.4f} {c_acc:>12.4f} {c0_acc:>12.4f} {c0_acc - b_acc:>+10.4f}")

        b_f1 = f1_score(b_true, b_pred, average="macro", zero_division=0)
        c_f1 = f1_score(c_true, c_pred, average="macro", zero_division=0)
        c0_f1 = f1_score(c0_true, c0_pred, average="macro", zero_division=0)
        out(f"{'Macro F1':<30} {b_f1:>12.4f} {c_f1:>12.4f} {c0_f1:>12.4f} {c0_f1 - b_f1:>+10.4f}")

        out(f"{'Скорость среднее (мс)':<30} {b_times.mean():>12.3f} {c_times.mean():>12.3f} {c0_times.mean():>12.3f}")
        out(f"{'Скорость P99 (мс)':<30} {np.percentile(b_times, 99):>12.3f} {np.percentile(c_times, 99):>12.3f} {np.percentile(c0_times, 99):>12.3f}")

        b_total_errors = sum(1 for t, p in zip(b_true, b_pred) if t != p)
        c0_total_errors = sum(1 for t, p in zip(c0_true, c0_pred) if t != p)
        out(f"{'Всего ошибок':<30} {b_total_errors:>12} {c0_total_errors:>24} {c0_total_errors - b_total_errors:>+10}")

    # ── Per-class F1 ──────────────────────────────────────────────────
    if has_baseline and has_combined:
        b_f1_cls = _per_class_f1(b_true, b_pred)
        c0_f1_cls = _per_class_f1(c0_true, c0_pred)
        all_labels = sorted(set(b_f1_cls) | set(c0_f1_cls))

        out(f"\n{'=' * 70}")
        out("ПОКЛАССОВЫЙ MACRO F1 (Baseline vs Combined t=0)")
        out("=" * 70)
        out(f"\n{'Класс':<10} {'Baseline':>12} {'Comb.(t=0)':>12} {'Δ':>10}")
        out("-" * 50)
        for lbl in all_labels:
            bf = b_f1_cls.get(lbl, 0.0)
            cf = c0_f1_cls.get(lbl, 0.0)
            delta = cf - bf
            marker = " *" if abs(delta) >= 0.005 else ""
            out(f"{lbl:<10} {bf:>12.4f} {cf:>12.4f} {delta:>+10.4f}{marker}")

    # ── Sensitive pairs ───────────────────────────────────────────────
    out(f"\n{'=' * 70}")
    out("ЧУВСТВИТЕЛЬНЫЕ ПАРЫ")
    out("=" * 70)

    b_pairs = c0_pairs = None
    if has_baseline:
        b_pairs = _pair_errors(b_true, b_pred, SENSITIVE_PAIRS)
    if has_combined:
        c0_pairs = _pair_errors(c0_true, c0_pred, SENSITIVE_PAIRS)

    if has_baseline and has_combined:
        out(f"\n{'Пара':<10} {'Баз.ош/вс':>12} {'Баз.%':>8} {'Комб.ош/вс':>12} {'Комб.%':>8} {'Δ ош.':>8}")
        out("-" * 65)
        for pair in SENSITIVE_PAIRS:
            bp = b_pairs[pair]
            cp = c0_pairs[pair]
            delta_err = cp["errors"] - bp["errors"]
            out(
                f"{pair[0]}-{pair[1]:<5}"
                f" {bp['errors']:>5}/{bp['total']:<5}"
                f" {bp['error_rate']:>7.2%}"
                f" {cp['errors']:>5}/{cp['total']:<5}"
                f" {cp['error_rate']:>7.2%}"
                f" {delta_err:>+8}"
            )
    elif has_baseline:
        out(f"\n{'Пара':<10} {'Ошибок':>8} {'Всего':>8} {'%':>8}")
        out("-" * 40)
        for pair in SENSITIVE_PAIRS:
            bp = b_pairs[pair]
            out(f"{pair[0]}-{pair[1]:<5} {bp['errors']:>8} {bp['total']:>8} {bp['error_rate']:>7.2%}")
    elif has_combined:
        out(f"\n{'Пара':<10} {'Ошибок':>8} {'Всего':>8} {'%':>8}")
        out("-" * 40)
        for pair in SENSITIVE_PAIRS:
            cp = c0_pairs[pair]
            out(f"{pair[0]}-{pair[1]:<5} {cp['errors']:>8} {cp['total']:>8} {cp['error_rate']:>7.2%}")

    if has_baseline and has_combined:
        out(f"\nДетализация перекрёстных ошибок (Comb.t=0):")
        for pair in SENSITIVE_PAIRS:
            bp = b_pairs[pair]
            cp = c0_pairs[pair]
            key1 = f"{pair[0]}->{pair[1]}"
            key2 = f"{pair[1]}->{pair[0]}"
            out(f"  {pair[0]}-{pair[1]}:")
            out(f"    Баз.:     {key1}={bp[key1]}, {key2}={bp[key2]}")
            out(f"    Комб.t=0: {key1}={cp[key1]}, {key2}={cp[key2]}")

    out("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("СРАВНЕНИЕ БАЗОВОЙ И КОМБИНИРОВАННОЙ МОДЕЛЕЙ\n")
        f.write("=" * 70 + "\n\n")

        if has_baseline:
            f.write("БАЗОВАЯ МОДЕЛЬ\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(b_true, b_pred, digits=4, zero_division=0))
            f.write("\n")

        if has_combined:
            f.write("КОМБИНИРОВАННАЯ МОДЕЛЬ (t=0)\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(c0_true, c0_pred, digits=4, zero_division=0))
            f.write("\n")

        f.write("\nСРАВНЕНИЕ\n")
        f.write("-" * 40 + "\n")
        f.write("\n".join(lines))

    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
