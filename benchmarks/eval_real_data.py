"""
Оценка моделей на реальных данных (real_data.csv).

Сравнивает baseline и combined модели на данных из продакшена.
real_data.csv содержит колонки task_language и rider_ml_message.

Запуск:
    docker-compose run --rm comparer_solutions python benchmarks/eval_real_data.py
    # или локально если модели собраны:
    python benchmarks/eval_real_data.py
"""

import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils.label_mapping import merge_label


def _read_real_data(path="real_data.csv"):
    import csv
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            lbl = row[0].strip().strip('"')
            txt = row[1].strip().strip('"')
            if not lbl or not txt:
                continue
            rows.append((lbl, txt))
    return rows


def _evaluate_baseline(model_path, data):
    import fasttext
    from scripts.data_processing.preprocess_text import preprocess_text

    model = fasttext.load_model(model_path)
    y_true, y_pred, times = [], [], []
    for true_lbl, text in data:
        processed = preprocess_text(text)
        processed = processed.replace("\n", " ").replace("\r", " ").strip()
        if not processed:
            continue
        t0 = time.time()
        labels, _ = model.predict(processed, k=1)
        times.append(time.time() - t0)
        pred = merge_label(labels[0].replace("__label__", ""))
        y_true.append(true_lbl)
        y_pred.append(pred)
    return y_true, y_pred, np.array(times) * 1000


def _evaluate_combined(model_path, classifiers_dir, threshold, data):
    from scripts.detection.detector import LanguageDetector

    detector = LanguageDetector(
        fasttext_model_path=model_path,
        sensitive_classifiers_dir=classifiers_dir,
        threshold=threshold,
        router_verbose=False,
    )
    y_true, y_pred, times = [], [], []
    for true_lbl, text in data:
        t0 = time.time()
        lang, _ = detector.detect(text)
        times.append(time.time() - t0)
        y_true.append(true_lbl)
        y_pred.append(lang)
    return y_true, y_pred, np.array(times) * 1000


SENSITIVE_PAIRS = [
    ("hy", "az"),
    ("ur", "hi"),
    ("ru", "uk"),
    ("he", "ar"),
    ("ar", "fa"),
]


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


def main():
    data_path = "real_data.csv"
    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        sys.exit(1)

    raw = _read_real_data(data_path)
    print(f"Загружено: {len(raw)} строк\n")

    valid_labels = {"hy", "uz", "en", "ur", "he", "sr", "ne", "ar", "am", "az",
                    "ka", "ro", "ru", "uk", "fr", "es", "tr", "hi", "kk", "fa", "pt", "other"}

    data = [(lbl, txt) for lbl, txt in raw if lbl in valid_labels]
    skipped = len(raw) - len(data)
    print(f"Валидных (известный язык): {len(data)}")
    if skipped:
        print(f"Пропущено (неизвестный язык/мусор): {skipped}")

    from collections import Counter
    lbl_counts = Counter(lbl for lbl, _ in data)
    print(f"\nРаспределение языков:")
    for lbl, cnt in lbl_counts.most_common():
        print(f"  {lbl:<8} {cnt:>5}  ({cnt/len(data)*100:.1f}%)")

    has_baseline = os.path.exists("output/baseline_model.bin")
    has_combined = os.path.exists("output/lang_detection_model.bin")

    b_true = b_pred = b_times = None
    c0_true = c0_pred = c0_times = None

    if has_baseline:
        print(f"\n{'=' * 70}")
        print("БАЗОВАЯ МОДЕЛЬ")
        print("=" * 70)
        b_true, b_pred, b_times = _evaluate_baseline("output/baseline_model.bin", data)
        print(f"  Accuracy:    {accuracy_score(b_true, b_pred):.4f}")
        print(f"  Macro F1:    {f1_score(b_true, b_pred, average='macro', zero_division=0):.4f}")
        print(f"  Скорость:    {b_times.mean():.3f} мс (среднее)")
    else:
        print("\nБазовая модель не найдена. Запустите: docker-compose run --rm baseline_trainer")

    if has_combined:
        print(f"\n{'=' * 70}")
        print("КОМБИНИРОВАННАЯ МОДЕЛЬ (threshold=0)")
        print("=" * 70)
        c0_true, c0_pred, c0_times = _evaluate_combined(
            "output/lang_detection_model.bin",
            "output/sensitive_classifiers",
            0.0,
            data,
        )
        print(f"  Accuracy:    {accuracy_score(c0_true, c0_pred):.4f}")
        print(f"  Macro F1:    {f1_score(c0_true, c0_pred, average='macro', zero_division=0):.4f}")
        print(f"  Скорость:    {c0_times.mean():.3f} мс (среднее)")
    else:
        print("\nКомбинированная модель не найдена. Запустите: docker-compose run --rm trainer")

    if not has_baseline and not has_combined:
        sys.exit(1)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'=' * 70}")
    out("СРАВНЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    out("=" * 70)

    if has_baseline and has_combined:
        out(f"\n{'Метрика':<30} {'Базовая':>12} {'Комб.(t=0)':>12} {'Δ':>10}")
        out("-" * 70)

        b_acc = accuracy_score(b_true, b_pred)
        c0_acc = accuracy_score(c0_true, c0_pred)
        out(f"{'Accuracy':<30} {b_acc:>12.4f} {c0_acc:>12.4f} {c0_acc - b_acc:>+10.4f}")

        b_f1 = f1_score(b_true, b_pred, average="macro", zero_division=0)
        c0_f1 = f1_score(c0_true, c0_pred, average="macro", zero_division=0)
        out(f"{'Macro F1':<30} {b_f1:>12.4f} {c0_f1:>12.4f} {c0_f1 - b_f1:>+10.4f}")

        b_err = sum(1 for t, p in zip(b_true, b_pred) if t != p)
        c0_err = sum(1 for t, p in zip(c0_true, c0_pred) if t != p)
        out(f"{'Всего ошибок':<30} {b_err:>12} {c0_err:>12} {c0_err - b_err:>+10}")

        if b_times is not None and c0_times is not None:
            out(f"{'Скорость среднее (мс)':<30} {b_times.mean():>12.3f} {c0_times.mean():>12.3f}")

    # ── Per-class F1 ──────────────────────────────────────────────────
    if has_baseline and has_combined:
        from sklearn.metrics import precision_recall_fscore_support
        all_labels = sorted(set(b_true) | set(c0_true) | set(b_pred) | set(c0_pred))

        _, _, b_f1_cls, _ = precision_recall_fscore_support(b_true, b_pred, labels=all_labels, average=None, zero_division=0)
        _, _, c0_f1_cls, _ = precision_recall_fscore_support(c0_true, c0_pred, labels=all_labels, average=None, zero_division=0)

        out(f"\n{'=' * 70}")
        out("ПОКЛАССОВЫЙ F1")
        out("=" * 70)
        out(f"\n{'Класс':<8} {'N':>6} {'Base F1':>10} {'Comb F1':>10} {'Δ':>10}")
        out("-" * 50)
        for i, lbl in enumerate(all_labels):
            n = lbl_counts.get(lbl, 0)
            bf = b_f1_cls[i]
            cf = c0_f1_cls[i]
            delta = cf - bf
            marker = " *" if abs(delta) >= 0.01 else ""
            out(f"{lbl:<8} {n:>6} {bf:>10.4f} {cf:>10.4f} {delta:>+10.4f}{marker}")

    # ── Sensitive pairs ───────────────────────────────────────────────
    out(f"\n{'=' * 70}")
    out("ЧУВСТВИТЕЛЬНЫЕ ПАРЫ")
    out("=" * 70)

    if has_baseline and has_combined:
        b_pairs = _pair_errors(b_true, b_pred, SENSITIVE_PAIRS)
        c0_pairs = _pair_errors(c0_true, c0_pred, SENSITIVE_PAIRS)

        out(f"\n{'Пара':<10} {'N':>5} {'Баз.ош':>8} {'Баз.%':>8} {'Комб.ош':>8} {'Комб.%':>8} {'Δ':>8}")
        out("-" * 65)
        for pair in SENSITIVE_PAIRS:
            bp = b_pairs[pair]
            cp = c0_pairs[pair]
            if bp["total"] == 0:
                continue
            delta_err = cp["errors"] - bp["errors"]
            out(
                f"{pair[0]}-{pair[1]:<5}"
                f" {bp['total']:>5}"
                f" {bp['errors']:>8}"
                f" {bp['error_rate']:>7.2%}"
                f" {cp['errors']:>8}"
                f" {cp['error_rate']:>7.2%}"
                f" {delta_err:>+8}"
            )

        out(f"\nПерекрёстные ошибки:")
        for pair in SENSITIVE_PAIRS:
            bp = b_pairs[pair]
            cp = c0_pairs[pair]
            if bp["total"] == 0:
                continue
            key1 = f"{pair[0]}->{pair[1]}"
            key2 = f"{pair[1]}->{pair[0]}"
            out(f"  {pair[0]}-{pair[1]}:")
            out(f"    Баз.:  {key1}={bp[key1]}, {key2}={bp[key2]}")
            out(f"    Комб.: {key1}={cp[key1]}, {key2}={cp[key2]}")

    # ── Error examples ────────────────────────────────────────────────
    if has_combined:
        out(f"\n{'=' * 70}")
        out("ПРИМЕРЫ ОШИБОК КОМБИНИРОВАННОЙ МОДЕЛИ (первые 20)")
        out("=" * 70)
        errors_shown = 0
        for i, (true_lbl, text) in enumerate(data):
            if c0_pred[i] != true_lbl:
                out(f"\n  [{true_lbl}] -> предсказано [{c0_pred[i]}]")
                out(f"  Текст: {text[:150]}")
                errors_shown += 1
                if errors_shown >= 20:
                    break

    # ── Save report ───────────────────────────────────────────────────
    report_path = "output/real_data_evaluation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ОЦЕНКА НА РЕАЛЬНЫХ ДАННЫХ (real_data.csv)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Всего строк: {len(raw)}, валидных: {len(data)}\n\n")

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

    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    main()
