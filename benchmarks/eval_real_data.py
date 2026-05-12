"""
Оценка моделей на реальных данных (real_data.csv).

Сравнивает baseline, combined (fastText+binary) и full pipeline (+ DistilBERT ONNX).
real_data.csv содержит колонки task_language и rider_ml_message.

Запуск:
    docker-compose run --rm eval_real
    python benchmarks/eval_real_data.py
"""

import csv
import os
import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils.label_mapping import merge_label

VALID_LABELS = {
    "hy", "uz", "en", "ur", "he", "sr", "ne", "ar", "am", "az",
    "ka", "ro", "ru", "uk", "fr", "es", "tr", "hi", "kk", "fa", "pt", "other",
}

ALL_SENSITIVE_PAIRS = [
    ("hy", "az"), ("ur", "hi"), ("ru", "uk"), ("he", "ar"), ("ar", "fa"),
]


def _read_real_data(path="real_data.csv"):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            if "request_text" in row and "result" in row:
                txt = (row.get("request_text") or "").strip()
                lbl = (row.get("result") or "").strip()
            elif "rider_ml_message" in row and "task_language" in row:
                txt = (row.get("rider_ml_message") or "").strip().strip('"')
                lbl = (row.get("task_language") or "").strip().strip('"')
            else:
                vals = list(row.values())
                if len(vals) < 2:
                    continue
                lbl = vals[0].strip().strip('"')
                txt = vals[1].strip().strip('"')
            lbl = lbl.strip('"')
            if lbl and txt:
                rows.append((lbl, txt))
    return rows


def _evaluate_baseline(model_path, data):
    import fasttext
    from scripts.data_processing.preprocess_text import preprocess_text

    model = fasttext.load_model(model_path)
    y_true, y_pred, times = [], [], []
    for true_lbl, text in data:
        processed = preprocess_text(text).replace("\n", " ").replace("\r", " ").strip()
        if not processed:
            continue
        t0 = time.time()
        labels, _ = model.predict(processed, k=1)
        times.append(time.time() - t0)
        pred = merge_label(labels[0].replace("__label__", ""))
        y_true.append(true_lbl)
        y_pred.append(pred)
    return y_true, y_pred, np.array(times) * 1000


def _evaluate_combined(model_path, classifiers_dir, threshold, data, onnx_path=None):
    from scripts.detection.detector import LanguageDetector

    onnx_config = None
    if onnx_path:
        onnx_config = onnx_path.replace(".onnx", "/label_config.json")
        if not os.path.exists(onnx_config):
            onnx_config = os.path.join(os.path.dirname(onnx_path), "distilbert_lang_detection", "label_config.json")

    detector = LanguageDetector(
        fasttext_model_path=model_path,
        sensitive_classifiers_dir=classifiers_dir,
        onnx_model_path=onnx_path,
        onnx_config_path=onnx_config,
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
    data_path = sys.argv[1] if len(sys.argv) > 1 else "real_data.csv"
    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        sys.exit(1)

    raw = _read_real_data(data_path)
    data = [(lbl, txt) for lbl, txt in raw if lbl in VALID_LABELS]
    skipped = len(raw) - len(data)
    print(f"Загружено: {len(raw)}, валидных: {len(data)}, пропущено: {skipped}\n")

    lbl_counts = Counter(lbl for lbl, _ in data)
    print("Распределение языков:")
    for lbl, cnt in lbl_counts.most_common():
        print(f"  {lbl:<8} {cnt:>5}  ({cnt/len(data)*100:.1f}%)")

    has_baseline = os.path.exists("output/baseline_model.bin")
    has_combined = os.path.exists("output/lang_detection_model.bin")
    has_onnx = os.path.exists("output/distilbert_lang_detection.onnx")

    results = {}
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ── Baseline ─────────────────────────────────────────────────────────
    if has_baseline:
        out(f"\n{'=' * 70}")
        out("БАЗОВАЯ МОДЕЛЬ (fastText only, 21 класс)")
        out("=" * 70)
        y_true, y_pred, times = _evaluate_baseline("output/baseline_model.bin", data)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out(f"  Accuracy:  {acc:.4f}")
        out(f"  Macro F1:  {f1:.4f}")
        out(f"  Ошибок:    {sum(1 for t, p in zip(y_true, y_pred) if t != p)}")
        out(f"  Скорость:  {times.mean():.3f} мс (среднее), {np.percentile(times, 95):.1f} мс (P95)")
        results["baseline"] = (y_true, y_pred, times)
    else:
        out("\nБазовая модель не найдена. Пропуск.")

    # ── Combined (fastText + 9 binary classifiers) ───────────────────────
    if has_combined:
        out(f"\n{'=' * 70}")
        out("КОМБИНИРОВАННАЯ МОДЕЛЬ (fastText + 5 бинарных, без Transformer)")
        out("=" * 70)
        y_true, y_pred, times = _evaluate_combined(
            "output/lang_detection_model.bin",
            "output/sensitive_classifiers",
            0.0,
            data,
            onnx_path=None,
        )
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out(f"  Accuracy:  {acc:.4f}")
        out(f"  Macro F1:  {f1:.4f}")
        out(f"  Ошибок:    {sum(1 for t, p in zip(y_true, y_pred) if t != p)}")
        out(f"  Скорость:  {times.mean():.3f} мс (среднее), {np.percentile(times, 95):.1f} мс (P95)")
        results["combined"] = (y_true, y_pred, times)
    else:
        out("\nКомбинированная модель не найдена. Пропуск.")

    # ── Full pipeline (+ DistilBERT ONNX) ────────────────────────────────
    if has_combined and has_onnx:
        out(f"\n{'=' * 70}")
        out("ПОЛНЫЙ ПАЙПЛАЙН (fastText + DistilBERT ONNX + 5 бинарных)")
        out("=" * 70)
        y_true, y_pred, times = _evaluate_combined(
            "output/lang_detection_model.bin",
            "output/sensitive_classifiers",
            0.0,
            data,
            onnx_path="output/distilbert_lang_detection.onnx",
        )
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out(f"  Accuracy:  {acc:.4f}")
        out(f"  Macro F1:  {f1:.4f}")
        out(f"  Ошибок:    {sum(1 for t, p in zip(y_true, y_pred) if t != p)}")
        out(f"  Скорость:  {times.mean():.3f} мс (среднее), {np.percentile(times, 95):.1f} мс (P95)")
        results["full"] = (y_true, y_pred, times)
    elif has_combined:
        out("\nONNX модель не найдена — полный пайплайн пропущен.")
        out("  Запустите: docker-compose run --rm trainer_distilbert && docker-compose run --rm export_onnx")

    # ── Сравнение ────────────────────────────────────────────────────────
    out(f"\n{'=' * 70}")
    out("СРАВНЕНИЕ МОДЕЛЕЙ")
    out("=" * 70)

    MIN_SAMPLES = 10
    meaningful_labels = sorted(l for l, c in lbl_counts.items() if c >= MIN_SAMPLES)
    tiny_labels = sorted(l for l, c in lbl_counts.items() if c < MIN_SAMPLES)
    if tiny_labels:
        out(f"\n  Классы с <{MIN_SAMPLES} семплами (исключены из Macro F1*): {', '.join(f'{l}({lbl_counts[l]})' for l in tiny_labels)}")

    if len(results) >= 2:
        names = list(results.keys())
        out(f"\n{'Метрика':<30} " + " ".join(f"{n:>14}" for n in names))
        out("-" * (30 + 16 * len(results)))

        for metric_name, metric_fn in [
            ("Accuracy", lambda yt, yp: accuracy_score(yt, yp)),
            ("Macro F1 (all)", lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
            (f"Macro F1* (N≥{MIN_SAMPLES})", lambda yt, yp: f1_score(yt, yp, labels=meaningful_labels, average="macro", zero_division=0)),
        ]:
            vals = {n: metric_fn(*r[:2]) for n, r in results.items()}
            out(f"{metric_name:<30} " + " ".join(f"{vals[n]:>14.4f}" for n in names))

        errs = {n: sum(1 for t, p in zip(r[0], r[1]) if t != p) for n, r in results.items()}
        out(f"{'Всего ошибок':<30} " + " ".join(f"{errs[n]:>14}" for n in names))

        speeds = {n: r[2].mean() for n, r in results.items()}
        out(f"{'Скорость (мс, среднее)':<30} " + " ".join(f"{speeds[n]:>14.3f}" for n in names))

        p95 = {n: np.percentile(r[2], 95) for n, r in results.items()}
        out(f"{'Скорость P95 (мс)':<30} " + " ".join(f"{p95[n]:>14.1f}" for n in names))

    # ── Per-class F1 ─────────────────────────────────────────────────────
    if len(results) >= 2:
        all_labels = sorted(set().union(*(set(r[0]) | set(r[1]) for r in results.values())))

        f1_per_model = {}
        for name, (yt, yp, _) in results.items():
            _, _, f1_cls, _ = precision_recall_fscore_support(
                yt, yp, labels=all_labels, average=None, zero_division=0,
            )
            f1_per_model[name] = f1_cls

        out(f"\n{'=' * 70}")
        out("ПОКЛАССОВЫЙ F1")
        out("=" * 70)
        header = f"{'Класс':<8} {'N':>6}"
        for name in results:
            header += f" {name + ' F1':>14}"
        out(header)
        out("-" * len(header))
        for i, lbl in enumerate(all_labels):
            n = lbl_counts.get(lbl, 0)
            row = f"{lbl:<8} {n:>6}"
            for name in results:
                row += f" {f1_per_model[name][i]:>14.4f}"
            out(row)

    # ── Sensitive pairs ──────────────────────────────────────────────────
    out(f"\n{'=' * 70}")
    out("ЧУВСТВИТЕЛЬНЫЕ ПАРЫ (5 пар)")
    out("=" * 70)

    pair_errors_per_model = {}
    for name, (yt, yp, _) in results.items():
        pair_errors_per_model[name] = _pair_errors(yt, yp, ALL_SENSITIVE_PAIRS)

    if results:
        header = f"{'Пара':<10} {'N':>5}"
        for name in results:
            header += f" {name + ' ош':>12} {name + ' %':>8}"
        out(f"\n{header}")
        out("-" * len(header))

        for pair in ALL_SENSITIVE_PAIRS:
            first_model = list(results.keys())[0]
            total = pair_errors_per_model[first_model][pair]["total"]
            if total == 0:
                continue
            row = f"{pair[0]}-{pair[1]:<5} {total:>5}"
            for name in results:
                pe = pair_errors_per_model[name][pair]
                row += f" {pe['errors']:>12} {pe['error_rate']:>7.1%}"
            out(row)

        out(f"\nПерекрёстные ошибки:")
        for pair in ALL_SENSITIVE_PAIRS:
            first_model = list(results.keys())[0]
            total = pair_errors_per_model[first_model][pair]["total"]
            if total == 0:
                continue
            out(f"  {pair[0]}-{pair[1]}:")
            for name in results:
                pe = pair_errors_per_model[name][pair]
                key1 = f"{pair[0]}->{pair[1]}"
                key2 = f"{pair[1]}->{pair[0]}"
                out(f"    {name:>12}: {key1}={pe[key1]}, {key2}={pe[key2]}")

    # ── Error examples ──────────────────────────────────────────────────
    if "full" in results:
        y_true, y_pred, _ = results["full"]
        model_label = "ПОЛНЫЙ ПАЙПЛАЙН"
    elif "combined" in results:
        y_true, y_pred, _ = results["combined"]
        model_label = "КОМБИНИРОВАННАЯ МОДЕЛЬ"
    elif "baseline" in results:
        y_true, y_pred, _ = results["baseline"]
        model_label = "БАЗОВАЯ МОДЕЛЬ"
    else:
        y_true = y_pred = None
        model_label = None

    if y_true is not None:
        out(f"\n{'=' * 70}")
        out(f"ПРИМЕРЫ ОШИБОК: {model_label} (первые 30)")
        out("=" * 70)
        shown = 0
        for i, (true_lbl, text) in enumerate(data):
            if y_pred[i] != true_lbl:
                out(f"\n  [{true_lbl}] -> [{y_pred[i]}]")
                out(f"  {text[:200]}")
                shown += 1
                if shown >= 30:
                    break

    # ── Save report ─────────────────────────────────────────────────────
    report_path = "output/real_data_evaluation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ОЦЕНКА НА РЕАЛЬНЫХ ДАННЫХ (real_data.csv)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Всего строк: {len(raw)}, валидных: {len(data)}, пропущено: {skipped}\n\n")

        for name, (yt, yp, _) in results.items():
            f.write(f"\n{name.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(yt, yp, digits=4, zero_division=0))
            f.write("\n")

        f.write("\nСРАВНЕНИЕ\n")
        f.write("-" * 40 + "\n")
        f.write("\n".join(lines))

    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    main()
