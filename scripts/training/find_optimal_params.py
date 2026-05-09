"""
Grid search оптимальных гиперпараметров для fastText.
Перебирает dim x (minn, maxn) x wordNgrams, для каждого — auto epoch.
Метрика: macro F1 на val.csv.

Запуск:
    docker-compose run --rm param_searcher
"""

import os
import itertools
import fasttext
import pandas as pd
from sklearn.metrics import f1_score

from scripts.utils.label_mapping import merge_label
from scripts.data_processing.preprocess_text import preprocess_text

PARAM_GRID = {
    "dim": [100, 150],
    "minn_maxn": [(2, 5), (3, 6)],
    "wordNgrams": [1, 2],
}

FIXED_PARAMS = {
    "lr": 0.1,
    "minCount": 2,
    "loss": "softmax",
    "seed": 42,
    "verbose": 0,
}

EPOCH_STEP = 5
EPOCH_MAX = 70
EPOCH_MIN_DELTA = 0.001

TRAIN_PATH = "output/train.csv"
VAL_PATH = "output/val.csv"
RESULTS_PATH = "output/param_search_results.txt"


def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, on_bad_lines="skip")
        except (UnicodeDecodeError, Exception):
            continue
    raise RuntimeError(f"Cannot read {path}")


def _prepare_ft_file(df, text_col, label_col, output_path, merge=False):
    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = preprocess_text(str(row[text_col]))
            text = text.replace("\n", " ").replace("\r", " ").strip()
            label_raw = str(row[label_col]).strip()
            if merge:
                label = merge_label(label_raw)
            else:
                label = label_raw.replace(" ", "_")
            if label and text:
                f.write(f"__label__{label} {text}\n")
                n += 1
    return n


def _eval_macro_f1(model, val_data):
    y_true, y_pred = [], []
    for text, true_label in val_data:
        pred_raw = model.predict(text, k=1)[0][0].replace("__label__", "")
        pred = merge_label(pred_raw)
        y_true.append(true_label)
        y_pred.append(pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _find_best_epoch(train_txt, val_data, params):
    best_f1 = 0.0
    best_epoch = EPOCH_STEP
    prev_f1 = None

    for epoch in range(EPOCH_STEP, EPOCH_MAX + 1, EPOCH_STEP):
        model = fasttext.train_supervised(input=train_txt, epoch=epoch, **params)
        f1 = _eval_macro_f1(model, val_data)
        delta = (f1 - prev_f1) if prev_f1 is not None else float("inf")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch

        if prev_f1 is not None and delta < EPOCH_MIN_DELTA:
            break

        prev_f1 = f1

    return best_epoch, best_f1


def find_optimal_params(
    train_file=TRAIN_PATH,
    val_file=VAL_PATH,
    text_col="request_text",
    label_col="result",
):
    print("=" * 70)
    print("GRID SEARCH ГИПЕРПАРАМЕТРОВ FASTTEXT")
    print(f"  Метрика: macro F1")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print("=" * 70)

    print("\nЗагрузка данных...")
    train_df = _read_csv(train_file)
    val_df = _read_csv(val_file)

    train_txt = "output/param_search_train.txt"
    _prepare_ft_file(train_df, text_col, label_col, train_txt, merge=False)

    val_data = []
    for _, row in val_df.iterrows():
        text = preprocess_text(str(row[text_col]))
        text = text.replace("\n", " ").replace("\r", " ").strip()
        label = merge_label(str(row[label_col]).strip())
        if text and label:
            val_data.append((text, label))
    print(f"Train: {len(train_df)}, Val: {len(val_data)}")

    combos = list(itertools.product(
        PARAM_GRID["dim"],
        PARAM_GRID["minn_maxn"],
        PARAM_GRID["wordNgrams"],
    ))
    print(f"Комбинаций: {len(combos)}")
    print()

    all_results = []
    best_overall_f1 = 0.0
    best_overall_params = None

    header = f"{'#':>3}  {'dim':>5}  {'minn':>5}  {'maxn':>5}  {'wNgr':>5}  {'epoch':>6}  {'macro F1':>10}  {'':}"
    print(header)
    print("-" * len(header))

    for idx, (dim, (minn, maxn), word_ngrams) in enumerate(combos, 1):
        params = {
            **FIXED_PARAMS,
            "dim": dim,
            "minn": minn,
            "maxn": maxn,
            "wordNgrams": word_ngrams,
        }

        best_epoch, best_f1 = _find_best_epoch(train_txt, val_data, params)

        is_best = best_f1 > best_overall_f1
        if is_best:
            best_overall_f1 = best_f1
            best_overall_params = {
                "dim": dim, "minn": minn, "maxn": maxn,
                "wordNgrams": word_ngrams, "epoch": best_epoch,
            }

        marker = " <- BEST" if is_best else ""
        print(f"{idx:>3}  {dim:>5}  {minn:>5}  {maxn:>5}  {word_ngrams:>5}  {best_epoch:>6}  {best_f1:>10.4f}  {marker}")

        all_results.append({
            "dim": dim, "minn": minn, "maxn": maxn,
            "wordNgrams": word_ngrams, "epoch": best_epoch,
            "macro_f1": best_f1,
        })

    print()
    print("=" * 70)
    bp = best_overall_params
    print(f"ЛУЧШИЕ ПАРАМЕТРЫ:")
    print(f"  dim={bp['dim']}  minn={bp['minn']}  maxn={bp['maxn']}")
    print(f"  wordNgrams={bp['wordNgrams']}  epoch={bp['epoch']}")
    print(f"  macro F1={best_overall_f1:.4f}")
    print("=" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("GRID SEARCH ГИПЕРПАРАМЕТРОВ FASTTEXT\n")
        f.write(f"Метрика: macro F1\n")
        f.write(f"Train: {train_file}  |  Val: {val_file}\n\n")
        f.write(f"{'#':>3}  {'dim':>5}  {'minn':>5}  {'maxn':>5}  {'wNgr':>5}  {'epoch':>6}  {'macro F1':>10}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(all_results, 1):
            marker = " <- BEST" if r["macro_f1"] == best_overall_f1 else ""
            f.write(f"{i:>3}  {r['dim']:>5}  {r['minn']:>5}  {r['maxn']:>5}  {r['wordNgrams']:>5}  {r['epoch']:>6}  {r['macro_f1']:>10.4f}{marker}\n")
        f.write(f"\nЛУЧШИЕ: dim={bp['dim']} minn={bp['minn']} maxn={bp['maxn']} "
                f"wordNgrams={bp['wordNgrams']} epoch={bp['epoch']} F1={best_overall_f1:.4f}\n")

    print(f"\nРезультаты: {RESULTS_PATH}")
    return best_overall_params, all_results


if __name__ == "__main__":
    find_optimal_params()
