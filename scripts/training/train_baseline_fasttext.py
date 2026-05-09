"""
Базовая fastText-модель на объединённых метках (21 класс).
Без бинарных классификаторов, без детектора скрипта, без роутера.
Гиперпараметры подбираются автоматически (grid search, macro F1).

Запуск:
    docker-compose run --rm baseline_trainer
"""

import os
import sys
import itertools

import fasttext
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from scripts.utils.label_mapping import merge_label
from scripts.data_processing.preprocess_text import preprocess_text

TRAIN_PATH = "output/train.csv"
VAL_PATH = "output/val.csv"
MODEL_PATH = "output/baseline_model.bin"
REPORT_PATH = "output/baseline_evaluation.txt"

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


def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, on_bad_lines="skip")
        except (UnicodeDecodeError, Exception):
            continue
    raise RuntimeError(f"Cannot read {path}")


def _to_ft(df, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = merge_label(str(row["result"]).strip())
            text = preprocess_text(str(row["request_text"]))
            text = text.replace("\n", " ").replace("\r", " ").strip()
            if label and text:
                f.write(f"__label__{label} {text}\n")
                n += 1
    return n


def _eval_macro_f1(model, val_texts, val_labels):
    y_pred = []
    for text in val_texts:
        pred = model.predict(text, k=1)[0][0].replace("__label__", "")
        y_pred.append(pred)
    return f1_score(val_labels, y_pred, average="macro", zero_division=0)


def _find_best_epoch(ft_train, val_texts, val_labels, params):
    best_f1 = 0.0
    best_epoch = EPOCH_STEP
    prev_f1 = None

    for epoch in range(EPOCH_STEP, EPOCH_MAX + 1, EPOCH_STEP):
        model = fasttext.train_supervised(input=ft_train, epoch=epoch, **params)
        f1 = _eval_macro_f1(model, val_texts, val_labels)
        delta = (f1 - prev_f1) if prev_f1 is not None else float("inf")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch

        if prev_f1 is not None and delta < EPOCH_MIN_DELTA:
            break

        prev_f1 = f1

    return best_epoch, best_f1


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"Файл не найден: {TRAIN_PATH}")
        print("Сначала: docker-compose run --rm splitter")
        sys.exit(1)

    train_df = _read_csv(TRAIN_PATH)
    ft_train = "output/baseline_train.txt"
    ft_val = "output/baseline_val.txt"

    print(f"Train: {len(train_df)} записей")
    n_train = _to_ft(train_df, ft_train)
    print(f"fastText train: {n_train} строк")

    best_params = {
        "dim": 150, "minn": 3, "maxn": 6,
        "wordNgrams": 2, "epoch": 40,
    }

    if os.path.exists(VAL_PATH):
        val_df = _read_csv(VAL_PATH)
        n_val = _to_ft(val_df, ft_val)
        print(f"Val: {n_val} строк")

        val_texts = []
        val_labels = []
        for _, row in val_df.iterrows():
            text = preprocess_text(str(row["request_text"]))
            text = text.replace("\n", " ").replace("\r", " ").strip()
            if not text:
                continue
            val_texts.append(text)
            val_labels.append(merge_label(str(row["result"]).strip()))

        combos = list(itertools.product(
            PARAM_GRID["dim"],
            PARAM_GRID["minn_maxn"],
            PARAM_GRID["wordNgrams"],
        ))
        print(f"\nGrid search ({len(combos)} комбинаций, метрика: macro F1)...")
        print(f"{'#':>3}  {'dim':>5}  {'minn':>5}  {'maxn':>5}  {'wNgr':>5}  {'epoch':>6}  {'macro F1':>10}")
        print("-" * 50)

        best_overall_f1 = 0.0
        for idx, (dim, (minn, maxn), word_ngrams) in enumerate(combos, 1):
            params = {
                **FIXED_PARAMS,
                "dim": dim,
                "minn": minn,
                "maxn": maxn,
                "wordNgrams": word_ngrams,
            }

            ep, f1 = _find_best_epoch(ft_train, val_texts, val_labels, params)

            is_best = f1 > best_overall_f1
            if is_best:
                best_overall_f1 = f1
                best_params = {
                    "dim": dim, "minn": minn, "maxn": maxn,
                    "wordNgrams": word_ngrams, "epoch": ep,
                }

            marker = " <- BEST" if is_best else ""
            print(f"{idx:>3}  {dim:>5}  {minn:>5}  {maxn:>5}  {word_ngrams:>5}  {ep:>6}  {f1:>10.4f}{marker}")

        print(f"\nЛучшие: {best_params}, F1={best_overall_f1:.4f}")
    else:
        print(f"Val не найден, используются параметры по умолчанию: {best_params}")

    dim = best_params["dim"]
    minn = best_params["minn"]
    maxn = best_params["maxn"]
    word_ngrams = best_params["wordNgrams"]
    best_epoch = best_params["epoch"]

    print(f"\nОбучение финальной модели (epoch={best_epoch}, dim={dim}, "
          f"minn={minn}, maxn={maxn}, wordNgrams={word_ngrams})...")
    model = fasttext.train_supervised(
        input=ft_train,
        epoch=best_epoch,
        lr=FIXED_PARAMS["lr"],
        wordNgrams=word_ngrams,
        dim=dim,
        minn=minn,
        maxn=maxn,
        minCount=FIXED_PARAMS["minCount"],
        loss=FIXED_PARAMS["loss"],
        seed=FIXED_PARAMS["seed"],
    )
    model.save_model(MODEL_PATH)
    print(f"Сохранено: {MODEL_PATH}")

    print(f"\nКлассов: {len(model.get_labels())}")
    print(f"Labels: {sorted(model.get_labels())}")

    report_text = ""
    if os.path.exists(VAL_PATH):
        print("\nОценка на val (объединённые метки)...")
        val_df = _read_csv(VAL_PATH)
        y_true, y_pred = [], []
        for _, row in val_df.iterrows():
            text = preprocess_text(str(row["request_text"]))
            text = text.replace("\n", " ").replace("\r", " ").strip()
            if not text:
                continue
            true = merge_label(str(row["result"]).strip())
            pred_lbl = model.predict(text, k=1)[0][0].replace("__label__", "")
            y_true.append(true)
            y_pred.append(pred_lbl)

        report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
        print(report_text)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(f"БАЗОВАЯ МОДЕЛЬ (fastText, 21 класс)\n")
            f.write(f"Параметры (auto): dim={dim} minn={minn} maxn={maxn} "
                    f"wordNgrams={word_ngrams} epoch={best_epoch}\n")
            f.write(f"Train: {n_train}\n\n")
            f.write(report_text)

        print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
