"""
Поиск оптимального числа эпох для fastText через output/val.csv.

ВАЖНО: fastText не поддерживает checkpoint per epoch — каждую точку
надо обучать заново от нуля. Поэтому проверяем каждые STEP_EPOCHS эпох,
а не каждую.

Использует output/val.csv (выделенная валидационная выборка),
а не повторный split внутри train.
"""

import os
import fasttext
import pandas as pd
from scripts.utils.label_mapping import merge_label

STEP_EPOCHS = 5        # шаг: 5, 10, 15 ...
MAX_EPOCHS  = 70
MIN_DELTA   = 0.001    # прирост accuracy < 0.1% → останавливаемся

BASE_PARAMS = {
    "lr": 0.1,
    "wordNgrams": 2,
    "dim": 150,
    "minn": 3,
    "maxn": 6,
    "minCount": 2,
    "loss": "softmax",
    "seed": 42,
    "verbose": 0,
}


def _read_val(val_file: str, text_col: str, label_col: str):
    """Reads val CSV with encoding fallback."""
    for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
        try:
            return pd.read_csv(val_file, sep=';', encoding=enc, on_bad_lines='skip')
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {val_file}")


def find_optimal_epochs(
    train_file: str = "output/train_preprocessed.csv",
    val_file: str   = "output/val.csv",
    text_col: str   = "request_text",
    label_col: str  = "result",
    step: int       = STEP_EPOCHS,
    max_epochs: int = MAX_EPOCHS,
    min_delta: float = MIN_DELTA,
    random_state: int = 42,
):
    """
    Перебирает числа эпох [step, 2*step, ..., max_epochs].
    Для каждого значения обучает fastText на train_file, оценивает на val_file.
    Останавливается, когда прирост accuracy < min_delta.

    Returns:
        (results dict, best_epoch int)
    """
    print("=" * 70)
    print("ПОИСК ОПТИМАЛЬНОГО ЧИСЛА ЭПОХ")
    print(f"  Шаг: каждые {step} эпох  |  Макс: {max_epochs}")
    print(f"  Причина шага: fastText обучается заново для каждой точки")
    print(f"  Val: {val_file}  (выделенная, не пересечётся с test)")
    print("=" * 70)

    # ── Подготовка val данных ────────────────────────────────────────
    val_df = _read_val(val_file, text_col, label_col)
    val_data = [
        (str(row[text_col]).replace("\n", " ").replace("\r", " ").strip(),
         merge_label(str(row[label_col])))
        for _, row in val_df.iterrows()
        if str(row[text_col]).strip() and str(row[label_col]).strip()
    ]
    print(f"Val примеров: {len(val_data)}")

    # ── Подготовка train в формате fastText ─────────────────────────
    train_txt = "output/train_for_epoch_search.txt"
    train_df = None
    for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
        try:
            train_df = pd.read_csv(train_file, sep=';', encoding=enc, on_bad_lines='skip')
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    if train_df is None:
        raise RuntimeError(f"Cannot read {train_file}")

    with open(train_txt, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            label = str(row[label_col]).strip().replace(" ", "_")
            text  = str(row[text_col]).strip()
            if label and text:
                f.write(f"__label__{label} {text}\n")

    print(f"Train примеров для fastText: {len(train_df)}")

    # ── Перебор эпох ────────────────────────────────────────────────
    epochs_to_try = list(range(step, max_epochs + 1, step))
    results = {}
    best_epoch    = epochs_to_try[0]
    best_accuracy = 0.0
    prev_accuracy = None

    print(f"\n{'Epoch':>6}  {'Val Accuracy':>14}  {'Δ':>8}  {'':}")
    print("-" * 45)

    for epoch in epochs_to_try:
        model = fasttext.train_supervised(input=train_txt, epoch=epoch, **BASE_PARAMS)

        correct = sum(
            1 for text, true in val_data
            if merge_label(
                model.predict(text.replace("\n", " ").replace("\r", " "), k=1)[0][0]
                .replace("__label__", "")
            ) == true
        )
        accuracy = correct / len(val_data)
        delta    = (accuracy - prev_accuracy) if prev_accuracy is not None else float("inf")

        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            best_epoch    = epoch

        results[epoch] = {"accuracy": accuracy, "correct": correct, "total": len(val_data)}

        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker    = "← BEST" if is_best else ""
        print(f"{epoch:>6}  {accuracy:>14.2%}  {delta_str:>8}  {marker}")

        if prev_accuracy is not None and delta < min_delta:
            print(f"\n  Прирост {delta:.4f} < {min_delta} — останавливаемся.")
            break

        prev_accuracy = accuracy

    # ── Итог ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ОПТИМУМ: epoch={best_epoch}  Accuracy={best_accuracy:.2%}")
    print(f"{'='*70}")

    results_file = "output/epoch_validation_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("ПОИСК ОПТИМАЛЬНОГО ЧИСЛА ЭПОХ\n")
        f.write(f"Train: {train_file}  |  Val: {val_file}\n\n")
        f.write(f"{'Epoch':>6}  {'Accuracy':>10}  {'Correct/Total':>15}\n")
        f.write("-" * 40 + "\n")
        for ep, r in sorted(results.items()):
            marker = " ← BEST" if ep == best_epoch else ""
            f.write(f"{ep:>6}  {r['accuracy']:>10.2%}  {r['correct']}/{r['total']}{marker}\n")
        f.write(f"\nОПТИМУМ: epoch={best_epoch}  Accuracy={best_accuracy:.2%}\n")

    print(f"Результаты: {results_file}")
    return results, best_epoch


if __name__ == "__main__":
    find_optimal_epochs()
