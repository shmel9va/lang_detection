"""
Пересобирает lang_detection_diploma.csv, заменяя повреждённые строки
на корректные Unicode-данные из data/*.xlsx.

Проблема: при сохранении в cp1251 все небуквенно-кириллические/латинские
символы (армянский, грузинский, деванагари, арабский, иврит, амхарский)
заменились на '?' — модель не может обучиться на этих языках.

Что исправляет этот скрипт:
  hy_arm  → data/armenian_dataset.xlsx  (Sheet «Armenian (Original)»,  col «sentence»)
  ka      → data/georgian_dataset.xlsx  (Sheet «Georgian (Original)»,  col «text»)
  ne_nep  → data/nepali_dataset.xlsx    (Sheet «Nepali (Original)»,    col «text»)

Что по-прежнему требует внешних данных (в CSV всё ещё '?'):
  ar, fa, he, am, hi, ur_ur

Результат: lang_detection_diploma.csv перезаписывается UTF-8 (без BOM).
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "lang_detection_diploma.csv"
DATA_DIR = ROOT / "data"


def _load_xlsx_texts(xlsx_path: Path, sheet: str, col: str, n: int = 2500) -> list[str]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    texts = df[col].dropna().astype(str).str.strip()
    texts = texts[texts != ""].reset_index(drop=True)
    if len(texts) < n:
        print(f"  WARNING: {xlsx_path.name} / {sheet} has only {len(texts)} rows (expected {n})")
    return texts.tolist()[:n]


def rebuild():
    print("=" * 70)
    print("ПЕРЕСБОРКА lang_detection_diploma.csv")
    print("=" * 70)

    # ── Читаем текущий CSV ────────────────────────────────────────────
    print(f"\nЧитаем: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=";", encoding="cp1251", on_bad_lines="skip")
    print(f"  Загружено: {len(df)} строк, столбцы: {list(df.columns)}")

    counts_before = df["result"].value_counts().sort_index()

    # ── Удаляем повреждённые метки ────────────────────────────────────
    to_replace = {"hy_arm", "ka", "ne_nep"}
    df_clean = df[~df["result"].isin(to_replace)].copy()
    print(f"\n  Удалено строк ({', '.join(sorted(to_replace))}): "
          f"{len(df) - len(df_clean)}")

    # ── Загружаем корректные данные из XLSX ───────────────────────────
    replacements = [
        (
            "hy_arm",
            DATA_DIR / "armenian_dataset.xlsx",
            "Armenian (Original)",
            "sentence",
        ),
        (
            "ka",
            DATA_DIR / "georgian_dataset.xlsx",
            "Georgian (Original)",
            "text",
        ),
        (
            "ne_nep",
            DATA_DIR / "nepali_dataset.xlsx",
            "Nepali (Original)",
            "text",
        ),
    ]

    new_parts = []
    for label, xlsx_path, sheet, col in replacements:
        print(f"\n  {label}: {xlsx_path.name} / {sheet!r} / col={col!r}")
        texts = _load_xlsx_texts(xlsx_path, sheet, col, n=2500)
        part = pd.DataFrame({"request_text": texts, "result": label})
        new_parts.append(part)
        print(f"    → {len(part)} строк добавлено")

    # ── Собираем итоговый датафрейм ───────────────────────────────────
    df_fixed = pd.concat([df_clean] + new_parts, ignore_index=True)
    df_fixed = df_fixed[["request_text", "result"]]

    # ── Перемешиваем ──────────────────────────────────────────────────
    df_fixed = df_fixed.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nИтого строк: {len(df_fixed)}")
    counts_after = df_fixed["result"].value_counts().sort_index()
    print(f"\n{'Метка':<12} {'До':>6} {'После':>7}")
    print("-" * 28)
    all_labels = sorted(set(counts_before.index) | set(counts_after.index))
    for lbl in all_labels:
        before = counts_before.get(lbl, 0)
        after  = counts_after.get(lbl, 0)
        flag   = " ← FIXED" if lbl in to_replace else ""
        print(f"{lbl:<12} {before:>6} {after:>7}{flag}")

    # ── Сохраняем UTF-8 ───────────────────────────────────────────────
    print(f"\nСохраняем: {CSV_PATH}  (UTF-8, sep=';')")
    df_fixed.to_csv(CSV_PATH, sep=";", index=False, encoding="utf-8")
    size_mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"  Размер файла: {size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("ГОТОВО. Следующий шаг: запустить полный pipeline заново.")
    print("\nВСЁ ЕЩЁ ПОВРЕЖДЕНЫ (нужны внешние данные):")
    for lbl in ["ar", "fa", "he", "am", "hi", "ur_ur"]:
        n = counts_after.get(lbl, 0)
        print(f"  {lbl}: {n} строк в датасете, текст — '?' (не обучится)")
    print("=" * 70)


if __name__ == "__main__":
    rebuild()
