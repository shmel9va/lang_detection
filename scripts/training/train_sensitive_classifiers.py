"""
Скрипт обучения бинарных классификаторов для чувствительных языковых пар.

Для каждой из 5 пар:
  ar–fa  |  ru–uk  |  hy–az  |  he–ar  |  ur–hi

скрипт:
  1. Берёт примеры только для двух языков из train.csv
  2. Обучает SensitivePairClassifier (char n-gram TF-IDF + LogReg)
  3. Оценивает на test.csv: accuracy + Precision/Recall/F1 по каждому классу
  4. Сохраняет модель в output/sensitive_classifiers/<pair>.pkl

Запуск:
    python -m scripts.training.train_sensitive_classifiers
    # или через docker-compose run --rm trainer_sensitive
"""

import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple, Type

import pandas as pd

from scripts.detection.sensitive_classifiers.ar_fa import ArFaClassifier
from scripts.detection.sensitive_classifiers.az_tr import AzTrClassifier
from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier
from scripts.detection.sensitive_classifiers.es_pt import EsPtClassifier
from scripts.detection.sensitive_classifiers.he_ar import HeArClassifier
from scripts.detection.sensitive_classifiers.hy_az import HyAzClassifier
from scripts.detection.sensitive_classifiers.ru_sr import RuSrClassifier
from scripts.detection.sensitive_classifiers.ru_uk import RuUkClassifier
from scripts.detection.sensitive_classifiers.ur_hi import UrHiClassifier
from scripts.detection.sensitive_classifiers.uz_kk import UzKkClassifier
from scripts.utils.label_mapping import merge_label
from scripts.data_processing.preprocess_text import normalize_for_detection

# Список пар для обучения: (lang1, lang2, класс классификатора)
PAIRS: List[Tuple[str, str, Type[SensitivePairClassifier]]] = [
    ("ar", "fa", ArFaClassifier),
    ("ru", "uk", RuUkClassifier),
    ("hy", "az", HyAzClassifier),
    ("he", "ar", HeArClassifier),
    ("ur", "hi", UrHiClassifier),
    ("az", "tr", AzTrClassifier),
    ("es", "pt", EsPtClassifier),
    ("ru", "sr", RuSrClassifier),
    ("uz", "kk", UzKkClassifier),
]


# ──────────────────────────────────────────────────────────────────────
# I/O утилиты
# ──────────────────────────────────────────────────────────────────────

def _read_csv(path: str):
    """Reads CSV (sep=;, cp1251/utf-8) files."""
    for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
        try:
            return pd.read_csv(path, sep=';', encoding=enc, on_bad_lines='skip')
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    print(f"  ERROR: cannot read {path}")
    return None


def _prepare_pair_data(
    df: pd.DataFrame,
    lang1: str,
    lang2: str,
    text_col: str = "request_text",
    label_col: str = "result",
) -> Tuple[List[str], List[str]]:
    """Фильтрует датасет по двум языкам, возвращает тексты и метки."""
    df = df.copy()
    df["_merged"] = df[label_col].apply(merge_label)
    mask = df["_merged"].isin([lang1, lang2])
    pair_df = df[mask].dropna(subset=[text_col, label_col]).copy()

    texts = pair_df[text_col].astype(str).apply(normalize_for_detection).tolist()
    labels = pair_df["_merged"].tolist()
    return texts, labels


# ──────────────────────────────────────────────────────────────────────
# Обучение одной пары
# ──────────────────────────────────────────────────────────────────────

def train_one_pair(
    lang1: str,
    lang2: str,
    clf_class: Type[SensitivePairClassifier],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: str,
) -> Optional[SensitivePairClassifier]:
    pair_name = f"{lang1}_{lang2}"
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {lang1.upper()} vs {lang2.upper()}")
    print(sep)

    # ── Обучающие данные ──────────────────────────────────────────
    train_texts, train_labels = _prepare_pair_data(
        train_df, lang1, lang2, text_col, label_col
    )

    if len(train_texts) < 10:
        counts = pd.Series(train_labels).value_counts().to_dict() if train_labels else {}
        print(f"  ПРОПУСК: слишком мало данных. Найдено: {counts}")
        print(f"  Добавьте примеры языков {lang1}/{lang2} в датасет.")
        return None

    # Проверяем, что оба класса присутствуют — LogReg требует минимум 2 класса
    unique_classes = set(train_labels)
    if len(unique_classes) < 2:
        missing = [l for l in [lang1, lang2] if l not in unique_classes]
        present = list(unique_classes)[0]
        print(f"  ПРОПУСК: только один класс в данных ('{present}'), нет данных для: {missing}")
        print(f"  Добавьте примеры для {missing} в датасет и запустите повторно.")
        print(f"  Быстрый путь (детерминированные правила) работает без обучения.")
        return None

    label_counts = pd.Series(train_labels).value_counts()
    print(f"\n  Обучающие примеры:")
    for lang, cnt in label_counts.items():
        print(f"    {lang}: {cnt}")

    # ── Обучение ──────────────────────────────────────────────────
    clf = clf_class()
    clf.fit(train_texts, train_labels)
    print(f"\n  Обучение завершено.")

    # ── Оценка на val ─────────────────────────────────────────────
    # Используем val, а не test — test остаётся полностью закрытым
    val_texts, val_labels = _prepare_pair_data(
        val_df, lang1, lang2, text_col, label_col
    )

    if val_texts:
        val_counts = pd.Series(val_labels).value_counts()
        print(f"\n  Val примеры:")
        for lang, cnt in val_counts.items():
            print(f"    {lang}: {cnt}")

        accuracy, _ = clf.evaluate(val_texts, val_labels)
        print(f"\n  Accuracy (val): {accuracy:.2%}")

        metrics = clf.per_class_metrics(val_texts, val_labels)
        print(f"\n  Метрики по классам (val):")
        header = f"  {'Язык':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}  TP   FP   FN"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lang, m in metrics.items():
            print(
                f"  {lang:<8} {m['precision']:>9.2%} {m['recall']:>9.2%} "
                f"{m['f1']:>9.2%}  {m['tp']:>3}  {m['fp']:>3}  {m['fn']:>3}"
            )
    else:
        print(f"\n  ВНИМАНИЕ: нет val данных для {lang1}/{lang2}.")

    # ── Сохранение ────────────────────────────────────────────────
    save_path = os.path.join(output_dir, f"{pair_name}.pkl")
    clf.save(save_path)
    print(f"\n  Сохранено: {save_path}")

    return clf


# ──────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────

def train_all(
    train_file: str = "output/train.csv",
    val_file: str = "output/val.csv",
    text_col: str = "request_text",
    label_col: str = "result",
    output_dir: str = "output/sensitive_classifiers",
) -> None:
    """
    Обучает все бинарные классификаторы.

    Оценка выполняется на val, а не на test — test остаётся закрытым
    до финальной оценки всей системы.
    """
    print("=" * 70)
    print("ОБУЧЕНИЕ БИНАРНЫХ КЛАССИФИКАТОРОВ ЧУВСТВИТЕЛЬНЫХ ПАР")
    print("=" * 70)

    for path in [train_file, val_file]:
        if not os.path.exists(path):
            print(f"ОШИБКА: файл не найден: {path}")
            print("Сначала запустите: splitter → preprocessor → trainer.")
            return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nЗагрузка данных...")
    train_df = _read_csv(train_file)
    val_df = _read_csv(val_file)

    if train_df is None or val_df is None:
        return

    print(f"Train: {len(train_df)} записей")
    print(f"Val:   {len(val_df)} записей")

    if label_col in train_df.columns:
        available = set(train_df[label_col].apply(merge_label).dropna().unique())
        print(f"\nДоступные языки в train: {sorted(available)}")
        print(f"\nПары с чувствительными языками:")
        for lang1, lang2, _ in PAIRS:
            has1 = lang1 in available
            has2 = lang2 in available
            status = "✓ готова" if (has1 and has2) else f"⚠ нет данных: {[l for l in [lang1, lang2] if l not in available]}"
            print(f"  {lang1}-{lang2}: {status}")

    trained_pairs = []
    skipped_pairs = []

    for lang1, lang2, clf_class in PAIRS:
        clf = train_one_pair(
            lang1, lang2, clf_class,
            train_df, val_df,
            text_col, label_col,
            output_dir,
        )
        if clf is not None:
            trained_pairs.append(f"{lang1}-{lang2}")
        else:
            skipped_pairs.append(f"{lang1}-{lang2}")

    # Итог
    print(f"\n{'=' * 70}")
    print("ИТОГ")
    print(f"{'=' * 70}")
    print(f"Обучено классификаторов: {len(trained_pairs)}")
    for p in trained_pairs:
        print(f"  + {p}")

    if skipped_pairs:
        print(f"\nПропущено (нет данных): {len(skipped_pairs)}")
        for p in skipped_pairs:
            print(f"  - {p}  ← добавьте данные и запустите повторно")

    print(f"\nМодели сохранены в: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    train_all()
