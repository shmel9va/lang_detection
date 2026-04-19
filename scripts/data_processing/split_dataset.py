"""
Разбиение датасета на train / val / test с сохранением распределения языков.

Схема:
  train (70%) — обучение fastText и бинарных классификаторов
  val   (15%) — подбор threshold, оценка бинарных классификаторов, отладка
  test  (15%) — финальная оценка; не трогаем до конца разработки

Стратификация выполняется по объединённым меткам (uz_lat/uz_cyr → uz и т.д.),
но в сами файлы сохраняются исходные метки (uz_lat, uz_cyr, sr_lat, ...).
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.utils.label_mapping import merge_labels_in_series


# ──────────────────────────────────────────────────────────────────────
# Чтение датасета
# ──────────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                header_line = f.readline().strip()
                headers = [h.strip("\ufeff") for h in header_line.split(";")]
            df = pd.read_csv(
                path, sep=";", skiprows=[0, 1], encoding=enc,
                on_bad_lines="skip", names=headers, engine="python"
            )
            print(f"  Файл загружен с кодировкой: {enc}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    raise RuntimeError(f"Не удалось прочитать файл: {path}")


def _find_language_column(df: pd.DataFrame) -> str:
    if "result" in df.columns:
        return "result"
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 100:
            sample = [str(v).lower() for v in unique_vals[:10]]
            if any(len(v) <= 5 for v in sample):
                return col
    raise RuntimeError("Не найден столбец с языковыми метками.")


# ──────────────────────────────────────────────────────────────────────
# Статистика
# ──────────────────────────────────────────────────────────────────────

def _print_stats(df: pd.DataFrame, col: str, title: str) -> pd.Series:
    counts = df[col].dropna().value_counts().sort_values(ascending=False)
    total = counts.sum()
    print(f"\n{'='*70}")
    print(f"  {title}  (всего: {total})")
    print(f"{'='*70}")
    print(f"  {'Метка':<25} {'N':>7}  {'%':>7}")
    print(f"  {'-'*42}")
    for lang, cnt in counts.items():
        print(f"  {str(lang):<25} {cnt:>7}  {cnt/total*100:>6.1f}%")
    return counts


# ──────────────────────────────────────────────────────────────────────
# Основная функция
# ──────────────────────────────────────────────────────────────────────

def split_dataset(
    file_path: str = "lang_detection_hackathon.csv",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Разбивает датасет на train / val / test.

    Args:
        file_path:    путь к исходному CSV.
        val_size:     доля validation (default 0.15).
        test_size:    доля test (default 0.15).
        random_state: seed для воспроизводимости.

    Returns:
        (train_df, val_df, test_df)
    """
    print("=" * 70)
    print("РАЗБИЕНИЕ ДАТАСЕТА: train / val / test")
    print("=" * 70)
    print(f"\n  train : {1 - val_size - test_size:.0%}")
    print(f"  val   : {val_size:.0%}")
    print(f"  test  : {test_size:.0%}")

    # ── Загрузка ──────────────────────────────────────────────────
    print(f"\nЗагрузка: {file_path}")
    df = _read_csv(file_path)
    print(f"Загружено записей: {len(df)}")

    lang_col = _find_language_column(df)
    text_col = "request_text"
    print(f"Столбец языков: '{lang_col}'")

    # ── Базовая фильтрация ────────────────────────────────────────
    df = df.dropna(subset=[lang_col, text_col]).copy()
    df = df[df[lang_col].astype(str).str.strip() != ""].copy()
    df = df[df[text_col].astype(str).str.strip() != ""].copy()
    print(f"После фильтрации пустых: {len(df)}")

    # Исходные метки сохраняем; merged используем только для стратификации
    df["_merged"] = merge_labels_in_series(df[lang_col])

    # Удаляем классы с < 2 примерами (stratified split требует ≥ 2)
    counts = df["_merged"].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        print(f"\n  Удалены классы с < 2 примерами (слишком мало для stratified split): {rare}")
        df = df[~df["_merged"].isin(rare)].copy()

    _print_stats(df, lang_col, "ИСХОДНЫЙ ДАТАСЕТ (исходные метки)")

    # ── Первое разбиение: отщепляем test ─────────────────────────
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["_merged"],
        random_state=random_state,
    )

    # ── Второе разбиение: отщепляем val из trainval ───────────────
    relative_val = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=relative_val,
        stratify=df_trainval["_merged"],
        random_state=random_state,
    )

    # Убираем служебный столбец
    for part in [df_train, df_val, df_test]:
        part.drop(columns=["_merged"], inplace=True, errors="ignore")

    total = len(df)
    print(f"\n  train : {len(df_train):>6} записей ({len(df_train)/total:.1%})")
    print(f"  val   : {len(df_val):>6} записей ({len(df_val)/total:.1%})")
    print(f"  test  : {len(df_test):>6} записей ({len(df_test)/total:.1%})")

    _print_stats(df_train, lang_col, "TRAIN")
    _print_stats(df_val,   lang_col, "VAL")
    _print_stats(df_test,  lang_col, "TEST")

    # ── Сохранение ────────────────────────────────────────────────
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, sep=";", index=False, encoding="utf-8")
    df_val.to_csv(val_path,     sep=";", index=False, encoding="utf-8")
    df_test.to_csv(test_path,   sep=";", index=False, encoding="utf-8")

    print(f"\n  Сохранено:")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"    {test_path}")

    # Статистика в файл
    stats_path = os.path.join(output_dir, "split_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("РАЗБИЕНИЕ ДАТАСЕТА\n")
        f.write(f"train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)}\n")
        f.write(f"random_state: {random_state}\n\n")
        for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
            f.write(f"[{name.upper()}]\n")
            counts = part[lang_col].value_counts()
            for lang, cnt in counts.items():
                f.write(f"  {lang:<25} {cnt}\n")
            f.write("\n")

    print(f"  Статистика: {stats_path}")
    print("=" * 70)

    return df_train, df_val, df_test


if __name__ == "__main__":
    split_dataset()
