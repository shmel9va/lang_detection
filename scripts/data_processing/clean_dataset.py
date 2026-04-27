"""
Скрипт для очистки датасета: удаление редких классов.
Для lang_detection_diploma.csv редких классов нет — все 27 меток имеют по 2500 примеров.
"""
import pandas as pd
import os


def _read_file(file_path: str):
    """Читает CSV (cp1251/utf-8, разделитель ;) или Excel."""
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            return None
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            df = pd.read_csv(file_path, sep=";", encoding=enc, on_bad_lines="skip")
            print(f"  CSV загружен с кодировкой: {enc}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Ошибка ({enc}): {e}")
            continue
    print("Ошибка: не удалось прочитать CSV файл")
    return None


def clean_dataset(
    file_path="lang_detection_diploma.csv",
    language_column="result",
    rare_classes=[],
    replacement="other",
):
    print("Загрузка датасета...")
    df = _read_file(file_path)
    if df is None:
        return None

    print(f"Загружено записей: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")

    if language_column not in df.columns:
        print(f"ОШИБКА: Столбец {language_column!r} не найден")
        return None

    languages_before = df[language_column].dropna()
    language_counts_before = languages_before.value_counts().sort_values(ascending=False)
    total_before = len(languages_before)

    print(f"Всего примеров: {total_before}, языков: {languages_before.nunique()}")
    print(f"Редкие классы для удаления: {rare_classes}")

    df_cleaned = df.copy()
    mask = df_cleaned[language_column].isin(rare_classes)
    removed_count = int(mask.sum())

    if removed_count > 0:
        df_cleaned = df_cleaned[~mask]
        print(f"Удалено записей: {removed_count}")
    else:
        print("Редкие классы не найдены, датасет не изменился")

    languages_after = df_cleaned[language_column].dropna()
    language_counts_after = languages_after.value_counts().sort_values(ascending=False)
    total_after = len(languages_after)

    print(f"
{chr(39)*80}")
    print(f"{'Язык':<30} {'Количество':<15} {'Процент':<15}")
    print("-" * 60)
    for lang, count in language_counts_after.items():
        print(f"{str(lang):<30} {count:<15} {count/total_after*100:.2f}%")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if file_path.endswith(".csv"):
        output_file = os.path.join(output_dir, "lang_detection_diploma_cleaned.csv")
        df_cleaned.to_csv(output_file, sep=";", index=False, encoding="utf-8")
    else:
        output_file = os.path.join(output_dir, "lang_detection_diploma_cleaned.xlsx")
        df_cleaned.to_excel(output_file, index=False)

    print(f"Сохранено: {output_file}")

    stats_file = os.path.join(output_dir, "cleaning_statistics.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"До: {total_before} записей, {languages_before.nunique()} языков
")
        f.write(f"После: {total_after} записей, удалено {removed_count}
")
        f.write(f"Редкие классы: {rare_classes}

")
        for lang, count in language_counts_after.items():
            f.write(f"{str(lang):<30} {count:<15} {count/total_after*100:.2f}%
")

    return df_cleaned


if __name__ == "__main__":
    clean_dataset()
