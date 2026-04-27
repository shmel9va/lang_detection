"""
Скрипт для анализа статистики датасета определения языка.
"""
import pandas as pd
from collections import Counter
import os
import datetime


def analyze_dataset(file_path='lang_detection_diploma.csv'):
    print('=' * 70)
    print('АНАЛИЗ ДАТАСЕТА')
    print('=' * 70)
    print(f'Файл: {file_path}')
    print(f'Существует: {os.path.exists(file_path)}')
    if os.path.exists(file_path):
        print(f'Размер: {os.path.getsize(file_path):,} байт')
        mt = os.path.getmtime(file_path)
        print(f'Изменён: {datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    print('Загрузка...')
    df_raw = None
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        try:
            df_raw = pd.read_excel(file_path)
            print(f'Загружено строк из Excel: {len(df_raw)}')
        except Exception as e:
            print(f'Ошибка Excel: {e}')
            return
    else:
        for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
            try:
                df_raw = pd.read_csv(file_path, sep=';', encoding=enc, on_bad_lines='skip')
                print(f'Загружено строк из CSV ({enc}): {len(df_raw)}')
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f'Ошибка CSV ({enc}): {e}')
                continue
        if df_raw is None:
            print('Ошибка: не удалось прочитать CSV')
            return

    print(f'
{"="*60}')
    print('ОБЩАЯ ИНФОРМАЦИЯ')
    print(f'{"="*60}')
    print(f'Записей: {len(df_raw)}')
    print(f'Столбцов: {len(df_raw.columns)}')
    print(f'Названия: {list(df_raw.columns)}')

    print(f'
{"="*60}')
    print('АНАЛИЗ ПУСТЫХ ЗНАЧЕНИЙ')
    print(f'{"="*60}')
    for col in df_raw.columns:
        null_count = df_raw[col].isna().sum()
        empty_count = (df_raw[col].astype(str).str.strip() == '').sum()
        print(f'{col}: NaN={null_count}, пустых строк={empty_count}')

    print(f'
{"="*60}')
    print('ПЕРВЫЕ 5 СТРОК:')
    print(f'{"="*60}')
    print(df_raw.head())

    # Определяем столбец с языками
    language_column = None
    if 'result' in df_raw.columns:
        language_column = 'result'
    else:
        for col in df_raw.columns:
            unique_vals = df_raw[col].dropna().unique()
            if len(unique_vals) < 100:
                sample = [str(v).lower() for v in unique_vals[:10]]
                if any(len(v) <= 6 for v in sample):
                    language_column = col
                    break

    if language_column is None:
        print('
Не удалось определить столбец с языками')
        return

    print(f'
Столбец с языками: {language_column!r}')

    languages = df_raw[language_column].dropna()
    language_counts = languages.value_counts().sort_values(ascending=False)
    total = len(languages)

    print(f'
{"="*60}')
    print('РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ:')
    print(f'{"="*60}')
    print(f'{"Язык":<30} {"Количество":<15} {"Процент":<15}')
    print('-' * 60)
    for lang, count in language_counts.items():
        print(f'{str(lang):<30} {count:<15} {count/total*100:.2f}%')

    print(f'
Среднее на язык: {total / languages.nunique():.1f}')
    print(f'Мин: {language_counts.min()}, Макс: {language_counts.max()}')

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'dataset_statistics.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('СТАТИСТИКА ДАТАСЕТА
')
        f.write('=' * 60 + '

')
        f.write(f'Файл: {file_path}
')
        f.write(f'Дата: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

')
        f.write(f'Записей: {len(df_raw)}
')
        f.write(f'Языков: {languages.nunique()}

')
        f.write('РАСПРЕДЕЛЕНИЕ:
')
        f.write('-' * 60 + '
')
        for lang, count in language_counts.items():
            f.write(f'{str(lang):<30} {count:<15} {count/total*100:.2f}%
')

    print(f'
Статистика сохранена: {output_file}')


if __name__ == '__main__':
    analyze_dataset()
