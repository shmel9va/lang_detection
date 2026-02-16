"""
Скрипт для анализа статистики датасета определения языка
"""
import pandas as pd
from collections import Counter
import os

def analyze_dataset(file_path='lang_detection_hackathon.csv'):
    """
    Анализирует датасет и выводит статистику по языкам
    
    Args:
        file_path: путь к файлу датасета
        force_refresh: если True, перезаписывает старые файлы статистики
    """
    print("=" * 80)
    print("АНАЛИЗ ДАТАСЕТА")
    print("=" * 80)
    print(f"Файл: {file_path}")
    print(f"Полный путь: {os.path.abspath(file_path)}")
    print(f"Файл существует: {os.path.exists(file_path)}")
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        mod_time = os.path.getmtime(file_path)
        import datetime
        mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Размер файла: {file_size:,} байт")
        print(f"Дата изменения: {mod_time_str}")
    print("=" * 80)
    print("\nЗагрузка датасета...")
    
    # Определяем формат файла по расширению
    if file_path.endswith('.csv'):
        try:
            # CSV файл с разделителем точка с запятой
            # Пробуем разные кодировки
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'iso-8859-1']
            df_raw = None
            
            for encoding in encodings:
                try:
                    # Читаем первую строку для получения заголовков
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        header_line = f.readline().strip()
                        headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
                    
                    # Читаем данные, пропуская первую строку (заголовки) и вторую (типы данных)
                    df_raw = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding=encoding, on_bad_lines='skip', names=headers, engine='python')
                    print(f"Загружено строк из CSV: {len(df_raw):,} (кодировка: {encoding})")
                    print(f"  Столбцы: {list(df_raw.columns)}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Ошибка при чтении с кодировкой {encoding}: {e}")
                    continue
            
            if df_raw is None:
                # Последняя попытка с обработкой ошибок
                with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    header_line = f.readline().strip()
                    headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
                df_raw = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding='utf-8', on_bad_lines='skip', names=headers, engine='python')
                print(f"Загружено строк из CSV: {len(df_raw)} (кодировка: utf-8 с заменой ошибок)")
        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Excel файл
        try:
            df_raw = pd.read_excel(file_path)
            print(f"Загружено строк из Excel: {len(df_raw)}")
        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            return
    
    print(f"\n{'='*60}")
    print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ (ДО ФИЛЬТРАЦИИ)")
    print(f"{'='*60}")
    print(f"Общее количество записей: {len(df_raw)}")
    print(f"Количество столбцов: {len(df_raw.columns)}")
    print(f"Названия столбцов: {list(df_raw.columns)}")
    
    # Проверяем наличие пустых значений
    print(f"\n{'='*60}")
    print("АНАЛИЗ ПУСТЫХ ЗНАЧЕНИЙ")
    print(f"{'='*60}")
    for col in df_raw.columns:
        null_count = df_raw[col].isna().sum()
        empty_count = (df_raw[col].astype(str).str.strip() == '').sum()
        print(f"{col}: пустых (NaN) = {null_count}, пустых строк = {empty_count}")
    
    # Используем полный датасет для анализа (не фильтруем пока)
    df = df_raw.copy()
    
    # Показываем первые несколько строк для понимания структуры
    print(f"\n{'='*60}")
    print("ПЕРВЫЕ 5 СТРОК ДАТАСЕТА:")
    print(f"{'='*60}")
    print(df.head())
    
    # Ищем столбец с языками (сначала проверяем 'result')
    language_column = None
    if 'result' in df.columns:
        language_column = 'result'
    else:
        possible_names = ['language', 'lang', 'язык', 'label', 'target', 'class']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(name in col_lower for name in possible_names):
                language_column = col
                break
    
    # Если не нашли по названию, пробуем определить по содержимому
    if language_column is None:
        # Проверяем каждый столбец на наличие языковых кодов или названий
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 100:  # Если уникальных значений немного, возможно это языки
                # Проверяем, похожи ли значения на языковые коды (2-3 символа) или названия
                sample_vals = [str(v).lower() for v in unique_vals[:10]]
                if any(len(v) <= 5 for v in sample_vals) or any(v in ['ru', 'en', 'de', 'fr', 'es', 'it', 'ru', 'en'] for v in sample_vals):
                    language_column = col
                    break
    
    if language_column is None:
        print(f"\n{'='*60}")
        print("ВНИМАНИЕ: Не удалось автоматически определить столбец с языками")
        print(f"{'='*60}")
        print("Доступные столбцы:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col} (уникальных значений: {df[col].nunique()})")
        print("\nПожалуйста, укажите номер столбца с языками:")
        try:
            col_num = int(input("Номер столбца: ")) - 1
            language_column = df.columns[col_num]
        except:
            print("Ошибка ввода. Анализ прерван.")
            return
    else:
        print(f"\n{'='*60}")
        print(f"Найден столбец с языками: '{language_column}'")
        print(f"{'='*60}")
    
    # Анализ языков
    print(f"\n{'='*60}")
    print("СТАТИСТИКА ПО ЯЗЫКАМ")
    print(f"{'='*60}")
    
    # Показываем статистику до и после фильтрации
    languages_all = df[language_column]
    languages_valid = df[language_column].dropna()
    
    print(f"Всего записей в столбце '{language_column}': {len(languages_all)}")
    print(f"Записей с пустыми метками (NaN): {languages_all.isna().sum()}")
    print(f"Записей с валидными метками: {len(languages_valid)}")
    print(f"Количество уникальных языков: {languages_valid.nunique()}")
    
    # Используем только валидные метки для дальнейшего анализа
    languages = languages_valid
    
    # Подсчет количества примеров на каждый язык
    language_counts = languages.value_counts().sort_values(ascending=False)
    
    print(f"\n{'='*60}")
    print("РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ:")
    print(f"{'='*60}")
    print(f"{'Язык':<30} {'Количество':<15} {'Процент':<15}")
    print("-" * 60)
    
    total = len(languages)
    for lang, count in language_counts.items():
        percentage = (count / total) * 100
        print(f"{str(lang):<30} {count:<15} {percentage:.2f}%")
    
    # Дополнительная статистика
    print(f"\n{'='*60}")
    print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"{'='*60}")
    print(f"Среднее количество примеров на язык: {total / languages.nunique():.2f}")
    print(f"Минимальное количество примеров: {language_counts.min()}")
    print(f"Максимальное количество примеров: {language_counts.max()}")
    print(f"Медианное количество примеров: {language_counts.median():.2f}")
    
    # Список всех языков
    print(f"\n{'='*60}")
    print("СПИСОК ВСЕХ ЯЗЫКОВ:")
    print(f"{'='*60}")
    for i, lang in enumerate(language_counts.index, 1):
        print(f"{i}. {lang}")
    
    # Сохраняем статистику в файл (перезаписываем старый файл)
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'dataset_statistics.txt')
    
    # Удаляем старый файл, если существует
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА ДАТАСЕТА\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Исходный файл: {file_path}\n")
        f.write(f"Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Общее количество записей (после фильтрации): {len(df)}\n")
        f.write(f"Записей с валидными метками: {len(languages)}\n")
        f.write(f"Количество уникальных языков: {languages.nunique()}\n\n")
        f.write("РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Язык':<30} {'Количество':<15} {'Процент':<15}\n")
        f.write("-" * 60 + "\n")
        for lang, count in language_counts.items():
            percentage = (count / total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
    
    print(f"\n{'='*60}")
    print(f"Статистика сохранена в файл: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    analyze_dataset()

