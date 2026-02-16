"""
Скрипт для очистки датасета: замена редких классов на 'other'
"""
import pandas as pd
import os

def clean_dataset(
    file_path='lang_detection_hackathon.csv',
    language_column='result',
    rare_classes=['mk', 'string', 'mr', 'bs', 'tg', 'de', 'ky', 'zh'],
    replacement='other'
):
    """
    Заменяет редкие классы на 'other' и сохраняет очищенный датасет
    """
    print("Загрузка датасета...")
    
    # Определяем формат файла по расширению
    if file_path.endswith('.csv'):
        try:
            # CSV файл с разделителем точка с запятой
            # Пробуем разные кодировки
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    # Читаем первую строку для получения заголовков
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        header_line = f.readline().strip()
                        headers = header_line.split(';')
                    
                    # Читаем данные, пропуская первую строку (заголовки) и вторую (типы данных)
                    df = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding=encoding, on_bad_lines='skip', names=headers)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if df is None:
                # Последняя попытка с обработкой ошибок
                with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    header_line = f.readline().strip()
                    headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
                df = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding='utf-8', on_bad_lines='skip', names=headers, engine='python')
        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            return None
    else:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            return None
    
    print(f"Загружено записей: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")
    
    # Проверяем наличие столбца с языками
    if language_column not in df.columns:
        print(f"\nОШИБКА: Столбец '{language_column}' не найден в датасете")
        print("Доступные столбцы:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        return None
    
    print(f"\nНайден столбец с языками: '{language_column}'")
    
    # Статистика до очистки
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ДО ОЧИСТКИ")
    print(f"{'='*80}")
    
    languages_before = df[language_column].dropna()
    language_counts_before = languages_before.value_counts().sort_values(ascending=False)
    total_before = len(languages_before)
    
    print(f"Общее количество примеров: {total_before}")
    print(f"Количество уникальных языков: {languages_before.nunique()}")
    
    # Показываем редкие классы
    print(f"\nРедкие классы для замены: {rare_classes}")
    rare_counts = {lang: language_counts_before.get(lang, 0) for lang in rare_classes}
    print("Количество примеров в редких классах:")
    for lang, count in rare_counts.items():
        if count > 0:
            print(f"  {lang}: {count}")
    
    df_cleaned = df.copy()
    
    # Удаляем редкие классы
    mask = df_cleaned[language_column].isin(rare_classes)
    removed_count = mask.sum()
    
    if removed_count > 0:
        df_cleaned = df_cleaned[~mask]
        print(f"\nУдалено записей с редкими классами: {removed_count}")
    else:
        print(f"\nРедкие классы не найдены в датасете")
    
    # Статистика после очистки
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ПОСЛЕ ОЧИСТКИ")
    print(f"{'='*80}")
    
    languages_after = df_cleaned[language_column].dropna()
    language_counts_after = languages_after.value_counts().sort_values(ascending=False)
    total_after = len(languages_after)
    
    print(f"Общее количество примеров: {total_after}")
    print(f"Количество уникальных языков: {languages_after.nunique()}")
    
    print(f"\n{'Язык':<30} {'Количество':<15} {'Процент':<15}")
    print("-" * 60)
    
    for lang, count in language_counts_after.items():
        percentage = (count / total_after) * 100
        print(f"{str(lang):<30} {count:<15} {percentage:.2f}%")
    
    # Сохраняем очищенный датасет
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Сохраняем в том же формате, что и исходный файл
    if file_path.endswith('.csv'):
        output_file = os.path.join(output_dir, 'lang_detection_hackathon_cleaned.csv')
        df_cleaned.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    else:
        output_file = os.path.join(output_dir, 'lang_detection_hackathon_cleaned.xlsx')
        df_cleaned.to_excel(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("СОХРАНЕНИЕ ДАННЫХ")
    print(f"{'='*80}")
    print(f"Очищенный датасет сохранен: {output_file}")
    
    # Сохраняем статистику
    stats_file = os.path.join(output_dir, 'cleaning_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА ОЧИСТКИ ДАТАСЕТА\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Исходный датасет: {total_before} записей, {languages_before.nunique()} языков\n")
        f.write(f"Очищенный датасет: {total_after} записей, {languages_after.nunique()} языков\n")
        f.write(f"Удалено записей: {removed_count}\n")
        f.write(f"Редкие классы (удалены): {', '.join(rare_classes)}\n\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - ДО ОЧИСТКИ\n")
        f.write("-" * 80 + "\n")
        for lang, count in language_counts_before.items():
            percentage = (count / total_before) * 100
            marker = " <-- заменено" if lang in rare_classes else ""
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%{marker}\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - ПОСЛЕ ОЧИСТКИ\n")
        f.write("-" * 80 + "\n")
        for lang, count in language_counts_after.items():
            percentage = (count / total_after) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
    
    print(f"Статистика сохранена: {stats_file}")
    print(f"{'='*80}")
    
    return df_cleaned

if __name__ == "__main__":
    clean_dataset()



