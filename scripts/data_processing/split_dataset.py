"""
Скрипт для разделения датасета на train/test выборки с сохранением распределения языков
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from scripts.utils.label_mapping import merge_labels_in_series

def find_language_column(df):
    """
    Находит столбец с языками в датасете
    """
    # Сначала проверяем столбец 'result'
    if 'result' in df.columns:
        return 'result'
    
    language_column = None
    possible_names = ['language', 'lang', 'язык', 'label', 'target', 'class']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(name in col_lower for name in possible_names):
            language_column = col
            break
    
    # Если не нашли по названию, пробуем определить по содержимому
    if language_column is None:
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 100:
                sample_vals = [str(v).lower() for v in unique_vals[:10]]
                if any(len(v) <= 5 for v in sample_vals) or any(v in ['ru', 'en', 'de', 'fr', 'es', 'it', 'ru', 'en'] for v in sample_vals):
                    language_column = col
                    break
    
    return language_column

def print_language_statistics(df, language_column, dataset_name):
    """
    Выводит статистику по языкам для датасета
    """
    languages = df[language_column].dropna()
    language_counts = languages.value_counts().sort_values(ascending=False)
    total = len(languages)
    
    print(f"\n{'='*80}")
    print(f"СТАТИСТИКА ПО ЯЗЫКАМ - {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Общее количество примеров: {total}")
    print(f"Количество уникальных языков: {languages.nunique()}")
    print(f"\n{'Язык':<30} {'Количество':<15} {'Процент':<15}")
    print("-" * 60)
    
    for lang, count in language_counts.items():
        percentage = (count / total) * 100
        print(f"{str(lang):<30} {count:<15} {percentage:.2f}%")
    
    return language_counts, total

def split_dataset(file_path='lang_detection_hackathon.csv', test_size=0.2, random_state=42, use_cleaned=True):
    """
    Разделяет датасет на train/test с сохранением распределения языков
    """
    # Пробуем использовать очищенный датасет, если он существует
    if use_cleaned:
        # Проверяем оба формата
        cleaned_path_csv = os.path.join('output', 'lang_detection_hackathon_cleaned.csv')
        cleaned_path_xlsx = os.path.join('output', 'lang_detection_hackathon_cleaned.xlsx')
        if os.path.exists(cleaned_path_csv):
            print(f"Используется очищенный датасет: {cleaned_path_csv}")
            file_path = cleaned_path_csv
        elif os.path.exists(cleaned_path_xlsx):
            print(f"Используется очищенный датасет: {cleaned_path_xlsx}")
            file_path = cleaned_path_xlsx
    
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
                    print(f"Файл загружен с кодировкой: {encoding}")
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
                print(f"Файл загружен с обработкой ошибок кодировки")
        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            return
    
    print(f"Загружено записей: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")
    
    # Находим столбец с языками
    language_column = find_language_column(df)
    
    if language_column is None:
        print("\nОШИБКА: Не удалось определить столбец с языками")
        print("Доступные столбцы:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col} (уникальных значений: {df[col].nunique()})")
        return None, None
    
    print(f"\nНайден столбец с языками: '{language_column}'")
    
    # Показываем статистику до фильтрации
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ДО ФИЛЬТРАЦИИ")
    print(f"{'='*80}")
    print(f"Всего записей: {len(df)}")
    print(f"Записей с пустыми метками (NaN): {df[language_column].isna().sum()}")
    print(f"Записей с пустыми строками в метках: {(df[language_column].astype(str).str.strip() == '').sum()}")
    
    # Удаляем строки с пропущенными значениями в столбце языков
    df_clean = df.dropna(subset=[language_column]).copy()
    
    # Также удаляем строки с пустыми строками в метках
    mask_empty = df_clean[language_column].astype(str).str.strip() == ''
    empty_count = mask_empty.sum()
    if empty_count > 0:
        print(f"Записей с пустыми строками в метках (будут удалены): {empty_count}")
        df_clean = df_clean[~mask_empty].copy()
    
    print(f"Записей после удаления пропусков: {len(df_clean)}")
    print(f"Удалено записей: {len(df) - len(df_clean)} ({((len(df) - len(df_clean)) / len(df) * 100):.2f}%)")
    
    # Сохраняем исходные метки в отдельном столбце
    df_clean['original_label'] = df_clean[language_column].copy()
    
    # Создаем столбец с объединенными метками для стратификации
    df_clean['merged_label'] = merge_labels_in_series(df_clean[language_column])
    
    # Статистика исходного датасета (по исходным меткам)
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ИСХОДНОГО ДАТАСЕТА (ИСХОДНЫЕ МЕТКИ)")
    print(f"{'='*80}")
    original_counts, original_total = print_language_statistics(df_clean, language_column, "Исходный датасет")
    
    # Статистика по объединенным меткам
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ИСХОДНОГО ДАТАСЕТА (ОБЪЕДИНЕННЫЕ МЕТКИ)")
    print(f"{'='*80}")
    merged_counts, merged_total = print_language_statistics(df_clean, 'merged_label', "Исходный датасет (объединенные)")
    
    # Обрабатываем редкие классы (менее 2 примеров) для stratified split
    # Используем объединенные метки для определения редких классов
    rare_classes_mask = merged_counts < 2
    rare_classes = merged_counts[rare_classes_mask].index.tolist() if rare_classes_mask.any() else []
    
    if len(rare_classes) > 0:
        print(f"\n{'='*80}")
        print("ОБРАБОТКА РЕДКИХ КЛАССОВ (ПО ОБЪЕДИНЕННЫМ МЕТКАМ)")
        print(f"{'='*80}")
        print(f"Найдены объединенные классы с менее чем 2 примерами: {rare_classes}")
        
        # ИЗМЕНЕНИЕ: Теперь УДАЛЯЕМ редкие классы вместо замены на 'other'
        mask = df_clean['merged_label'].isin(rare_classes)
        total_rare = mask.sum()
        
        print(f"Всего примеров в редких классах: {total_rare}")
        print(f"Редкие классы будут УДАЛЕНЫ из датасета (не заменены на 'other')")
        
        df_clean = df_clean[~mask].copy()
        
        print(f"Записей после удаления редких классов: {len(df_clean)}")
        print(f"\nПРИМЕЧАНИЕ: Класс 'other' будет предсказываться через threshold на уверенность модели")
        print(f"См. модуль predict_with_threshold.py для использования")
        
        # Обновляем статистику после удаления
        updated_counts, updated_total = print_language_statistics(df_clean, 'merged_label', "После удаления редких классов")
    else:
        print(f"\nРедких классов не найдено, пропускаем обработку")
        updated_counts = merged_counts
        updated_total = merged_total
    
    # Разделяем данные с сохранением распределения языков (stratified split)
    # Используем объединенные метки для стратификации
    print(f"\n{'='*80}")
    print(f"РАЗДЕЛЕНИЕ ДАННЫХ ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)")
    print(f"{'='*80}")
    print("Стратификация выполняется по объединенным меткам, но исходные метки сохраняются в данных")
    
    X = df_clean.drop(columns=[language_column, 'original_label', 'merged_label'])
    y_stratify = df_clean['merged_label']  # Для стратификации используем объединенные метки
    y_original = df_clean[language_column]  # Сохраняем исходные метки
    
    # Используем stratified split для сохранения распределения языков (по объединенным меткам)
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X, y_original, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_stratify  # Стратификация по объединенным меткам
    )
    
    # Собираем обратно в DataFrame с исходными метками
    train_df = X_train.copy()
    train_df[language_column] = y_train_orig.values
    
    test_df = X_test.copy()
    test_df[language_column] = y_test_orig.values
    
    # Добавляем столбец с объединенными метками в test для удобства
    # (основной столбец остается с исходными метками для обучения)
    test_df['merged_label'] = merge_labels_in_series(test_df[language_column])
    
    print(f"Train выборка: {len(train_df)} записей ({len(train_df)/len(df_clean)*100:.2f}%)")
    print(f"Test выборка: {len(test_df)} записей ({len(test_df)/len(df_clean)*100:.2f}%)")
    
    # Статистика train выборки (по исходным меткам)
    print(f"\n{'='*80}")
    print("СТАТИСТИКА TRAIN ВЫБОРКИ (ИСХОДНЫЕ МЕТКИ)")
    print(f"{'='*80}")
    train_counts_orig, train_total = print_language_statistics(train_df, language_column, "Train")
    
    # Статистика test выборки (по исходным меткам)
    print(f"\n{'='*80}")
    print("СТАТИСТИКА TEST ВЫБОРКИ (ИСХОДНЫЕ МЕТКИ)")
    print(f"{'='*80}")
    test_counts_orig, test_total = print_language_statistics(test_df, language_column, "Test")
    
    # Создаем объединенные метки для train и test для сравнения
    train_df_temp = train_df.copy()
    train_df_temp['merged_label'] = merge_labels_in_series(train_df[language_column])
    train_counts_merged, _ = print_language_statistics(train_df_temp, 'merged_label', "Train (объединенные)")
    
    test_df_temp = test_df.copy()
    test_df_temp['merged_label'] = merge_labels_in_series(test_df[language_column])
    test_counts_merged, _ = print_language_statistics(test_df_temp, 'merged_label', "Test (объединенные)")
    
    # Сравнение распределений (по объединенным меткам)
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ЯЗЫКОВ (ОБЪЕДИНЕННЫЕ МЕТКИ)")
    print(f"{'='*80}")
    print(f"{'Язык':<30} {'Исходный %':<15} {'Train %':<15} {'Test %':<15} {'Разница':<15}")
    print("-" * 90)
    
    all_languages = set(updated_counts.index) | set(train_counts_merged.index) | set(test_counts_merged.index)
    
    for lang in sorted(all_languages, key=lambda x: updated_counts.get(x, 0), reverse=True):
        orig_pct = (updated_counts.get(lang, 0) / updated_total) * 100
        train_pct = (train_counts_merged.get(lang, 0) / train_total) * 100
        test_pct = (test_counts_merged.get(lang, 0) / test_total) * 100
        diff = abs(train_pct - test_pct)
        
        print(f"{str(lang):<30} {orig_pct:>13.2f}% {train_pct:>13.2f}% {test_pct:>13.2f}% {diff:>13.2f}%")
    
    # Сохраняем разделенные данные
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Сохраняем в том же формате, что и исходный файл
    if file_path.endswith('.csv'):
        train_file = os.path.join(output_dir, 'train.csv')
        test_file = os.path.join(output_dir, 'test.csv')
        train_df.to_csv(train_file, sep=';', index=False, encoding='utf-8')
        test_df.to_csv(test_file, sep=';', index=False, encoding='utf-8')
    else:
        train_file = os.path.join(output_dir, 'train.xlsx')
        test_file = os.path.join(output_dir, 'test.xlsx')
        train_df.to_excel(train_file, index=False)
        test_df.to_excel(test_file, index=False)
    
    print(f"\n{'='*80}")
    print("СОХРАНЕНИЕ ДАННЫХ")
    print(f"{'='*80}")
    print(f"Train выборка сохранена: {train_file}")
    print(f"Test выборка сохранена: {test_file}")
    
    # Сохраняем статистику в файл
    stats_file = os.path.join(output_dir, 'split_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА РАЗДЕЛЕНИЯ ДАТАСЕТА\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Исходный датасет: {len(df_clean)} записей\n")
        f.write(f"Train выборка: {len(train_df)} записей ({len(train_df)/len(df_clean)*100:.2f}%)\n")
        f.write(f"Test выборка: {len(test_df)} записей ({len(test_df)/len(df_clean)*100:.2f}%)\n")
        f.write(f"Random state: {random_state}\n")
        if rare_classes:
            f.write(f"Редкие классы заменены на 'other': {', '.join(rare_classes)}\n")
        f.write("\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - ИСХОДНЫЙ ДАТАСЕТ (ИСХОДНЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in original_counts.items():
            percentage = (count / original_total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - ИСХОДНЫЙ ДАТАСЕТ (ОБЪЕДИНЕННЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in merged_counts.items():
            percentage = (count / merged_total) * 100
            marker = " <-- заменено на 'other'" if lang in rare_classes else ""
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%{marker}\n")
        
        if rare_classes:
            f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - ПОСЛЕ ОБРАБОТКИ РЕДКИХ КЛАССОВ (ОБЪЕДИНЕННЫЕ)\n")
            f.write("-" * 80 + "\n")
            for lang, count in updated_counts.items():
                percentage = (count / updated_total) * 100
                f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - TRAIN (ИСХОДНЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in train_counts_orig.items():
            percentage = (count / train_total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - TRAIN (ОБЪЕДИНЕННЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in train_counts_merged.items():
            percentage = (count / train_total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - TEST (ИСХОДНЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in test_counts_orig.items():
            percentage = (count / test_total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ - TEST (ОБЪЕДИНЕННЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        for lang, count in test_counts_merged.items():
            percentage = (count / test_total) * 100
            f.write(f"{str(lang):<30} {count:<15} {percentage:.2f}%\n")
        
        f.write("\nСРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ (ОБЪЕДИНЕННЫЕ МЕТКИ)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Язык':<30} {'Исходный %':<15} {'Train %':<15} {'Test %':<15} {'Разница':<15}\n")
        f.write("-" * 90 + "\n")
        for lang in sorted(all_languages, key=lambda x: updated_counts.get(x, 0), reverse=True):
            orig_pct = (updated_counts.get(lang, 0) / updated_total) * 100
            train_pct = (train_counts_merged.get(lang, 0) / train_total) * 100
            test_pct = (test_counts_merged.get(lang, 0) / test_total) * 100
            diff = abs(train_pct - test_pct)
            f.write(f"{str(lang):<30} {orig_pct:>13.2f}% {train_pct:>13.2f}% {test_pct:>13.2f}% {diff:>13.2f}%\n")
    
    print(f"Статистика сохранена: {stats_file}")
    print(f"{'='*80}")
    
    return train_df, test_df

if __name__ == "__main__":
    split_dataset()

