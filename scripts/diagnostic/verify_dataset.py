"""
Скрипт для проверки конфигурации датасета
"""
import pandas as pd
import os

def verify_dataset():
    """
    Проверяет структуру датасета и настройки
    """
    file_path = 'lang_detection_hackathon.csv'
    
    print("=" * 80)
    print("ПРОВЕРКА КОНФИГУРАЦИИ ДАТАСЕТА")
    print("=" * 80)
    
    # Читаем CSV
    print(f"\n1. Чтение CSV файла: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            header_line = f.readline().strip()
            headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
        
        df = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding='utf-8', on_bad_lines='skip', names=headers, engine='python')
        print(f"   Загружено строк: {len(df)}")
        print(f"   Столбцы: {list(df.columns)}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")
        return
    
    # Проверяем необходимые столбцы
    print(f"\n2. Проверка необходимых столбцов:")
    required_columns = {
        'request_text': 'Столбец с текстом для обучения',
        'result': 'Столбец с метками языков'
    }
    
    all_ok = True
    for col, description in required_columns.items():
        if col in df.columns:
            print(f"   {col}: найден ({description})")
        else:
            print(f"   {col}: НЕ НАЙДЕН! ({description})")
            all_ok = False
    
    if not all_ok:
        print("\n   ОШИБКА: Не все необходимые столбцы найдены!")
        return
    
    # Анализ данных
    print(f"\n3. Анализ данных:")
    
    # Анализ столбца result
    if 'result' in df.columns:
        total = len(df)
        null_count = df['result'].isna().sum()
        empty_count = (df['result'].astype(str).str.strip() == '').sum()
        valid_count = total - null_count - empty_count
        
        print(f"   Столбец 'result':")
        print(f"     Всего строк: {total}")
        print(f"     Пустых (NaN): {null_count} ({null_count/total*100:.2f}%)")
        print(f"     Пустых строк: {empty_count} ({empty_count/total*100:.2f}%)")
        print(f"     Валидных меток: {valid_count} ({valid_count/total*100:.2f}%)")
        
        # Показываем уникальные метки
        if valid_count > 0:
            valid_labels = df[df['result'].notna() & (df['result'].astype(str).str.strip() != '')]['result']
            unique_labels = valid_labels.unique()
            print(f"     Уникальных языков: {len(unique_labels)}")
            print(f"     Языки: {sorted(unique_labels)}")
    
    # Анализ столбца request_text
    if 'request_text' in df.columns:
        total = len(df)
        null_count = df['request_text'].isna().sum()
        empty_count = (df['request_text'].astype(str).str.strip() == '').sum()
        valid_count = total - null_count - empty_count
        
        print(f"\n   Столбец 'request_text':")
        print(f"     Всего строк: {total}")
        print(f"     Пустых (NaN): {null_count} ({null_count/total*100:.2f}%)")
        print(f"     Пустых строк: {empty_count} ({empty_count/total*100:.2f}%)")
        print(f"     Валидных текстов: {valid_count} ({valid_count/total*100:.2f}%)")
    
    # Проверка комбинации
    if 'result' in df.columns and 'request_text' in df.columns:
        print(f"\n4. Комбинация 'result' и 'request_text':")
        total = len(df)
        
        mask_valid = (
            df['result'].notna() & 
            (df['result'].astype(str).str.strip() != '') &
            df['request_text'].notna() & 
            (df['request_text'].astype(str).str.strip() != '')
        )
        valid_both = mask_valid.sum()
        
        print(f"   Строк с валидными result И request_text: {valid_both} ({valid_both/total*100:.2f}%)")
        print(f"   Будет потеряно при фильтрации: {total - valid_both} ({(total - valid_both)/total*100:.2f}%)")
        
        if valid_both > 0:
            df_valid = df[mask_valid].copy()
            print(f"\n5. Распределение по языкам (только валидные данные):")
            lang_counts = df_valid['result'].value_counts()
            print(f"   Всего валидных записей: {len(df_valid)}")
            print(f"   Уникальных языков: {df_valid['result'].nunique()}")
            print(f"\n   Распределение по языкам:")
            for lang, count in lang_counts.items():
                print(f"     {lang}: {count} ({count/len(df_valid)*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)

if __name__ == "__main__":
    verify_dataset()

