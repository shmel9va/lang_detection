"""
Скрипт для проверки потери данных при обработке CSV
"""
import pandas as pd
import os

def check_data_loss():
    """
    Проверяет, сколько данных теряется на каждом этапе
    """
    file_path = 'lang_detection_hackathon.csv'
    
    print("=" * 80)
    print("ПРОВЕРКА ПОТЕРИ ДАННЫХ")
    print("=" * 80)
    
    # Читаем CSV
    print(f"\n1. Чтение CSV файла: {file_path}")
    try:
        # Пробуем разные кодировки
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'iso-8859-1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                # Читаем первую строку для получения заголовков
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    header_line = f.readline().strip()
                    headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
                
                # Читаем данные, пропуская первую строку (заголовки) и вторую (типы данных)
                df = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding=encoding, on_bad_lines='skip', names=headers, engine='python')
                used_encoding = encoding
                print(f"   Успешно загружено с кодировкой: {encoding}")
                break
            except Exception as e:
                continue
        
        if df is None:
            # Последняя попытка с обработкой ошибок
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                header_line = f.readline().strip()
                headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
            df = pd.read_csv(file_path, sep=';', skiprows=[0, 1], encoding='utf-8', on_bad_lines='skip', names=headers, engine='python')
            used_encoding = 'utf-8 (с заменой ошибок)'
            print(f"   Загружено с обработкой ошибок кодировки")
        
        print(f"   Загружено строк: {len(df)}")
        print(f"   Использована кодировка: {used_encoding}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n2. Анализ столбцов:")
    print(f"   Столбцы: {list(df.columns)}")
    
    # Проверяем столбец result
    if 'result' in df.columns:
        print(f"\n3. Анализ столбца 'result':")
        total = len(df)
        null_count = df['result'].isna().sum()
        empty_count = (df['result'].astype(str).str.strip() == '').sum()
        valid_count = total - null_count - empty_count
        
        print(f"   Всего строк: {total}")
        print(f"   Пустых (NaN): {null_count} ({null_count/total*100:.2f}%)")
        print(f"   Пустых строк: {empty_count} ({empty_count/total*100:.2f}%)")
        print(f"   Валидных меток: {valid_count} ({valid_count/total*100:.2f}%)")
        
        # Показываем примеры пустых строк
        if null_count > 0 or empty_count > 0:
            print(f"\n   Примеры строк с пустыми метками (первые 5):")
            mask = df['result'].isna() | (df['result'].astype(str).str.strip() == '')
            empty_rows = df[mask].head(5)
            for idx, row in empty_rows.iterrows():
                print(f"     Строка {idx}: result='{row['result']}', request_text='{str(row.get('request_text', ''))[:50]}...'")
    
    # Проверяем столбец request_text
    if 'request_text' in df.columns:
        print(f"\n4. Анализ столбца 'request_text':")
        total = len(df)
        null_count = df['request_text'].isna().sum()
        empty_count = (df['request_text'].astype(str).str.strip() == '').sum()
        valid_count = total - null_count - empty_count
        
        print(f"   Всего строк: {total}")
        print(f"   Пустых (NaN): {null_count} ({null_count/total*100:.2f}%)")
        print(f"   Пустых строк: {empty_count} ({empty_count/total*100:.2f}%)")
        print(f"   Валидных текстов: {valid_count} ({valid_count/total*100:.2f}%)")
    
    # Проверяем комбинацию
    if 'result' in df.columns and 'request_text' in df.columns:
        print(f"\n5. Анализ комбинации 'result' и 'request_text':")
        total = len(df)
        
        # Строки с валидными обоими полями
        mask_valid = (
            df['result'].notna() & 
            (df['result'].astype(str).str.strip() != '') &
            df['request_text'].notna() & 
            (df['request_text'].astype(str).str.strip() != '')
        )
        valid_both = mask_valid.sum()
        
        print(f"   Всего строк: {total}")
        print(f"   С валидными result И request_text: {valid_both} ({valid_both/total*100:.2f}%)")
        print(f"   Будет потеряно при фильтрации: {total - valid_both} ({(total - valid_both)/total*100:.2f}%)")
        
        # Показываем распределение по языкам для валидных данных
        if valid_both > 0:
            df_valid = df[mask_valid].copy()
            print(f"\n6. Распределение по языкам (только валидные данные):")
            lang_counts = df_valid['result'].value_counts()
            print(f"   Всего валидных записей: {len(df_valid)}")
            print(f"   Уникальных языков: {df_valid['result'].nunique()}")
            print(f"\n   Топ-20 языков:")
            for lang, count in lang_counts.head(20).items():
                print(f"     {lang}: {count} ({count/len(df_valid)*100:.2f}%)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_data_loss()

