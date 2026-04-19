"""
Скрипт для предобработки текста: очистка, приведение к нижнему регистру, токенизация
"""
import pandas as pd
import re
import os
import unicodedata

# Паттерн для удаления emoji
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF"
    "\U00010000-\U0010FFFF"  # дополнительные символы Unicode
    "]+",
    flags=re.UNICODE,
)


def normalize_for_detection(text):
    """
    Лёгкая нормализация текста перед определением языка.
    Применяется ДО детектора скрипта и fastText.

    Выполняет:
    - Unicode NFKC нормализацию (приводит арабские/еврейские формы к базовым)
    - Удаление URL и email
    - Удаление телефонных номеров
    - Удаление emoji
    - Нормализацию пробелов

    НЕ делает lowercase и НЕ удаляет Unicode-буквы —
    детектор скрипта работает с оригинальными символами.
    """
    if not text:
        return ""
    text = str(text)

    # 1. NFKC нормализация: приводит презентационные формы к базовым
    text = unicodedata.normalize("NFKC", text)

    # 2. Удаляем URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)

    # 3. Удаляем email
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Удаляем телефонные номера (7+ цифр, возможно со знаком + и разделителями)
    text = re.sub(r"\+?[\d][\d\s\-\(\)\.]{5,}\d", " ", text)

    # 5. Удаляем emoji
    text = _EMOJI_RE.sub(" ", text)

    # 6. Нормализуем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_html_tags(text):
    """
    Удаляет HTML/XML теги из текста
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Удаляем HTML/XML теги
    text = re.sub(r'<[^>]+>', '', text)
    return text


def clean_text(text):
    """
    Очищает текст от специальных символов, оставляя только буквы, цифры и пробелы
    """
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Удаляем HTML/XML теги
    text = remove_html_tags(text)
    
    # Оставляем только буквы, цифры и пробелы (поддерживаем Unicode для разных языков)
    # Это сохранит буквы из разных алфавитов (кириллица, арабский, иврит и т.д.)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    
    # Нормализуем пробелы: заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    
    # Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def tokenize_text(text):
    """
    Разбивает текст на токены (слова)
    Использует простой токенизатор, аналогичный fastText
    """
    if pd.isna(text) or text == "":
        return []
    
    text = str(text)
    
    # fastText использует простую токенизацию: разбиение по пробелам
    # Также разделяем по некоторым знакам препинания, если они остались
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    
    return tokens


def preprocess_text(text):
    """
    Полная предобработка текста:
    1. Очистка от специальных символов и тегов
    2. Приведение к нижнему регистру
    3. Токенизация
    """
    if pd.isna(text):
        return ""
    
    # Очистка
    cleaned = clean_text(text)
    
    # Приведение к нижнему регистру
    cleaned = cleaned.lower()
    
    # Токенизация
    tokens = tokenize_text(cleaned)
    
    # Возвращаем токены как строку, разделенную пробелами (стандартный формат для fastText)
    return ' '.join(tokens)


def preprocess_dataset(
    train_file='output/train.csv',
    text_column='request_text',
    output_file='output/train_preprocessed.csv'
):
    """
    Предобрабатывает текст в train выборке
    """
    print("Загрузка train выборки...")
    
    # Определяем формат файла по расширению
    if train_file.endswith('.csv'):
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    # Читаем первую строку для получения заголовков
                    with open(train_file, 'r', encoding=encoding, errors='replace') as f:
                        header_line = f.readline().strip()
                        headers = header_line.split(';')
                    
                    # Читаем данные, пропуская первую строку (заголовки) и вторую (типы данных)
                    df = pd.read_csv(train_file, sep=';', skiprows=[0, 1], encoding=encoding, on_bad_lines='skip', names=headers)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if df is None:
                # Последняя попытка с обработкой ошибок
                with open(train_file, 'r', encoding='utf-8-sig', errors='replace') as f:
                    header_line = f.readline().strip()
                    headers = [h.strip('\ufeff') for h in header_line.split(';')]  # Убираем BOM
                df = pd.read_csv(train_file, sep=';', skiprows=[0, 1], encoding='utf-8', on_bad_lines='skip', names=headers, engine='python')
        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        try:
            df = pd.read_excel(train_file)
        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            return None
    
    print(f"Загружено записей: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")
    
    # Проверяем наличие столбца с текстом
    if text_column not in df.columns:
        print(f"\nОШИБКА: Столбец '{text_column}' не найден в датасете")
        print("Доступные столбцы:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        return None
    
    print(f"\nНайден столбец с текстом: '{text_column}'")
    
    # Статистика до предобработки
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ДО ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")
    
    original_texts = df[text_column].dropna()
    print(f"Записей с текстом: {len(original_texts)}")
    
    # Примеры текста до обработки
    print(f"\nПримеры текста ДО обработки:")
    for i in range(min(3, len(original_texts))):
        sample_text = str(original_texts.iloc[i])
        preview = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
        print(f"  {i+1}. {preview}")
    
    # Предобработка
    print(f"\n{'='*80}")
    print("ПРЕДОБРАБОТКА ТЕКСТА")
    print(f"{'='*80}")
    print("Выполняется очистка, приведение к нижнему регистру и токенизация...")
    
    df_processed = df.copy()
    df_processed[text_column] = df_processed[text_column].apply(preprocess_text)
    
    # Статистика после предобработки
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ПОСЛЕ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")
    
    processed_texts = df_processed[text_column].dropna()
    processed_texts = processed_texts[processed_texts != ""]
    print(f"Записей с обработанным текстом: {len(processed_texts)}")
    
    # Примеры текста после обработки
    print(f"\nПримеры текста ПОСЛЕ обработки:")
    for i in range(min(3, len(processed_texts))):
        sample_text = str(processed_texts.iloc[i])
        preview = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
        print(f"  {i+1}. {preview}")
    
    # Статистика по длине текстов
    text_lengths_before = original_texts.apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    text_lengths_after = processed_texts.apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    print(f"\nСтатистика по длине текстов:")
    print(f"  До обработки: средняя длина = {text_lengths_before.mean():.1f}, медиана = {text_lengths_before.median():.1f}")
    print(f"  После обработки: средняя длина = {text_lengths_after.mean():.1f}, медиана = {text_lengths_after.median():.1f}")
    
    # Сохраняем обработанные данные
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Сохраняем в том же формате, что и исходный файл
    if output_file.endswith('.csv'):
        df_processed.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    else:
        df_processed.to_excel(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("СОХРАНЕНИЕ ДАННЫХ")
    print(f"{'='*80}")
    print(f"Обработанный датасет сохранен: {output_file}")
    
    # Сохраняем статистику
    stats_file = os.path.join(output_dir, 'preprocessing_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА ПРЕДОБРАБОТКИ ТЕКСТА\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Исходный файл: {train_file}\n")
        f.write(f"Выходной файл: {output_file}\n")
        f.write(f"Обработано записей: {len(df_processed)}\n\n")
        
        f.write("СТАТИСТИКА ПО ДЛИНЕ ТЕКСТОВ\n")
        f.write("-" * 80 + "\n")
        f.write(f"До обработки:\n")
        f.write(f"  Средняя длина: {text_lengths_before.mean():.1f} символов\n")
        f.write(f"  Медианная длина: {text_lengths_before.median():.1f} символов\n")
        f.write(f"  Минимальная длина: {text_lengths_before.min()} символов\n")
        f.write(f"  Максимальная длина: {text_lengths_before.max()} символов\n\n")
        
        f.write(f"После обработки:\n")
        f.write(f"  Средняя длина: {text_lengths_after.mean():.1f} символов\n")
        f.write(f"  Медианная длина: {text_lengths_after.median():.1f} символов\n")
        f.write(f"  Минимальная длина: {text_lengths_after.min()} символов\n")
        f.write(f"  Максимальная длина: {text_lengths_after.max()} символов\n\n")
        
        f.write("ПРИМЕРЫ ПРЕДОБРАБОТКИ\n")
        f.write("-" * 80 + "\n")
        for i in range(min(5, len(original_texts))):
            original = str(original_texts.iloc[i])
            processed = str(processed_texts.iloc[i]) if i < len(processed_texts) else ""
            f.write(f"\nПример {i+1}:\n")
            f.write(f"  ДО:  {original[:200]}{'...' if len(original) > 200 else ''}\n")
            f.write(f"  ПОСЛЕ: {processed[:200]}{'...' if len(processed) > 200 else ''}\n")
    
    print(f"Статистика сохранена: {stats_file}")
    print(f"{'='*80}")
    
    return df_processed


if __name__ == "__main__":
    preprocess_dataset()


