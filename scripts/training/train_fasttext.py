"""
Скрипт для обучения модели fastText для классификации языков
"""
import pandas as pd
import fasttext
import os
from scripts.data_processing.preprocess_text import preprocess_text
from scripts.utils.label_mapping import merge_label


def prepare_fasttext_data(df, text_column='request_text', label_column='result', output_file='train.txt'):
    """
    Подготавливает данные в формате fastText: __label__<class> <text>
    """
    print(f"Подготовка данных для fastText...")
    
    # Фильтруем строки с валидными данными
    df_valid = df.dropna(subset=[text_column, label_column]).copy()
    
    # Применяем предобработку к тексту (если еще не обработано)
    # Проверяем, нужно ли предобрабатывать (если текст содержит заглавные буквы или спецсимволы)
    sample_text = str(df_valid[text_column].iloc[0]) if len(df_valid) > 0 else ""
    needs_preprocessing = any(c.isupper() for c in sample_text[:100]) or any(c in sample_text[:100] for c in ['<', '>', '&', '\n', '\t'])
    
    if needs_preprocessing:
        print("Применяется предобработка текста...")
        df_valid[text_column] = df_valid[text_column].apply(preprocess_text)
    else:
        print("Текст уже предобработан, пропускаем предобработку")
    
    # Удаляем пустые тексты после предобработки
    df_valid = df_valid[df_valid[text_column].astype(str).str.strip() != ''].copy()
    
    print(f"Валидных записей: {len(df_valid)}")
    
    # Формируем данные в формате fastText
    fasttext_lines = []
    skipped = 0
    for idx, row in df_valid.iterrows():
        # Извлекаем метку из столбца result
        label_raw = row[label_column]
        if pd.isna(label_raw):
            skipped += 1
            continue
        
        label = str(label_raw).strip()
        # Удаляем лишние пробелы и символы из метки
        label = ' '.join(label.split())  # Нормализуем пробелы
        
        # Извлекаем текст из столбца request_text
        text_raw = row[text_column]
        if pd.isna(text_raw):
            skipped += 1
            continue
        
        text = str(text_raw).strip()
        
        # Проверяем, что метка и текст не пустые
        if not label or not text:
            skipped += 1
            continue
        
        # Формат fastText: __label__<class> <text>
        if ' ' in label:
            label = label.replace(' ', '_')
        
        line = f"__label__{label} {text}\n"
        fasttext_lines.append(line)
    
    if skipped > 0:
        print(f"Пропущено записей с пустыми метками или текстом: {skipped}")
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fasttext_lines)
    
    print(f"Данные сохранены в: {output_file}")
    print(f"Всего строк: {len(fasttext_lines)}")
    
    return output_file, len(fasttext_lines)


def read_data_file(file_path):
    """
    Читает данные из файла (CSV или Excel)
    """
    if file_path.endswith('.csv'):
        try:
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
            
            return df
        except Exception as e:
            print(f"Ошибка при чтении CSV файла {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"Ошибка при чтении Excel файла {file_path}: {e}")
            return None

def train_fasttext_model(
    train_file='output/train_preprocessed.csv',
    test_file='output/test.csv',
    text_column='request_text',
    label_column='result',
    model_output='output/lang_detection_model.bin',
    find_optimal_epoch=True,  # НОВЫЙ ПАРАМЕТР
    validation_size=0.15,
    epochs_to_try=None
):
    """
    Обучает модель fastText для классификации языков
    """
    print("=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛИ FASTTEXT")
    print("=" * 80)
    
    # Проверяем наличие предобработанных данных, иначе используем обычные
    if not os.path.exists(train_file):
        # Пробуем найти файл в разных форматах
        train_file_csv = 'output/train.csv'
        train_file_xlsx = 'output/train.xlsx'
        if os.path.exists(train_file_csv):
            train_file = train_file_csv
            print(f"Предобработанный файл не найден, используем: {train_file}")
        elif os.path.exists(train_file_xlsx):
            train_file = train_file_xlsx
            print(f"Предобработанный файл не найден, используем: {train_file}")
        else:
            print(f"ОШИБКА: Не найден файл train данных!")
            return None
    
    print(f"\nЗагрузка train данных: {train_file}")
    df_train = read_data_file(train_file)
    if df_train is None:
        return None
    
    print(f"Загружено train записей: {len(df_train)}")
    print(f"Столбцы: {list(df_train.columns)}")
    
    # Проверяем наличие нужных столбцов
    if text_column not in df_train.columns:
        print(f"ОШИБКА: Столбец '{text_column}' не найден в train данных!")
        print(f"Доступные столбцы: {list(df_train.columns)}")
        return None
    
    if label_column not in df_train.columns:
        print(f"ОШИБКА: Столбец '{label_column}' не найден в train данных!")
        print(f"Доступные столбцы: {list(df_train.columns)}")
        return None
    
    # Показываем примеры меток из train
    print(f"\nПримеры меток из столбца '{label_column}' (первые 10 уникальных):")
    unique_labels_train = df_train[label_column].dropna().unique()[:10]
    for lbl in unique_labels_train:
        print(f"  - {lbl}")
    
    print(f"\nЗагрузка test данных: {test_file}")
    # Пробуем найти файл в разных форматах
    if not os.path.exists(test_file):
        test_file_xlsx = 'output/test.xlsx'
        if os.path.exists(test_file_xlsx):
            test_file = test_file_xlsx
    
    df_test = read_data_file(test_file)
    if df_test is None:
        return None
    
    print(f"Загружено test записей: {len(df_test)}")
    print(f"Столбцы: {list(df_test.columns)}")
    
    # Проверяем наличие нужных столбцов в test
    if text_column not in df_test.columns:
        print(f"ОШИБКА: Столбец '{text_column}' не найден в test данных!")
        print(f"Доступные столбцы: {list(df_test.columns)}")
        return None
    
    if label_column not in df_test.columns:
        print(f"ОШИБКА: Столбец '{label_column}' не найден в test данных!")
        print(f"Доступные столбцы: {list(df_test.columns)}")
        return None
    
    # Показываем примеры меток из test
    print(f"\nПримеры меток из столбца '{label_column}' (первые 10 уникальных):")
    unique_labels_test = df_test[label_column].dropna().unique()[:10]
    for lbl in unique_labels_test:
        print(f"  - {lbl}")
    
    # Подготавливаем данные в формате fastText
    print(f"\n{'='*80}")
    print("ПОДГОТОВКА ДАННЫХ")
    print(f"{'='*80}")
    
    train_txt, train_count = prepare_fasttext_data(
        df_train, text_column, label_column, 'output/train_fasttext.txt'
    )
    
    test_txt, test_count = prepare_fasttext_data(
        df_test, text_column, label_column, 'output/test_fasttext.txt'
    )
    
    optimal_epoch = 40
    
    if find_optimal_epoch:
        from scripts.training.find_optimal_epochs import find_optimal_epochs
        
        if epochs_to_try is None:
            epochs_to_try = [10, 20, 30, 40, 50, 60, 70]
        
        try:
            results, optimal_epoch = find_optimal_epochs(
                train_file=train_file,
                val_file="output/val.csv",
                text_col=text_column,
                label_col=label_column,
            )
            
            print(f"\nНайдено оптимальное количество эпох: {optimal_epoch}")
            print(f"  Будет использовано для финального обучения на полном train")
        except Exception as e:
            print(f"Ошибка при поиске оптимальных эпох: {e}")
            print(f"  Используется значение по умолчанию: epoch={optimal_epoch}")
    else:
        print(f"\n  Поиск оптимальных эпох отключен, используется epoch={optimal_epoch}")
    
    # Обучаем модель
    print(f"\n{'='*80}")
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print(f"{'='*80}")
    
    print(f"Обучаем модель на {train_count} примерах...")
    
    # Параметры обучения (УЛУЧШЕННЫЕ для tr/az)
    print(f"\n{'='*80}")
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ПОЛНОМ TRAIN")
    print(f"{'='*80}")
    print(f"""
Параметры модели (улучшенные для различения похожих языков):
  - Эпох: {optimal_epoch} (определено через validation)
  - Word N-grams: 2 - биграммы слов
  - Размерность: 150 - больше признаков
  - Subword: 3-6 символов - морфология
  - Min count: 2 - игнорировать редкие слова
  - Loss: softmax - стандарт для multiclass
""")
    
    model = fasttext.train_supervised(
        input=train_txt,
        lr=0.1,
        epoch=optimal_epoch,
        wordNgrams=2,
        dim=150,
        minn=3,
        maxn=6,
        minCount=2,
        loss='softmax',
        seed=42
    )
    
    print("Обучение завершено!")
    
    # Сохраняем модель
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_model(model_output)
    print(f"Модель сохранена: {model_output}")
    
    # Оценка на тестовом наборе
    print(f"\n{'='*80}")
    print("ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
    print(f"{'='*80}")
    
    # Загружаем тестовые данные для оценки
    test_data = []
    invalid_lines = 0
    with open(test_txt, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Проверяем, что строка начинается с __label__
            if not line.startswith('__label__'):
                invalid_lines += 1
                if invalid_lines <= 5:  # Показываем первые 5 ошибок
                    print(f"ВНИМАНИЕ: Строка {line_num} не начинается с __label__: {line[:100]}")
                continue
            
            # Извлекаем метку и текст
            # Формат: __label__<label> <text>
            parts = line.split(' ', 1)
            if len(parts) != 2:
                invalid_lines += 1
                if invalid_lines <= 5:
                    print(f"ВНИМАНИЕ: Строка {line_num} имеет неправильный формат: {line[:100]}")
                continue
            
            label = parts[0].replace('__label__', '').strip()
            text = parts[1].strip()
            
            if not label or not text:
                invalid_lines += 1
                if invalid_lines <= 5:
                    print(f"ВНИМАНИЕ: Строка {line_num} имеет пустую метку или текст")
                continue
            
            test_data.append((label, text))
    
    if invalid_lines > 0:
        print(f"ВНИМАНИЕ: Найдено {invalid_lines} некорректных строк в test_fasttext.txt")
    
    print(f"Тестовых примеров: {len(test_data)}")
    
    # Получаем список всех валидных меток из обучающего набора
    valid_labels = set()
    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('__label__'):
                continue
            
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label = parts[0].replace('__label__', '').strip()
                if label:  # Проверяем, что метка не пустая
                    valid_labels.add(label)
    
    print(f"Найдено валидных меток в обучающем наборе: {len(valid_labels)}")
    print(f"Валидные метки: {sorted(valid_labels)}")
    
    correct = 0
    total = len(test_data)
    invalid_predictions = 0
    
    # Для метрик: собираем confusion matrix данные
    true_labels = []
    predicted_labels = []
    
    # Для подсчета метрик по классам (по объединенным меткам)
    true_positives = {}  # TP[class] = количество правильно предсказанных для класса
    false_positives = {}  # FP[class] = количество неправильно предсказанных как этот класс
    false_negatives = {}  # FN[class] = количество неправильно пропущенных для класса
    all_labels = set()
    
    print("\nВыполняется оценка...")
    print("ВАЖНО: Модель предсказывает исходные метки, но оценка выполняется по объединенным меткам")
    
    # Счетчик для отладки
    debug_count = 0
    max_debug = 10
    
    for true_label_orig, text in test_data:
        # Получаем предсказание (top-1) - модель вернет исходную метку (uz_lat, uz_kir и т.д.)
        try:
            predictions = model.predict(text, k=1)
            # predictions[0] - это список меток, predictions[1] - это список вероятностей
            if len(predictions[0]) > 0:
                predicted_label_orig = predictions[0][0].replace('__label__', '').strip()
            else:
                predicted_label_orig = 'other'
        except Exception as e:
            if debug_count < max_debug:
                print(f"Ошибка при предсказании для текста '{text[:50]}...': {e}")
                debug_count += 1
            predicted_label_orig = 'other'
        
        # Валидация предсказанной метки - если она не валидна, используем 'other'
        if not predicted_label_orig or predicted_label_orig not in valid_labels:
            if invalid_predictions < max_debug:
                print(f"Невалидное предсказание: '{predicted_label_orig}' (заменено на 'other')")
                print(f"  Текст: {text[:100]}...")
                print(f"  Истинная метка: {true_label_orig}")
            invalid_predictions += 1
            predicted_label_orig = 'other'
        
        # Объединяем метки для сравнения
        true_label_merged = merge_label(true_label_orig)
        predicted_label_merged = merge_label(predicted_label_orig)
        
        true_labels.append(true_label_merged)
        predicted_labels.append(predicted_label_merged)
        all_labels.add(true_label_merged)
        all_labels.add(predicted_label_merged)
        
        # Инициализация счетчиков (по объединенным меткам)
        if true_label_merged not in true_positives:
            true_positives[true_label_merged] = 0
            false_negatives[true_label_merged] = 0
        if predicted_label_merged not in false_positives:
            false_positives[predicted_label_merged] = 0
        
        # Подсчет TP, FP, FN (по объединенным меткам)
        if predicted_label_merged == true_label_merged:
            correct += 1
            true_positives[true_label_merged] += 1
        else:
            false_negatives[true_label_merged] += 1
            false_positives[predicted_label_merged] += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    if invalid_predictions > 0:
        print(f"\nВНИМАНИЕ: Найдено {invalid_predictions} невалидных предсказаний (заменены на 'other')")
    
    # Вычисление метрик по классам
    precision_by_class = {}
    recall_by_class = {}
    f1_by_class = {}
    
    for label in all_labels:
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_by_class[label] = precision
        recall_by_class[label] = recall
        f1_by_class[label] = f1
    
    # Macro average (среднее по классам)
    macro_precision = sum(precision_by_class.values()) / len(precision_by_class) if precision_by_class else 0
    macro_recall = sum(recall_by_class.values()) / len(recall_by_class) if recall_by_class else 0
    macro_f1 = sum(f1_by_class.values()) / len(f1_by_class) if f1_by_class else 0
    
    # Micro average (глобальные метрики)
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Вывод результатов
    print(f"\n{'='*80}")
    print("ОБЩИЕ МЕТРИКИ")
    print(f"{'='*80}")
    print(f"Accuracy (точность): {accuracy:.2f}% ({correct}/{total})")
    print(f"\nMacro Average:")
    print(f"  Precision: {macro_precision*100:.2f}%")
    print(f"  Recall: {macro_recall*100:.2f}%")
    print(f"  F1-score: {macro_f1*100:.2f}%")
    print(f"\nMicro Average:")
    print(f"  Precision: {micro_precision*100:.2f}%")
    print(f"  Recall: {micro_recall*100:.2f}%")
    print(f"  F1-score: {micro_f1*100:.2f}%")
    
    # Метрики по классам
    print(f"\n{'='*80}")
    print("МЕТРИКИ ПО КЛАССАМ")
    print(f"{'='*80}")
    print(f"{'Язык':<30} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 100)
    
    for label in sorted(all_labels):
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)
        prec = precision_by_class[label]
        rec = recall_by_class[label]
        f1 = f1_by_class[label]
        
        print(f"{label:<30} {prec*100:>10.2f}% {rec*100:>10.2f}% {f1*100:>10.2f}% {tp:>7} {fp:>7} {fn:>7}")
    
    # Confusion Matrix (упрощенная версия - топ ошибок)
    print(f"\n{'='*80}")
    print("ТОП ОШИБОК КЛАССИФИКАЦИИ")
    print(f"{'='*80}")
    
    error_counts = {}
    for true_label, pred_label in zip(true_labels, predicted_labels):
        if true_label != pred_label:
            error_key = f"{true_label} -> {pred_label}"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
    
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"{'Истинный класс':<30} {'Предсказанный класс':<30} {'Количество':<15}")
    print("-" * 75)
    for error, count in top_errors:
        parts = error.split(' -> ')
        if len(parts) == 2:
            print(f"{parts[0]:<30} {parts[1]:<30} {count:<15}")
    
    # Сохраняем статистику
    stats_file = os.path.join(output_dir, 'model_evaluation.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("ОЦЕНКА МОДЕЛИ FASTTEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Модель: {model_output}\n")
        f.write(f"Train файл: {train_file}\n")
        f.write(f"Test файл: {test_file}\n")
        f.write(f"Train примеров: {train_count}\n")
        f.write(f"Test примеров: {test_count}\n")
        if invalid_predictions > 0:
            f.write(f"Невалидных предсказаний (заменены на 'other'): {invalid_predictions}\n")
        f.write("\n")
        
        f.write("ПАРАМЕТРЫ ОБУЧЕНИЯ\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning rate: 0.1\n")
        f.write(f"Epochs: 25\n")
        f.write(f"Word n-grams: 2\n")
        f.write(f"Dimension: 100\n")
        f.write(f"Loss: one-vs-all\n")
        f.write(f"\nВАЖНО: Модель обучалась на исходных метках (uz_lat, uz_kir, ur_lat, ur_arab, ne, ne_lat)\n")
        f.write(f"Оценка выполняется по объединенным меткам (uz, ur, ne)\n\n")
        
        f.write("ОБЩИЕ МЕТРИКИ\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy (точность): {accuracy:.2f}% ({correct}/{total})\n\n")
        
        f.write("MACRO AVERAGE (среднее по классам)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {macro_precision*100:.2f}%\n")
        f.write(f"Recall: {macro_recall*100:.2f}%\n")
        f.write(f"F1-score: {macro_f1*100:.2f}%\n\n")
        
        f.write("MICRO AVERAGE (глобальные метрики)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {micro_precision*100:.2f}%\n")
        f.write(f"Recall: {micro_recall*100:.2f}%\n")
        f.write(f"F1-score: {micro_f1*100:.2f}%\n\n")
        
        f.write("МЕТРИКИ ПО КЛАССАМ\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Язык':<30} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'TP':<8} {'FP':<8} {'FN':<8}\n")
        f.write("-" * 100 + "\n")
        for label in sorted(all_labels):
            tp = true_positives.get(label, 0)
            fp = false_positives.get(label, 0)
            fn = false_negatives.get(label, 0)
            prec = precision_by_class[label]
            rec = recall_by_class[label]
            f1 = f1_by_class[label]
            
            f.write(f"{label:<30} {prec*100:>10.2f}% {rec*100:>10.2f}% {f1*100:>10.2f}% {tp:>7} {fp:>7} {fn:>7}\n")
        
        f.write("\nТОП ОШИБОК КЛАССИФИКАЦИИ\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Истинный класс':<30} {'Предсказанный класс':<30} {'Количество':<15}\n")
        f.write("-" * 75 + "\n")
        for error, count in top_errors:
            parts = error.split(' -> ')
            if len(parts) == 2:
                f.write(f"{parts[0]:<30} {parts[1]:<30} {count:<15}\n")
    
    print(f"\nСтатистика сохранена: {stats_file}")
    print(f"{'='*80}")
    
    return model, accuracy


if __name__ == "__main__":
    train_fasttext_model()

