"""
Модуль для поиска оптимального количества эпох через validation set
"""
import fasttext
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scripts.utils.label_mapping import merge_label


def find_optimal_epochs(
    train_file='output/train_preprocessed.csv',
    text_column='request_text',
    label_column='result',
    validation_size=0.15,
    epochs_to_try=[10, 20, 30, 40, 50, 60, 70],
    min_delta=0.003,
    random_state=42
):
    """
    Находит оптимальное количество эпох через validation set.
    Останавливается, когда прирост accuracy становится меньше min_delta.
    
    Args:
        train_file: путь к train файлу
        text_column: столбец с текстом
        label_column: столбец с метками
        validation_size: размер validation set (0.15 = 15%)
        epochs_to_try: список эпох для проверки
        min_delta: минимальный прирост accuracy для продолжения (0.003 = 0.3%)
        random_state: random seed
    
    Returns:
        dict: результаты для каждого epoch
        int: оптимальное количество эпох
    """
    print("=" * 80)
    print("ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА ЭПОХ")
    print("=" * 80)
    
    print(f"\nЗагрузка данных: {train_file}")
    
    if train_file.endswith('.csv'):
        with open(train_file, 'r', encoding='utf-8', errors='replace') as f:
            header_line = f.readline().strip()
            headers = [h.strip('\ufeff') for h in header_line.split(';')]
        
        df = pd.read_csv(train_file, sep=';', skiprows=[0, 1], encoding='utf-8', 
                        on_bad_lines='skip', names=headers, engine='python')
    else:
        df = pd.read_excel(train_file)
    
    print(f"Загружено записей: {len(df)}")
    
    print(f"\nРазделение на train/validation ({int((1-validation_size)*100)}% / {int(validation_size*100)}%)")
    
    df_valid = df.dropna(subset=[text_column, label_column]).copy()
    print(f"Валидных записей: {len(df_valid)}")
    
    df_valid['merged_label'] = df_valid[label_column].apply(merge_label)
    
    df_train, df_val = train_test_split(
        df_valid,
        test_size=validation_size,
        stratify=df_valid['merged_label'],
        random_state=random_state
    )
    
    print(f"Train: {len(df_train)} записей")
    print(f"Validation: {len(df_val)} записей")
    
    print("\nПодготовка данных для fastText...")
    
    train_txt = 'output/train_for_validation.txt'
    val_data = []
    
    with open(train_txt, 'w', encoding='utf-8') as f:
        for _, row in df_train.iterrows():
            label = str(row[label_column]).strip().replace(' ', '_')
            text = str(row[text_column]).strip()
            if text and label:
                f.write(f"__label__{label} {text}\n")
    
    for _, row in df_val.iterrows():
        true_label = merge_label(str(row[label_column]))
        text = str(row[text_column]).strip()
        if text and true_label:
            val_data.append((text, true_label))
    
    print(f"Train примеров: {len(df_train)}")
    print(f"Validation примеров: {len(val_data)}")
    
    print(f"\nОбучение моделей с разными количествами эпох")
    print("-" * 80)
    
    results = {}
    best_epoch = epochs_to_try[0]
    best_accuracy = 0
    prev_accuracy = 0
    early_stopped = False
    
    base_params = {
        'lr': 0.1,
        'wordNgrams': 2,
        'dim': 150,
        'minn': 3,
        'maxn': 6,
        'minCount': 2,
        'loss': 'softmax',
        'seed': 42,
        'verbose': 0
    }
    
    for i, epoch in enumerate(epochs_to_try):
        print(f"\nEpoch = {epoch}:")
        
        model = fasttext.train_supervised(
            input=train_txt,
            epoch=epoch,
            **base_params
        )
        
        correct = 0
        for text, true_label in val_data:
            predictions = model.predict(text, k=1)
            predicted_label_orig = predictions[0][0].replace('__label__', '')
            predicted_label = merge_label(predicted_label_orig)
            
            if predicted_label == true_label:
                correct += 1
        
        accuracy = correct / len(val_data)
        
        results[epoch] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(val_data)
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            status = "NEW BEST"
        else:
            status = ""
        
        improvement = accuracy - prev_accuracy if i > 0 else accuracy
        print(f"  Validation Accuracy: {accuracy:.2%} ({correct}/{len(val_data)}) {status}")
        
        if i > 0 and improvement < min_delta:
            print(f"  Прирост {improvement:.4f} < {min_delta:.4f} - останавливаем поиск")
            early_stopped = True
            break
        
        prev_accuracy = accuracy
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    if early_stopped:
        print(f"\nОстановлено досрочно: прирост accuracy < {min_delta:.4f}")
    
    print(f"\n{'Epoch':<10} {'Accuracy':<15} {'Correct/Total':<20} {'Status':<10}")
    print("-" * 80)
    
    for epoch in results.keys():
        acc = results[epoch]['accuracy']
        correct = results[epoch]['correct']
        total = results[epoch]['total']
        status = "← BEST" if epoch == best_epoch else ""
        
        print(f"{epoch:<10} {acc:<15.2%} {correct}/{total:<15} {status}")
    
    print("\n" + "=" * 80)
    print(f"ОПТИМАЛЬНОЕ КОЛИЧЕСТВО ЭПОХ: {best_epoch}")
    print(f"Validation Accuracy: {best_accuracy:.2%}")
    print("=" * 80)
    
    print(f"\nГрафик Validation Accuracy:")
    print("-" * 80)
    
    max_bar_length = 50
    for epoch in results.keys():
        acc = results[epoch]['accuracy']
        bar_length = int(acc * max_bar_length)
        bar = "█" * bar_length
        marker = "◄" if epoch == best_epoch else ""
        print(f"Epoch {epoch:3d}: {bar} {acc:.2%} {marker}")
    
    print("-" * 80)
    
    print(f"\nАнализ:")
    print("-" * 80)
    
    tested_epochs = list(results.keys())
    accuracies = [results[epoch]['accuracy'] for epoch in tested_epochs]
    max_acc_idx = accuracies.index(max(accuracies))
    
    if early_stopped:
        print(f"Оптимум найден на epoch={best_epoch}")
        print(f"  Прирост accuracy стал меньше {min_delta:.4f} - дальнейшее обучение не эффективно")
    elif max_acc_idx < len(tested_epochs) - 1:
        print(f"Оптимум найден на epoch={best_epoch}")
        print(f"  После этого accuracy не улучшается")
        
        after_peak = accuracies[max_acc_idx+1:]
        if any(acc < best_accuracy - 0.01 for acc in after_peak):
            print(f"Accuracy падает после epoch={best_epoch} - переобучение!")
        else:
            print(f"Accuracy стабильна после пика")
    else:
        print(f"Лучший результат на последней проверенной эпохе (epoch={best_epoch})")
        print(f"  Рекомендация: увеличить диапазон epochs_to_try")
    
    print("=" * 80)
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_file = os.path.join(output_dir, 'epoch_validation_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ПОИСКА ОПТИМАЛЬНОГО КОЛИЧЕСТВА ЭПОХ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Параметры:\n")
        f.write(f"  Train размер: {len(df_train)}\n")
        f.write(f"  Validation размер: {len(val_data)}\n")
        f.write(f"  Validation ratio: {validation_size:.1%}\n\n")
        
        f.write(f"{'Epoch':<10} {'Accuracy':<15} {'Correct/Total':<20}\n")
        f.write("-" * 80 + "\n")
        
        for epoch in results.keys():
            acc = results[epoch]['accuracy']
            correct = results[epoch]['correct']
            total = results[epoch]['total']
            marker = " ← BEST" if epoch == best_epoch else ""
            f.write(f"{epoch:<10} {acc:<15.2%} {correct}/{total:<15}{marker}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"ОПТИМАЛЬНОЕ КОЛИЧЕСТВО ЭПОХ: {best_epoch}\n")
        f.write(f"Validation Accuracy: {best_accuracy:.2%}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nРезультаты сохранены: {results_file}")
    
    return results, best_epoch


if __name__ == "__main__":
    # Запуск поиска оптимального количества эпох
    results, best_epoch = find_optimal_epochs(
        train_file='output/train_preprocessed.csv',
        validation_size=0.15,
        epochs_to_try=[10, 20, 30, 40, 50],
        random_state=42
    )
    
    print(f"\nОптимальное количество эпох: {best_epoch}")
    print(f"  Используйте это значение для финального обучения")

