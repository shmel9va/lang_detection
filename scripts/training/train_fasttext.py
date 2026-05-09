"""
Скрипт для обучения модели fastText для классификации языков.

Оценка проводится на val.csv. test.csv НЕ используется.
Обучает на ИСХОДНЫХ метках (hy_arm, hy_lat, ...), предсказывает их,
затем объединяет через merge_label для оценки.
Гиперпараметры подбираются автоматически через grid search (macro F1).
"""
import pandas as pd
import fasttext
import os
from scripts.data_processing.preprocess_text import preprocess_text
from scripts.utils.label_mapping import merge_label


def prepare_fasttext_data(df, text_column='request_text', label_column='result', output_file='train.txt'):
    print(f"Подготовка данных для fastText...")
    
    df_valid = df.dropna(subset=[text_column, label_column]).copy()
    df_valid[text_column] = df_valid[text_column].apply(preprocess_text)
    df_valid = df_valid[df_valid[text_column].astype(str).str.strip() != ''].copy()
    
    print(f"Валидных записей: {len(df_valid)}")
    
    fasttext_lines = []
    skipped = 0
    for idx, row in df_valid.iterrows():
        label_raw = row[label_column]
        if pd.isna(label_raw):
            skipped += 1
            continue
        
        label = str(label_raw).strip()
        label = ' '.join(label.split())
        
        text_raw = row[text_column]
        if pd.isna(text_raw):
            skipped += 1
            continue
        
        text = str(text_raw).strip()
        
        if not label or not text:
            skipped += 1
            continue
        
        if ' ' in label:
            label = label.replace(' ', '_')
        
        line = f"__label__{label} {text}\n"
        fasttext_lines.append(line)
    
    if skipped > 0:
        print(f"Пропущено записей с пустыми метками или текстом: {skipped}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fasttext_lines)
    
    print(f"Данные сохранены в: {output_file}")
    print(f"Всего строк: {len(fasttext_lines)}")
    
    return output_file, len(fasttext_lines)


def read_data_file(file_path):
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"Excel read error {file_path}: {e}")
            return None
    for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
        try:
            return pd.read_csv(file_path, sep=';', encoding=enc, on_bad_lines='skip')
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"CSV read error ({enc}) {file_path}: {e}")
            continue
    print(f"Cannot read file: {file_path}")
    return None


def train_fasttext_model(
    train_file='output/train.csv',
    text_column='request_text',
    label_column='result',
    model_output='output/lang_detection_model.bin',
):
    """
    Обучает модель fastText для классификации языков.
    Оценка ТОЛЬКО на val.csv. test.csv НЕ трогается.
    Гиперпараметры подбираются автоматически.
    """
    print("=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛИ FASTTEXT (комбинированная, исходные метки)")
    print("=" * 80)
    
    if not os.path.exists(train_file):
        print(f"ОШИБКА: Не найден файл train данных: {train_file}")
        return None
    
    print(f"\nЗагрузка train данных: {train_file}")
    df_train = read_data_file(train_file)
    if df_train is None:
        return None
    
    print(f"Загружено train записей: {len(df_train)}")
    
    if text_column not in df_train.columns:
        print(f"ОШИБКА: Столбец '{text_column}' не найден!")
        return None
    if label_column not in df_train.columns:
        print(f"ОШИБКА: Столбец '{label_column}' не найден!")
        return None

    print(f"\nУникальные метки ({label_column}):")
    for lbl in sorted(df_train[label_column].dropna().unique()):
        print(f"  - {lbl}")
    
    print(f"\n{'='*80}")
    print("ПОДГОТОВКА ДАННЫХ")
    print(f"{'='*80}")
    
    train_txt, train_count = prepare_fasttext_data(
        df_train, text_column, label_column, 'output/train_fasttext.txt'
    )
    
    val_file = 'output/val.csv'
    val_txt = 'output/val_fasttext.txt'
    val_count = 0
    if os.path.exists(val_file):
        df_val = read_data_file(val_file)
        if df_val is not None:
            val_txt, val_count = prepare_fasttext_data(
                df_val, text_column, label_column, val_txt
            )

    valid_labels = set()
    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('__label__'):
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                lbl = parts[0].replace('__label__', '').strip()
                if lbl:
                    valid_labels.add(lbl)
    print(f"Валидных меток: {len(valid_labels)}: {sorted(valid_labels)}")

    best_params = {
        "dim": 150, "minn": 3, "maxn": 6,
        "wordNgrams": 2, "epoch": 25,
    }
    try:
        from scripts.training.find_optimal_params import find_optimal_params
        bp, _ = find_optimal_params(
            train_file=train_file,
            val_file=val_file,
            text_col=text_column,
            label_col=label_column,
        )
        best_params = bp
        print(f"\nЛучшие параметры: {best_params}")
    except Exception as e:
        print(f"Ошибка при поиске параметров: {e}")
        print(f"  Используются параметры по умолчанию: {best_params}")

    lr = 0.1
    min_count = 2
    loss = 'softmax'
    dim = best_params["dim"]
    minn = best_params["minn"]
    maxn = best_params["maxn"]
    word_ngrams = best_params["wordNgrams"]
    optimal_epoch = best_params["epoch"]
    
    print(f"\n{'='*80}")
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ")
    print(f"{'='*80}")
    print(f"  Примеров: {train_count}")
    print(f"  lr={lr}  epoch={optimal_epoch}  wordNgrams={word_ngrams}  dim={dim}")
    print(f"  minn={minn}  maxn={maxn}  minCount={min_count}  loss={loss}")
    
    model = fasttext.train_supervised(
        input=train_txt,
        lr=lr,
        epoch=optimal_epoch,
        wordNgrams=word_ngrams,
        dim=dim,
        minn=minn,
        maxn=maxn,
        minCount=min_count,
        loss=loss,
        seed=42,
    )
    
    print("Обучение завершено!")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(model_output)
    print(f"Модель сохранена: {model_output}")
    
    accuracy = 0.0
    if val_count > 0:
        print(f"\n{'='*80}")
        print("ОЦЕНКА НА VALIDATION")
        print(f"{'='*80}")
        
        accuracy = 0.0
        y_true, y_pred = [], []
        correct = 0
        total = 0
        with open(val_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('__label__'):
                    continue
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                true_orig = parts[0].replace('__label__', '').strip()
                text = parts[1].strip()
                if not true_orig or not text:
                    continue
                true_merged = merge_label(true_orig)
                try:
                    pred_orig = model.predict(text, k=1)[0][0].replace('__label__', '').strip()
                except Exception:
                    pred_orig = 'other'
                pred_merged = merge_label(pred_orig)
                y_true.append(true_merged)
                y_pred.append(pred_merged)
                correct += pred_merged == true_merged
                total += 1

        accuracy = correct / total if total else 0
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        print(report)

        stats_file = os.path.join(output_dir, 'model_evaluation.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ОЦЕНКА МОДЕЛИ FASTTEXT (на validation)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Модель: {model_output}\n")
            f.write(f"Train: {train_file} ({train_count} примеров)\n")
            f.write(f"Val:   {val_file} ({val_count} примеров)\n")
            f.write(f"Параметры (auto): dim={dim} epoch={optimal_epoch} wordNgrams={word_ngrams}")
            f.write(f"  minn={minn} maxn={maxn} lr={lr} minCount={min_count} loss={loss}\n\n")
            f.write(report)

        print(f"\nСтатистика сохранена: {stats_file}")
    else:
        print(f"\nВНИМАНИЕ: val.csv не найден, оценка пропущена")
    
    print(f"{'='*80}")
    acc = accuracy if val_count > 0 else 0
    return model, acc


if __name__ == "__main__":
    train_fasttext_model()
