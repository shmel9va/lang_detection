"""
Fine-tune DistilBERT для классификации языков (22 класса).

Двухэтапный подход:
  Этап 1 — Grid Search: перебор learning_rate × weight_decay,
           3 эпохи каждый, выбираем лучшую по macro F1 на val.
  Этап 2 — Финальное обучение: лучшая конфигурация,
           до 10 эпох с early stopping (patience=3).

Оценка на val.csv. test.csv НЕ трогается.

Использование:
    python -m scripts.training.train_distilbert
    docker-compose run --rm trainer_distilbert

RTX 5060: этап 1 ~10 мин, этап 2 ~15 мин, итог ~25 мин.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from scripts.utils.label_mapping import merge_label
from scripts.data_processing.preprocess_text import normalize_for_detection


# ── Конфигурация ──────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 128
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "distilbert_lang_detection"
SEARCH_DIR = PROJECT_ROOT / "output" / "distilbert_search"

TEXT_COL = "request_text"
LABEL_COL = "result"

# Grid search: перебираем эти значения (4 × 2 = 8 комбинаций, 3 эпохи каждая)
LR_CANDIDATES = [2e-5, 3e-5, 5e-5, 8e-5]
WD_CANDIDATES = [0.01, 0.1]
SEARCH_EPOCHS = 3

# Финальное обучение
FINAL_MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3


# ── 22 выходных класса ───────────────────────────────────────────────────────
def get_label_list() -> List[str]:
    return sorted([
        "am", "ar", "az", "en", "es", "fa", "fr", "he", "hi", "hy",
        "ka", "kk", "ne", "other", "pt", "ro", "ru", "sr", "tr",
        "uk", "ur", "uz",
    ])


# ── Чтение CSV ───────────────────────────────────────────────────────────────
def read_csv(path: str) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, on_bad_lines="skip")
        except (UnicodeDecodeError, Exception):
            continue
    print(f"ERROR: cannot read {path}")
    return None


# ── Dataset с токенизацией в __getitem__ ──────────────────────────────────────
class LanguageDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Подготовка данных ────────────────────────────────────────────────────────
def load_data(csv_path: str, label2id: dict, tokenizer) -> Optional[LanguageDataset]:
    df = read_csv(csv_path)
    if df is None:
        return None
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df["_merged"] = df[LABEL_COL].apply(merge_label)
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(normalize_for_detection)
    df = df[df[TEXT_COL].str.strip() != ""]
    df = df[df["_merged"].isin(label2id)]
    print(f"  {os.path.basename(csv_path)}: {len(df)} примеров")
    return LanguageDataset(df[TEXT_COL].tolist(), [label2id[l] for l in df["_merged"].tolist()], tokenizer)


# ── Метрики ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ── Создаём свежую модель ────────────────────────────────────────────────────
def make_model(label_list, label2id, id2label):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )


# ── Этап 1: Grid Search ─────────────────────────────────────────────────────
def grid_search(train_dataset, val_dataset, label_list, label2id, id2label):
    print("\n" + "=" * 70)
    print("ЭТАП 1: GRID SEARCH")
    print(f"  Learning rates: {LR_CANDIDATES}")
    print(f"  Weight decays:  {WD_CANDIDATES}")
    print(f"  Комбинаций: {len(LR_CANDIDATES) * len(WD_CANDIDATES)}")
    print(f"  Эпох на комбинацию: {SEARCH_EPOCHS}")
    print("=" * 70)

    search_dir = str(SEARCH_DIR)
    os.makedirs(search_dir, exist_ok=True)

    results: List[Dict] = []
    total = len(LR_CANDIDATES) * len(WD_CANDIDATES)

    for i, lr in enumerate(LR_CANDIDATES):
        for j, wd in enumerate(WD_CANDIDATES):
            run_idx = i * len(WD_CANDIDATES) + j + 1
            run_name = f"lr{lr}_wd{wd}"
            run_dir = os.path.join(search_dir, run_name)

            print(f"\n── Прогон {run_idx}/{total}: lr={lr}, wd={wd} ──")

            model = make_model(label_list, label2id, id2label)

            args = TrainingArguments(
                output_dir=run_dir,
                num_train_epochs=SEARCH_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE * 2,
                learning_rate=lr,
                weight_decay=wd,
                warmup_ratio=0.1,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),
                logging_steps=100,
                save_total_limit=1,
                report_to="none",
                dataloader_num_workers=0,
                disable_tqdm=False,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            metrics = trainer.evaluate()

            result = {
                "learning_rate": lr,
                "weight_decay": wd,
                "macro_f1": metrics["eval_macro_f1"],
                "accuracy": metrics["eval_accuracy"],
            }
            results.append(result)

            print(f"  → macro_f1={result['macro_f1']:.4f}, accuracy={result['accuracy']:.4f}")

            # Освобождаем память GPU
            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Итоги grid search ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ИТОГИ GRID SEARCH")
    print("=" * 70)
    print(f"  {'LR':<10} {'WD':<8} {'Macro F1':>10} {'Accuracy':>10}")
    print("  " + "-" * 40)

    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    for r in results:
        marker = " ★" if r == results[0] else ""
        print(f"  {r['learning_rate']:<10.0e} {r['weight_decay']:<8.3f} "
              f"{r['macro_f1']:>10.4f} {r['accuracy']:>10.4f}{marker}")

    best = results[0]
    print(f"\n  ЛУЧШАЯ КОМБИНАЦИЯ: lr={best['learning_rate']}, wd={best['weight_decay']}")
    print(f"  macro_f1={best['macro_f1']:.4f}, accuracy={best['accuracy']:.4f}")

    # Сохраняем результаты
    with open(os.path.join(search_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {search_dir}/search_results.json")

    # Удаляем временные чекпоинты
    shutil.rmtree(search_dir, ignore_errors=True)

    return best["learning_rate"], best["weight_decay"]


# ── Этап 2: Финальное обучение ───────────────────────────────────────────────
def final_train(train_dataset, val_dataset, label_list, label2id, id2label, best_lr, best_wd):
    print("\n" + "=" * 70)
    print("ЭТАП 2: ФИНАЛЬНОЕ ОБУЧЕНИЕ")
    print(f"  Learning rate: {best_lr}")
    print(f"  Weight decay:  {best_wd}")
    print(f"  Max эпох:      {FINAL_MAX_EPOCHS}")
    print(f"  Early stopping: patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print("=" * 70)

    model = make_model(label_list, label2id, id2label)
    output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FINAL_MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=best_lr,
        weight_decay=best_wd,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # ── Финальная оценка ──────────────────────────────────────────────────
    print("\nФинальная оценка на val...")
    metrics = trainer.evaluate()
    print(f"  Accuracy:  {metrics['eval_accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['eval_macro_f1']:.4f}")

    # ── Per-class report ──────────────────────────────────────────────────
    from sklearn.metrics import classification_report
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    report = classification_report(
        labels, preds,
        target_names=label_list,
        zero_division=0,
        digits=3,
    )
    print(f"\n  Per-class report (val):")
    for line in report.split("\n"):
        print(f"  {line}")

    # ── Сохранение ────────────────────────────────────────────────────────
    print(f"\nСохраняем модель: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer = val_dataset.tokenizer
    tokenizer.save_pretrained(output_dir)

    config = {
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "best_learning_rate": best_lr,
        "best_weight_decay": best_wd,
        "final_macro_f1": metrics["eval_macro_f1"],
        "final_accuracy": metrics["eval_accuracy"],
    }
    with open(os.path.join(output_dir, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"DistilBERT Fine-tune Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Learning rate: {best_lr}\n")
        f.write(f"Weight decay: {best_wd}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Max length: {MAX_LENGTH}\n")
        f.write(f"Accuracy: {metrics['eval_accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['eval_macro_f1']:.4f}\n\n")
        f.write(report)

    print(f"\n  label_config.json — маппинг меток + гиперпараметры")
    print(f"  classification_report.txt — per-class метрики")
    print(f"  model.safetensors — веса модели")
    print("=" * 70)


# ── Точка входа ───────────────────────────────────────────────────────────────
def train():
    print("=" * 70)
    print("DISTILBERT FINE-TUNE: ЯЗЫКОВАЯ КЛАССИФИКАЦИЯ (22 КЛАССА)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nУстройство: {device}")
    if device == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    label_list = get_label_list()
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"\nКлассов: {len(label_list)}")
    print(f"  {', '.join(label_list)}")

    # ── Токенизатор ───────────────────────────────────────────────────────
    print(f"\nЗагружаем токенизатор: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Данные ────────────────────────────────────────────────────────────
    train_csv = str(PROJECT_ROOT / "output" / "train.csv")
    val_csv = str(PROJECT_ROOT / "output" / "val.csv")

    for p in [train_csv, val_csv]:
        if not os.path.exists(p):
            print(f"ОШИБКА: {p} не найден. Сначала запустите splitter.")
            return

    print("\nЗагрузка данных...")
    train_dataset = load_data(train_csv, label2id, tokenizer)
    val_dataset = load_data(val_csv, label2id, tokenizer)

    if train_dataset is None or val_dataset is None:
        print("ОШИБКА: не удалось загрузить данные.")
        return

    # ── Этап 1: Grid Search ───────────────────────────────────────────────
    best_lr, best_wd = grid_search(train_dataset, val_dataset, label_list, label2id, id2label)

    # ── Этап 2: Финальное обучение ────────────────────────────────────────
    final_train(train_dataset, val_dataset, label_list, label2id, id2label, best_lr, best_wd)


if __name__ == "__main__":
    train()
