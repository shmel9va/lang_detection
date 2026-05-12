"""
Экспорт fine-tuned DistilBERT в ONNX для быстрого инференса.

Использование:
    python -m scripts.training.export_distilbert_onnx

Требования:
    pip install onnx onnxruntime optimum

Результат:
    output/distilbert_lang_detection.onnx
"""

import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "output" / "distilbert_lang_detection"
ONNX_PATH = PROJECT_ROOT / "output" / "distilbert_lang_detection.onnx"


def export():
    print("=" * 70)
    print("ЭКСПОРТ DistilBERT → ONNX")
    print("=" * 70)

    if not MODEL_DIR.exists():
        print(f"ОШИБКА: модель не найдена: {MODEL_DIR}")
        print("Сначала запустите: python -m scripts.training.train_distilbert")
        return

    config_path = MODEL_DIR / "label_config.json"
    if not config_path.exists():
        print(f"ОШИБКА: конфиг не найден: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    max_length = config.get("max_length", 128)
    print(f"  Классов: {len(config['label_list'])}")
    print(f"  Max length: {max_length}")

    # ── Загрузка модели ───────────────────────────────────────────────────
    print(f"\nЗагружаем модель из: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    # ── Dummy input ───────────────────────────────────────────────────────
    dummy_text = "Hello world, this is a test"
    inputs = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # ── Экспорт ───────────────────────────────────────────────────────────
    print(f"Экспортируем в ONNX: {ONNX_PATH}")
    os.makedirs(ONNX_PATH.parent, exist_ok=True)

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"\nГотово! ONNX модель: {ONNX_PATH}")
    print(f"  Размер: {size_mb:.1f} MB")

    # ── Верификация ───────────────────────────────────────────────────────
    try:
        import onnxruntime as ort

        print("\nВерификация ONNX Runtime...")
        sess = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        ort_output = sess.run(None, ort_inputs)[0]

        with torch.no_grad():
            pt_output = model(inputs["input_ids"], inputs["attention_mask"]).logits.numpy()

        max_diff = abs(ort_output - pt_output).max()
        print(f"  Max разница PyTorch vs ONNX: {max_diff:.6f}")
        print(f"  Providers: {sess.get_providers()}")
        if max_diff < 1e-4:
            print("  Верификация пройдена!")
        else:
            print("  ВНИМАНИЕ: большая разница между PyTorch и ONNX!")
    except ImportError:
        print("  onnxruntime не установлен — верификация пропущена")

    print("=" * 70)


if __name__ == "__main__":
    export()
