# Комбинированная модель идентификации языка текста

Модель для определения языка текста в чатах поддержки международных компаний.  
Устойчива к нестандартному пользовательскому вводу: опечатки, транслитерация,  
смешение алфавитов, короткие сообщения.

Целевой тайминг: **≤ 30 мс** на классификацию.  
Целевое качество: **macro F1 ≥ 85%** на реальных данных (с ошибками, короткие тексты,  
транслитерация).

---

## Архитектура (fastText + Transformer fallback + Binary classifiers)

```
Входной текст
     │
     ▼
┌─────────────────────────────────────────────────┐
│ Level 0: Phrase Dictionary                       │
│   Точные совпадения частых фраз                  │
│   Покрытие: ~15% текстов, 0.01 мс               │
└──────────────┬──────────────────────────────────┘
               │ нет совпадения
               ▼
┌─────────────────────────────────────────────────┐
│ Level 1: Script Detector                         │
│   Уникальные алфавиты → немедленный ответ        │
│   hy (армянский), ka (грузинский), he, am, CJK   │
│   Покрытие: ~10% текстов, 0.1 мс                │
└──────────────┬──────────────────────────────────┘
               │ скрипт неоднозначен (латиница / кириллица)
               ▼
┌─────────────────────────────────────────────────┐
│ Level 2: fastText (обучен на 41 метку)           │
│   Если confidence ≥ 0.95                         │
│   И пара НЕ чувствительная → ответ               │
│   Покрытие: ~50% текстов, 0.05 мс               │
└──────────────┬──────────────────────────────────┘
               │ confidence < 0.95 ИЛИ чувствительная пара
               ▼
┌─────────────────────────────────────────────────┐
│ Level 3: DistilBERT (ONNX, 22 класса)            │
│   Fallback для неоднозначных случаев             │
│   Покрытие: ~25% текстов, 8–10 мс               │
└──────────────┬──────────────────────────────────┘
               │ топ-2 = чувствительная пара
               ▼
┌─────────────────────────────────────────────────┐
│ Level 4: 5 бинарных классификаторов              │
│   hy-az · he-ar · ur-hi · ar-fa · ru-uk          │
│   Покрытие: ~5% текстов, 0.5 мс                 │
└─────────────────────────────────────────────────┘

Среднее: ~2.6 мс   Максимум: ~12 мс   (при инференсе на CPU через ONNX)
```

Каждый бинарный классификатор двухуровневый:
1. **Быстрый путь** — детерминированные Unicode-правила (< 0.1 мс)
2. **LogReg на char n-gram** — TF-IDF (2–4-граммы) для романизованных текстов

---

## Датасет

Файл: `lang_detection_diploma.csv`  
Поля: `request_text`, `result`  
Размер: **~88 800 строк, 41 метка → 22 выходных класса**

### Принцип разделения на метки

Для языков, где пользователи пишут в разных алфавитах, созданы отдельные метки  
по схеме `{lang}_{script}`. fastText обучается на всех 41 метках — это позволяет  
модели связать символьные n-граммы конкретного алфавита с нужным языком.  
DistilBERT обучается на 22 объединённых классах (маппинг через `label_mapping.py`).

### Метки датасета (41 → 22)

| Исходная метка | → Выход | Язык | Алфавит |
|---------------|---------|------|---------|
| `hy_arm` | `hy` | Армянский | Армянский |
| `hy_lat` | `hy` | Армянский | Латиница |
| `hy_cyr` | `hy` | Армянский | Кириллица |
| `ka_geo` | `ka` | Грузинский | Грузинский |
| `ka_lat` | `ka` | Грузинский | Латиница |
| `uz_lat` | `uz` | Узбекский | Латиница |
| `uz_cyr` | `uz` | Узбекский | Кириллица |
| `ur_ur` | `ur` | Урду | Арабский (насталик) |
| `ur_lat` | `ur` | Урду | Латиница |
| `ne_nep` | `ne` | Непальский | Деванагари |
| `ne_lat` | `ne` | Непальский | Латиница |
| `sr_cyr` | `sr` | Сербский | Кириллица |
| `sr_lat` | `sr` | Сербский | Латиница |
| `hi_hi` | `hi` | Хинди | Деванагари |
| `hi_lat` | `hi` | Хинди | Латиница (Hinglish) |
| `kk` | `kk` | Казахский | Кириллица |
| `kk_lat` | `kk` | Казахский | Латиница (стандарт 2021) |
| `ru_cyr` | `ru` | Русский | Кириллица |
| `ru_lat` | `ru` | Русский | Латиница |
| `am` | `am` | Амхарский | Эфиопский |
| `am_lat` | `am` | Амхарский | Латиница |
| `en` | `en` | Английский | Латиница |
| `he` | `he` | Иврит | Иврит |
| `ar` | `ar` | Арабский | Арабское письмо |
| `az` | `az` | Азербайджанский | Латиница |
| `ro` | `ro` | Румынский | Латиница |
| `uk` | `uk` | Украинский | Кириллица |
| `fr` | `fr` | Французский | Латиница |
| `es` | `es` | Испанский | Латиница |
| `tr` | `tr` | Турецкий | Латиница |
| `fa` | `fa` | Персидский | Арабское письмо |
| `pt` | `pt` | Португальский | Латиница |
| `de` | `other` | Немецкий | — |
| `it` | `other` | Итальянский | — |
| `nl` | `other` | Нидерландский | — |
| `pl` | `other` | Польский | — |
| `bg` | `other` | Болгарский | — |
| `ja` | `other` | Японский | — |
| `vi` | `other` | Вьетнамский | — |
| `ko` | `other` | Корейский | — |

### 5 чувствительных пар (обязательных по заданию)

| Пара | Почему чувствительна | Быстрый путь (Unicode) |
|------|---------------------|----------------------|
| hy–az | Политический конфликт (Армения–Азербайджан) | Армянский алфавит → hy |
| he–ar | Иврит vs арабский — разные Unicode-блоки | Иврит/арабский подсчёт |
| ur–hi | Оба могут быть на латинице | Деванагари → hi, арабский → ur |
| ar–fa | Оба — арабское письмо | پچژگ → fa |
| ru–uk | Оба — кириллица | іїєґ → uk |

---

## Полный pipeline запуска

### Шаг 0. Установка зависимостей

```bash
docker-compose build
docker-compose build trainer_distilbert   # отдельный образ с CUDA (для GPU)
```

Для GPU-сервисов нужен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Шаг 1. Разбиение датасета

```bash
docker-compose run --rm splitter
```

Результат: `output/train.csv` (70%), `output/val.csv` (15%), `output/test.csv` (15%)

### Шаг 2. Обучение fastText

```bash
docker-compose run --rm trainer
```

Результат: `output/lang_detection_model.bin`

### Шаг 3. Обучение бинарных классификаторов

```bash
docker-compose run --rm trainer_sensitive
```

Результат: `output/sensitive_classifiers/*.pkl` (5 классификаторов)

### Шаг 4. Fine-tune DistilBERT (требует GPU)

```bash
docker-compose run --rm trainer_distilbert
```

Результат: `output/distilbert_lang_detection/`

### Шаг 5. Экспорт DistilBERT → ONNX

```bash
docker-compose run --rm export_onnx
```

Результат: `output/distilbert_lang_detection.onnx`

### Шаг 6. Подбор threshold

```bash
docker-compose run --rm threshold_finder
```

### Шаг 7. Финальная оценка

```bash
docker-compose run --rm evaluator
docker-compose run --rm eval_real
```

### Полный pipeline одной командой

```bash
docker-compose build && \
docker-compose run --rm splitter && \
docker-compose run --rm trainer && \
docker-compose run --rm trainer_sensitive && \
docker-compose run --rm trainer_distilbert && \
docker-compose run --rm export_onnx && \
docker-compose run --rm threshold_finder
```

---

## Использование модели

### Полный пайплайн

```python
from scripts.detection.detector import LanguageDetector

detector = LanguageDetector(
    fasttext_model_path='output/lang_detection_model.bin',
    sensitive_classifiers_dir='output/sensitive_classifiers',
    onnx_model_path='output/distilbert_lang_detection.onnx',
    threshold=0.5,
)

lang, conf = detector.detect("Привет, как дела?")
# → ('ru', 0.97)

lang, conf = detector.detect("Բարի՜ Ծաղիկ")
# → ('hy', 0.99)  — армянский алфавит, быстрый путь (Level 1)

lang, conf = detector.detect("Ес чем асканум ес инч ек анум")
# → ('hy', 0.90)  — армянский на кириллице, fastText (Level 2)

lang, conf = detector.detect("Salam, necəsən?")
# → ('az', 0.95)  — ə → az (Level 4, бинарный классификатор)
```

### Batch

```python
results = detector.detect_batch(["Hello", "Salam", "مرحبا"])
```

---

## Docker-сервисы

| Сервис | Команда | Что делает |
|--------|---------|-----------|
| `splitter` | `docker-compose run --rm splitter` | 70/15/15 split |
| `trainer` | `docker-compose run --rm trainer` | Обучение fastText |
| `trainer_sensitive` | `docker-compose run --rm trainer_sensitive` | 5 бинарных классификаторов |
| `trainer_distilbert` | `docker-compose run --rm trainer_distilbert` | Fine-tune DistilBERT (GPU) |
| `export_onnx` | `docker-compose run --rm export_onnx` | Экспорт DistilBERT → ONNX |
| `threshold_finder` | `docker-compose run --rm threshold_finder` | Подбор threshold |
| `evaluator` | `docker-compose run --rm evaluator` | Финальная оценка на test |
| `eval_real` | `docker-compose run --rm eval_real` | Оценка на реальных данных |
| `baseline_trainer` | `docker-compose run --rm baseline_trainer` | Baseline fastText |
| `comparer_solutions` | `docker-compose run --rm comparer_solutions` | Сравнение моделей |
| `benchmark` | `docker-compose run --rm benchmark` | Замер скорости |

---

## Скорость

| Компонент | Среднее | Максимум |
|-----------|---------|----------|
| Level 0: Phrase Dictionary | 0.01 мс | 0.01 мс |
| Level 1: Script Detector | 0.1 мс | 0.1 мс |
| Level 2: fastText | 0.05 мс | 0.5 мс |
| Level 3: DistilBERT ONNX | 8 мс | 10 мс |
| Level 4: Binary classifiers | 0.5 мс | 1 мс |
| **Пайплайн (среднее)** | **~2.6 мс** | **~12 мс** |

---

## Поддерживаемые языки (22 выхода)

| Код | Язык | Скрипты |
|-----|------|---------|
| `am` | Амхарский | Эфиопский, Латиница |
| `ar` | Арабский | Арабское письмо |
| `az` | Азербайджанский | Латиница |
| `en` | Английский | Латиница |
| `es` | Испанский | Латиница |
| `fa` | Персидский | Арабское письмо |
| `fr` | Французский | Латиница |
| `he` | Иврит | Иврит |
| `hi` | Хинди | Деванагари, Латиница (Hinglish) |
| `hy` | Армянский | Армянский, Латиница, Кириллица |
| `ka` | Грузинский | Грузинский, Латиница |
| `kk` | Казахский | Кириллица, Латиница (2021) |
| `ne` | Непальский | Деванагари, Латиница |
| `pt` | Португальский | Латиница |
| `ro` | Румынский | Латиница |
| `ru` | Русский | Кириллица, Латиница |
| `sr` | Сербский | Кириллица, Латиница |
| `tr` | Турецкий | Латиница |
| `uk` | Украинский | Кириллица |
| `ur` | Урду | Арабский, Латиница |
| `uz` | Узбекский | Латиница, Кириллица |
| `other` | Неподдерживаемые | CJK/Hangul/и т.д. |

---

## Структура проекта

```
lang_detection/
├── data/                              # сырые данные по языкам (xlsx)
│   ├── armenian_dataset.xlsx
│   ├── armenian_cyrillic_dataset.xlsx
│   ├── russian_latin_dataset.xlsx
│   ├── amharic_latin_dataset.xlsx
│   ├── georgian_dataset.xlsx
│   ├── nepali_dataset.xlsx
│   └── kazakh_dataset.xlsx
├── benchmarks/
│   ├── benchmark_speed.py
│   ├── compare_solutions.py
│   ├── eval_real_data.py
│   └── collect_errors.py
├── scripts/
│   ├── dataset_collection/            # сбор данных с HuggingFace
│   │   ├── armenian_dataset_to_xlsx.py
│   │   ├── armenian_cyrillic_dataset_to_xlsx.py
│   │   ├── russian_latin_dataset_to_xlsx.py
│   │   ├── amharic_latin_dataset_to_xlsx.py
│   │   ├── georgian_dataset_to_xlsx.py
│   │   ├── nepali_dataset_to_xlsx.py
│   │   ├── kazakh_dataset_to_xlsx.py
│   │   └── rebuild_diploma_csv.py
│   ├── data_processing/
│   │   ├── preprocess_text.py
│   │   ├── split_dataset.py
│   │   └── clean_dataset.py
│   ├── detection/
│   │   ├── detector.py                # LanguageDetector — точка входа
│   │   ├── script_detector.py         # уникальные алфавиты → быстрый путь
│   │   ├── sensitive_router.py        # роутер чувствительных пар (5 пар)
│   │   └── sensitive_classifiers/
│   │       ├── base.py                # char n-gram TF-IDF + LogReg
│   │       ├── hy_az.py
│   │       ├── he_ar.py
│   │       ├── ur_hi.py
│   │       ├── ar_fa.py
│   │       ├── ru_uk.py
│   │       ├── az_tr.py
│   │       ├── es_pt.py
│   │       ├── ru_sr.py
│   │       └── uz_kk.py
│   ├── training/
│   │   ├── train_fasttext.py
│   │   ├── train_baseline_fasttext.py
│   │   ├── train_sensitive_classifiers.py
│   │   ├── train_distilbert.py        # DistilBERT fine-tune (GPU)
│   │   ├── export_distilbert_onnx.py  # Экспорт → ONNX
│   │   ├── find_optimal_params.py
│   │   ├── find_threshold.py
│   │   └── evaluate_final.py
│   └── utils/
│       ├── label_mapping.py           # 41→22 маппинг меток
│       └── predict_with_threshold.py
├── output/                            # создаётся автоматически
├── lang_detection_diploma.csv         # датасет (вход пайплайна)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Требования

```
Python 3.11+  ·  Docker  ·  Docker Compose  ·  CUDA GPU (для DistilBERT)
pandas · scikit-learn · fasttext · torch · transformers · onnxruntime
numpy · joblib · openpyxl · datasets (HuggingFace)
```
