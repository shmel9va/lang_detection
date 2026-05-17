# Комбинированная модель идентификации языка текста

Модель для определения языка текста в чатах международной поддержки.
Устойчива к опечаткам, транслитерации, смешению алфавитов, коротким сообщениям.

**22 языка** · **~5.5 мс** среднее время · **macro F1 = 0.88** на реальных данных (без `other`)

---

## Быстрый старт (без обучения)

Модели уже обучены и включены в репозиторий через Git LFS.

### Требования

- Docker + Docker Compose
- [Git LFS](https://git-lfs.github.com/)

### Запуск

```bash
git lfs install
git clone https://github.com/shmel9va/lang_detection.git
cd lang_detection
docker-compose build
docker-compose run --rm eval_real
```

Результат — оценка на реальных данных: accuracy, macro F1, поклассовый F1, скорость.

### Тестирование на своём датасете

1. Положите CSV файл в корень проекта (рядом с `docker-compose.yml`)
2. Формат CSV: разделитель `;`, колонки `task_language` и `rider_ml_message`
3. Коды языков: `am ar az en es fa fr he hi hy ka kk ne other pt ro ru sr tr uk ur uz`

```bash
docker-compose run --rm eval_real python benchmarks/eval_real_data.py ваш_файл.csv
```

### Предсказание для одного текста

```python
from scripts.detection.detector import LanguageDetector

detector = LanguageDetector(
    fasttext_model_path='output/lang_detection_model.bin',
    sensitive_classifiers_dir='output/sensitive_classifiers',
    onnx_model_path='output/distilbert_lang_detection.onnx',
)

lang, conf = detector.detect("Привет, как дела?")
# → ('ru', 0.97)

lang, conf = detector.detect("barev dzez")
# → ('hy', 0.95)

lang, conf = detector.detect("Salam, necesen?")
# → ('az', 0.92)
```

---

## Архитектура пайплайна

```
Входной текст
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Level 1: Script Detector                              │
│   Уникальные алфавиты: hy → hy, ka → ka,              │
│   he → he, am → am  → немедленный ответ               │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│ Level 2: fastText (42+ метки → 22 класса)             │
│   confidence ≥ 0.95 И не чувствительная пара → ответ   │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│ Level 4: 3 бинарных классификатора                     │
│   hy–az · he–ar · ru–uk                               │
│   Unicode-правила + LogReg на TF-IDF char n-граммах    │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│ PersoArabicLID (top-1 ∈ {ar, fa})                     │
│   Специализированная fasttext модель (2 МБ)            │
│   Фильтрует top-3: ar→ar, fa→fa                        │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│ IndicLID-FTR (top-1 ∈ {hi, ur, ne})                   │
│   Специализированная fasttext модель (357 МБ)          │
│   hin_Latn→ur, urd_Latn→ur, nep_Latn→ne                │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│ DistilBERT ONNX (22 класса, fallback)                  │
│   Пропускается если top-1 = "hi"                       │
└──────────────┬───────────────────────────────────────┘
               ▼
         fastText top-1 (финальный fallback)
```

---

## Результаты на реальных данных (~18 900 примеров)

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.9336 |
| Macro F1 (все 22 класса) | 0.8521 |
| Macro F1 (без `other`) | 0.8796 |
| Среднее время | ~5.5 мс |
| P95 время | ~25 мс |

### Поклассовый F1

| Класс | F1 | | Класс | F1 | | Класс | F1 |
|-------|----|-|-------|----|-|-------|----|
| es | 0.97 | | en | 0.96 | | hi | 0.95 |
| fr | 0.97 | | uz | 0.94 | | he | 0.93 |
| hy | 0.97 | | tr | 0.89 | | ar | 0.87 |
| ka | 0.97 | | az | 0.85 | | uk | 0.85 |
| ru | 0.91 | | pt | 0.89 | | ur | 0.82 |
| ro | 0.89 | | am | 0.83 | | fa | 0.79 |
| ne | 0.69 | | kk | 0.60 | | other | 0.27 |

---

## Обучение с нуля

```bash
docker-compose build

# 1. Split датасета
docker-compose run --rm splitter

# 2. Обучение fastText (~10 мин, CPU)
docker-compose run --rm trainer

# 3. Бинарные классификаторы (~2 мин, CPU)
docker-compose run --rm trainer_sensitive

# 4. DistilBERT (~25 мин, нужен GPU + NVIDIA Container Toolkit)
docker-compose run --rm trainer_distilbert
docker-compose run --rm export_onnx

# 5. Оценка
docker-compose run --rm evaluator
docker-compose run --rm eval_real
```

---

## Датасет

**Файл:** `lang_detection_diploma.csv` (~88 800 строк, 42+ меток → 22 класса)

Для языков с несколькими алфавитами созданы отдельные метки (`hy_arm`, `hy_lat`, `hy_cyr`, `kk_lat`, `ur_ur`, `ur_lat`, `hi_hi`, `hi_lat` и т.д.).

- **fastText** обучается на разделённых метках — изучает n-граммы каждого алфавита отдельно
- **DistilBERT** обучается на 22 объединённых классах

### 3 чувствительные пары (решаются бинарными классификаторами)

| Пара | Быстрый путь |
|------|-------------|
| hy–az | Армянский алфавит → hy |
| he–ar | Иврит/арабский подсчёт символов |
| ru–uk | іїєґ → uk |

### 2 специализированных модели

| Модель | Языки | Размер | Назначение |
|--------|-------|--------|------------|
| PersoArabicLID | ar, fa | 2 МБ | Различение арабского и фарси |
| IndicLID-FTR | hi, ur, ne | 357 МБ | Различение хинди/урду/непали (вкл. латиницу) |

### Дизайнерские решения

- **hi_lat → ur**: Хинглиш (Hinglish) и романизированный урду на латинице неразличимы — оба мапятся на `ur`. Метка `hi` предсказывается только на деванагари.

---

## Поддерживаемые языки (22)

`am` · `ar` · `az` · `en` · `es` · `fa` · `fr` · `he` · `hi` · `hy` · `ka` · `kk` · `ne` · `other` · `pt` · `ro` · `ru` · `sr` · `tr` · `uk` · `ur` · `uz`

---

## Структура проекта

```
lang_detection/
├── output/
│   ├── lang_detection_model.bin              # fastText (Git LFS)
│   ├── distilbert_lang_detection.onnx        # DistilBERT ONNX (Git LFS)
│   ├── distilbert_lang_detection.onnx.data   # ONNX weights (Git LFS)
│   ├── distilbert_lang_detection/            # tokenizer + config + label_config
│   ├── sensitive_classifiers/                # бинарные классификаторы .pkl
│   ├── indiclid-ftr/                         # IndicLID-FTR модель
│   └── persoarabic/                          # PersoArabicLID модель
├── benchmarks/
│   ├── eval_real_data.py                     # оценка на реальных данных
│   ├── export_predictions.py                 # выгрузка предсказаний в CSV
│   └── collect_errors.py                     # сбор ошибок для анализа
├── scripts/
│   ├── detection/
│   │   ├── detector.py                       # LanguageDetector — точка входа
│   │   ├── script_detector.py                # уникальные алфавиты
│   │   ├── sensitive_router.py               # роутер чувствительных пар
│   │   ├── indiclid_wrapper.py               # IndicLID-FTR обёртка
│   │   ├── persoarabic_wrapper.py            # PersoArabicLID обёртка
│   │   └── sensitive_classifiers/            # hy_az, he_ar, ru_uk (+ 6 неактивных)
│   ├── training/                             # скрипты обучения
│   ├── data_processing/                      # preprocess, split, clean
│   ├── dataset_collection/                   # сбор данных с HuggingFace
│   └── utils/
│       └── label_mapping.py                  # 42+ → 22 маппинг
├── lang_detection_diploma.csv                # обучающий датасет
├── real_data.csv                             # реальные данные для оценки
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Требования

```
Docker · Docker Compose · Git LFS
(для обучения DistilBERT: CUDA GPU + NVIDIA Container Toolkit)
```
