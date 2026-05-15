# Комбинированная модель идентификации языка текста

Модель для определения языка текста в чатах поддержки.  
Устойчива к опечаткам, транслитерации, смешению алфавитов, коротким сообщениям.

**22 языка** · **≤ 30 мс** на классификацию · **macro F1 = 0.77** на реальных данных

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

Пример содержимого CSV:
```
task_language;rider_ml_message
en;Hello, how are you?
ru;Привет, как дела?
az;Salam, necesen?
```

### Предсказание для одного текста

```python
from scripts.detection.detector import LanguageDetector

detector = LanguageDetector(
    fasttext_model_path='output/lang_detection_model.bin',
    sensitive_classifiers_dir='output/sensitive_classifiers',
    onnx_model_path='output/distilbert_lang_detection.onnx',
    threshold=0.0,
)

lang, conf = detector.detect("Привет, как дела?")
# → ('ru', 0.97)

lang, conf = detector.detect("barev dzez")
# → ('hy', 0.95)  — армянский на латинице

lang, conf = detector.detect("Ес чем асканум")
# → ('hy', 0.90)  — армянский на кириллице
```

### Выгрузка предсказаний в CSV

```bash
docker-compose run --rm eval_real python benchmarks/export_predictions.py
```

Результат: `output/predictions.csv` (true_label, predicted_label, confidence, correct, message)

---

## Архитектура

```
Входной текст
     │
     ▼
┌─────────────────────────────────────────────────┐
│ Level 0: Phrase Dictionary                       │
└──────────────┬──────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────┐
│ Level 1: Script Detector                         │
│   hy, ka, he, am, CJK → немедленный ответ        │
└──────────────┬──────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────┐
│ Level 2: fastText (41 метка, 22 класса)           │
│   confidence ≥ 0.95 И не чувствительная → ответ   │
└──────────────┬──────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────┐
│ Level 4: 5 бинарных классификаторов              │
│   hy-az · he-ar · ur-hi · ar-fa · ru-uk          │
└──────────────┬──────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────┐
│ Level 3: DistilBERT ONNX (22 класса)             │
│   Fallback для неоднозначных случаев             │
└─────────────────────────────────────────────────┘
```

Каждый бинарный классификатор: Unicode-правила (быстрый путь) + LogReg на TF-IDF char n-граммах.

---

## Обучение с нуля (если нужно переобучить)

```bash
docker-compose build

# 1. Split датасета
docker-compose run --rm splitter

# 2. Обучение fastText (~10 мин, CPU)
docker-compose run --rm trainer

# 3. Бинарные классификаторы (~2 мин, CPU)
docker-compose run --rm trainer_sensitive

# 4. Подбор threshold (~1 мин, CPU)
docker-compose run --rm threshold_finder

# 5. DistilBERT (~25 мин, нужен GPU + NVIDIA Container Toolkit)
docker-compose run --rm trainer_distilbert
docker-compose run --rm export_onnx

# 6. Оценка
docker-compose run --rm evaluator
docker-compose run --rm eval_real
```

---

## Датасет

**Файл:** `lang_detection_diploma.csv` (~88 800 строк, 41 метка → 22 класса)

Для языков с несколькими алфавитами созданы отдельные метки (`hy_arm`, `hy_lat`, `hy_cyr`).
fastText обучается на 41 метке (изучает n-граммы каждого алфавита).
DistilBERT обучается на 22 объединённых классах.

### 5 чувствительных пар

| Пара | Быстрый путь |
|------|-------------|
| hy–az | Армянский алфавит → hy |
| he–ar | Иврит/арабский подсчёт |
| ur–hi | Деванагари → hi, арабский → ur |
| ar–fa | پچژگ → fa |
| ru–uk | іїєґ → uk |

---

## Поддерживаемые языки (22)

am · ar · az · en · es · fa · fr · he · hi · hy · ka · kk · ne · other · pt · ro · ru · sr · tr · uk · ur · uz

---

## Структура проекта

```
lang_detection/
├── output/
│   ├── lang_detection_model.bin              # fastText (Git LFS)
│   ├── distilbert_lang_detection.onnx        # DistilBERT ONNX (Git LFS)
│   ├── distilbert_lang_detection.onnx.data   # ONNX weights (Git LFS)
│   ├── distilbert_lang_detection/            # tokenizer + config
│   └── sensitive_classifiers/                # 9 бинарных классификаторов .pkl
├── benchmarks/
│   ├── eval_real_data.py                     # оценка на реальных данных
│   └── export_predictions.py                 # выгрузка предсказаний в CSV
├── scripts/
│   ├── detection/
│   │   ├── detector.py                       # LanguageDetector — точка входа
│   │   ├── script_detector.py                # уникальные алфавиты
│   │   ├── sensitive_router.py               # роутер чувствительных пар
│   │   └── sensitive_classifiers/            # hy_az, he_ar, ur_hi, ar_fa, ru_uk
│   ├── training/                             # скрипты обучения
│   ├── data_processing/                      # preprocess, split, clean
│   ├── dataset_collection/                   # сбор данных с HuggingFace
│   └── utils/
│       └── label_mapping.py                  # 41→22 маппинг
├── lang_detection_diploma.csv                # датасет
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
