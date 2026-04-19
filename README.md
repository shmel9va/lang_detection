# Комбинированная модель определения языка

Модель для определения языка текста в чатах поддержки Яндекс Такси.  
Целевой тайминг: ≤ 30 мс на одну классификацию.

---

## Архитектура

```
Входной текст
     │
     ▼
┌──────────────────────────────────────────────┐
│ 1. Препроцессор                              │
│    NFKC-нормализация · удаление URL/         │
│    телефонов / эмодзи                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. Детектор скрипта           БЫСТРЫЙ ПУТЬ   │
│    Уникальные алфавиты → немедленный ответ   │
│      hy (армянский)   ka (грузинский)        │
│      he (иврит)       am (амхарский)         │
└───────────┬──────────────────────────────────┘
            │ если скрипт неоднозначен
            ▼
┌──────────────────────────────────────────────┐
│ 3. fastText                                  │
│    Топ-2 языка с вероятностями               │
│    Субклассы: uz_lat, uz_cyr, sr_lat, sr_cyr │
│    confidence < threshold → "other"          │
└───────────┬──────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ 4. Роутер чувствительных пар                 │
│    Если топ-2 — чувствительная пара:         │
│      hy–az · he–ar · ur–hi                   │
│      ar–fa  (fast-path: پچژگ → fa)          │
│      ru–uk  (fast-path: іїєґ → uk)          │
│    Иначе → результат fastText                │
└───────────┬──────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ 5. Постпроцессор                             │
│    uz_lat / uz_cyr → uz                      │
│    sr_lat / sr_cyr → sr                      │
│    → ISO 639-1 код + confidence              │
└──────────────────────────────────────────────┘
```

Каждый бинарный классификатор двухуровневый:
1. **Быстрый путь** — детерминированные Unicode-правила (< 0.1 мс)
2. **LogReg на char n-gram** — TF-IDF (2–4-граммы) для романизованных текстов

---

## Разбиение данных

```
lang_detection_hackathon.csv
         │  docker-compose run --rm splitter
         ▼
  train.csv  70%   ← обучение fastText + LogReg
  val.csv    15%   ← подбор threshold, оценка LogReg
  test.csv   15%   ← ФИНАЛЬНАЯ ОЦЕНКА (один раз в самом конце)
```

**Правило:** `test.csv` не открывать до завершения всей разработки.

---

## Последовательность запуска

### Шаг 0. Сборка образа (один раз)

```bash
docker-compose build
```

Пересборка нужна только при изменении `requirements.txt` или `Dockerfile`.

---

### Шаг 1. Разбиение датасета

```bash
docker-compose run --rm splitter
```

Производит три файла со стратифицированным распределением языков:
- `output/train.csv` (70%)
- `output/val.csv` (15%)
- `output/test.csv` (15%)

Результат: `output/split_statistics.txt`

> Сервис `cleaner` **не используется** — датасет берётся целиком, редкие языки не удаляются.

---

### Шаг 2. Предобработка текста

```bash
docker-compose run --rm preprocessor
```

Читает `output/train.csv`, удаляет HTML, нормализует пробелы, применяет lowercase.  
Результат: `output/train_preprocessed.csv`

---

### Шаг 3. Обучение fastText

```bash
docker-compose run --rm trainer
```

Автоматически подбирает число эпох (10–70) через внутренний validation на 15% от train,
затем обучает итоговую модель на полном train.

Результаты:
- `output/lang_detection_model.bin`
- `output/model_evaluation.txt`
- `output/epoch_validation_results.txt`

**Время:** 5–15 минут.

---

### Шаг 4. Подбор threshold

```bash
docker-compose run --rm threshold_finder
```

Перебирает пороги [0.3 … 0.9] на `val.csv` и выбирает оптимальный по accuracy.

Результаты: `output/threshold_results.txt`  
Оптимальный порог выводится в stdout — его нужно прописать в `detector.py`:
```python
detector = LanguageDetector(..., threshold=<best_threshold>)
```

---

### Шаг 5. Обучение бинарных классификаторов

```bash
docker-compose run --rm trainer_sensitive
```

Для каждой из 5 чувствительных пар (ar–fa, ru–uk, hy–az, he–ar, ur–hi):
- обучает LogReg на char n-gram из `train.csv`
- оценивает на `val.csv` (Accuracy / Precision / Recall / F1)
- если языка нет в датасете — выводит `ПРОПУСК` и продолжает, не падает

Результаты: `output/sensitive_classifiers/ar_fa.pkl`, `hy_az.pkl`, `he_ar.pkl`, `ur_hi.pkl`  
(`ru_uk.pkl` появится после добавления украинского)

**Время:** < 1 минуты.

---

### Шаг 6. Бенчмарк скорости

```bash
docker-compose run --rm benchmark
```

Измеряет:
- fastText-only: среднее / медиана / min / max, throughput, влияние длины текста
- Полный пайплайн `LanguageDetector`: среднее, P95, P99, сравнение с порогом 30 мс

Результат: `output/speed_benchmark.txt`

---

### Шаг 7. Финальная оценка (только в самом конце)

Когда разработка завершена и больше ничего не будет меняться:

```bash
docker-compose run --rm evaluator
```

Threshold читается автоматически из `output/threshold_results.txt`.  
Переопределить: `docker-compose run --rm -e THRESHOLD=0.6 evaluator`

Результат: `output/final_evaluation.txt`

---

### Полный pipeline одной командой

```bash
docker-compose build && \
docker-compose run --rm splitter && \
docker-compose run --rm preprocessor && \
docker-compose run --rm trainer && \
docker-compose run --rm threshold_finder && \
docker-compose run --rm trainer_sensitive && \
docker-compose run --rm benchmark
```

После — вручную запустить подбор threshold (шаг 4).

---

## Docker-сервисы

| Сервис | Команда | Что делает |
|--------|---------|-----------|
| `splitter` | `docker-compose run --rm splitter` | 70/15/15 split → train/val/test |
| `preprocessor` | `docker-compose run --rm preprocessor` | Очистка train |
| `trainer` | `docker-compose run --rm trainer` | Обучение fastText |
| `threshold_finder` | `docker-compose run --rm threshold_finder` | Подбор threshold на val |
| `trainer_sensitive` | `docker-compose run --rm trainer_sensitive` | Бинарные классификаторы |
| `benchmark` | `docker-compose run --rm benchmark` | Замер скорости |
| `evaluator` | `docker-compose run --rm evaluator` | Финальная оценка на test (один раз) |
| `analyzer` | `docker-compose run --rm analyzer` | Анализ датасета (опционально) |
| `comparer` | `docker-compose run --rm comparer` | Сравнение версий моделей |

---

## Использование модели

### Полный пайплайн (рекомендуется)

```python
from scripts.detection.detector import LanguageDetector

detector = LanguageDetector(
    fasttext_model_path='output/lang_detection_model.bin',
    sensitive_classifiers_dir='output/sensitive_classifiers',
    threshold=0.5,  # заменить на результат find_optimal_threshold
)

lang, conf = detector.detect("Привет, как дела?")
# → ('ru', 0.97)

lang, conf = detector.detect("Բարի՜ Ծաղիկ")
# → ('hy', 0.99)  — армянский алфавит, быстрый путь

lang, conf = detector.detect("این متن فارسی است. پدر چه گفت؟")
# → ('fa', 0.99)  — персидские символы پچژگ

lang, conf = detector.detect("Доброго ранку! Як справи?")
# → ('uk', 0.99)  — украинские символы іїєґ, быстрый путь
```

### Batch

```python
results = detector.detect_batch(["Hello", "Salam", "مرحبا"])
# → [('en', 0.99), ('az', 0.95), ('ar', 0.98)]
```

### Отладка — топ-3 кандидата

```python
top3 = detector.detect_top_k("Xush kelibsiz", k=3)
# → [('uz', 0.82), ('tr', 0.10), ('az', 0.05)]
```

### fastText-only (без бинарных классификаторов)

```python
from scripts.utils.predict_with_threshold import LanguageDetectorWithThreshold

detector = LanguageDetectorWithThreshold('output/lang_detection_model.bin', threshold=0.7)
lang, conf = detector.predict("Hello world", return_confidence=True)
```

---

## Экспорт модели

Скопировать для деплоя:

```
output/
├── lang_detection_model.bin          # fastText (~50 MB)
└── sensitive_classifiers/
    ├── ar_fa.pkl
    ├── hy_az.pkl
    ├── he_ar.pkl
    ├── ur_hi.pkl
    └── ru_uk.pkl                     # после добавления uk

scripts/
├── detection/
├── utils/label_mapping.py
└── data_processing/preprocess_text.py
```

Упаковать:

```bash
tar -czf lang_detection_v1.tar.gz \
  output/lang_detection_model.bin \
  output/sensitive_classifiers/ \
  scripts/detection/ \
  scripts/utils/ \
  scripts/data_processing/preprocess_text.py \
  requirements.txt
```

---

## Скорость

| Компонент | Min | Среднее | Max |
|-----------|-----|---------|-----|
| fastText-only | 0.01 мс | 0.07 мс | 0.64 мс* |
| Полный пайплайн | < 0.1 мс** | ~0.5 мс | ~5–10 мс* |
| Целевой порог | — | — | **30 мс ✓** |

\* тексты > 500 символов  
\** быстрый путь: армянский, грузинский, иврит, амхарский

Запусти `docker-compose run --rm benchmark` для актуальных цифр.

---

## Метрики качества

| Метрика | Значение |
|---------|---------|
| Accuracy | 96.12% |
| Macro F1 | 96.01% |
| Macro Precision | 96.07% |
| Macro Recall | 96.00% |

Лучшие: he 100%, ar 99.35%, fa 99.01%  
Проблемные: tr 84.30%, az 85.78%, pt 92.89%

---

## Поддерживаемые языки

| Код | Язык | Скрипт |
|-----|------|--------|
| am | Амхарский | Эфиопский |
| ar | Арабский | Арабское письмо |
| az | Азербайджанский | Латиница |
| en | Английский | Латиница |
| es | Испанский | Латиница |
| fa | Персидский | Арабское письмо |
| fr | Французский | Латиница |
| he | Иврит | Иврит |
| hi | Хинди | Деванагари |
| hy | Армянский | Армянский |
| ka | Грузинский | Грузинский |
| kk | Казахский | Кириллица |
| ne | Непальский | Деванагари |
| pt | Португальский | Латиница |
| ro | Румынский | Латиница |
| ru | Русский | Кириллица |
| sr | Сербский (sr_lat / sr_cyr) | Латиница + Кириллица |
| tr | Турецкий | Латиница |
| uk | Украинский ⚠ | Кириллица | ← добавить в датасет |
| ur | Урду | Арабское письмо |
| uz | Узбекский (uz_lat / uz_cyr) | Латиница + Кириллица |

---

## Структура проекта

```
lang_detection/
├── benchmarks/
│   ├── benchmark_speed.py        # замер скорости (fastText + полный пайплайн)
│   └── compare_models.py
├── scripts/
│   ├── data_processing/
│   │   ├── analyze_dataset.py
│   │   ├── preprocess_text.py    # очистка + normalize_for_detection
│   │   └── split_dataset.py      # 70/15/15 → train/val/test
│   ├── detection/
│   │   ├── detector.py           # LanguageDetector — точка входа
│   │   ├── script_detector.py    # уникальные алфавиты → быстрый путь
│   │   ├── sensitive_router.py   # роутер чувствительных пар
│   │   └── sensitive_classifiers/
│   │       ├── base.py           # char n-gram TF-IDF + LogReg
│   │       ├── ar_fa.py          # арабский / персидский
│   │       ├── ru_uk.py          # русский / украинский
│   │       ├── hy_az.py          # армянский / азербайджанский
│   │       ├── he_ar.py          # иврит / арабский
│   │       └── ur_hi.py          # урду / хинди
│   ├── training/
│   │   ├── train_fasttext.py
│   │   ├── find_optimal_epochs.py
│   │   └── train_sensitive_classifiers.py
│   └── utils/
│       ├── label_mapping.py               # uz_lat/uz_cyr→uz, sr_lat/sr_cyr→sr
│       └── predict_with_threshold.py      # LanguageDetectorWithThreshold
│                                          # + find_optimal_threshold()
├── output/                        # создаётся автоматически
├── lang_detection_hackathon.csv
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Требования

```
Python 3.11+  ·  Docker  ·  Docker Compose
pandas>=2.0.0  ·  scikit-learn>=1.3.0  ·  fasttext>=0.9.2
numpy<2.0.0  ·  joblib>=1.2.0
```
