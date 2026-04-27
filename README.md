# Комбинированная модель определения языка

Модель для определения языка текста в чатах поддержки такси.  
Целевой тайминг: ≤ 30 мс на одну классификацию.

---

## Датасет

Файл: `lang_detection_diploma.csv`  
Поля: `request_text`, `result`  
Размер: **27 меток × 2 500 примеров = 70 000 строк** ( — 5 000, т.к. входит в две чувствительные пары)

| Метка | Язык | Группа |
|-------|------|--------|
| `hy_arm` | Армянский (армянский алфавит) | → `hy` |
| `hy_lat` | Армянский (латинская транскрипция) | → `hy` |
| `ka` | Грузинский (грузинский алфавит) | — |
| `ka_lat` | Грузинский (латинская транскрипция) | → `ka` |
| `uz_lat` | Узбекский (латиница) | → `uz` |
| `uz_cyr` | Узбекский (кириллица) | → `uz` |
| `ur_ur` | Урду (арабский скрипт / насталик) | → `ur` |
| `ur_lat` | Урду (латинская транскрипция) | → `ur` |
| `ne_nep` | Непальский (деванагари) | → `ne` |
| `ne_lat` | Непальский (латинская транскрипция) | → `ne` |
| `sr_cyr` | Сербский (кириллица) | → `sr` |
| `sr_lat` | Сербский (латиница) | → `sr` |
| `he` | Иврит | — |
| `ar` | Арабский | — |
| `am` | Амхарский | — |
| `az` | Азербайджанский | — |
| `en` | Английский | — |
| `es` | Испанский | — |
| `fa` | Персидский | — |
| `fr` | Французский | — |
| `hi` | Хинди | — |
| `kk` | Казахский | — |
| `pt` | Португальский | — |
| `ro` | Румынский | — |
| `ru` | Русский | — |
| `tr` | Турецкий | — |
| `uk` | Украинский | — |

Метки с суффиксами объединяются постпроцессором: `hy_arm`/`hy_lat` → `hy`, `ka_lat` → `ka` и т.д.

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
│      hy_arm (армянский)   ka (грузинский)│
│      he (иврит)           am (амхарский)     │
└───────────┬──────────────────────────────────┘
            │ если скрипт неоднозначен
            ▼
┌──────────────────────────────────────────────┐
│ 3. fastText                                  │
│    Топ-2 языка с вероятностями               │
│    Обучен на всех 27 исходных метках         │
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
│ 5. Постпроцессор (merge_label)               │
│    hy_arm / hy_lat → hy                      │
│    ka_lat → ka                      │
│    uz_lat / uz_cyr → uz                      │
│    ur_ur  / ur_lat → ur                      │
│    ne_nep / ne_lat → ne                      │
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
lang_detection_diploma.csv
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

> Сервис `cleaner` **не используется** — все 27 классов имеют по 2 500 примеров, редких меток нет.

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
- все 5 пар имеют данные в новом датасете — `ru_uk.pkl` теперь обучается полноценно

Результаты: `output/sensitive_classifiers/ar_fa.pkl`, `hy_az.pkl`, `he_ar.pkl`, `ur_hi.pkl`, `ru_uk.pkl`

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

После — вручную запустить финальную оценку (шаг 7).

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
    threshold=0.5,  # заменить на результат find_threshold
)

lang, conf = detector.detect("Привет, как дела?")
# → ('ru', 0.97)

lang, conf = detector.detect("Բարի՜ Ծաղիկ")
# → ('hy', 0.99)  — армянский алфавит (hy_arm), быстрый путь

lang, conf = detector.detect("Salam, necəsən?")
# → ('az', 0.95)  — латиница, fastText

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
    └── ru_uk.pkl

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

> Актуальные метрики получить после переобучения на `lang_detection_diploma.csv`.  
> Запусти шаги 1–7 и смотри `output/final_evaluation.txt`.

Ориентировочные цели (датасет сбалансирован, 2 500 примеров на класс):

| Метрика | Цель |
|---------|------|
| Accuracy | ≥ 97% |
| Macro F1 | ≥ 97% |

---

## Поддерживаемые языки (на выходе детектора)

| Код | Язык | Исходные метки | Скрипт |
|-----|------|----------------|--------|
| `am` | Амхарский | `am` | Эфиопский (быстрый путь) |
| `ar` | Арабский | `ar` | Арабское письмо |
| `az` | Азербайджанский | `az` | Латиница |
| `en` | Английский | `en` | Латиница |
| `es` | Испанский | `es` | Латиница |
| `fa` | Персидский | `fa` | Арабское письмо |
| `fr` | Французский | `fr` | Латиница |
| `he` | Иврит | `he` | Иврит (быстрый путь) |
| `hi` | Хинди | `hi` | Деванагари |
| `hy` | Армянский | `hy_arm`, `hy_lat` | Армянский / Латиница |
| `ka` | Грузинский | `ka`, `ka_lat` | Грузинский / Латиница |
| `kk` | Казахский | `kk` | Кириллица |
| `ne` | Непальский | `ne_nep`, `ne_lat` | Деванагари / Латиница |
| `pt` | Португальский | `pt` | Латиница |
| `ro` | Румынский | `ro` | Латиница |
| `ru` | Русский | `ru` | Кириллица |
| `sr` | Сербский | `sr_cyr`, `sr_lat` | Кириллица / Латиница |
| `tr` | Турецкий | `tr` | Латиница |
| `uk` | Украинский | `uk` | Кириллица |
| `ur` | Урду | `ur_ur`, `ur_lat` | Арабское письмо / Латиница |
| `uz` | Узбекский | `uz_lat`, `uz_cyr` | Латиница / Кириллица |

Итого: **21 выходной код** (27 меток в датасете, 6 групп объединяются).

---

## Что ещё нужно проекту

- [ ] **Переобучить модели** — после появления нового датасета все `output/*.bin` и `output/sensitive_classifiers/*.pkl` устарели. Запустить полный pipeline (шаги 1–7).
- [ ] **Подобрать новый threshold** — после переобучения fastText запустить `threshold_finder` и обновить `detector.py`.
- [ ] **Оценить качество** `hy_lat` и `ka_lat` — латинские транскрипции армянского и грузинского нетипичны; если accuracy ниже 90%, рассмотреть повышение числа эпох или n-gram диапазона в fastText.
- [ ] **Добавить детектор скрипта для `ne_nep`** — деванагари уникален, но shared с `hi`. В `script_detector.py` можно добавить быстрый путь для Devanagari → отправлять в `UrHiClassifier` вместо fastText.
- [ ] **Расширить SensitiveRouter** — пара `hy_lat` vs `az` (латиница) может смешиваться; рассмотреть добавление LogReg-классификатора для этого случая.
- [ ] **Тест-бенчмарк на производственных данных** — реальные чаты поддержки могут иметь иное распределение длин и ошибок, чем учебный датасет.

---

## Структура проекта

```
lang_detection/
├── data/                          # сырые источники данных по языкам
│   ├── armenian_dataset.xlsx
│   ├── georgian_dataset.xlsx
│   └── nepali_dataset.xlsx
├── benchmarks/
│   ├── benchmark_speed.py         # замер скорости (fastText + полный пайплайн)
│   └── compare_models.py
├── scripts/
│   ├── dataset_collection/        # скрипты сбора и подготовки сырых данных
│   │   ├── armenian_dataset_to_xlsx.py
│   │   ├── georgian_dataset_to_xlsx.py
│   │   └── nepali_dataset_to_xlsx.py
│   ├── data_processing/           # обработка собранного датасета
│   │   ├── analyze_dataset.py
│   │   ├── clean_dataset.py       # удаление редких классов (в новом датасете не нужен)
│   │   ├── preprocess_text.py     # очистка + normalize_for_detection
│   │   └── split_dataset.py       # 70/15/15 → train/val/test
│   ├── detection/
│   │   ├── detector.py            # LanguageDetector — точка входа
│   │   ├── script_detector.py     # уникальные алфавиты → быстрый путь
│   │   ├── sensitive_router.py    # роутер чувствительных пар
│   │   └── sensitive_classifiers/
│   │       ├── base.py            # char n-gram TF-IDF + LogReg
│   │       ├── ar_fa.py           # арабский / персидский
│   │       ├── ru_uk.py           # русский / украинский
│   │       ├── hy_az.py           # армянский / азербайджанский
│   │       ├── he_ar.py           # иврит / арабский
│   │       └── ur_hi.py           # урду / хинди
│   ├── diagnostic/                # диагностика датасета
│   │   ├── check_data_loss.py
│   │   └── verify_dataset.py
│   ├── training/
│   │   ├── train_fasttext.py
│   │   ├── find_optimal_epochs.py
│   │   ├── train_sensitive_classifiers.py
│   │   ├── find_threshold.py      # docker-compose run --rm threshold_finder
│   │   └── evaluate_final.py      # docker-compose run --rm evaluator
│   └── utils/
│       ├── label_mapping.py       # hy_arm/hy_lat→hy, ka_lat→ka, uz_lat/uz_cyr→uz, …
│       └── predict_with_threshold.py
├── output/                        # создаётся автоматически
├── lang_detection_diploma.csv    # финальный датасет (вход пайплайна)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Требования

```
Python 3.11+  ·  Docker  ·  Docker Compose
pandas>=2.0.0  ·  scikit-learn>=1.3.0  ·  fasttext>=0.9.2
numpy<2.0.0  ·  joblib>=1.2.0  ·  openpyxl>=3.1.0
```
