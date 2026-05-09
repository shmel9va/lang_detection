# Комбинированная модель определения языка

Модель для определения языка текста в чатах поддержки такси.  
Целевой тайминг: ≤ 30 мс на одну классификацию.

---

## Датасет

Файл: `lang_detection_diploma.csv`  
Поля: `request_text`, `result`  
Размер: **35 меток: 27 × 2500 + 8 × 300 = 69 900 строк**

### Поддерживаемые языки (21 + other)

| Метка | Язык | Группа |
|-------|------|--------|
| `hy_arm` | Армянский (армянский алфавит) | → `hy` |
| `hy_lat` | Армянский (латинская транскрипция) | → `hy` |
| `ka_geo` | Грузинский (грузинский алфавит) | → `ka` |
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

### Неподдерживаемые языки → `other` (×300)

| Метка | Язык | Почему добавлен |
|-------|------|-----------------|
| `de` | Немецкий | Путается с `en`, `nl` |
| `it` | Итальянский | Путается с `es`, `pt` |
| `nl` | Нидерландский | Путается с `en`, `de` |
| `pl` | Польский | Путается с `ro`, `en` |
| `bg` | Болгарский | Кириллица — путается с `ru`, `uk` |
| `ja` | Японский | CJK — быстрый путь |
| `vi` | Вьетнамский | Латиница с диакритиками |
| `ko` | Корейский | Hangul — быстрый путь |

Метки объединяются постпроцессором: `hy_arm`/`hy_lat` → `hy`, `ka_geo`/`ka_lat` → `ka`, `de`/`ja`/... → `other`.

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
│      CJK/Hangul → other                      │
└───────────┬──────────────────────────────────┘
             │ если скрипт неоднозначен
             ▼
┌──────────────────────────────────────────────┐
│ 3. fastText                                  │
│    Топ-2 языка с вероятностями               │
│    Обучен на 35 исходных метках              │
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
│    Если confidence ≥ 0.90 → fastText ответ   │
│    Иначе → результат fastText                │
└───────────┬──────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ 5. Постпроцессор (merge_label)               │
│    hy_arm / hy_lat → hy                      │
│    ka_geo / ka_lat → ka                      │
│    uz_lat / uz_cyr → uz                      │
│    ur_ur  / ur_lat → ur                      │
│    ne_nep / ne_lat → ne                      │
│    sr_lat / sr_cyr → sr                      │
│    de / it / ja / ko / ... → other           │
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

### Шаг 2. Обучение fastText

```bash
docker-compose run --rm trainer
```

Читает `output/train.csv`, применяет `preprocess_text` один раз, подбирает число эпох на val,
затем обучает итоговую модель. Шаг `preprocessor` **не нужен** — предобработка встроена в trainer.

Результаты:
- `output/lang_detection_model.bin`
- `output/model_evaluation.txt`

**Время:** 5–15 минут.

---

### Шаг 3. Подбор threshold

```bash
docker-compose run --rm threshold_finder
```

Перебирает пороги [0.3 … 0.9] на `val.csv`. Выбирает минимальный порог в пределах 0.5% от лучшего accuracy.

Результаты: `output/threshold_results.txt`

---

### Шаг 4. Обучение бинарных классификаторов

```bash
docker-compose run --rm trainer_sensitive
```

Тексты нормализуются через `normalize_for_detection` (как в пайплайне).

Результаты: `output/sensitive_classifiers/ar_fa.pkl`, `hy_az.pkl`, `he_ar.pkl`, `ur_hi.pkl`, `ru_uk.pkl`

**Время:** < 1 минуты.

---

### Шаг 5. Бенчмарк скорости

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
| `baseline_trainer` | `docker-compose run --rm baseline_trainer` | Обучение baseline fastText (21 класс) |
| `comparer_solutions` | `docker-compose run --rm comparer_solutions` | Сравнение baseline vs combined |

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
| `ka` | Грузинский | `ka_geo`, `ka_lat` | Грузинский / Латиница |
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
| `other` | Неподдерживаемые | `de`, `it`, `nl`, `pl`, `bg`, `ja`, `vi`, `ko` | CJK/Hangul → быстрый путь |

Итого: **22 выходных кода** (35 меток в датасете, 14 групп объединяются).

---

## Исправленные ошибки

| # | Статус | Исправление |
|---|--------|-------------|
| 1 | **Исправлено** | `train_fasttext.py` — оценка только на val.csv, test не трогается |
| 2 | **Исправлено** | `preprocess_text.py` — убран `skiprows=[0,1]`, нормальное чтение CSV |
| 3 | **Исправлено** | `train_fasttext.py` — реальные параметры (epoch, dim, loss) в отчёте |
| 4 | **Исправлено** | `predict_with_threshold.py` — модель загружается 1 раз, перебираются только пороги |
| 5 | **Исправлено** | `predict_with_threshold.py` — добавлены `normalize_for_detection` + `preprocess_text` |
| 6 | **Исправлено** | `sensitive_router.py` — при `probs[0] >= 0.90` fastText не переопределяется |
| 7 | **Исправлено** | Добавлены 8 языков (de/it/nl/pl/bg/ja/vi/ko) → `other`, script_detector для CJK/Hangul |
| 8 | **Исправлено** | `train_fasttext.py` — `preprocess_text` применяется всегда |
| 9 | **Исправлено** | `benchmark_speed.py` — убран `skiprows=[0,1]` |
| 10 | **Исправлено** | `base.py` — необученный классификатор возвращает `None` → fallback на fastText |
| 11 | **Исправлено** | `_pick_threshold` — выбирает порог с лучшим accuracy (минимальный в пределах 0.5%) |
| 12 | **Исправлено** | `train_fasttext.py` — читает `train.csv` напрямую, preprocess_text 1 раз (не 2) |
| 13 | **Исправлено** | `find_optimal_epochs.py` — добавлена preprocess_text для train и val |
| 14 | **Исправлено** | `sensitive_router.py` — если top1 входит в чувствительную пару (но top2 нет) — прогоняет через LogReg |
| 15 | **Исправлено** | `train_sensitive_classifiers.py` — тексты нормализуются через `normalize_for_detection` |
| 16 | **Исправлено** | `base.py` — `evaluate` не падает если predict вернул None |

---

## Базовая модель (baseline)

Простой fastText на 21 объединённом классе. Без бинарных классификаторов, детектора скрипта и роутера.

```bash
docker-compose run --rm splitter           # шаг 1 — разделение данных
docker-compose run --rm baseline_trainer   # обучение baseline
```

Результат: `output/baseline_model.bin`, `output/baseline_evaluation.txt`

---

## Сравнение моделей

Сравнение базовой и комбинированной моделей на `test.csv`:

```bash
docker-compose run --rm comparer_solutions
```

Выводит: accuracy, macro F1, ошибки по чувствительным парам, скорость.

Результат: `output/solutions_comparison.txt`

---

## Что ещё нужно проекту

- [x] **Базовая модель** — `train_baseline_fasttext.py`
- [x] **Скрипт сравнения** — `compare_solutions.py`
- [x] **Исправить ошибки #1–#10** — все 10 исправлены
- [x] **Класс «other»** — 8 языков × 300 примеров, CJK/Hangul в script_detector
- [ ] **Переобучить обе модели** — `docker-compose build` → полный pipeline
- [ ] **Подобрать threshold** — `docker-compose run --rm threshold_finder`
- [ ] **Сравнить модели** — `docker-compose run --rm comparer_solutions`

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
│   ├── compare_models.py
│   └── compare_solutions.py       # сравнение baseline vs combined
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
│   │   ├── train_baseline_fasttext.py  # baseline: 21 класс, без роутера
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
