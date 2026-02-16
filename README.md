# Language Detection

Модель для определения языка текста на основе FastText.

## Метрики

- **Accuracy:** 96.12%
- **Macro F1:** 96.01%
- **Precision:** 96.07%
- **Recall:** 96.00%
- **Epochs:** 40 (определено автоматически)
- **Скорость:** 0.08 мс/текст

## Использование

### Базовый вариант

```python
import fasttext
from scripts.utils.label_mapping import merge_label

model = fasttext.load_model('output/lang_detection_model.bin')

text = "Привет, как дела?"
predictions = model.predict(text, k=1)
label = predictions[0][0].replace('__label__', '')
language = merge_label(label)
confidence = predictions[1][0]

print(f"Язык: {language}, Уверенность: {confidence:.2%}")
```

### С threshold

```python
from scripts.utils.predict_with_threshold import LanguageDetectorWithThreshold

detector = LanguageDetectorWithThreshold(
    model_path='output/lang_detection_model.bin',
    threshold=0.7
)

language, confidence = detector.predict("Hello world", return_confidence=True)
print(f"Язык: {language}, Уверенность: {confidence:.2f}")
```

Если `confidence < 0.7`, модель вернет `'other'`.

## Обучение

### Сборка

```bash
docker-compose build
```

### Запуск пайплайна

```bash
docker-compose run --rm analyzer
docker-compose run --rm cleaner
docker-compose run --rm splitter
docker-compose run --rm preprocessor
docker-compose run --rm trainer
```

### Бенчмарк

```bash
docker-compose run --rm benchmark
```

## Параметры

- **Epochs:** 40 (early stopping, прирост < 0.3%)
- **Dimension:** 150
- **Word N-grams:** 2
- **Subword:** 3-6 символов
- **Min count:** 2
- **Loss:** softmax
- **Seed:** 42

## Датасет

- **Всего:** 26,143 примеров
- **Train:** 20,914 (80%)
  - Train для обучения: 17,575 (85%)
  - Validation для подбора эпох: 3,102 (15%)
- **Test:** 5,229 (20%)
- **Языки:** 20 (после объединения меток)

### Объединение меток

- `uz_lat` + `uz_kir` = `uz`
- `ur` + `ur_arab` = `ur`
- `ne` + `ne_lat` = `ne`

## Производительность

### Скорость

- **Одиночное предсказание:** 0.08 мс (среднее)
- **Throughput:** 22,000 текстов/сек
- **Загрузка модели:** 5.5 сек

### По длине текста

- Короткие (<50 символов): 0.03 мс
- Средние (50-200): 0.08 мс
- Длинные (200-500): 0.17 мс
- Очень длинные (>500): 0.62 мс

## Проблемные пары

Модель иногда путает похожие языки:

- **tr / az** (турецкий / азербайджанский): 39 ошибок
- **tr / uz** (турецкий / узбекский): 8 ошибок
- **az / uz** (азербайджанский / узбекский): 8 ошибок

## Метрики по языкам

Лучшие (F1 > 99%):
- **he** (иврит): 100%
- **ar** (арабский): 99.35%
- **fa** (фарси): 99.01%

Проблемные (F1 < 95%):
- **tr** (турецкий): 84.30%
- **az** (азербайджанский): 85.78%
- **pt** (португальский): 92.89%

## Логика 'other'

Редкие языки (`mk`, `mr`, `bs`, `tg`, `de`, `ky`, `zh`) удаляются из обучения.

'other' определяется через threshold:
- Если `confidence < 0.7` = 'other'
- Порог настраивается в `LanguageDetectorWithThreshold`

## Поддерживаемые языки

am, ar, az, en, es, fa, fr, he, hi, hy, ka, kk, ne, pt, ro, ru, sr, tr, ur, uz

## Требования

- Python 3.11+
- Docker
- FastText, pandas, scikit-learn

## Установка

```bash
pip install -r requirements.txt
```

Или через Docker:

```bash
docker-compose build
```
