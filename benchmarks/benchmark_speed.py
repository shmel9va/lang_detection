"""
Бенчмарк для измерения скорости предсказания модели.

Разделён на два блока:
  A. Бенчмарк fastText-only (существующий, не изменён)
  B. Бенчмарк полного пайплайна LanguageDetector (новый)
"""
import fasttext
import os
import pandas as pd
import time
import numpy as np
from scripts.utils.predict_with_threshold import LanguageDetectorWithThreshold


def benchmark_model_loading(model_path='output/lang_detection_model.bin'):
    """
    Измеряет время загрузки модели
    """
    print("=" * 80)
    print("1. БЕНЧМАРК: ЗАГРУЗКА МОДЕЛИ")
    print("=" * 80)
    
    start = time.time()
    model = fasttext.load_model(model_path)
    load_time = time.time() - start
    
    print(f"\nВремя загрузки модели: {load_time:.4f} секунд")
    print(f"  Размер модели: {model.get_dimension()} dimensions")
    print(f"  Количество слов: {len(model.get_words())}")
    print(f"  Количество меток: {len(model.get_labels())}")
    
    return model, load_time


def benchmark_single_prediction(model, texts):
    """
    Измеряет время предсказания для 1 текста
    """
    print("\n" + "=" * 80)
    print("2. БЕНЧМАРК: ОДИНОЧНОЕ ПРЕДСКАЗАНИЕ")
    print("=" * 80)
    
    # Прогрев (первое предсказание может быть медленнее)
    _ = model.predict(texts[0], k=1)
    
    # Измеряем на разных текстах
    times = []
    for text in texts[:100]:  # Первые 100 текстов
        start = time.time()
        _ = model.predict(text, k=1)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    print(f"\nСтатистика (100 предсказаний):")
    print(f"  Среднее время: {times.mean()*1000:.2f} мс")
    print(f"  Медиана: {np.median(times)*1000:.2f} мс")
    print(f"  Минимум: {times.min()*1000:.2f} мс")
    print(f"  Максимум: {times.max()*1000:.2f} мс")
    print(f"  Стандартное отклонение: {times.std()*1000:.2f} мс")
    
    print(f"\nСреднее время предсказания 1 текста: {times.mean()*1000:.2f} мс")
    
    return times.mean()


def benchmark_batch_prediction(model, texts, batch_sizes=[10, 50, 100, 500, 1000]):
    """
    Измеряет время предсказания для batch
    """
    print("\n" + "=" * 80)
    print("3. БЕНЧМАРК: BATCH ПРЕДСКАЗАНИЕ")
    print("=" * 80)
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(texts):
            continue
        
        batch = texts[:batch_size]
        
        start = time.time()
        for text in batch:
            _ = model.predict(text, k=1)
        elapsed = time.time() - start
        
        time_per_text = (elapsed / batch_size) * 1000
        throughput = batch_size / elapsed
        
        results[batch_size] = {
            'total_time': elapsed,
            'time_per_text': time_per_text,
            'throughput': throughput
        }
        
        print(f"\nBatch size: {batch_size}")
        print(f"  Общее время: {elapsed:.4f} сек")
        print(f"  Время на 1 текст: {time_per_text:.2f} мс")
        print(f"  Throughput: {throughput:.1f} текстов/сек")
    
    return results


def benchmark_with_threshold(model_path, texts, thresholds=[0.5, 0.7, 0.9]):
    """
    Измеряет время предсказания с threshold
    """
    print("\n" + "=" * 80)
    print("4. БЕНЧМАРК: ПРЕДСКАЗАНИЕ С THRESHOLD")
    print("=" * 80)
    
    results = {}
    
    for threshold in thresholds:
        detector = LanguageDetectorWithThreshold(model_path, threshold=threshold)
        
        times = []
        for text in texts[:100]:
            start = time.time()
            _ = detector.predict(text, return_confidence=True)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        avg_time = times.mean() * 1000
        
        results[threshold] = avg_time
        
        print(f"\nThreshold: {threshold}")
        print(f"  Среднее время: {avg_time:.2f} мс")
    
    return results


def benchmark_text_length_impact(model, texts):
    """
    Измеряет влияние длины текста на скорость
    """
    print("\n" + "=" * 80)
    print("5. БЕНЧМАРК: ВЛИЯНИЕ ДЛИНЫ ТЕКСТА")
    print("=" * 80)
    
    # Группируем тексты по длине
    length_groups = {
        'Короткие (< 50 символов)': [t for t in texts if len(t) < 50],
        'Средние (50-200)': [t for t in texts if 50 <= len(t) < 200],
        'Длинные (200-500)': [t for t in texts if 200 <= len(t) < 500],
        'Очень длинные (> 500)': [t for t in texts if len(t) >= 500],
    }
    
    results = {}
    
    for group_name, group_texts in length_groups.items():
        if len(group_texts) < 10:
            continue
        
        times = []
        for text in group_texts[:50]:  # Первые 50 из группы
            start = time.time()
            _ = model.predict(text, k=1)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        avg_length = np.mean([len(t) for t in group_texts[:50]])
        avg_time = times.mean() * 1000
        
        results[group_name] = {
            'avg_length': avg_length,
            'avg_time': avg_time,
            'count': len(group_texts)
        }
        
        print(f"\n{group_name}:")
        print(f"  Средняя длина: {avg_length:.0f} символов")
        print(f"  Количество текстов: {len(group_texts)}")
        print(f"  Среднее время: {avg_time:.2f} мс")
    
    return results


def benchmark_top_k(model, texts, k_values=[1, 3, 5]):
    """
    Измеряет влияние параметра k на скорость
    """
    print("\n" + "=" * 80)
    print("6. БЕНЧМАРК: TOP-K ПРЕДСКАЗАНИЯ")
    print("=" * 80)
    
    results = {}
    
    for k in k_values:
        times = []
        for text in texts[:100]:
            start = time.time()
            _ = model.predict(text, k=k)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        avg_time = times.mean() * 1000
        
        results[k] = avg_time
        
        print(f"\nTop-{k} предсказания:")
        print(f"  Среднее время: {avg_time:.2f} мс")
    
    return results


def generate_report(results):
    """
    Генерирует итоговый отчет
    """
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    single_ms = results['single_time'] * 1000
    
    print(f"\nВремя загрузки модели: {results['load_time']:.4f} сек")
    print(f"Среднее время предсказания: {single_ms:.2f} мс")
    print(f"Максимальный throughput: {results['max_throughput']:.1f} текстов/сек")
    
    output_file = 'output/speed_benchmark.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("БЕНЧМАРК СКОРОСТИ МОДЕЛИ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Время загрузки: {results['load_time']:.4f} сек\n")
        f.write(f"Среднее время предсказания: {single_ms:.2f} мс\n")
        f.write(f"Максимальный throughput: {results['max_throughput']:.1f} текстов/сек\n")
    
    print(f"\nОтчет сохранен: {output_file}")


def benchmark_full_pipeline(texts, model_path="output/lang_detection_model.bin",
                            classifiers_dir="output/sensitive_classifiers",
                            n_warmup=10, n_measure=200):
    """
    Бенчмарк полного пайплайна LanguageDetector:
    normalize → script_detector → fastText → router → postprocess.

    Включает все уровни, которые видит реальный запрос в проде.
    """
    print("\n" + "=" * 80)
    print("7. БЕНЧМАРК: ПОЛНЫЙ ПАЙПЛАЙН (LanguageDetector)")
    print("=" * 80)

    try:
        from scripts.detection.detector import LanguageDetector
    except ImportError as e:
        print(f"  Не удалось импортировать LanguageDetector: {e}")
        return None

    if not os.path.exists(model_path):
        print(f"  Модель не найдена: {model_path}")
        return None

    print(f"\nЗагрузка LanguageDetector...")
    t0 = time.time()
    detector = LanguageDetector(
        fasttext_model_path=model_path,
        sensitive_classifiers_dir=classifiers_dir,
        threshold=0.5,
        router_verbose=False,
    )
    load_time = time.time() - t0
    print(f"Время загрузки: {load_time:.3f} сек")

    sample = texts[:max(n_warmup + n_measure, len(texts))]

    # Прогрев
    for text in sample[:n_warmup]:
        _ = detector.detect(text)

    # Измерение
    measure_texts = sample[n_warmup: n_warmup + n_measure]
    if not measure_texts:
        measure_texts = sample[:50]

    times = []
    for text in measure_texts:
        t0 = time.time()
        _ = detector.detect(text)
        times.append(time.time() - t0)

    times = np.array(times) * 1000  # → мс

    print(f"\nСтатистика ({len(times)} измерений, мс):")
    print(f"  Среднее:    {times.mean():.3f} мс")
    print(f"  Медиана:    {np.median(times):.3f} мс")
    print(f"  P95:        {np.percentile(times, 95):.3f} мс")
    print(f"  P99:        {np.percentile(times, 99):.3f} мс")
    print(f"  Минимум:    {times.min():.3f} мс")
    print(f"  Максимум:   {times.max():.3f} мс")

    throughput = 1000.0 / times.mean()
    print(f"\nThroughput:   {throughput:.0f} текстов/сек")

    target_ms = 30.0
    ok = times.mean() < target_ms
    print(f"\nЦелевой порог: {target_ms} мс → {'✓ ВЫПОЛНЕН' if ok else '✗ превышен'}")

    # Замеряем разбивку по длине
    groups = {
        "< 50 символов":    [t for t in measure_texts if len(t) < 50],
        "50–200 символов":  [t for t in measure_texts if 50 <= len(t) < 200],
        "200–500 символов": [t for t in measure_texts if 200 <= len(t) < 500],
        "> 500 символов":   [t for t in measure_texts if len(t) >= 500],
    }

    print(f"\nВлияние длины текста:")
    for group, gtexts in groups.items():
        if len(gtexts) < 3:
            continue
        gtimes = []
        for text in gtexts[:50]:
            t0 = time.time()
            _ = detector.detect(text)
            gtimes.append((time.time() - t0) * 1000)
        print(f"  {group:<22}: {np.mean(gtimes):.3f} мс  (n={len(gtimes)})")

    return {"mean_ms": float(times.mean()), "p99_ms": float(np.percentile(times, 99)),
            "throughput": throughput, "load_time": load_time}


def main():
    """
    Запуск всех бенчмарков
    """
    print("\n" + "=" * 80)
    print("БЕНЧМАРК СКОРОСТИ МОДЕЛИ ОПРЕДЕЛЕНИЯ ЯЗЫКА")
    print("=" * 80)
    
    model_path = 'output/lang_detection_model.bin'
    test_file = 'output/test.csv'
    
    # Загрузка тестовых данных
    print(f"\nЗагрузка тестовых данных: {test_file}")
    
    try:
        df = pd.read_csv(test_file, sep=';', encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        for enc in ('utf-8-sig', 'cp1251', 'latin-1'):
            try:
                df = pd.read_csv(test_file, sep=';', encoding=enc, on_bad_lines='skip')
                break
            except Exception:
                continue
        else:
            df = pd.read_csv(test_file, sep=';')
    
    texts = df['request_text'].dropna().astype(str).tolist()
    # Очищаем тексты от символов новой строки (fastText требование)
    texts = [text.replace('\n', ' ').replace('\r', ' ').strip() for text in texts]
    print(f"Загружено {len(texts)} текстов\n")
    
    # Запуск бенчмарков
    results = {}
    
    # 1. Загрузка модели
    model, load_time = benchmark_model_loading(model_path)
    results['load_time'] = load_time
    
    # 2. Одиночное предсказание
    single_time = benchmark_single_prediction(model, texts)
    results['single_time'] = single_time
    
    # 3. Batch предсказание
    batch_results = benchmark_batch_prediction(model, texts)
    results['max_throughput'] = max(r['throughput'] for r in batch_results.values())
    
    # 4. С threshold
    threshold_results = benchmark_with_threshold(model_path, texts)
    
    # 5. Влияние длины текста
    length_results = benchmark_text_length_impact(model, texts)
    
    # 6. Top-k
    topk_results = benchmark_top_k(model, texts)
    
    # Итоговый отчет
    generate_report(results)

    # 7. Полный пайплайн LanguageDetector
    pipeline_results = benchmark_full_pipeline(texts)
    if pipeline_results:
        results["pipeline_mean_ms"] = pipeline_results["mean_ms"]
        results["pipeline_p99_ms"] = pipeline_results["p99_ms"]
        results["pipeline_throughput"] = pipeline_results["throughput"]

        # Дописываем сравнение в файл отчёта
        output_file = "output/speed_benchmark.txt"
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n\nПОЛНЫЙ ПАЙПЛАЙН (LanguageDetector)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Среднее время:  {pipeline_results['mean_ms']:.3f} мс\n")
            f.write(f"P99:            {pipeline_results['p99_ms']:.3f} мс\n")
            f.write(f"Throughput:     {pipeline_results['throughput']:.0f} текстов/сек\n")
            f.write(f"Целевой порог 30 мс: {'выполнен' if pipeline_results['mean_ms'] < 30 else 'превышен'}\n")

    print("\n" + "=" * 80)
    print("БЕНЧМАРК ЗАВЕРШЕН")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

