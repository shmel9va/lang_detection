"""
Скрипт для сравнения результатов разных моделей
"""
import os


def parse_evaluation_file(filepath):
    """
    Парсит файл с оценкой модели и извлекает ключевые метрики
    """
    if not os.path.exists(filepath):
        return None
    
    metrics = {
        'accuracy': None,
        'macro_f1': None,
        'macro_precision': None,
        'macro_recall': None,
        'model_name': os.path.basename(filepath)
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Ищем Accuracy
        for line in content.split('\n'):
            if 'Accuracy' in line and ':' in line:
                try:
                    acc_str = line.split(':')[1].strip().replace('%', '')
                    metrics['accuracy'] = float(acc_str)
                except:
                    pass
            
            # Ищем Macro F1
            if ('F1-score' in line or 'F1' in line) and 'MACRO' in content[:content.index(line) + 100]:
                if ':' in line:
                    try:
                        f1_str = line.split(':')[1].strip().replace('%', '').replace('⭐', '').strip()
                        if metrics['macro_f1'] is None:
                            metrics['macro_f1'] = float(f1_str)
                    except:
                        pass
            
            # Ищем Macro Precision
            if 'Precision' in line and 'MACRO' in content[:content.index(line) + 100]:
                if ':' in line:
                    try:
                        p_str = line.split(':')[1].strip().replace('%', '')
                        if metrics['macro_precision'] is None:
                            metrics['macro_precision'] = float(p_str)
                    except:
                        pass
            
            # Ищем Macro Recall
            if 'Recall' in line and 'MACRO' in content[:content.index(line) + 100]:
                if ':' in line:
                    try:
                        r_str = line.split(':')[1].strip().replace('%', '')
                        if metrics['macro_recall'] is None:
                            metrics['macro_recall'] = float(r_str)
                    except:
                        pass
    
    return metrics


def compare_models():
    """
    Сравнивает результаты всех обученных моделей
    """
    print("=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    models = [
        ('Базовая модель', 'output/model_evaluation.txt'),
    ]
    
    results = []
    
    for model_name, filepath in models:
        metrics = parse_evaluation_file(filepath)
        if metrics:
            results.append((model_name, metrics))
            print(f"\n{model_name}: найдены результаты")
        else:
            print(f"\n{model_name}: результаты не найдены ({filepath})")
    
    if not results:
        print("\nНет данных для сравнения. Обучите модели сначала.")
        return
    
    # Сортируем по Macro F1
    results.sort(key=lambda x: x[1]['macro_f1'] if x[1]['macro_f1'] else 0, reverse=True)
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ (отсортировано по Macro F1)")
    print("=" * 80)
    
    print(f"\n{'Модель':<30} {'Accuracy':>12} {'Macro F1':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 80)
    
    for model_name, metrics in results:
        acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] else "N/A"
        f1 = f"{metrics['macro_f1']:.2f}%" if metrics['macro_f1'] else "N/A"
        prec = f"{metrics['macro_precision']:.2f}%" if metrics['macro_precision'] else "N/A"
        rec = f"{metrics['macro_recall']:.2f}%" if metrics['macro_recall'] else "N/A"
        
        print(f"{model_name:<30} {acc:>12} {f1:>12} {prec:>12} {rec:>12}")
    
    # Лучшая модель
    if results:
        best_model_name, best_metrics = results[0]
        print("\n" + "=" * 80)
        print("🏆 ЛУЧШАЯ МОДЕЛЬ")
        print("=" * 80)
        print(f"\nМодель: {best_model_name}")
        print(f"Macro F1: {best_metrics['macro_f1']:.2f}%")
        print(f"Accuracy: {best_metrics['accuracy']:.2f}%")
        
        # Сравнение с базовой
        baseline = None
        for model_name, metrics in results:
            if 'Базовая' in model_name:
                baseline = metrics
                break
        
        if baseline and best_model_name != 'Базовая модель':
            improvement = best_metrics['macro_f1'] - baseline['macro_f1']
            print(f"\n✨ Улучшение по сравнению с базовой моделью: {improvement:+.2f}%")
    
    # Сохраняем отчет
    output_file = 'output/models_comparison.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("СРАВНЕНИЕ МОДЕЛЕЙ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Модель':<30} {'Accuracy':>12} {'Macro F1':>12} {'Precision':>12} {'Recall':>12}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, metrics in results:
            acc = f"{metrics['accuracy']:.2f}%" if metrics['accuracy'] else "N/A"
            f1 = f"{metrics['macro_f1']:.2f}%" if metrics['macro_f1'] else "N/A"
            prec = f"{metrics['macro_precision']:.2f}%" if metrics['macro_precision'] else "N/A"
            rec = f"{metrics['macro_recall']:.2f}%" if metrics['macro_recall'] else "N/A"
            
            f.write(f"{model_name:<30} {acc:>12} {f1:>12} {prec:>12} {rec:>12}\n")
        
        if results:
            best_model_name, best_metrics = results[0]
            f.write(f"\n\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}\n")
            f.write(f"Macro F1: {best_metrics['macro_f1']:.2f}%\n")
    
    print(f"\nОтчет сохранен: {output_file}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    compare_models()

