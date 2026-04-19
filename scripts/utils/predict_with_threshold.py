"""
Модуль для предсказания языка с поддержкой threshold для определения 'other'
"""
import fasttext
from scripts.utils.label_mapping import merge_label


class LanguageDetectorWithThreshold:
    """
    Детектор языка с поддержкой определения неизвестных языков через threshold
    """
    
    def __init__(self, model_path, threshold=0.7):
        """
        Инициализация детектора
        
        Args:
            model_path: путь к модели fastText
            threshold: порог уверенности (0.0-1.0)
                      Если confidence < threshold, возвращается 'other'
        """
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold
        print(f"Модель загружена: {model_path}")
        print(f"Threshold: {threshold}")
    
    def predict(self, text, return_confidence=False, merge_labels=True):
        """
        Предсказание языка с учетом threshold
        
        Args:
            text: текст для классификации
            return_confidence: вернуть также уверенность модели
            merge_labels: объединить метки (uz_lat/uz_kir -> uz)
        
        Returns:
            str: язык ('ru', 'en', ...) или 'other' если не уверена
            или tuple (язык, confidence) если return_confidence=True
        """
        # Получаем предсказание и уверенность
        text = text.replace("\n", " ").replace("\r", " ").strip()
        labels, confidences = self.model.predict(text, k=1)
        
        # Извлекаем метку и уверенность
        predicted_label = labels[0].replace('__label__', '')
        confidence = float(confidences[0])
        
        # Объединяем метки если нужно
        if merge_labels:
            predicted_label = merge_label(predicted_label)
        
        # Проверяем threshold
        if confidence < self.threshold:
            final_label = 'other'
        else:
            final_label = predicted_label
        
        if return_confidence:
            return final_label, confidence
        else:
            return final_label
    
    def predict_top_k(self, text, k=3, merge_labels=True):
        """
        Предсказание топ-K языков с уверенностью
        
        Args:
            text: текст для классификации
            k: количество топ предсказаний
            merge_labels: объединить метки
        
        Returns:
            list of tuples: [(язык, confidence), ...]
        """
        text = text.replace("\n", " ").replace("\r", " ").strip()
        labels, confidences = self.model.predict(text, k=k)
        
        results = []
        for label, conf in zip(labels, confidences):
            lang = label.replace('__label__', '')
            if merge_labels:
                lang = merge_label(lang)
            results.append((lang, float(conf)))
        
        return results
    
    def set_threshold(self, threshold):
        """
        Изменить threshold
        
        Args:
            threshold: новый порог (0.0-1.0)
        """
        self.threshold = threshold
        print(f"Threshold изменен на: {threshold}")


def find_optimal_threshold(model_path, test_data, thresholds=None):
    """
    Находит оптимальный threshold на тестовых данных
    
    Args:
        model_path: путь к модели
        test_data: list of tuples [(text, true_label), ...]
        thresholds: список порогов для проверки
    
    Returns:
        dict: результаты для каждого threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {}
    
    print("=" * 80)
    print("ПОИСК ОПТИМАЛЬНОГО THRESHOLD")
    print("=" * 80)
    
    for threshold in thresholds:
        print(f"\nТестирование threshold = {threshold}")
        
        detector = LanguageDetectorWithThreshold(model_path, threshold=threshold)
        
        total = len(test_data)
        correct = 0
        predicted_as_other = 0
        correctly_predicted_other = 0
        
        for text, true_label in test_data:
            predicted = detector.predict(text, merge_labels=True)
            
            if predicted == 'other':
                predicted_as_other += 1
                if true_label == 'other':
                    correctly_predicted_other += 1
            
            if predicted == true_label:
                correct += 1
        
        accuracy = correct / total
        other_ratio = predicted_as_other / total
        
        results[threshold] = {
            'accuracy': accuracy,
            'predicted_as_other': predicted_as_other,
            'other_ratio': other_ratio,
            'correctly_predicted_other': correctly_predicted_other
        }
        
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Предсказано как 'other': {predicted_as_other} ({other_ratio:.2%})")
        if predicted_as_other > 0:
            other_precision = correctly_predicted_other / predicted_as_other
            print(f"  Precision для 'other': {other_precision:.2%}")
    
    # Находим лучший threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['accuracy'])
    
    print("\n" + "=" * 80)
    print(f"ЛУЧШИЙ THRESHOLD: {best_threshold}")
    print(f"Accuracy: {results[best_threshold]['accuracy']:.2%}")
    print("=" * 80)
    
    return results, best_threshold


# Пример использования
if __name__ == "__main__":
    # Загрузка модели
    detector = LanguageDetectorWithThreshold(
        model_path='output/lang_detection_model.bin',
        threshold=0.7
    )
    
    # Примеры текстов
    test_texts = [
        ("Привет, как дела?", "ru"),
        ("Hello, how are you?", "en"),
        ("Salam, necəsən?", "az"),
        ("こんにちは", "other"),  # Японский - нет в обучении
    ]
    
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    for text, expected in test_texts:
        predicted, confidence = detector.predict(text, return_confidence=True)
        status = "OK" if predicted == expected else "FAIL"
        print(f"\n{status} Текст: {text}")
        print(f"  Ожидается: {expected}")
        print(f"  Предсказано: {predicted} (confidence: {confidence:.3f})")
        
        # Показываем топ-3
        top3 = detector.predict_top_k(text, k=3)
        print(f"  Топ-3: {top3}")

