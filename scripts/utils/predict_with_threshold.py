"""
Модуль для предсказания языка с поддержкой threshold для определения 'other'.
Использует ту же нормализацию, что и LanguageDetector.
"""
import fasttext
from sklearn.metrics import f1_score
from scripts.utils.label_mapping import merge_label
from scripts.data_processing.preprocess_text import normalize_for_detection, preprocess_text


class LanguageDetectorWithThreshold:
    """
    Детектор языка с поддержкой определения неизвестных языков через threshold.
    Нормализация текста совпадает с полным пайплайном LanguageDetector.
    """
    
    def __init__(self, model_path, threshold=0.7):
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold
        print(f"Модель загружена: {model_path}")
        print(f"Threshold: {threshold}")
    
    def _normalize(self, text):
        normalized = normalize_for_detection(text)
        processed = preprocess_text(normalized)
        return processed.replace("\n", " ").replace("\r", " ").strip()
    
    def predict(self, text, return_confidence=False, merge_labels=True):
        prepared = self._normalize(text)
        if not prepared:
            if return_confidence:
                return 'other', 0.0
            return 'other'
        
        labels, confidences = self.model.predict(prepared, k=1)
        
        predicted_label = labels[0].replace('__label__', '')
        confidence = float(confidences[0])
        
        if merge_labels:
            predicted_label = merge_label(predicted_label)
        
        if confidence < self.threshold:
            final_label = 'other'
        else:
            final_label = predicted_label
        
        if return_confidence:
            return final_label, confidence
        else:
            return final_label
    
    def predict_top_k(self, text, k=3, merge_labels=True):
        prepared = self._normalize(text)
        if not prepared:
            return [('other', 0.0)]
        
        labels, confidences = self.model.predict(prepared, k=k)
        
        results = []
        for label, conf in zip(labels, confidences):
            lang = label.replace('__label__', '')
            if merge_labels:
                lang = merge_label(lang)
            results.append((lang, float(conf)))
        
        return results
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        print(f"Threshold изменен на: {threshold}")


def find_optimal_threshold(model_path, test_data, thresholds=None):
    """
    Находит оптимальный threshold на тестовых данных.
    Загружает модель ОДИН раз.
    
    Args:
        model_path: путь к модели fastText
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
    
    model = fasttext.load_model(model_path)
    
    all_preds = []
    for text, _ in test_data:
        normalized = normalize_for_detection(text)
        processed = preprocess_text(normalized)
        prepared = processed.replace("\n", " ").replace("\r", " ").strip()
        if not prepared:
            all_preds.append(('other', 0.0, None))
            continue
        labels, confs = model.predict(prepared, k=1)
        pred = merge_label(labels[0].replace('__label__', ''))
        conf = float(confs[0])
        all_preds.append((pred, conf, None))
    
    for threshold in thresholds:
        print(f"\nТестирование threshold = {threshold}")
        
        total = len(test_data)
        y_true_thresh, y_pred_thresh = [], []
        predicted_as_other = 0
        correctly_predicted_other = 0
        
        for i, (text, true_label) in enumerate(test_data):
            pred, conf, _ = all_preds[i]
            
            if conf < threshold:
                predicted = 'other'
            else:
                predicted = pred
            
            if predicted == 'other':
                predicted_as_other += 1
                if true_label == 'other':
                    correctly_predicted_other += 1
            
            y_true_thresh.append(true_label)
            y_pred_thresh.append(predicted)
        
        accuracy = sum(1 for t, p in zip(y_true_thresh, y_pred_thresh) if t == p) / total
        macro_f1 = f1_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0)
        other_ratio = predicted_as_other / total
        
        results[threshold] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'predicted_as_other': predicted_as_other,
            'other_ratio': other_ratio,
            'correctly_predicted_other': correctly_predicted_other
        }
        
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Macro F1:  {macro_f1:.2%}")
        print(f"  Предсказано как 'other': {predicted_as_other} ({other_ratio:.2%})")
        if predicted_as_other > 0:
            other_precision = correctly_predicted_other / predicted_as_other
            print(f"  Precision для 'other': {other_precision:.2%}")
    
    best_threshold = max(results.keys(), key=lambda t: results[t]['macro_f1'])
    
    print("\n" + "=" * 80)
    print(f"ЛУЧШИЙ THRESHOLD: {best_threshold}")
    print(f"Macro F1:  {results[best_threshold]['macro_f1']:.2%}")
    print(f"Accuracy:  {results[best_threshold]['accuracy']:.2%}")
    print("=" * 80)
    
    return results, best_threshold


if __name__ == "__main__":
    detector = LanguageDetectorWithThreshold(
        model_path='output/lang_detection_model.bin',
        threshold=0.7
    )
    
    test_texts = [
        ("Привет, как дела?", "ru"),
        ("Hello, how are you?", "en"),
        ("Salam, necəsən?", "az"),
        ("こんにちは", "other"),
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
        
        top3 = detector.predict_top_k(text, k=3)
        print(f"  Топ-3: {top3}")
