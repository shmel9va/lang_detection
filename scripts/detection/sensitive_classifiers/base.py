"""
Базовый класс бинарного классификатора чувствительной языковой пары.

Архитектура двухуровневая:
  1. Быстрый путь (_fast_predict) — детерминированные правила на основе
     уникальных символов алфавита или диагностических символов.
     Переопределяется в каждом подклассе.
  2. Логистическая регрессия на char n-gram TF-IDF — fallback для случаев,
     когда быстрый путь не дал ответа (романизация, смешанный скрипт).

Сохранение/загрузка через joblib: объект сериализуется целиком.
"""

import os
import unicodedata
from collections import defaultdict
from typing import List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SensitivePairClassifier:
    """
    Базовый класс для бинарных классификаторов чувствительных пар.

    Subclass обязан:
    - передать lang1, lang2 в super().__init__()
    - переопределить _fast_predict(text) при необходимости
    """

    def __init__(
        self,
        lang1: str,
        lang2: str,
        ngram_range: Tuple[int, int] = (2, 4),
        max_features: int = 50_000,
    ):
        self.lang1 = lang1
        self.lang2 = lang2
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.logreg = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Определить язык текста.

        Returns:
            (iso_code, confidence)
        """
        normalized = _nfkc(text)

        # Уровень 1: детерминированный быстрый путь
        fast = self._fast_predict(normalized)
        if fast is not None:
            return fast

        # Уровень 2: логистическая регрессия
        if not self.is_trained:
            return None

        return self._logreg_predict(normalized)

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Обучить на парных данных."""
        normalized = [_nfkc(t) for t in texts]
        X = self.vectorizer.fit_transform(normalized)
        self.logreg.fit(X, labels)
        self.is_trained = True

    def evaluate(
        self, texts: List[str], labels: List[str]
    ) -> Tuple[float, List[dict]]:
        """
        Оценить качество на списке примеров.

        Returns:
            accuracy (float), список результатов по каждому примеру
        """
        correct = 0
        results = []
        for text, true_label in zip(texts, labels):
            pred_result = self.predict(text)
            if pred_result is None:
                pred, conf = true_label, 0.0
            else:
                pred, conf = pred_result
            ok = pred == true_label
            correct += ok
            results.append(
                {
                    "text": text[:60],
                    "true": true_label,
                    "pred": pred,
                    "conf": round(conf, 4),
                    "correct": ok,
                }
            )
        accuracy = correct / len(texts) if texts else 0.0
        return accuracy, results

    def per_class_metrics(self, texts: List[str], labels: List[str]) -> dict:
        """Precision / Recall / F1 по каждому классу."""
        _, results = self.evaluate(texts, labels)
        tp: dict = defaultdict(int)
        fp: dict = defaultdict(int)
        fn: dict = defaultdict(int)
        for r in results:
            if r["correct"]:
                tp[r["true"]] += 1
            else:
                fn[r["true"]] += 1
                fp[r["pred"]] += 1

        metrics = {}
        for lang in sorted({self.lang1, self.lang2}):
            p = tp[lang] / (tp[lang] + fp[lang]) if (tp[lang] + fp[lang]) > 0 else 0.0
            r = tp[lang] / (tp[lang] + fn[lang]) if (tp[lang] + fn[lang]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            metrics[lang] = {"precision": p, "recall": r, "f1": f1,
                              "tp": tp[lang], "fp": fp[lang], "fn": fn[lang]}
        return metrics

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SensitivePairClassifier":
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Внутренние методы (переопределяются в подклассах)
    # ------------------------------------------------------------------

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Детерминированный быстрый путь.
        Переопределить в подклассе.
        Returns (lang, confidence) или None.
        """
        return None

    def _logreg_predict(self, text: str) -> Tuple[str, float]:
        X = self.vectorizer.transform([text])
        proba = self.logreg.predict_proba(X)[0]
        idx = int(proba.argmax())
        return str(self.logreg.classes_[idx]), float(proba[idx])

    def __repr__(self) -> str:
        trained = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}({self.lang1}/{self.lang2}, {trained})"


def _nfkc(text: str) -> str:
    """Unicode NFKC нормализация — приводит презентационные формы к базовым."""
    return unicodedata.normalize("NFKC", str(text))
