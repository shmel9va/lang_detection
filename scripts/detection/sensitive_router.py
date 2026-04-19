"""
Роутер чувствительных языковых пар.

После того как fastText выдал топ-2 языка, роутер проверяет:
является ли пара {lang1, lang2} чувствительной.

Если да — активируется соответствующий бинарный классификатор.
Если нет — возвращается результат fastText без изменений.

Загрузка классификаторов при инициализации:
  - Если файл .pkl существует → загружается обученная модель.
  - Если файла нет → создаётся необученный экземпляр (работает только быстрый путь).
    Это позволяет запускать детектор ещё до обучения бинарных классификаторов.
"""

import os
from typing import Dict, List, Optional, Tuple

from scripts.detection.sensitive_classifiers.ar_fa import ArFaClassifier
from scripts.detection.sensitive_classifiers.he_ar import HeArClassifier
from scripts.detection.sensitive_classifiers.hy_az import HyAzClassifier
from scripts.detection.sensitive_classifiers.ru_uk import RuUkClassifier
from scripts.detection.sensitive_classifiers.ur_hi import UrHiClassifier
from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Таблица чувствительных пар: frozenset → имя классификатора
SENSITIVE_PAIRS: Dict[frozenset, str] = {
    frozenset({"hy", "az"}): "hy_az",
    frozenset({"he", "ar"}): "he_ar",
    frozenset({"ur", "hi"}): "ur_hi",
    frozenset({"ar", "fa"}): "ar_fa",
    frozenset({"ru", "uk"}): "ru_uk",
}

# Имя файла pkl для каждого классификатора
_PKL_FILES: Dict[str, str] = {
    "hy_az": "hy_az.pkl",
    "he_ar": "he_ar.pkl",
    "ur_hi": "ur_hi.pkl",
    "ar_fa": "ar_fa.pkl",
    "ru_uk": "ru_uk.pkl",
}

# Класс для каждого классификатора (для создания необученного экземпляра)
_CLASSIFIER_CLASSES: Dict[str, type] = {
    "hy_az": HyAzClassifier,
    "he_ar": HeArClassifier,
    "ur_hi": UrHiClassifier,
    "ar_fa": ArFaClassifier,
    "ru_uk": RuUkClassifier,
}


class SensitiveRouter:
    """
    Роутер чувствительных языковых пар.

    Args:
        classifiers_dir: директория с сохранёнными .pkl файлами классификаторов.
        verbose: логировать загрузку классификаторов.
    """

    def __init__(
        self,
        classifiers_dir: str = "output/sensitive_classifiers",
        verbose: bool = True,
    ):
        self.classifiers_dir = classifiers_dir
        self.classifiers: Dict[str, SensitivePairClassifier] = {}
        self._load_classifiers(verbose=verbose)

    def route(
        self,
        text: str,
        top2_langs: List[str],
        top2_probs: List[float],
    ) -> Tuple[str, float]:
        """
        Определить язык с учётом чувствительных пар.

        Args:
            text:        оригинальный нормализованный текст (после normalize_for_detection).
            top2_langs:  список из ≤2 языков от fastText (уже прошедших merge_label).
            top2_probs:  вероятности от fastText для top2_langs.

        Returns:
            (iso_code, confidence)
        """
        if not top2_langs:
            return "other", 0.0

        best_lang = top2_langs[0]
        best_prob = top2_probs[0] if top2_probs else 0.5

        if len(top2_langs) < 2:
            return best_lang, best_prob

        pair = frozenset({top2_langs[0], top2_langs[1]})

        if pair not in SENSITIVE_PAIRS:
            return best_lang, best_prob

        clf_name = SENSITIVE_PAIRS[pair]
        clf = self.classifiers.get(clf_name)

        if clf is None:
            return best_lang, best_prob

        return clf.predict(text)

    def is_sensitive_pair(self, lang1: str, lang2: str) -> bool:
        return frozenset({lang1, lang2}) in SENSITIVE_PAIRS

    # ------------------------------------------------------------------
    # Загрузка классификаторов
    # ------------------------------------------------------------------

    def _load_classifiers(self, verbose: bool = True) -> None:
        for name, filename in _PKL_FILES.items():
            path = os.path.join(self.classifiers_dir, filename)
            if os.path.exists(path):
                try:
                    clf = SensitivePairClassifier.load(path)
                    self.classifiers[name] = clf
                    if verbose:
                        print(f"  [router] Loaded classifier: {name} ({path})")
                except Exception as exc:
                    if verbose:
                        print(f"  [router] Failed to load {name}: {exc}. Using untrained instance.")
                    self.classifiers[name] = _CLASSIFIER_CLASSES[name]()
            else:
                # Создаём необученный экземпляр — быстрый путь уже работает
                self.classifiers[name] = _CLASSIFIER_CLASSES[name]()
                if verbose:
                    print(f"  [router] No saved model for {name}, using fast-path only.")
