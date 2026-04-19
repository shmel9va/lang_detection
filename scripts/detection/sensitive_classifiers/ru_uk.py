"""
Бинарный классификатор: русский (ru) vs украинский (uk).

Оба языка используют кириллицу — нужны дополнительные признаки.
Алгоритм быстрого пути:
  Украинские буквы, отсутствующие в русском:
    і  (U+0456 / U+0406)   — «і десятеричне»
    ї  (U+0457 / U+0407)   — только в украинском
    є  (U+0454 / U+0404)   — только в украинском
    ґ  (U+0491 / U+0490)   — только в украинском

  Если в тексте (после приведения к нижнему регистру) есть хотя бы одна
  из этих букв → uk, confidence 0.99.
  Иначе → логистическая регрессия на char n-gram.

Примечание: украинского языка ещё нет в обучающем датасете.
Классификатор заработает полноценно после добавления данных и обучения.
До тех пор быстрый путь уже перехватывает тексты с украинскими символами.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Украинские буквы (строчные после .lower())
_UKRAINIAN_UNIQUE: frozenset = frozenset("іїєґ")


class RuUkClassifier(SensitivePairClassifier):
    """Классификатор пары русский / украинский."""

    def __init__(self):
        super().__init__("ru", "uk")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        lower = text.lower()
        if any(c in _UKRAINIAN_UNIQUE for c in lower):
            return "uk", 0.99
        return None
