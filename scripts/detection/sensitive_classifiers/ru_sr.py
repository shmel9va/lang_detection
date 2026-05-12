"""
Бинарный классификатор: русский (ru) vs сербский (sr).

Оба языка могут использовать кириллицу.
Алгоритм быстрого пути:
  Сербские буквы, отсутствующие в русском:
    ђ (U+0452)  ћ (U+045B)  љ (U+0459)  њ (U+045A)  џ (U+045F)
  Русские буквы, отсутствующие в сербском:
    ъ ы э ё Щ

  Если найдена сербская буква → sr, confidence 0.99.
  Иначе → логистическая регрессия на char n-gram.

Примечание: сербский также использует латиницу (gaevica), но тогда
скрипт-детектор не направит текст к этой паре.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

_SR_UNIQUE: frozenset = frozenset("ђћљњџЂЋЉЊЏ")


class RuSrClassifier(SensitivePairClassifier):
    """Классификатор пары русский / сербский."""

    def __init__(self):
        super().__init__("ru", "sr")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if any(c in _SR_UNIQUE for c in text):
            return "sr", 0.99
        return None
