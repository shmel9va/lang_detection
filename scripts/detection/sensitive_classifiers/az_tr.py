"""
Бинарный классификатор: азербайджанский (az) vs турецкий (tr).

Оба языка — тюркские, используют латиницу с общими диакритиками:
  ə ö ü ğ ı ş ç (az)  vs  ö ü ğ ı ş ç (tr)

Алгоритм быстрого пути:
  Azerbaijani имеет уникальную букву ə (schwa, U+0259 / U+018F),
  которая отсутствует в турецком алфавите.
  Если ə или Ä найдена → az, confidence 0.99.

  Иначе → логистическая регрессия на char n-gram.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

_AZ_UNIQUE: frozenset = frozenset("əäƏÄ")


class AzTrClassifier(SensitivePairClassifier):
    """Классификатор пары азербайджанский / турецкий."""

    def __init__(self):
        super().__init__("az", "tr")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if any(c in _AZ_UNIQUE for c in text):
            return "az", 0.99
        return None
