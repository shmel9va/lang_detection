"""
Бинарный классификатор: испанский (es) vs португальский (pt).

Оба языка — романские, используют латиницу.
Алгоритм быстрого пути:
  Португальские символы, отсутствующие в испанском:
    ã õ Ã Õ — тильда над гласными (nasalisation), уникальна для pt.
  Испанские символы:
    ñ Ñ — тильда над n, уникальна для es.

  Если ã/õ → pt, confidence 0.99.
  Если ñ → es, confidence 0.99.
  Иначе → логистическая регрессия на char n-gram.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

_PT_UNIQUE: frozenset = frozenset("ãõÃÕ")
_ES_UNIQUE: frozenset = frozenset("ñÑ")


class EsPtClassifier(SensitivePairClassifier):
    """Классификатор пары испанский / португальский."""

    def __init__(self):
        super().__init__("es", "pt")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if any(c in _PT_UNIQUE for c in text):
            return "pt", 0.99
        if any(c in _ES_UNIQUE for c in text):
            return "es", 0.99
        return None
