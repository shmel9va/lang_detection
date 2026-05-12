"""
Бинарный классификатор: узбекский (uz) vs казахский (kk).

Оба языка — тюркские, оба могут использоваться в кириллице.
Алгоритм быстрого пути:
  Казахские буквы, отсутствующие в узбекском:
    ә (U+04D9)  ң (U+04A3)  ө (U+04E9)  ұ (U+04B1)  ү (U+04AF)  і (U+0456)
    (қ и ғ общие для обоих, һ/ҳ — тоже общие)
  Узбекские буквы, отсутствующие в казахском:
    ў (U+045E) — уникальна для узбекского кириллицы

  Если найдена казахская уникальная буква → kk, confidence 0.99.
  Если найдена ў → uz, confidence 0.99.
  Иначе → логистическая регрессия на char n-gram.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

_KK_UNIQUE: frozenset = frozenset("әңөұүіӘҢӨҰҮІ")
_UZ_UNIQUE: frozenset = frozenset("ўЎ")


class UzKkClassifier(SensitivePairClassifier):
    """Классификатор пары узбекский / казахский."""

    def __init__(self):
        super().__init__("uz", "kk")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if any(c in _KK_UNIQUE for c in text):
            return "kk", 0.99
        if any(c in _UZ_UNIQUE for c in text):
            return "uz", 0.99
        return None
