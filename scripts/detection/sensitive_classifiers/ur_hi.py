"""
Бинарный классификатор: урду (ur) vs хинди (hi).

Алгоритм быстрого пути:
  Урду:    арабское письмо (U+0600–U+06FF) + специфичные урду-буквы
           ٹ (U+0679)  ڈ (U+0688)  ڑ (U+0691)  ں (U+06BA)  ہ (U+06C1)
  Хинди:  деванагари (U+0900–U+097F)

  Если есть символы деванагари → hi, conf 0.99.
  Если есть символы арабского письма (без деванагари) → ur, conf 0.99.
  Если ничего нет (романизованный текст, латиница) → логистическая регрессия.

Почему нужна LogReg:
  Romanized Urdu (Urdu written in Latin) и Romanized Hindi (Hinglish)
  внешне очень похожи. Char n-gram LogReg улавливает типичные
  транслитерационные паттерны каждого языка.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Деванагари (хинди, непальский)
_DEVA_RANGE = (0x0900, 0x097F)
# Арабское письмо (урду, арабский, персидский)
_ARAB_RANGE = (0x0600, 0x06FF)


def _count_range_simple(text: str, start: int, end: int) -> int:
    return sum(1 for c in text if start <= ord(c) <= end)


class UrHiClassifier(SensitivePairClassifier):
    """Классификатор пары урду / хинди."""

    def __init__(self):
        super().__init__("ur", "hi")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        deva_count = _count_range_simple(text, *_DEVA_RANGE)
        arab_count = _count_range_simple(text, *_ARAB_RANGE)

        if deva_count == 0 and arab_count == 0:
            return None  # романизованный текст → LogReg

        if deva_count >= arab_count:
            return "hi", 0.99
        return "ur", 0.99
