"""
Бинарный классификатор: арабский (ar) vs персидский/фарси (fa).

Оба языка используют арабское письмо — скрипт-детектор не помогает.
Алгоритм быстрого пути:
  - Персидские буквы  پ (U+067E)  چ (U+0686)  ژ (U+0698)  گ (U+06AF)
    присутствуют только в персидском, отсутствуют в арабском.
    Если хотя бы одна такая буква найдена → fa, confidence 0.99.
  - Иначе → логистическая регрессия на char n-gram.

Почему именно эти буквы:
  Они встречаются примерно в 70-80% персидских текстов (слова  پدر, چه, ژاپن, گفت и т.д.).
  В классическом арабском их нет вообще.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Символы, уникальные для персидского (отсутствуют в арабском)
_PERSIAN_UNIQUE: frozenset = frozenset("پچژگ")


class ArFaClassifier(SensitivePairClassifier):
    """Классификатор пары арабский / персидский."""

    def __init__(self):
        super().__init__("ar", "fa")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if any(c in _PERSIAN_UNIQUE for c in text):
            return "fa", 0.99
        return None
