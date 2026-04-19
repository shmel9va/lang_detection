"""
Бинарный классификатор: иврит (he) vs арабский (ar).

Алгоритм быстрого пути:
  Иврит:   U+05D0–U+05EA (основные буквы),  U+FB1D–U+FB4F (презентационные формы).
  Арабский: U+0600–U+06FF.

  Оба алфавита — разные Unicode-блоки, пересечений нет.
  Считаем символы каждого алфавита:
    - Если есть ивритские ИЛИ арабские символы → возвращаем тот, которых больше.
    - Если ничего нет (романизованный текст) → логистическая регрессия.

Примечание:
  Детектор скрипта в script_detector.py уже возвращает 'he' для ивритских текстов
  до того, как они доходят до роутера. HeArClassifier активируется лишь когда
  fastText выдаёт {he, ar} в топ-2 при отсутствии явных ивритских символов
  (очень короткие тексты, романизация).
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Иврит: основные буквы + презентационные формы
_HE_RANGES = [(0x05D0, 0x05EA), (0xFB1D, 0xFB4F)]
# Арабское письмо
_AR_RANGES = [(0x0600, 0x06FF)]


def _count_range(text: str, ranges: list) -> int:
    total = 0
    for ch in text:
        cp = ord(ch)
        for s, e in ranges:
            if s <= cp <= e:
                total += 1
                break
    return total


class HeArClassifier(SensitivePairClassifier):
    """Классификатор пары иврит / арабский."""

    def __init__(self):
        super().__init__("he", "ar")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        he_count = _count_range(text, _HE_RANGES)
        ar_count = _count_range(text, _AR_RANGES)

        if he_count == 0 and ar_count == 0:
            return None  # романизованный текст → LogReg

        if he_count >= ar_count:
            return "he", 0.99
        return "ar", 0.99
