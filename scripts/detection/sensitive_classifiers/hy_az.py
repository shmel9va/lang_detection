"""
Бинарный классификатор: армянский (hy) vs азербайджанский (az).

PR-риск: ошибка в этой паре недопустима — страны находятся в состоянии конфликта.

Алгоритм быстрого пути:
  Армянский алфавит (U+0531–U+0587) — полностью уникален, никакой другой
  язык из поддерживаемых его не использует.
  Если ≥1 армянский символ И (текст ≤5 букв ИЛИ доля ≥30%) → hy, conf 0.99.

  Азербайджанский использует латиницу с диакритиками: ə ö ü ğ ı ş ç
  Но эти же буквы есть в турецком → нельзя использовать как однозначный признак.
  Поэтому: если армянских символов нет → логистическая регрессия.

На практике армянский текст почти всегда написан армянским алфавитом,
поэтому быстрый путь покрывает >99% реальных случаев.
LogReg нужен для редких романизаций и очень коротких текстов.
"""

from typing import Optional, Tuple

from scripts.detection.sensitive_classifiers.base import SensitivePairClassifier

# Армянский Unicode-блок: U+0531–U+0587
_ARM_START = 0x0531
_ARM_END = 0x0587

# Минимальная доля армянских символов для детерминированного ответа
_MIN_RATIO = 0.30
# Порог «короткий текст» — для него достаточно 1 символа
_SHORT_TEXT = 5


class HyAzClassifier(SensitivePairClassifier):
    """Классификатор пары армянский / азербайджанский."""

    def __init__(self):
        super().__init__("hy", "az")

    def _fast_predict(self, text: str) -> Optional[Tuple[str, float]]:
        total_letters = sum(1 for c in text if c.isalpha())
        if total_letters == 0:
            return None

        arm_count = sum(1 for c in text if _ARM_START <= ord(c) <= _ARM_END)
        if arm_count == 0:
            return None

        ratio = arm_count / total_letters
        if total_letters <= _SHORT_TEXT or ratio >= _MIN_RATIO:
            return "hy", 0.99

        return None
