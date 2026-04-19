"""
Детектор скрипта (алфавита) на основе Unicode-диапазонов.

Возвращает результат немедленно для языков с уникальными алфавитами,
не обращаясь к ML-модели. Это самый быстрый уровень в пайплайне.

Уникальные алфавиты (нет перекрытий с другими поддерживаемыми языками):
  hy  — армянский      (U+0531–U+0587)
  ka  — грузинский     (U+10A0–U+10FF, U+2D00–U+2D2F)
  he  — иврит          (U+05D0–U+05EA, U+FB1D–U+FB4F)
  am  — амхарский      (U+1200–U+137F)

Неоднозначные алфавиты (передаются в fastText):
  латиница   — en, tr, az, uz_lat, sr_lat, ro, es, fr, pt, ...
  кириллица  — ru, uk, kk, uz_cyr, sr_cyr, ...
  арабское   — ar, fa, ur, ...
  деванагари — hi, ne, ...
"""

from typing import Optional, Tuple

# Диапазоны Unicode для уникальных алфавитов.
# Формат: { iso_639_1: [(start, end), ...] }
UNIQUE_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "hy": [  # Армянский
        (0x0531, 0x0587),  # прописные (U+0531–U+0556) + строчные (U+0561–U+0587)
    ],
    "ka": [  # Грузинский
        (0x10A0, 0x10FF),  # основной блок
        (0x2D00, 0x2D2F),  # Mkhedruli supplement
    ],
    "he": [  # Иврит
        (0x05D0, 0x05EA),  # еврейские буквы
        (0xFB1D, 0xFB4F),  # презентационные формы-A
    ],
    "am": [  # Амхарский (эфиопский силлабарий)
        (0x1200, 0x137F),
    ],
}

# Минимальное количество символов уникального скрипта для уверенного ответа
_MIN_COUNT = 1
# Минимальная доля символов уникального скрипта от всех букв текста
_MIN_RATIO = 0.30


class ScriptDetector:
    """
    Определяет язык по уникальному Unicode-алфавиту без ML.

    Алгоритм:
    1. Считаем все буквы в тексте.
    2. Для каждого уникального алфавита считаем совпадающие символы.
    3. Если символов >= _MIN_COUNT И (текст короткий ≤5 букв ИЛИ доля >= _MIN_RATIO) →
       возвращаем язык с confidence 0.99.
    4. Если ни один алфавит не подходит → возвращаем None (текст идёт в fastText).
    """

    def detect(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Попытка определить язык по уникальному скрипту.

        Args:
            text: текст после normalize_for_detection (NFKC, без URL/эмодзи).

        Returns:
            (iso_code, confidence) если язык однозначно определён,
            None если скрипт неоднозначен — текст нужно передать в fastText.
        """
        if not text:
            return None

        total_letters = sum(1 for c in text if c.isalpha())
        if total_letters == 0:
            return None

        for lang, ranges in UNIQUE_SCRIPT_RANGES.items():
            count = _count_in_ranges(text, ranges)
            if count == 0:
                continue
            ratio = count / total_letters
            if total_letters <= 5 or ratio >= _MIN_RATIO:
                return lang, 0.99

        return None


def _count_in_ranges(text: str, ranges: list[tuple[int, int]]) -> int:
    """Считает символы текста, попадающие в заданные Unicode-диапазоны."""
    total = 0
    for ch in text:
        cp = ord(ch)
        for start, end in ranges:
            if start <= cp <= end:
                total += 1
                break
    return total
