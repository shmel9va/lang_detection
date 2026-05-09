"""
Главный класс LanguageDetector — точка входа для определения языка.

Пайплайн (5 уровней):
  1. Препроцессор        — NFKC, удаление URL/телефонов/эмодзи.
  2. Детектор скрипта    — уникальные алфавиты: hy, ka, he, am → немедленный ответ.
  3. fastText            — топ-2 языка + вероятности.
  4. Роутер пар          — если топ-2 образуют чувствительную пару → бинарный классификатор.
  5. Постпроцессор       — merge_label: uz_lat/uz_cyr → uz, sr_lat/sr_cyr → sr, …

Использование:
    detector = LanguageDetector('output/lang_detection_model.bin')
    lang, conf = detector.detect("Привет, как дела?")
    # → ('ru', 0.97)

    lang, conf = detector.detect("Բարի՜ Оr Hello")
    # → ('hy', 0.99)  — армянские символы, быстрый путь
"""

from typing import List, Tuple

import fasttext

from scripts.data_processing.preprocess_text import normalize_for_detection, preprocess_text
from scripts.detection.script_detector import ScriptDetector
from scripts.detection.sensitive_router import SensitiveRouter
from scripts.utils.label_mapping import merge_label


class LanguageDetector:
    """
    Комбинированная модель определения языка.

    Args:
        fasttext_model_path:    путь к обученной fastText-модели (.bin).
        sensitive_classifiers_dir: директория с .pkl файлами бинарных классификаторов.
        threshold:              минимальная уверенность fastText; при меньшем значении
                                возвращается 'other'. Не влияет на быстрый путь скрипт-детектора.
        router_verbose:         печатать ли сообщения о загрузке классификаторов.
    """

    def __init__(
        self,
        fasttext_model_path: str = "output/lang_detection_model.bin",
        sensitive_classifiers_dir: str = "output/sensitive_classifiers",
        threshold: float = 0.0,
        router_verbose: bool = True,
    ):
        self.threshold = threshold

        print(f"Загрузка fastText модели: {fasttext_model_path}")
        self.ft_model = fasttext.load_model(fasttext_model_path)

        self.script_detector = ScriptDetector()
        self.router = SensitiveRouter(
            classifiers_dir=sensitive_classifiers_dir,
            verbose=router_verbose,
        )

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Определить язык текста.

        Args:
            text: входной текст (произвольный, с ошибками, эмодзи, URL, ...).

        Returns:
            (iso_639_1_code, confidence)
            Если язык не определён с достаточной уверенностью → ('other', conf).
        """
        if not text or not str(text).strip():
            return "other", 0.0

        # ── Уровень 1: нормализация ──────────────────────────────────
        normalized = normalize_for_detection(text)
        if not normalized.strip():
            return "other", 0.0

        # ── Уровень 2: детектор скрипта (быстрый путь) ───────────────
        script_result = self.script_detector.detect(normalized)
        if script_result is not None:
            return script_result

        # ── Уровень 3: fastText ──────────────────────────────────────
        ft_text = self._prepare_for_fasttext(normalized)
        if not ft_text.strip():
            return "other", 0.0

        labels, probs = self.ft_model.predict(ft_text, k=2)
        langs = [merge_label(lbl.replace("__label__", "")) for lbl in labels]
        probs_list = [float(p) for p in probs]

        # Порог уверенности
        if probs_list[0] < self.threshold:
            return "other", probs_list[0]

        # ── Уровень 4: роутер чувствительных пар ────────────────────
        result = self.router.route(normalized, langs, probs_list)
        if result is not None:
            return result
        return langs[0], probs_list[0]

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Определить язык для списка текстов."""
        return [self.detect(t) for t in texts]

    def detect_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Вернуть топ-k кандидатов от fastText (без роутера пар).
        Полезно для отладки и анализа неопределённых случаев.
        """
        if not text or not str(text).strip():
            return [("other", 0.0)]

        normalized = normalize_for_detection(text)
        script_result = self.script_detector.detect(normalized)
        if script_result is not None:
            lang, conf = script_result
            return [(lang, conf)]

        ft_text = self._prepare_for_fasttext(normalized)
        if not ft_text.strip():
            return [("other", 0.0)]

        labels, probs = self.ft_model.predict(ft_text, k=k)
        return [
            (merge_label(lbl.replace("__label__", "")), float(p))
            for lbl, p in zip(labels, probs)
        ]

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _prepare_for_fasttext(self, normalized_text: str) -> str:
        """
        Применяет тяжёлую предобработку (lowercase, удаление спецсимволов,
        токенизация) для передачи в fastText.
        Работает с уже нормализованным текстом.
        """
        processed = preprocess_text(normalized_text)
        # fastText не поддерживает символы новой строки
        return processed.replace("\n", " ").replace("\r", " ")


# ------------------------------------------------------------------
# Быстрый ручной тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    model_path = "output/lang_detection_model.bin"
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        sys.exit(1)

    detector = LanguageDetector(model_path, router_verbose=False)

    test_cases = [
        ("Привет, как дела?",                          "ru"),
        ("Hello, how are you?",                        "en"),
        ("Salam, necəsən?",                            "az"),
        ("Բարի՜ Ծաղիկ",                               "hy"),   # армянский → быстрый путь
        ("שלום עולם",                                  "he"),   # иврит → быстрый путь
        ("مرحبا بالعالم",                              "ar"),
        ("این یک متن فارسی است. پدر چه گفت؟",         "fa"),   # персидский диагностический символ
        ("Доброго ранку, як справи?",                  "uk"),   # украинские символи
        ("Xush kelibsiz",                              "uz"),
        ("こんにちは",                                  "other"), # японский — не поддерживается
    ]

    print("\n" + "=" * 70)
    print("ТЕСТ LanguageDetector")
    print("=" * 70)
    print(f"{'Текст':<40} {'Ожид':<7} {'Пред':<7} {'Conf':<7} {'OK?'}")
    print("-" * 70)

    ok = 0
    for text, expected in test_cases:
        lang, conf = detector.detect(text)
        status = "✓" if lang == expected else "✗"
        ok += lang == expected
        print(f"{text[:38]:<40} {expected:<7} {lang:<7} {conf:<7.3f} {status}")

    print("-" * 70)
    print(f"Итого: {ok}/{len(test_cases)} правильно")
