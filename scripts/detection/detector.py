"""
Главный класс LanguageDetector — точка входа для определения языка.

Пайплайн (4 уровня):
  Level 0: Phrase Dictionary  — точные совпадения частых фраз
  Level 1: Script Detector    — уникальные алфавиты: hy, ka, he, am → немедленный ответ
  Level 2: fastText           — если confidence ≥ 0.95 И не чувствительная пара → ответ
  Level 4: Binary classifiers — для чувствительных пар (вызываются ПЕРЕД Transformer)
  Level 3: DistilBERT (ONNX)  — fallback для неоднозначных случаев

Использование:
    detector = LanguageDetector(
        fasttext_model_path='output/lang_detection_model.bin',
        onnx_model_path='output/distilbert_lang_detection.onnx',
    )
    lang, conf = detector.detect("Привет, как дела?")
    # → ('ru', 0.97)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import fasttext
import numpy as np

from scripts.data_processing.preprocess_text import normalize_for_detection, preprocess_text
from scripts.detection.indiclid_wrapper import IndicLIDWrapper, INDICLID_LANGUAGES
from scripts.detection.persoarabic_wrapper import PersoArabicLIDWrapper, _PALI_LANGUAGES
from scripts.detection.script_detector import ScriptDetector
from scripts.detection.sensitive_router import SensitiveRouter, SENSITIVE_PAIRS
from scripts.utils.label_mapping import merge_label

# Конфигурация ONNX (ленивый импорт)
_ONNX_CONFIG_PATH = "label_config.json"


class LanguageDetector:
    """
    Комбинированная модель определения языка (6-уровневый пайплайн).

    Args:
        fasttext_model_path:       путь к fastText модели (.bin).
        onnx_model_path:           путь к ONNX модели (.onnx). Если None — Level 3b пропускается.
        onnx_config_path:          путь к label_config.json. Если None — ищется рядом с onnx.
        sensitive_classifiers_dir: директория с .pkl файлов бинарных классификаторов.
        threshold:                 минимальная уверенность fastText для 'other'.
        fasttext_confidence_keep:  если fastText уверен ≥ этого значения и пара не чувствительная → ответ.
        router_verbose:            печатать ли сообщения загрузки.
    """

    def __init__(
        self,
        fasttext_model_path: str = "output/lang_detection_model.bin",
        onnx_model_path: Optional[str] = "output/distilbert_lang_detection.onnx",
        onnx_config_path: Optional[str] = None,
        sensitive_classifiers_dir: str = "output/sensitive_classifiers",
        threshold: float = 0.0,
        fasttext_confidence_keep: float = 0.95,
        router_verbose: bool = True,
    ):
        self.threshold = threshold
        self.fasttext_confidence_keep = fasttext_confidence_keep

        # ── Level 0: Phrase Dictionary ──────────────────────────────────
        self.phrase_dict = self._load_phrase_dict()

        # ── Level 1: Script Detector ────────────────────────────────────
        self.script_detector = ScriptDetector()

        # ── Level 2: fastText ───────────────────────────────────────────
        print(f"Загрузка fastText: {fasttext_model_path}")
        self.ft_model = fasttext.load_model(fasttext_model_path)

        # ── Level 2b: IndicLID-FTR (для hi/ur/ne) ──────────────────────
        self.indiclid = IndicLIDWrapper(verbose=router_verbose)

        # ── Level 2c: PersoArabicLID (для ar/fa) ──────────────────────
        self.persoarabic = PersoArabicLIDWrapper(verbose=router_verbose)

        # ── Level 3: DistilBERT ONNX ───────────────────────────────────
        self.onnx_session = None
        self.onnx_tokenizer = None
        self.onnx_id2label: Optional[Dict[int, str]] = None
        self.onnx_max_length: int = 128

        if onnx_model_path and os.path.exists(onnx_model_path):
            self._load_onnx(onnx_model_path, onnx_config_path)
        else:
            print(f"ONNX модель не найдена: {onnx_model_path}")
            print("  Level 3b (DistilBERT) пропускается.")

        # ── Level 4: Binary classifiers ─────────────────────────────────
        self.router = SensitiveRouter(
            classifiers_dir=sensitive_classifiers_dir,
            verbose=router_verbose,
        )

    # ------------------------------------------------------------------
    # Загрузка ONNX
    # ------------------------------------------------------------------

    def _load_onnx(self, onnx_path: str, config_path: Optional[str]):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError:
            print("  WARNING: onnxruntime/transformers не установлены — Level 3 недоступен.")
            return

        # Config: ищем рядом с ONNX файлом, в подкаталоге модели
        if config_path is None:
            onnx_dir = os.path.dirname(onnx_path)
            model_subdir = onnx_dir if os.path.exists(os.path.join(onnx_dir, "label_config.json")) else os.path.join(onnx_dir, "distilbert_lang_detection")
            config_path = os.path.join(model_subdir, _ONNX_CONFIG_PATH)

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.onnx_id2label = {int(k): v for k, v in config["id2label"].items()}
            self.onnx_max_length = config.get("max_length", 128)
            tokenizer_name = config.get("model_name", "distilbert-base-multilingual-cased")
        else:
            print(f"  WARNING: {config_path} не найден — Level 3 недоступен.")
            return

        # ONNX session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        active_provider = self.onnx_session.get_providers()[0]
        print(f"ONNX загружен: {onnx_path} (provider: {active_provider})")

        # Tokenizer: сначала локальный, потом из конфига модели
        model_dir = os.path.dirname(onnx_path)
        local_tokenizer = os.path.join(model_dir, "distilbert_lang_detection")
        if os.path.exists(os.path.join(local_tokenizer, "tokenizer_config.json")):
            tokenizer_path = local_tokenizer
        elif os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
            tokenizer_path = model_dir
        else:
            tokenizer_path = tokenizer_name
        self.onnx_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ------------------------------------------------------------------
    # Level 0: Phrase Dictionary
    # ------------------------------------------------------------------

    @staticmethod
    def _load_phrase_dict() -> Dict[str, Tuple[str, float]]:
        return {}

    def _phrase_lookup(self, text: str) -> Optional[Tuple[str, float]]:
        if not self.phrase_dict:
            return None
        key = text.lower().strip()
        return self.phrase_dict.get(key)

    # ------------------------------------------------------------------
    # Level 3: ONNX inference
    # ------------------------------------------------------------------

    def _onnx_predict(self, text: str) -> Optional[Tuple[str, float]]:
        if self.onnx_session is None or self.onnx_tokenizer is None:
            return None

        inputs = self.onnx_tokenizer(
            text,
            max_length=self.onnx_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        logits = self.onnx_session.run(None, ort_inputs)[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())

        if self.onnx_id2label:
            label = self.onnx_id2label[idx]
        else:
            label = str(idx)

        return label, float(probs[idx])

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Определить язык текста.

        Returns:
            (iso_code, confidence)
        """
        if not text or not str(text).strip():
            return "other", 0.0

        # ── Предобработка ───────────────────────────────────────────
        normalized = normalize_for_detection(text)
        if not normalized.strip():
            return "other", 0.0

        # ── Level 0: Phrase Dictionary ──────────────────────────────
        phrase_result = self._phrase_lookup(normalized)
        if phrase_result is not None:
            return phrase_result

        # ── Level 1: Script Detector ────────────────────────────────
        script_result = self.script_detector.detect(normalized)
        if script_result is not None:
            return script_result

        # ── Level 2: fastText ───────────────────────────────────────
        ft_text = preprocess_text(normalized).replace("\n", " ").replace("\r", " ")
        if not ft_text.strip():
            return "other", 0.0

        labels, probs = self.ft_model.predict(ft_text, k=2)
        langs = [merge_label(lbl.replace("__label__", "")) for lbl in labels]
        probs_list = [float(p) for p in probs]

        if probs_list[0] < self.threshold:
            return "other", probs_list[0]

        is_sensitive = self.router.is_sensitive_pair(langs[0], langs[1]) if len(langs) >= 2 else False

        if probs_list[0] >= self.fasttext_confidence_keep and not is_sensitive:
            return langs[0], probs_list[0]

        # ── Level 4 (ПЕРЕД Transformer): чувствительные пары ───────
        if is_sensitive and len(langs) >= 2:
            binary_result = self.router.route(normalized, langs, probs_list)
            if binary_result is not None:
                return binary_result

        # ── PersoArabicLID: top-1 ∈ {ar, fa} ──────────────────────
        if langs[0] in _PALI_LANGUAGES and self.persoarabic.model is not None:
            pali_result = self.persoarabic.predict(normalized)
            if pali_result is not None:
                return pali_result

        # ── IndicLID-FTR: top-1 ∈ {hi, ur, ne} ────────────────────
        if langs[0] in INDICLID_LANGUAGES and self.indiclid.model is not None:
            indiclid_result = self.indiclid.predict(normalized)
            if indiclid_result is not None:
                return indiclid_result

        # ── Level 3: DistilBERT ONNX ───────────────────────────────
        if langs[0] != "hi":
            onnx_result = self._onnx_predict(normalized)
            if onnx_result is not None:
                return onnx_result

        # ── Fallback: fastText ответ ───────────────────────────────
        return langs[0], probs_list[0]

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Определить язык для списка текстов."""
        return [self.detect(t) for t in texts]

    def detect_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Вернуть топ-k кандидатов (от fastText, без роутера)."""
        if not text or not str(text).strip():
            return [("other", 0.0)]

        normalized = normalize_for_detection(text)
        script_result = self.script_detector.detect(normalized)
        if script_result is not None:
            lang, conf = script_result
            return [(lang, conf)]

        ft_text = preprocess_text(normalized).replace("\n", " ").replace("\r", " ")
        if not ft_text.strip():
            return [("other", 0.0)]

        labels, probs = self.ft_model.predict(ft_text, k=k)
        return [
            (merge_label(lbl.replace("__label__", "")), float(p))
            for lbl, p in zip(labels, probs)
        ]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ------------------------------------------------------------------
# Быстрый ручной тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model_path = "output/lang_detection_model.bin"
    onnx_path = "output/distilbert_lang_detection.onnx"

    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        sys.exit(1)

    detector = LanguageDetector(
        fasttext_model_path=model_path,
        onnx_model_path=onnx_path if os.path.exists(onnx_path) else None,
        router_verbose=False,
    )

    test_cases = [
        ("Привет, как дела?",                          "ru"),
        ("Hello, how are you?",                        "en"),
        ("Salam, necəsən?",                            "az"),
        ("Բարի՜ Ծաղիկ",                               "hy"),
        ("שלום עולם",                                  "he"),
        ("مرحبا بالعالم",                              "ar"),
        ("این یک متن فارسی است. پدر چه گفت؟",         "fa"),
        ("Доброго ранку, як справи?",                  "uk"),
        ("Xush kelibsiz",                              "uz"),
        ("こんにちは",                                  "other"),
    ]

    print("\n" + "=" * 70)
    print("ТЕСТ LanguageDetector")
    print("=" * 70)
    print(f"{'Текст':<40} {'Ожид':<7} {'Пред':<7} {'Conf':<7} {'OK?'}")
    print("-" * 70)

    ok = 0
    for text, expected in test_cases:
        lang, conf = detector.detect(text)
        status = "OK" if lang == expected else "XX"
        ok += lang == expected
        print(f"{text[:38]:<40} {expected:<7} {lang:<7} {conf:<7.3f} {status}")

    print("-" * 70)
    print(f"Итого: {ok}/{len(test_cases)} правильно")
