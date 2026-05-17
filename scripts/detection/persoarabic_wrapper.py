"""
Обёртка для PersoArabicLID — fasttext-модель для языков на арабской вязи.

Специализированная модель от PALI (Ahmadi et al., 2023), обученная на 19 языках
персидско-арабской письменности. Нас интересуют: arb → ar, fas → fa.
Остальные языки игнорируются (возвращаем None → fallback на DistilBERT).

Модель очень компактная (~2MB, .ftz), скорость ~0.03ms.
"""

import os
from typing import Optional, Tuple

import fasttext

_PALI_TO_ISO = {
    "ar": "ar",
    "fa": "fa",
}

_PALI_LANGUAGES = {"ar", "fa"}


class PersoArabicLIDWrapper:
    def __init__(
        self,
        model_path: str = "output/persoarabic/LID_model_merged.ftz",
        verbose: bool = True,
    ):
        self.model = None
        if os.path.exists(model_path):
            self.model = fasttext.load_model(model_path)
            if verbose:
                print(f"PersoArabicLID загружен: {model_path}")
        elif verbose:
            print(f"PersoArabicLID не найден: {model_path}")

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Предсказать язык текста (ar/fa) на арабской вязи.

        Returns:
            (iso_code, confidence) или None.
        """
        if self.model is None:
            return None

        clean = text.replace("\n", " ").replace("\r", " ").strip()
        if not clean:
            return None

        labels, probs = self.model.predict(clean, k=3)

        best_iso = None
        best_prob = -1.0

        for lbl, p in zip(labels, probs):
            label = lbl.replace("__label__", "")
            if label in _PALI_TO_ISO and float(p) > best_prob:
                best_iso = _PALI_TO_ISO[label]
                best_prob = float(p)

        if best_iso is not None:
            return best_iso, best_prob

        return None
