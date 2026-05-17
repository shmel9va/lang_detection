"""
Обёртка для IndicLID-FTR — fasttext-модель для romanized индийских языков.

Используется когда fastText top-1 ∈ {hi, ur, ne}.
Правило: hin_Latn → ur (латинский хинди = урду), urd_Latn → ur, nep_Latn → ne.
"""

import os
from typing import Optional, Tuple

import fasttext

_INDICLID_TO_ISO = {
    "hin_Latn": "ur",
    "urd_Latn": "ur",
    "nep_Latn": "ne",
}

INDICLID_LANGUAGES = {"hi", "ne", "ur"}


class IndicLIDWrapper:
    def __init__(
        self,
        model_path: str = "output/indiclid-ftr/model_baseline_roman.bin",
        verbose: bool = True,
    ):
        self.model = None
        if os.path.exists(model_path):
            self.model = fasttext.load_model(model_path)
            if verbose:
                print(f"IndicLID-FTR загружен: {model_path}")
        elif verbose:
            print(f"IndicLID-FTR не найден: {model_path}")

    _TOP_K = 3

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        if self.model is None:
            return None

        clean = text.replace("\n", " ").replace("\r", " ").strip()
        if not clean:
            return None

        labels, probs = self.model.predict(clean, k=self._TOP_K)

        best_iso = None
        best_prob = -1.0

        for lbl, p in zip(labels, probs):
            label = lbl.replace("__label__", "")
            if label in _INDICLID_TO_ISO and float(p) > best_prob:
                best_iso = _INDICLID_TO_ISO[label]
                best_prob = float(p)

        if best_iso is not None:
            return best_iso, best_prob

        return None
