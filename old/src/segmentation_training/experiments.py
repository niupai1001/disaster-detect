from __future__ import annotations

import numpy as np


def trivial_background_prediction(label: np.ndarray) -> np.ndarray:
    prediction = np.zeros_like(label, dtype=np.uint8)
    prediction[label == 255] = 255
    return prediction

