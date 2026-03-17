"""
preprocess.py
-------------
Preprocessing utilities for OCR.

Improved version:
- better contrast
- controlled upscale
- noise removal
- OCR-friendly threshold
"""

import cv2
import numpy as np


# ------------------------------------------------------------
# OCR Preprocessing
# ------------------------------------------------------------
def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE (better than equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Resize image (OCR works better on larger text)
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Slight blur to remove noise
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Binary threshold (clean characters)
    _, th = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY
    )

    # Convert back to BGR for EasyOCR
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
