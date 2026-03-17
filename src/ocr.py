"""
ocr.py
------
OCR helper using EasyOCR.

Stable version:
- Multi-text merge
- Character allowlist
- Optional character segmentation
- Robust cleaning
"""

import re
import numpy as np
import easyocr
import cv2


# ------------------------------------------------------------
# Initialize EasyOCR Reader (load once)
# ------------------------------------------------------------
_READER = easyocr.Reader(["en"], gpu=False)


# ------------------------------------------------------------
# Clean OCR Text
# ------------------------------------------------------------
def clean_plate_text(text: str) -> str:

    if not text:
        return ""

    text = text.upper()

    # remove spaces and special characters
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ------------------------------------------------------------
# Character Segmentation (fallback)
# ------------------------------------------------------------
def segment_characters(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    chars = []

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        # filter small noise
        if h > image.shape[0] * 0.4 and w > 8:

            char = gray[y:y + h, x:x + w]

            char = cv2.resize(char, (40, 60))

            chars.append((x, char))

    # sort characters left → right
    chars = sorted(chars, key=lambda x: x[0])

    return [c[1] for c in chars]


# ------------------------------------------------------------
# Main OCR Function
# ------------------------------------------------------------
def ocr_easyocr(image_bgr: np.ndarray):

    # Run EasyOCR
    results = _READER.readtext(
        image_bgr,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    texts = []
    confs = []

    for r in results:

        text = clean_plate_text(r[1])

        if text:
            texts.append(text)
            confs.append(float(r[2]))

    # Merge detected text pieces
    merged_text = "".join(texts)

    if merged_text:
        return merged_text, max(confs)

    # --------------------------------------------------------
    # Fallback: character segmentation
    # --------------------------------------------------------

    chars = segment_characters(image_bgr)

    segmented_text = ""

    for ch in chars:

        res = _READER.readtext(
            ch,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        if res:

            segmented_text += clean_plate_text(res[0][1])

    if segmented_text:

        return segmented_text, 0.85

    return "", None