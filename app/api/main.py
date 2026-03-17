"""
ANPR FastAPI Backend
--------------------

End-to-end Automatic Number Plate Recognition API.

Pipeline:
Upload image → Plate detection → OCR → Fine computation
→ Database logging → Gemini email generation → SMTP email send

This backend simulates a real-world traffic enforcement pipeline
using a modular ML + API architecture.
"""

import os
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ---------------------------------------------------------------------
# ML pipeline imports
# ---------------------------------------------------------------------

from src.detector import PlateDetector
from src.cropper import crop_bbox, crop_center_region
from src.preprocess import preprocess_for_ocr
from src.ocr import ocr_easyocr


# ---------------------------------------------------------------------
# Business logic imports
# ---------------------------------------------------------------------

from src.db_client import DBClient
from src.fine_engine import compute_fine
from src.email_sender import send_email_smtp
from src.gemini_client import draft_fine_email_with_gemini


# ---------------------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------------------

app = FastAPI(title="ANPR API")


# Load trained YOLO model
detector = PlateDetector("models/best.pt")


# Database client
db = DBClient(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
    database=os.getenv("DB_NAME", "anpr_db"),
)


# Demo email
DEMO_EMAIL = os.getenv("SMTP_USER")


# ---------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    """Check if API is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------
# Main ANPR endpoint
# ---------------------------------------------------------------------

@app.post("/anpr")
async def anpr(
    file: UploadFile = File(...),
    violation_type: str = Form("No Violation"),
):

    # -----------------------------------------------------------------
    # Decode uploaded image
    # -----------------------------------------------------------------

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "fail", "reason": "Invalid image"}


    # -----------------------------------------------------------------
    # Plate detection
    # -----------------------------------------------------------------

    bbox, _ = detector.detect_best_plate(img)

    crop = crop_bbox(img, bbox)

    if crop is not None:
        plate_img = crop
    else:
        # fallback center crop
        plate_img = crop_center_region(img, width_ratio=0.95, height_ratio=0.55)


    # -----------------------------------------------------------------
    # DEBUG IMAGE (helps diagnose OCR issues)
    # -----------------------------------------------------------------

    cv2.imwrite("debug_plate.png", plate_img)


    # -----------------------------------------------------------------
    # Preprocess for OCR
    # -----------------------------------------------------------------

    plate_img_pp = preprocess_for_ocr(plate_img)


    # -----------------------------------------------------------------
    # OCR extraction
    # -----------------------------------------------------------------

    ocr_text, ocr_conf = ocr_easyocr(plate_img_pp)


    # Ignore low confidence OCR
    if ocr_conf is not None and ocr_conf < 0.4:
        ocr_text = ""


    # -----------------------------------------------------------------
    # Fine computation
    # -----------------------------------------------------------------

    is_fined, fine_amount = compute_fine(violation_type)

    db_log = None
    email_result = None


    # -----------------------------------------------------------------
    # Database logging
    # -----------------------------------------------------------------

    if ocr_text:
        try:
            db_log = db.insert_fine_log(
                plate=ocr_text,
                violation_type=violation_type,
                fine_amount=fine_amount,
                is_fined=is_fined,
                ocr_text=ocr_text,
                ocr_conf=float(ocr_conf) if ocr_conf else None,
                email_sent=0,
            )
        except Exception as e:
            print("DB log error:", e)


    # -----------------------------------------------------------------
    # Generate email using Gemini
    # -----------------------------------------------------------------

    if DEMO_EMAIL and ocr_text:

        subject = f"Traffic Violation Notice - Vehicle {ocr_text}"

        email_draft = draft_fine_email_with_gemini(
            owner_name="Vehicle Owner",
            plate=ocr_text,
            violation=violation_type,
            fine_amount=fine_amount,
        )

        body = email_draft.get("draft")

        email_result = send_email_smtp(DEMO_EMAIL, subject, body)

        if (
            email_result.get("sent")
            and db_log
            and isinstance(db_log, dict)
            and "fine_id" in db_log
        ):
            db.mark_email_sent(db_log["fine_id"])


    # -----------------------------------------------------------------
    # API response
    # -----------------------------------------------------------------

    return {
        "status": "success",
        "plate": ocr_text,
        "violation": violation_type,
        "fine": fine_amount,
        "email_sent": email_result.get("sent") if email_result else False
    }