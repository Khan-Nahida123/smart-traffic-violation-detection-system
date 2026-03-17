"""
gemini_client.py
----------------
Gemini (Google GenAI) helper for drafting a traffic violation email notice.

Uses:
- Google Gemini API
- Professional government-style email format
- Safe fallback if Gemini fails

Environment variable required:
GEMINI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, Dict
from datetime import datetime

try:
    # Google official SDK
    from google import genai
except Exception:
    genai = None


# ------------------------------------------------------------
# Fallback draft (used if Gemini fails / not configured)
# ------------------------------------------------------------
def _fallback_draft(owner_name: str, plate: str, violation: str, fine_amount: int) -> str:

    current_time = datetime.now().strftime("%d %B %Y, %H:%M")
    location = "Automated Traffic Monitoring Zone"

    return (
        f"Subject: Traffic Violation Notice – {plate}\n\n"
        f"Dear {owner_name},\n\n"
        f"A traffic violation has been detected by the Automated Number Plate "
        f"Recognition (ANPR) monitoring system.\n\n"
        f"Vehicle Number : {plate}\n"
        f"Violation Type : {violation}\n"
        f"Fine Amount    : INR {fine_amount}\n"
        f"Date & Time    : {current_time}\n"
        f"Location       : {location}\n\n"
        f"Please ensure compliance with traffic regulations and clear the "
        f"applicable fine as per traffic authority guidelines.\n\n"
        f"Regards,\n"
        f"Traffic Monitoring Authority\n"
    )


# ------------------------------------------------------------
# Public function
# ------------------------------------------------------------
def draft_fine_email_with_gemini(
    owner_name: str,
    plate: str,
    violation: str,
    fine_amount: int
) -> Dict[str, Any]:
    """
    Generate a traffic violation notice using Gemini.

    Returns:
    {
      "ok": True/False,
      "draft": "...",
      "mode": "gemini" or "fallback",
      "error": "..." or ""
    }
    """

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if (genai is None) or (not api_key):
        return {
            "ok": False,
            "draft": _fallback_draft(owner_name, plate, violation, fine_amount),
            "mode": "fallback",
            "error": "Gemini not configured."
        }

    try:
        client = genai.Client(api_key=api_key)

        current_time = datetime.now().strftime("%d %B %Y, %H:%M")
        location = "Automated Traffic Monitoring Zone"

        prompt = f"""
You are drafting a professional traffic violation notice email from a traffic monitoring authority.

Write a short and clear formal notice.

Include the following information clearly:

Owner Name: {owner_name}
Vehicle Number: {plate}
Violation Type: {violation}
Fine Amount: INR {fine_amount}
Date & Time: {current_time}
Location: {location}

Rules:
- Use a professional government notice tone
- Keep the message concise
- Do not invent payment links
- Do not add unnecessary explanations
"""

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        draft = (getattr(resp, "text", "") or "").strip()

        if not draft:
            return {
                "ok": False,
                "draft": _fallback_draft(owner_name, plate, violation, fine_amount),
                "mode": "fallback",
                "error": "Gemini returned empty text."
            }

        return {
            "ok": True,
            "draft": draft,
            "mode": "gemini",
            "error": ""
        }

    except Exception as e:
        return {
            "ok": False,
            "draft": _fallback_draft(owner_name, plate, violation, fine_amount),
            "mode": "fallback",
            "error": str(e)
        }