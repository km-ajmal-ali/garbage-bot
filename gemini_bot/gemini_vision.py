"""
GeminiBot Vision Module
========================
Handles all communication with the Google Gemini API for image analysis.
Encodes images as small JPEGs and sends targeted prompts to minimise
token usage and API cost.
"""

import io
import base64
import json
import time
import cv2
import numpy as np

from core.logger import get_logger
from gemini_bot.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    JPEG_QUALITY, RESIZE_WIDTH, RESIZE_HEIGHT,
)

log = get_logger("gemini_vision")

# ── Lazy-import google-generativeai ──────────────────────────────────
_client = None


def _get_client():
    """Initialise the Gemini client once on first use."""
    global _client
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=GEMINI_API_KEY)
            log.info("Gemini client initialised (model=%s)", GEMINI_MODEL)
        except ImportError:
            log.error("google-genai package not installed! "
                      "Run: pip install google-genai")
            raise
    return _client


# ─────────────────────────────────────────────────────────────────────
# Image preparation (cost optimisation)
# ─────────────────────────────────────────────────────────────────────

def prepare_image(frame: np.ndarray) -> bytes:
    """
    Resize and JPEG-compress a frame for the API.

    - Resizes to RESIZE_WIDTH×RESIZE_HEIGHT (much smaller than 800×800)
    - JPEG quality is kept low to minimise payload bytes
    - Returns raw JPEG bytes
    """
    small = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        log.error("JPEG encode failed")
        return b""
    jpeg_bytes = buf.tobytes()
    log.debug("Image prepared: %dx%d → %dx%d  JPEG %d bytes (q=%d)",
              frame.shape[1], frame.shape[0],
              RESIZE_WIDTH, RESIZE_HEIGHT,
              len(jpeg_bytes), JPEG_QUALITY)
    return jpeg_bytes


def _image_to_part(jpeg_bytes: bytes) -> dict:
    """Wrap JPEG bytes into the Gemini inline_data part format."""
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(jpeg_bytes).decode("utf-8"),
        }
    }


# ─────────────────────────────────────────────────────────────────────
# Scan prompt  (used during SCANNING phase)
# ─────────────────────────────────────────────────────────────────────

SCAN_PROMPT = """\
You are an object detection assistant for a garbage-collecting robot.
Analyse this image and determine if there is any garbage, trash, litter,
matchbox, plastic wrapper, paper waste, bottle, can, or any collectible
waste object visible on the ground.

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "found": true or false,
  "objects": [
    {
      "label": "short name of the object",
      "confidence": 0.0 to 1.0,
      "position": "left", "center", or "right",
      "size": "small", "medium", or "large",
      "distance": "far", "medium", or "close"
    }
  ]
}

If nothing is found, respond: {"found": false, "objects": []}
Keep your response minimal — only the JSON.
"""


def scan_image(frame: np.ndarray) -> dict:
    """
    Send a single frame to Gemini and ask whether garbage is visible.

    Returns:
        dict with keys 'found' (bool) and 'objects' (list).
        On error returns {"found": False, "objects": [], "error": "..."}.
    """
    jpeg = prepare_image(frame)
    if not jpeg:
        return {"found": False, "objects": [], "error": "encode_failed"}

    client = _get_client()

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                SCAN_PROMPT,
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(jpeg).decode("utf-8"),
                    }
                },
            ],
        )
        elapsed = time.time() - t0
        raw = response.text.strip()
        log.info("Gemini SCAN response (%.1fs): %s", elapsed, raw[:200])

        result = _parse_json(raw)
        return result

    except Exception as e:
        log.error("Gemini API error during scan: %s", e)
        return {"found": False, "objects": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Approach prompt  (used during APPROACH / COLLECT phase)
# ─────────────────────────────────────────────────────────────────────

APPROACH_PROMPT = """\
You are guiding a garbage-collecting robot that is driving toward
a waste object on the ground. Analyse this image.

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "object_visible": true or false,
  "position": "left", "center", or "right",
  "distance": "far", "medium", "close", or "very_close",
  "collected": true or false,
  "action": "forward", "left", "right", or "stop"
}

Rules for "action":
- "forward"  : object is centred, keep driving toward it
- "left"     : object is on the left side, steer left
- "right"    : object is on the right side, steer right
- "stop"     : object is very close / directly beneath robot (collected)

If object is no longer visible and was very close before, set
"collected": true and "action": "stop".

Keep your response minimal — only the JSON.
"""


def approach_image(frame: np.ndarray) -> dict:
    """
    Send a frame during approach and get navigation guidance.

    Returns:
        dict with keys: object_visible, position, distance, collected, action.
    """
    jpeg = prepare_image(frame)
    if not jpeg:
        return {"object_visible": False, "action": "stop",
                "collected": False, "error": "encode_failed"}

    client = _get_client()

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                APPROACH_PROMPT,
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(jpeg).decode("utf-8"),
                    }
                },
            ],
        )
        elapsed = time.time() - t0
        raw = response.text.strip()
        log.info("Gemini APPROACH response (%.1fs): %s", elapsed, raw[:200])

        result = _parse_json(raw)
        return result

    except Exception as e:
        log.error("Gemini API error during approach: %s", e)
        return {"object_visible": False, "action": "stop",
                "collected": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# JSON parser helper
# ─────────────────────────────────────────────────────────────────────

def _parse_json(raw_text: str) -> dict:
    """
    Robustly parse JSON from Gemini response.
    Handles markdown code-fencing and other common wrappers.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s — raw: %s", e, text[:300])
        # Attempt to find JSON within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"found": False, "objects": [], "error": "parse_failed",
                "raw": text[:200]}
