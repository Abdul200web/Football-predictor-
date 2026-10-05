"""nlu.py - reads whatever someone types in plain English and turns it into a
structured request our engine can act on, using Google's Gemini model.
"""
import json
import time

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are the request parser for a sports prediction app called BoomBig.
Read the user's message and output ONLY a JSON object, nothing else, with this shape:

{
  "intent": "single_match" | "accumulator" | "other",
  "sport": "football" | "basketball" | null,
  "home_team": string or null,
  "away_team": string or null,
  "target_odds": number or null,
  "cutoff_hour": integer 0-23 or null,
  "cutoff_minute": integer 0-59 or null,
  "date": "today" | "tomorrow" | null
}

Rules:
- "single_match": the person names exactly two teams and wants a prediction, form, or head-to-head for that one game.
  Fill home_team and away_team with the team names as written by the user (do not translate or correct spelling).
- "accumulator": the person wants a combination/accumulator/multi-bet, usually asking for a target combined odds
  figure (e.g. "safest 20 odds", "combo worth 40", "accumulator around 15"). Fill target_odds with that number.
  If they give a time like "before 4pm" or "by 16:00", fill cutoff_hour/cutoff_minute (24-hour). If no time is
  given, leave cutoff_hour and cutoff_minute null.
- "other": anything else (greetings, unrelated questions, unclear requests).
- date defaults to "today" if not stated.
- sport defaults to "football" unless basketball, NBA, or similar is mentioned.
- Only output the JSON object. No explanation, no markdown fences.
"""

# Tried in order. Temporary overloads (503) are retried; missing/retired models (404) are skipped.
# If every listed model fails, we ask Google which models this key can really use and try those.
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
MAX_RETRIES_PER_MODEL = 2


def _call_model(client, model_name, text):
    return client.models.generate_content(
        model=model_name,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0,
        ),
    )


def _available_flash_models(client):
    """Ask Google which models this key can use; keep fast 'flash' text models, newest name first."""
    names = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(
                    x in name for x in ("image", "tts", "live", "audio", "embedding", "thinking")):
                names.append(name)
    except Exception:
        pass
    return sorted(set(names), reverse=True)


def _try_models(client, models, text, errors):
    for model_name in models:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                resp = _call_model(client, model_name, text)
                return json.loads(resp.text)
            except Exception as err:
                msg = str(err)
                errors[model_name] = msg[:160]
                if "404" in msg or "NOT_FOUND" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break  # retrying will not help; go to the next model
                time.sleep(1.5 * (attempt + 1))
    return None


def parse_request(text, api_key):
    """Returns a dict per the schema above. Falls back to intent='other' if every attempt fails."""
    fallback = {"intent": "other", "sport": None, "home_team": None, "away_team": None,
                "target_odds": None, "cutoff_hour": None, "cutoff_minute": None, "date": "today"}
    client = genai.Client(api_key=api_key)
    errors = {}
    data = _try_models(client, MODEL_CANDIDATES, text, errors)
    if data is None:
        extra = [m for m in _available_flash_models(client) if m not in MODEL_CANDIDATES][:4]
        data = _try_models(client, extra, text, errors)
    if data is not None:
        fallback.update({k: v for k, v in data.items() if k in fallback})
        return fallback
    fallback["_error"] = " | ".join(f"{m}: {e}" for m, e in errors.items()) or "unknown error"
    return fallback
