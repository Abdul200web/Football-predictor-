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

# Tried in order. If the first is overloaded (a common, temporary state for free models),
# we fall back to the next one rather than giving up.
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


def parse_request(text, api_key):
    """Returns a dict per the schema above. Falls back to intent='other' if every attempt fails."""
    fallback = {"intent": "other", "sport": None, "home_team": None, "away_team": None,
                "target_odds": None, "cutoff_hour": None, "cutoff_minute": None, "date": "today"}
    client = genai.Client(api_key=api_key)
    last_error = None
    for model_name in MODEL_CANDIDATES:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                resp = _call_model(client, model_name, text)
                data = json.loads(resp.text)
                fallback.update({k: v for k, v in data.items() if k in fallback})
                return fallback
            except Exception as err:
                last_error = err
                msg = str(err)
                # 503/UNAVAILABLE = temporarily overloaded, worth a quick retry or the next model.
                # 404/NOT_FOUND = this model name doesn't exist for us, skip straight to the next one.
                if "404" in msg or "NOT_FOUND" in msg:
                    break
                time.sleep(1.5 * (attempt + 1))
    fallback["_error"] = str(last_error)
    return fallback
