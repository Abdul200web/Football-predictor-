"""nlu.py - reads whatever someone types in plain English and turns it into a
structured request our engine can act on, using Google's Gemini model.
"""
import json

import google.generativeai as genai

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

MODEL_NAME = "gemini-2.5-flash"


def parse_request(text, api_key):
    """Returns a dict per the schema above. Falls back to intent='other' if parsing fails."""
    fallback = {"intent": "other", "sport": None, "home_team": None, "away_team": None,
                "target_odds": None, "cutoff_hour": None, "cutoff_minute": None, "date": "today"}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(
            text, generation_config={"response_mime_type": "application/json", "temperature": 0})
        data = json.loads(resp.text)
        fallback.update({k: v for k, v in data.items() if k in fallback})
        return fallback
    except Exception:
        return fallback
