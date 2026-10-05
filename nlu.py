"""nlu.py - reads whatever someone types in plain English and turns it into a
structured request our engine can act on, using Google's Gemini model. It also answers
general chat (no predictions) so the app can hold a normal conversation.
"""
import json
import time

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are the request parser for a sports prediction app called BoomBig.
Read the user's NEW message (earlier messages are context only, e.g. for follow-ups like
"what about tomorrow?") and output ONLY a JSON object with this shape:

{
  "intent": "single_match" | "tips" | "accumulator" | "chat",
  "sport": "football" | "basketball" | null,
  "home_team": string or null,
  "away_team": string or null,
  "league": string or null,
  "market": string or null,
  "num_picks": integer or null,
  "target_odds": number or null,
  "after_hour": integer 0-23 or null,
  "cutoff_hour": integer 0-23 or null,
  "cutoff_minute": integer 0-59 or null,
  "date": "today" | "tomorrow" | "YYYY-MM-DD" | null
}

Rules:
- "single_match": exactly two teams named and a prediction/form/head-to-head wanted. Copy team names as written.
- "tips": the person wants picks/tips/predictions across several games, e.g. "give me 5 tips for the
  Premier League tomorrow", "best over 2.5 games tonight", "any safe games in Nigeria's friendly".
  Put the competition or country in "league" as written (null = all leagues). Put how many picks in "num_picks".
- "accumulator": wants a combination/multi-bet with a target combined odds (e.g. "safest 20 odds", "combo worth 40").
  Fill target_odds. Also use "league" if they limit it to a competition.
- "chat": anything else (greetings, how things work, general football or betting questions).
- "market": only if they ask for one specific market, using exactly one of: "Home win", "Draw", "Away win",
  "Home or draw", "Away or draw", "Over 1.5 goals", "Over 2.5 goals", "Both teams to score",
  "Over 9.5 corners". Otherwise null.
- Times are Nigerian local time. "tonight"/"this evening" = after_hour 17. "before 4pm" = cutoff_hour 16, cutoff_minute 0.
  "afternoon" = after_hour 12 and cutoff_hour 17. Leave unused time fields null.
- date defaults to "today". "weekend"/"Saturday" etc.: use the matching YYYY-MM-DD only if you are sure of today's date
  from the context given; otherwise "today".
- sport defaults to "football" unless basketball, NBA or similar is mentioned.
- Only output the JSON object. No explanation, no markdown fences.
"""

CHAT_PROMPT = """You are BoomBig, a friendly football prediction assistant inside a chat app.
Answer the person's message in at most 4 short sentences. You can explain betting and football concepts
(odds, accumulators, value, form, double chance, etc.). You do NOT have live data in this reply, so never state
fixtures, scores, odds or predictions from memory. If they want predictions, tell them to ask for something like
"5 tips for the Premier League tomorrow", "Arsenal vs Chelsea" or "safest 10 odds tonight". Never promise wins;
betting always carries risk."""

# Tried in order. Temporary overloads (503) are retried; missing/retired models (404) are skipped.
# If every listed model fails, we ask Google which models this key can really use and try those.
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
MAX_RETRIES_PER_MODEL = 2


def _call_model(client, model_name, contents, system, as_json):
    cfg = dict(system_instruction=system, temperature=0 if as_json else 0.5)
    if as_json:
        cfg["response_mime_type"] = "application/json"
    return client.models.generate_content(model=model_name, contents=contents,
                                          config=types.GenerateContentConfig(**cfg))


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


def _try_models(client, models, contents, system, as_json, errors):
    for model_name in models:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                resp = _call_model(client, model_name, contents, system, as_json)
                return json.loads(resp.text) if as_json else resp.text
            except Exception as err:
                msg = str(err)
                errors[model_name] = msg[:160]
                if "404" in msg or "NOT_FOUND" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break  # retrying will not help; go to the next model
                time.sleep(1.5 * (attempt + 1))
    return None


def _run(api_key, contents, system, as_json):
    client = genai.Client(api_key=api_key)
    errors = {}
    out = _try_models(client, MODEL_CANDIDATES, contents, system, as_json, errors)
    if out is None:
        extra = [m for m in _available_flash_models(client) if m not in MODEL_CANDIDATES][:4]
        out = _try_models(client, extra, contents, system, as_json, errors)
    err = " | ".join(f"{m}: {e}" for m, e in errors.items()) or "unknown error"
    return out, err


def parse_request(text, api_key, context=""):
    """Returns a dict per the schema above. Falls back to intent='chat' if every attempt fails."""
    fallback = {"intent": "chat", "sport": None, "home_team": None, "away_team": None, "league": None,
                "market": None, "num_picks": None, "target_odds": None, "after_hour": None,
                "cutoff_hour": None, "cutoff_minute": None, "date": "today"}
    contents = (f"Earlier conversation (context only):\n{context}\n\n" if context else "") + \
               f"New message to parse:\n{text}"
    data, err = _run(api_key, contents, SYSTEM_PROMPT, True)
    if isinstance(data, dict):
        fallback.update({k: v for k, v in data.items() if k in fallback and v is not None})
        return fallback
    fallback["_error"] = err
    return fallback


def chat_reply(text, api_key, context=""):
    """A short conversational answer with no predictions. Returns (text, error_or_None)."""
    contents = (f"Earlier conversation:\n{context}\n\n" if context else "") + f"Person says: {text}"
    out, err = _run(api_key, contents, CHAT_PROMPT, False)
    return (out.strip(), None) if out else (None, err)
