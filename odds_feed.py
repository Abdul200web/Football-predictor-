"""odds_feed.py - upcoming fixtures + bookmaker odds from The Odds API (the-odds-api.com).

Returns a table shaped exactly like engine.load_fixtures() so the rest of BoomBig can use it,
plus a 'Kickoff' column (local time) so "before 6pm" / "tonight" can work.

Credit use (free plan = 500 credits/month): the sports list is free; each league's odds call
costs about 1 credit when it returns matches. We only ask for leagues that are in season, capped
at MAX_LEAGUES, and the app should cache the result for several hours.
"""
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://api.the-odds-api.com/v4"
LOCAL_TZ = "Africa/Lagos"
MAX_LEAGUES = 12

# Odds-API league key -> football-data.co.uk league code used by engine.py
DIV_BY_KEY = {
    "soccer_epl": "E0", "soccer_efl_champ": "E1", "soccer_spain_la_liga": "SP1",
    "soccer_germany_bundesliga": "D1", "soccer_italy_serie_a": "I1", "soccer_france_ligue_one": "F1",
    "soccer_netherlands_eredivisie": "N1", "soccer_belgium_first_div": "B1",
    "soccer_portugal_primeira_liga": "P1", "soccer_spl": "SC0", "soccer_turkey_super_league": "T1",
    "soccer_greece_super_league": "G1",
}
COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC",
           "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", "Kickoff"]


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["ODDS_API_KEY"]
        except Exception:
            key = None
    return key


def _get(path, **params):
    params["apiKey"] = _api_key()
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8")), resp.headers.get("x-requests-remaining")


def _h2h_odds(event):
    """Home/draw/away decimal odds: prefer bet365, otherwise the first bookmaker that has all three."""
    books = sorted(event.get("bookmakers", []), key=lambda b: b.get("key") != "bet365")
    for b in books:
        for m in b.get("markets", []):
            if m.get("key") == "h2h":
                prices = {o["name"]: o["price"] for o in m.get("outcomes", [])}
                h, a, d = prices.get(event["home_team"]), prices.get(event["away_team"]), prices.get("Draw")
                if h and a and d:
                    return h, d, a
    return np.nan, np.nan, np.nan


def parse_events(league_title, key, events, hours_ahead=48, now=None):
    """Turn raw Odds-API events into rows. Kept separate so it can be tested without internet."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    rows = []
    for e in events:
        ko = pd.Timestamp(e["commence_time"])
        if ko.tzinfo is None:
            ko = ko.tz_localize("UTC")
        if ko < now - pd.Timedelta(hours=3) or ko > now + pd.Timedelta(hours=hours_ahead):
            continue
        local = ko.tz_convert(LOCAL_TZ).tz_localize(None)
        h, d, a = _h2h_odds(e)
        rows.append({"Div": DIV_BY_KEY.get(key, league_title), "Date": local.normalize(),
                     "HomeTeam": e["home_team"], "AwayTeam": e["away_team"],
                     "FTHG": np.nan, "FTAG": np.nan, "HC": np.nan, "AC": np.nan,
                     "B365H": h, "B365D": d, "B365A": a, "B365>2.5": np.nan, "B365<2.5": np.nan,
                     "Kickoff": local})
    return rows


def harmonize_names(fx, teams):
    """Rename the Odds-API team names to the names used in the history data, when we can tell."""
    from engine import resolve_team  # imported here to avoid a circular import

    cache, unmatched = {}, set()

    def fix(name):
        if name in cache:
            return cache[name]
        found, _ = resolve_team(name, teams)
        if found is None:  # e.g. "Brighton and Hove Albion" contains the history name "Brighton"
            low = re.sub(r"[^a-z0-9' ]", " ", name.lower())
            subs = [t for t in teams if len(t) > 3 and re.search(rf"\b{re.escape(t.lower())}\b", low)]
            found = subs[0] if len(subs) == 1 else None
        if found is None:
            unmatched.add(name)
        cache[name] = found or name
        return cache[name]

    fx = fx.copy()
    fx["HomeTeam"], fx["AwayTeam"] = fx["HomeTeam"].map(fix), fx["AwayTeam"].map(fix)
    return fx, sorted(unmatched)


def load_fixtures_odds(hours_ahead=48, teams=None):
    """Returns (fixtures_dataframe, info_dict). Never raises: on any problem the table is empty
    and info['error'] explains why, so the app can show the real reason instead of 'no fixtures'."""
    info = {"leagues_checked": 0, "credits_remaining": None, "unmatched": [], "error": None}
    if not _api_key():
        info["error"] = "ODDS_API_KEY is missing from Streamlit secrets."
        return pd.DataFrame(columns=COLUMNS), info
    try:
        sports, _ = _get("/sports", all="false")
        leagues = [s for s in sports if s.get("group") == "Soccer" and not s.get("has_outrights")]
        # known big leagues first, then everything else (internationals, other countries)
        leagues.sort(key=lambda s: s["key"] not in DIV_BY_KEY)
        leagues = leagues[:MAX_LEAGUES]
        end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for lg in leagues:
            try:
                events, remaining = _get(f"/sports/{lg['key']}/odds", regions="eu", markets="h2h",
                                         oddsFormat="decimal", dateFormat="iso", commenceTimeTo=end)
            except Exception:
                continue
            info["credits_remaining"] = remaining
            info["leagues_checked"] += 1
            rows += parse_events(lg["title"], lg["key"], events, hours_ahead)
        fx = pd.DataFrame(rows, columns=COLUMNS)
        if teams is not None and len(fx):
            fx, info["unmatched"] = harmonize_names(fx, teams)
        return fx, info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=COLUMNS), info
