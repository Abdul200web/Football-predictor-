"""engine.py - downloads football data, builds features, trains models,
backtests them honestly, and produces picks for upcoming fixtures.

Data source: football-data.co.uk (free CSV files: results, corners, bookmaker odds,
plus a fixtures.csv with upcoming matches for the main leagues).
"""
import difflib
import io
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "SC0": "Scotland Premiership",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}
SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
RESULTS_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

KEEP = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC",
        "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)


# ----------------------------------------------------------------- data ----
def _read_csv(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")


def _clean(df):
    df = df.reindex(columns=KEEP)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in KEEP[4:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_history(leagues=None, seasons=SEASONS):
    """Download all past results (several files at once). Skips any file that cannot be fetched."""
    jobs = [(lg, s) for lg in (leagues or list(LEAGUES)) for s in seasons]

    def fetch(job):
        lg, s = job
        try:
            return _clean(_read_csv(RESULTS_URL.format(season=s, league=lg))), None
        except Exception:
            return None, f"{lg} {s}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, jobs))
    frames = [df for df, _ in results if df is not None]
    failed = [name for _, name in results if name]
    if not frames:
        raise RuntimeError("Could not download any data. Check the internet connection.")
    hist = pd.concat(frames, ignore_index=True)
    return hist.dropna(subset=["FTHG", "FTAG"]), failed


def load_fixtures(leagues=None):
    """Download upcoming fixtures (may be empty during international breaks)."""
    try:
        fx = _clean(_read_csv(FIXTURES_URL))
    except Exception:
        return pd.DataFrame(columns=KEEP)
    fx = fx[fx["Div"].isin(leagues or list(LEAGUES))].copy()
    fx["FTHG"], fx["FTAG"] = np.nan, np.nan
    return fx


# ------------------------------------------------------------- features ----
def _rolling_form(df, windows=(5, 10)):
    home = pd.DataFrame({"mid": df.index, "date": df["Date"], "team": df["HomeTeam"], "side": "h",
                         "gf": df["FTHG"], "ga": df["FTAG"], "cf": df["HC"], "ca": df["AC"]})
    away = pd.DataFrame({"mid": df.index, "date": df["Date"], "team": df["AwayTeam"], "side": "a",
                         "gf": df["FTAG"], "ga": df["FTHG"], "cf": df["AC"], "ca": df["HC"]})
    t = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "mid"]).reset_index(drop=True)
    t["pts"] = np.where(t["gf"] > t["ga"], 3.0, np.where(t["gf"] == t["ga"], 1.0, 0.0))
    t.loc[t["gf"].isna(), "pts"] = np.nan
    g = t.groupby("team")
    t["rest"] = g["date"].diff().dt.days.clip(upper=14)
    cols = []
    for w in windows:
        for c in ("gf", "ga", "pts", "cf", "ca"):
            name = f"{c}{w}"
            # shift(1) means we only ever use matches played BEFORE this one
            t[name] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean())
            cols.append(name)
    cols.append("rest")
    out = df[[]].copy()
    for side, prefix in (("h", "home_"), ("a", "away_")):
        part = t[t["side"] == side].set_index("mid")[cols].add_prefix(prefix)
        out = out.join(part)
    return out


def _elo(df, k=20.0, home_adv=60.0):
    ratings, eh, ea = {}, [], []
    for r in df.itertuples():
        rh, ra = ratings.get(r.HomeTeam, 1500.0), ratings.get(r.AwayTeam, 1500.0)
        eh.append(rh)
        ea.append(ra)
        if pd.notna(r.FTHG):
            exp = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
            s = 1.0 if r.FTHG > r.FTAG else 0.5 if r.FTHG == r.FTAG else 0.0
            ratings[r.HomeTeam] = rh + k * (s - exp)
            ratings[r.AwayTeam] = ra - k * (s - exp)
    return pd.DataFrame({"elo_home": eh, "elo_away": ea, "elo_diff": np.array(eh) - np.array(ea)}, index=df.index)


def _head_to_head(df, n=5):
    hist, rows = {}, []
    for r in df.itertuples():
        key = frozenset((r.HomeTeam, r.AwayTeam))
        past = hist.get(key, [])[-n:]
        if past:
            pts, gd, tot = [], [], []
            for ht, hg, ag in past:
                gf, ga = (hg, ag) if ht == r.HomeTeam else (ag, hg)
                pts.append(3 if gf > ga else 1 if gf == ga else 0)
                gd.append(gf - ga)
                tot.append(gf + ga)
            rows.append((np.mean(pts), np.mean(gd), np.mean(tot), len(past)))
        else:
            rows.append((np.nan, np.nan, np.nan, 0))
        if pd.notna(r.FTHG):
            hist.setdefault(key, []).append((r.HomeTeam, r.FTHG, r.FTAG))
    return pd.DataFrame(rows, columns=["h2h_pts", "h2h_gd", "h2h_goals", "h2h_n"], index=df.index)


def build_features(raw):
    """raw = played matches + (optionally) upcoming fixtures. Returns frame with features."""
    df = raw.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    feats = pd.concat([_rolling_form(df), _elo(df), _head_to_head(df)], axis=1)
    for code in LEAGUES:
        feats[f"lg_{code}"] = (df["Div"] == code).astype(float)
    return pd.concat([df, feats], axis=1)


def feature_columns(frame):
    return [c for c in frame.columns if c.startswith(("home_", "away_", "elo_", "h2h_", "lg_"))]


# -------------------------------------------------------------- targets ----
def add_targets(df):
    df = df.copy()
    tot = df["FTHG"] + df["FTAG"]
    df["y_1x2"] = np.where(df["FTHG"] > df["FTAG"], 0, np.where(df["FTHG"] == df["FTAG"], 1, 2))
    df["y_o15"] = (tot > 1.5).astype(float)
    df["y_o25"] = (tot > 2.5).astype(float)
    df["y_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(float)
    corners = df["HC"] + df["AC"]
    df["y_c95"] = np.where(corners.isna(), np.nan, (corners > 9.5).astype(float))
    return df


BINARY = {"y_o15": ("Over 1.5 goals", "Under 1.5 goals"),
          "y_o25": ("Over 2.5 goals", "Under 2.5 goals"),
          "y_btts": ("Both teams to score", "Both teams NOT to score"),
          "y_c95": ("Over 9.5 corners", "Under 9.5 corners")}


def _pipe():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.3, max_iter=3000))


def fit_models(train, cols):
    train = add_targets(train)
    models = {"1x2": _pipe().fit(train[cols], train["y_1x2"])}
    for target in BINARY:
        ok = train[target].notna()
        if ok.sum() > 200:
            models[target] = _pipe().fit(train.loc[ok, cols], train.loc[ok, target])
    return models


def predict_markets(models, X):
    p = pd.DataFrame(index=X.index)
    m = models["1x2"].predict_proba(X)
    p["Home win"], p["Draw"], p["Away win"] = m[:, 0], m[:, 1], m[:, 2]
    p["Home or draw"] = p["Home win"] + p["Draw"]
    p["Away or draw"] = p["Away win"] + p["Draw"]
    for target, (yes, no) in BINARY.items():
        if target in models:
            q = models[target].predict_proba(X)[:, 1]
            p[yes], p[no] = q, 1 - q
    return p


def market_truth(df):
    df = add_targets(df)
    t = pd.DataFrame(index=df.index)
    t["Home win"] = (df["y_1x2"] == 0).astype(float)
    t["Draw"] = (df["y_1x2"] == 1).astype(float)
    t["Away win"] = (df["y_1x2"] == 2).astype(float)
    t["Home or draw"] = (df["y_1x2"] != 2).astype(float)
    t["Away or draw"] = (df["y_1x2"] != 0).astype(float)
    for target, (yes, no) in BINARY.items():
        t[yes], t[no] = df[target], 1 - df[target]
    return t


ODDS_COL = {"Home win": "B365H", "Draw": "B365D", "Away win": "B365A",
            "Over 2.5 goals": "B365>2.5", "Under 2.5 goals": "B365<2.5"}


# ------------------------------------------------------------- backtest ----
def backtest(feat, test_frac=0.25):
    """Train on the older matches, test on the newest ones the model has never seen."""
    played = add_targets(feat[feat["FTHG"].notna()].sort_values("Date"))
    played = played.dropna(subset=["home_gf5", "away_gf5"])
    cut = int(len(played) * (1 - test_frac))
    train, test = played.iloc[:cut], played.iloc[cut:]
    cols = feature_columns(played)
    models = fit_models(train, cols)
    probs = predict_markets(models, test[cols])
    truth = market_truth(test)[probs.columns]

    rows = []
    for m in probs.columns:
        for thr in THRESHOLDS:
            sel = (probs[m] >= thr) & truth[m].notna()
            n = int(sel.sum())
            if n == 0:
                continue
            row = {"Market": m, "Min confidence": thr, "Picks": n, "Hit rate": truth.loc[sel, m].mean()}
            if m in ODDS_COL:
                odds = test.loc[sel, ODDS_COL[m]]
                won = truth.loc[sel, m] == 1
                ok = odds.notna()
                if ok.sum() > 0:
                    row["Avg odds"] = odds[ok].mean()
                    row["ROI (flat stake)"] = np.where(won[ok], odds[ok] - 1, -1).mean()
            rows.append(row)
    table = pd.DataFrame(rows)

    # all markets pooled: how often does "confidence >= X" really come true?
    pooled = []
    for thr in THRESHOLDS:
        n = hits = 0
        for m in probs.columns:
            sel = (probs[m] >= thr) & truth[m].notna()
            n += int(sel.sum())
            hits += float(truth.loc[sel, m].sum())
        if n:
            pooled.append({"Min confidence": thr, "Picks": n, "Hit rate": hits / n,
                           "Picks per week": n / max(1, (test["Date"].max() - test["Date"].min()).days / 7)})
    pooled = pd.DataFrame(pooled)

    # plain win/draw/loss accuracy, compared with simply backing the bookmaker favourite
    pick = probs[["Home win", "Draw", "Away win"]].values.argmax(axis=1)
    acc_model = float((pick == test["y_1x2"].values).mean())
    odds3 = test[["B365H", "B365D", "B365A"]]
    ok = odds3.notna().all(axis=1)
    acc_book = float((odds3[ok].values.argmin(axis=1) == test.loc[ok, "y_1x2"].values).mean()) if ok.any() else np.nan

    return {"train_n": len(train), "test_n": len(test),
            "test_from": test["Date"].min(), "test_to": test["Date"].max(),
            "table": table, "pooled": pooled, "acc_1x2_model": acc_model, "acc_1x2_bookmaker": acc_book}


# ------------------------------------------------------ upcoming picks ----
def upcoming_picks(history, fixtures, min_conf=0.70, days_ahead=7):
    """Train on everything played so far and score the upcoming fixtures."""
    if fixtures.empty:
        return pd.DataFrame()
    key = lambda d: d["Date"].dt.strftime("%Y-%m-%d") + d["HomeTeam"] + d["AwayTeam"]
    fixtures = fixtures[~key(fixtures).isin(set(key(history)))].copy()
    today = pd.Timestamp.today().normalize()
    fixtures = fixtures[(fixtures["Date"] >= today) & (fixtures["Date"] <= today + pd.Timedelta(days=days_ahead))]
    if fixtures.empty:
        return pd.DataFrame()
    history, fixtures = history.copy(), fixtures.copy()
    history["_fx"], fixtures["_fx"] = False, True
    feat = build_features(pd.concat([history, fixtures], ignore_index=True))
    cols = feature_columns(feat)
    played = feat[~feat["_fx"]].dropna(subset=["home_gf5", "away_gf5"])
    upcoming = feat[feat["_fx"]]
    models = fit_models(played, cols)
    probs = predict_markets(models, upcoming[cols])

    rows = []
    for idx, r in upcoming.iterrows():
        for m in probs.columns:
            conf = probs.loc[idx, m]
            if conf >= min_conf:
                odds = r.get(ODDS_COL.get(m, ""), np.nan) if m in ODDS_COL else np.nan
                rows.append({"Date": r["Date"].strftime("%a %d %b"), "League": LEAGUES.get(r["Div"], r["Div"]),
                             "Match": f"{r['HomeTeam']} vs {r['AwayTeam']}", "Pick": m,
                             "Confidence": conf, "Bookmaker odds": odds})
    out = pd.DataFrame(rows)
    return out.sort_values("Confidence", ascending=False).reset_index(drop=True) if len(out) else out


# ------------------------------------------------- single-match analysis ----
ALIASES = {
    "man utd": "Man United", "man u": "Man United", "manchester united": "Man United", "mufc": "Man United",
    "manchester city": "Man City", "city": "Man City", "spurs": "Tottenham", "tottenham hotspur": "Tottenham",
    "wolverhampton": "Wolves", "newcastle united": "Newcastle", "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest", "forest": "Nott'm Forest", "west ham united": "West Ham",
    "leeds united": "Leeds", "leicester city": "Leicester", "sheffield utd": "Sheffield United",
    "sheffield wed": "Sheffield Weds", "sheffield wednesday": "Sheffield Weds",
    "barca": "Barcelona", "atletico madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao", "real sociedad": "Sociedad",
    "real betis": "Betis", "bayern": "Bayern Munich", "bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund", "bvb": "Dortmund", "bayer leverkusen": "Leverkusen",
    "gladbach": "M'gladbach", "monchengladbach": "M'gladbach", "eintracht frankfurt": "Ein Frankfurt",
    "frankfurt": "Ein Frankfurt", "inter milan": "Inter", "internazionale": "Inter", "ac milan": "Milan",
    "psg": "Paris SG", "paris saint germain": "Paris SG", "paris saint-germain": "Paris SG",
}


def team_list(hist):
    return sorted(set(hist["HomeTeam"]) | set(hist["AwayTeam"]))


def split_match(text):
    """'Arsenal vs Chelsea' -> ('Arsenal', 'Chelsea'). Returns None if it cannot tell."""
    parts = re.split(r"\s+(?:vs\.?|v|versus|against|x|-)\s+", text.strip(), maxsplit=1, flags=re.I)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 and all(p.strip() for p in parts) else None


def resolve_team(text, teams):
    """Turn what the person typed into an exact team name. Returns (team or None, suggestions)."""
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()
    if not t:
        return None, []
    lower = {x.lower(): x for x in teams}
    if t in lower:
        return lower[t], []
    if t in ALIASES and ALIASES[t] in teams:
        return ALIASES[t], []
    subs = [x for x in teams if t in x.lower()]
    if len(subs) == 1:
        return subs[0], []
    close = difflib.get_close_matches(t, list(lower), n=4, cutoff=0.55)
    if close and difflib.SequenceMatcher(None, t, close[0]).ratio() >= 0.8:
        return lower[close[0]], []
    return None, (subs[:4] or [lower[c] for c in close])


def train_models(hist):
    """Train once on every played match; reused for any single-match question."""
    feat = build_features(hist)
    played = feat.dropna(subset=["home_gf5", "away_gf5"])
    cols = feature_columns(played)
    return fit_models(played, cols), cols


def recent_matches(hist, team, n=5, venue=None):
    d = hist[(hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)]
    if venue == "home":
        d = d[d["HomeTeam"] == team]
    elif venue == "away":
        d = d[d["AwayTeam"] == team]
    d = d.sort_values("Date").tail(n).iloc[::-1]  # newest first
    rows = []
    for r in d.itertuples():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        corners = r.HC if home else r.AC
        rows.append({"Date": r.Date.strftime("%d %b %y"),
                     "Opponent": ("vs " if home else "at ") + (r.AwayTeam if home else r.HomeTeam),
                     "Score": f"{int(gf)}-{int(ga)}",
                     "Result": "W" if gf > ga else "D" if gf == ga else "L",
                     "gf": gf, "ga": ga, "corners": corners})
    return pd.DataFrame(rows, columns=["Date", "Opponent", "Score", "Result", "gf", "ga", "corners"])


def form_summary(df):
    if df.empty:
        return {"W": 0, "D": 0, "L": 0, "gf": 0.0, "ga": 0.0, "corners": np.nan, "string": "-"}
    return {"W": int((df["Result"] == "W").sum()), "D": int((df["Result"] == "D").sum()),
            "L": int((df["Result"] == "L").sum()), "gf": float(df["gf"].mean()), "ga": float(df["ga"].mean()),
            "corners": float(df["corners"].mean()) if df["corners"].notna().any() else np.nan,
            "string": " ".join(df["Result"])}


def head_to_head(hist, home, away, n=5):
    d = hist[((hist["HomeTeam"] == home) & (hist["AwayTeam"] == away)) |
             ((hist["HomeTeam"] == away) & (hist["AwayTeam"] == home))]
    d = d.sort_values("Date").tail(n).iloc[::-1]
    return pd.DataFrame({"Date": d["Date"].dt.strftime("%d %b %y"), "Home": d["HomeTeam"],
                         "Score": d["FTHG"].astype(int).astype(str) + "-" + d["FTAG"].astype(int).astype(str),
                         "Away": d["AwayTeam"]})


def analyze_match(models, cols, hist, home, away, fixtures=None):
    """Everything the app shows for one match: form, head to head, model probabilities."""
    mine = hist[(hist["HomeTeam"] == home) | (hist["AwayTeam"] == home)].sort_values("Date")
    div = mine["Div"].iloc[-1]
    date, odds = pd.Timestamp.today().normalize(), {}
    if fixtures is not None and len(fixtures):
        m = fixtures[(fixtures["HomeTeam"] == home) & (fixtures["AwayTeam"] == away)]
        if len(m):
            date = m.iloc[0]["Date"]
            odds = {k: float(m.iloc[0][v]) for k, v in ODDS_COL.items() if pd.notna(m.iloc[0][v])}
    date = max(date, hist["Date"].max() + pd.Timedelta(days=1))
    row = pd.DataFrame([{c: np.nan for c in KEEP}])
    row["Div"], row["Date"], row["HomeTeam"], row["AwayTeam"] = div, date, home, away
    row[KEEP[4:]] = row[KEEP[4:]].astype(float)
    both = pd.concat([hist.assign(_fx=False), row.assign(_fx=True)], ignore_index=True)
    feat = build_features(both)
    X = feat[feat["_fx"]][cols]
    probs = predict_markets(models, X).iloc[0]
    ranked = probs.sort_values(ascending=False)
    picks = []
    for market, p in ranked.items():
        o = odds.get(market)
        picks.append({"market": market, "prob": float(p), "odds": o, "edge": (p * o - 1) if o else None})
    h_all, a_all = recent_matches(hist, home), recent_matches(hist, away)
    h_home, a_away = recent_matches(hist, home, venue="home"), recent_matches(hist, away, venue="away")
    return {"home": home, "away": away, "league": LEAGUES.get(div, div), "date": date if odds else None,
            "picks": picks, "has_odds": bool(odds),
            "elo_home": float(X["elo_home"].iloc[0]), "elo_away": float(X["elo_away"].iloc[0]),
            "home_form": h_all, "away_form": a_all, "home_at_home": h_home, "away_away": a_away,
            "home_sum": form_summary(h_all), "away_sum": form_summary(a_all),
            "home_home_sum": form_summary(h_home), "away_away_sum": form_summary(a_away),
            "h2h": head_to_head(hist, home, away)}


# --------------------------------------------------------- accumulator ----
def best_market_for_fixture(probs_row, min_prob=0.55):
    ranked = probs_row.sort_values(ascending=False)
    for market, p in ranked.items():
        if p >= min_prob:
            return market, float(p)
    return None, None


def estimate_odds(prob, margin=1.07):
    """Fair odds with a small bookmaker-style margin, used only when we have no real quoted price."""
    return round(1 / (prob * margin), 2) if prob else None


def daily_candidates(hist, fixtures, date=None, min_prob=0.55):
    """One safest market per fixture, for a given day. No precise kickoff-time filter yet (see note below)."""
    if fixtures.empty:
        return []
    date = (date or pd.Timestamp.today()).normalize()
    fx = fixtures[fixtures["Date"].dt.normalize() == date].copy()
    if fx.empty:
        return []
    history2, fx2 = hist.copy(), fx.copy()
    history2["_fx"], fx2["_fx"] = False, True
    feat = build_features(pd.concat([history2, fx2], ignore_index=True))
    cols = feature_columns(feat)
    played = feat[~feat["_fx"]].dropna(subset=["home_gf5", "away_gf5"])
    upcoming = feat[feat["_fx"]]
    if played.empty or upcoming.empty:
        return []
    models = fit_models(played, cols)
    probs = predict_markets(models, upcoming[cols])
    out = []
    for idx, r in upcoming.iterrows():
        market, prob = best_market_for_fixture(probs.loc[idx], min_prob=min_prob)
        if market is None:
            continue
        odds, estimated = np.nan, False
        if market in ODDS_COL:
            odds = r.get(ODDS_COL[market], np.nan)
        if pd.isna(odds):
            odds, estimated = estimate_odds(prob), True
        out.append({"match": f"{r['HomeTeam']} vs {r['AwayTeam']}", "league": LEAGUES.get(r["Div"], r["Div"]),
                    "market": market, "prob": prob, "odds": float(odds), "estimated_odds": estimated,
                    "fixture_key": idx})
    return out


def build_accumulator(candidates, target_odds, max_legs=12):
    """Greedily add the safest available legs, tier by tier, until reaching close to the target odds."""
    chosen, cum_odds, cum_prob, used = [], 1.0, 1.0, set()
    if not candidates:
        return {"legs": [], "combined_odds": 1.0, "combined_prob": 1.0, "reached": False}
    for tier in (0.80, 0.75, 0.70, 0.65, 0.60, 0.55):
        pool = sorted((c for c in candidates if c["prob"] >= tier and c["fixture_key"] not in used),
                     key=lambda c: -c["prob"])
        for c in pool:
            if len(chosen) >= max_legs or cum_odds >= target_odds * 0.97:
                break
            projected = cum_odds * c["odds"]
            if projected <= target_odds * 1.25:
                chosen.append(c)
                cum_odds, cum_prob = projected, cum_prob * c["prob"]
                used.add(c["fixture_key"])
        if cum_odds >= target_odds * 0.97 or len(chosen) >= max_legs:
            break
    return {"legs": chosen, "combined_odds": round(cum_odds, 2), "combined_prob": cum_prob,
            "reached": cum_odds >= target_odds * 0.8}
