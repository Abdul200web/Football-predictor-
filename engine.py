"""engine.py - downloads football data, builds features, trains models,
backtests them honestly, and produces picks for upcoming fixtures.

Data source: football-data.co.uk (free CSV files: results, corners, bookmaker odds,
plus a fixtures.csv with upcoming matches for the main leagues).
"""
import io
import urllib.request

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
    """Download all past results. Skips any file that cannot be fetched."""
    frames, failed = [], []
    for lg in (leagues or list(LEAGUES)):
        for s in seasons:
            try:
                frames.append(_clean(_read_csv(RESULTS_URL.format(season=s, league=lg))))
            except Exception:
                failed.append(f"{lg} {s}")
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
