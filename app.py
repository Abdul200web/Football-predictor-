"""app.py - the phone-friendly web page. Streamlit turns this file into a website."""
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Football Predictor")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading match data (first time takes a minute)...")
def get_data():
    hist, failed = E.load_history()
    fixtures = E.load_fixtures()
    return hist, fixtures, failed


@st.cache_data(ttl=6 * 3600, show_spinner="Testing the model on past matches...")
def get_backtest(hist):
    return E.backtest(E.build_features(hist))


@st.cache_data(ttl=3 * 3600, show_spinner="Scoring upcoming matches...")
def get_picks(hist, fixtures, min_conf, days):
    return E.upcoming_picks(hist, fixtures, min_conf=min_conf, days_ahead=days)


try:
    hist, fixtures, failed = get_data()
except Exception as err:  # shown in plain words instead of a scary crash
    st.error(f"Could not load data: {err}")
    st.stop()

bt = get_backtest(hist)
tab_picks, tab_test, tab_about = st.tabs(["Picks", "Track record", "About"])

# ------------------------------------------------------------------ picks --
with tab_picks:
    conf = st.select_slider("Minimum confidence", options=[60, 65, 70, 75, 80], value=70, format_func=lambda v: f"{v}%")
    days = st.slider("Days ahead", 1, 14, 7)
    picks = get_picks(hist, fixtures, conf / 100, days)

    row = bt["pooled"][bt["pooled"]["Min confidence"].round(2) == conf / 100]
    if len(row):
        r = row.iloc[0]
        st.info(f"On past matches the model has never seen, picks at {conf}%+ confidence came true "
                f"{r['Hit rate']:.0%} of the time (about {r['Picks per week']:.0f} picks a week across all six leagues).")

    if picks.empty:
        st.warning("No qualifying picks in this window. Either there are no fixtures (international break) "
                   "or nothing reaches this confidence. Try a lower confidence or more days.")
    else:
        leagues = st.multiselect("Leagues", sorted(picks["League"].unique()), default=sorted(picks["League"].unique()))
        show = picks[picks["League"].isin(leagues)].copy()
        show["Confidence"] = (show["Confidence"] * 100).round(0).astype(int).astype(str) + "%"
        show["Bookmaker odds"] = show["Bookmaker odds"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        st.dataframe(show, hide_index=True)
        st.caption("Odds shown are Bet365 prices from football-data.co.uk, only for win/draw/loss and 2.5 goals. "
                   "Check the real price on your bookmaker before placing anything.")

# ------------------------------------------------------------ track record --
with tab_test:
    st.write(f"The model learned from {bt['train_n']:,} older matches, then was tested on the newest "
             f"{bt['test_n']:,} ({bt['test_from']:%d %b %Y} to {bt['test_to']:%d %b %Y}), which it had never seen.")
    c1, c2 = st.columns(2)
    c1.metric("Model, every match", f"{bt['acc_1x2_model']:.1%}", help="Win/draw/loss accuracy if forced to pick every game")
    c2.metric("Bookmaker favourite", f"{bt['acc_1x2_bookmaker']:.1%}", help="Accuracy of always backing the shortest odds")

    st.subheader("How often does high confidence come true?")
    pooled = bt["pooled"].copy()
    pooled["Min confidence"] = (pooled["Min confidence"] * 100).astype(int).astype(str) + "%"
    pooled["Hit rate"] = pooled["Hit rate"].map("{:.1%}".format)
    pooled["Picks per week"] = pooled["Picks per week"].round(0).astype(int)
    st.dataframe(pooled, hide_index=True)

    st.subheader("By market")
    thr = st.select_slider("Confidence level", options=[60, 65, 70, 75, 80], value=70, format_func=lambda v: f"{v}%", key="t2")
    tbl = bt["table"][bt["table"]["Min confidence"].round(2) == thr / 100].drop(columns="Min confidence").copy()
    tbl["Hit rate"] = tbl["Hit rate"].map("{:.1%}".format)
    for c in ("Avg odds",):
        if c in tbl:
            tbl[c] = tbl[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    if "ROI (flat stake)" in tbl:
        tbl["ROI (flat stake)"] = tbl["ROI (flat stake)"].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "-")
    st.dataframe(tbl, hide_index=True)
    st.caption("ROI = profit per unit staked if you had backed every pick at Bet365's price. "
               "A hit rate alone tells you nothing about profit: low odds need a very high hit rate to break even.")
    if failed:
        st.caption("Some data files could not be downloaded: " + ", ".join(failed))

# ------------------------------------------------------------------- about --
with tab_about:
    st.markdown(
        """
**What this is.** A statistical model that looks at each team's last 5 and 10 matches (goals, points, corners),
an Elo strength rating, the last 5 meetings between the two teams, rest days and the league, and turns that into
probabilities for results, double chances, goals, both teams to score and corners.

**What it can't see yet.** Injuries and suspensions, lineups, weather, and basketball. Those come in later steps.

**Please remember.** No model wins every time. Bookmakers already price in most of this information, so use the
Track record tab to judge the model honestly, and never stake money you cannot afford to lose.
"""
    )
