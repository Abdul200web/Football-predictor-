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


@st.cache_resource(ttl=6 * 3600, show_spinner="Training the model (first time only)...")
def get_models(_hist):
    return E.train_models(_hist)


@st.cache_data(ttl=6 * 3600, show_spinner="Testing the model on past matches...")
def get_backtest(hist):
    return E.backtest(E.build_features(hist))


@st.cache_data(ttl=3 * 3600, show_spinner="Scoring upcoming matches...")
def get_picks(hist, fixtures, min_conf, days):
    return E.upcoming_picks(hist, fixtures, min_conf=min_conf, days_ahead=days)


def confidence_label(p):
    if p >= 0.80:
        return "very high"
    if p >= 0.70:
        return "high"
    if p >= 0.60:
        return "moderate"
    return "low"


def show_form(team, summ, venue_summ, venue_word, df, df_venue):
    st.markdown(f"**{team}**")
    st.write(f"Last 5 games (newest first): **{summ['string']}**  \n"
             f"{summ['W']} wins, {summ['D']} draws, {summ['L']} losses. "
             f"Scored {summ['gf']:.1f} and conceded {summ['ga']:.1f} goals per game.")
    if venue_summ["string"] != "-":
        st.write(f"{venue_word} form (last 5): **{venue_summ['string']}**, "
                 f"scoring {venue_summ['gf']:.1f} and conceding {venue_summ['ga']:.1f} per game.")
    st.dataframe(df[["Date", "Opponent", "Score", "Result"]], hide_index=True)


try:
    hist, fixtures, failed = get_data()
except Exception as err:  # shown in plain words instead of a scary crash
    st.error(f"Could not load data: {err}")
    st.stop()

teams = E.team_list(hist)
tab_match, tab_picks, tab_test, tab_about = st.tabs(["Analyze a match", "Weekly picks", "Track record", "About"])

# ---------------------------------------------------------- analyze a match --
with tab_match:
    st.write("Type two teams, like **Arsenal vs Chelsea**, then tap Analyze.")
    with st.form("match_form"):
        text = st.text_input("Match", placeholder="Arsenal vs Chelsea", label_visibility="collapsed")
        go = st.form_submit_button("Analyze", type="primary")
    with st.expander("Or choose the teams from a list"):
        c1, c2 = st.columns(2)
        pick_home = c1.selectbox("Home team", [""] + teams)
        pick_away = c2.selectbox("Away team", [""] + teams)
        go_list = st.button("Analyze these teams")
    with st.expander("Which teams and leagues are covered?"):
        st.write("Leagues covered: " + ", ".join(E.LEAGUES.values()) + ". Other competitions are not covered yet.")
        st.write(", ".join(teams))

    home = away = None
    if go and text.strip():
        parts = E.split_match(text)
        if not parts:
            st.warning("Please write it as two teams with 'vs' between them, for example: Arsenal vs Chelsea")
        else:
            (h, hs), (a, as_) = E.resolve_team(parts[0], teams), E.resolve_team(parts[1], teams)
            for typed, found, sugg in ((parts[0], h, hs), (parts[1], a, as_)):
                if found is None:
                    msg = f"I could not find a team called '{typed}'."
                    if sugg:
                        msg += " Did you mean: " + ", ".join(sugg) + "?"
                    st.warning(msg)
            if h and a:
                home, away = str(h), str(a)
    elif go_list and pick_home and pick_away:
        home, away = pick_home, pick_away

    if home and away:
        if home == away:
            st.warning("Please choose two different teams.")
        else:
            models, cols = get_models(hist)
            with st.spinner("Analyzing..."):
                st.session_state["result"] = E.analyze_match(models, cols, hist, home, away, fixtures)

    res = st.session_state.get("result")
    if res:
        st.divider()
        st.subheader(f"{res['home']} vs {res['away']}")
        line = res["league"]
        if res["date"] is not None:
            line += f" · {res['date']:%a %d %b}"
        st.caption(line)

        best = res["picks"][0]
        others = [p for p in res["picks"][1:] if p["prob"] >= 0.60][:3]
        if best["prob"] >= 0.60:
            st.success(f"**Best pick: {best['market']}**  \n"
                       f"Confidence {best['prob']:.0%} ({confidence_label(best['prob'])})")
            odds_txt = f" Bet365 price: {best['odds']:.2f}." if best["odds"] else ""
            st.caption("Confidence is the model's estimated chance this comes true." + odds_txt)
        else:
            st.warning(f"**No strong pick for this match.** The model's most likely outcome "
                       f"({best['market']}) is only {best['prob']:.0%}. Skipping this one is the safer choice.")
        if others:
            st.write("**Other options:** " + "; ".join(f"{p['market']} ({p['prob']:.0%})" for p in others))

        allp = {p["market"]: p["prob"] for p in res["picks"]}
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{res['home']} win", f"{allp['Home win']:.0%}")
        m2.metric("Draw", f"{allp['Draw']:.0%}")
        m3.metric(f"{res['away']} win", f"{allp['Away win']:.0%}")

        goals_bits = [f"{k} {allp[k]:.0%}" for k in ("Over 1.5 goals", "Over 2.5 goals", "Both teams to score",
                                                      "Over 9.5 corners") if k in allp]
        st.write("**Goals and corners:** " + " · ".join(goals_bits))
        st.write(f"**Strength rating:** {res['home']} {res['elo_home']:.0f} vs {res['away']} {res['elo_away']:.0f} "
                 f"(higher is stronger; 1500 is average)")

        st.markdown("### Recent form")
        show_form(res["home"], res["home_sum"], res["home_home_sum"], "Home", res["home_form"], res["home_at_home"])
        show_form(res["away"], res["away_sum"], res["away_away_sum"], "Away", res["away_form"], res["away_away"])

        st.markdown("### Last meetings")
        if res["h2h"].empty:
            st.write("No meetings found in the data.")
        else:
            st.dataframe(res["h2h"], hide_index=True)

        edge_rows = [p for p in res["picks"] if p["odds"]]
        if edge_rows:
            st.markdown("### Bookmaker prices (Bet365)")
            st.dataframe(pd.DataFrame({"Market": [p["market"] for p in edge_rows],
                                       "Odds": [f"{p['odds']:.2f}" for p in edge_rows],
                                       "Model chance": [f"{p['prob']:.0%}" for p in edge_rows],
                                       "Worth it?": ["Yes" if p["edge"] > 0 else "No" for p in edge_rows]}),
                         hide_index=True)
            st.caption("'Worth it' means the model thinks the odds pay more than the true risk.")
        st.caption("This is a statistical estimate, not a guarantee. It cannot see injuries or lineups yet. "
                   "Never bet money you cannot afford to lose.")

# ------------------------------------------------------------ weekly picks --
bt = get_backtest(hist)
with tab_picks:
    conf = st.select_slider("Minimum confidence", options=[60, 65, 70, 75, 80], value=70, format_func=lambda v: f"{v}%")
    days = st.slider("Days ahead", 1, 14, 7)
    picks = get_picks(hist, fixtures, conf / 100, days)
    st.caption(f"{len(fixtures)} upcoming fixtures were found in the data source.")

    row = bt["pooled"][bt["pooled"]["Min confidence"].round(2) == conf / 100]
    if len(row):
        r = row.iloc[0]
        st.info(f"On past matches the model has never seen, picks at {conf}%+ confidence came true "
                f"{r['Hit rate']:.0%} of the time (about {r['Picks per week']:.0f} picks a week across all covered leagues).")

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
    if "Avg odds" in tbl:
        tbl["Avg odds"] = tbl["Avg odds"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
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

**How to use it.** On the first tab, type a match such as "Arsenal vs Chelsea" and tap Analyze. You get the best
pick, the chances of each result, both teams' recent form and their last meetings.

**What it can't see yet.** Injuries and suspensions, lineups, weather, and basketball. Those come in later steps.

**Please remember.** No model wins every time. Bookmakers already price in most of this information, so use the
Track record tab to judge the model honestly, and never stake money you cannot afford to lose.
"""
    )
