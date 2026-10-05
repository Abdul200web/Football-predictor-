"""app.py - BoomBig: a chat-style front end. Type a question in plain English,
an AI model (Gemini) works out what you're asking, and our own engine (engine.py)
does the actual prediction using real match data. No outside AI ever picks the bets.
"""
import pandas as pd
import streamlit as st

import engine as E
import nlu

st.set_page_config(page_title="BoomBig", page_icon="⚡", layout="centered")

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

st.markdown("""
<style>
.stApp { background-color: #0b0f1a; }
</style>
""", unsafe_allow_html=True)

st.markdown("## ⚡ BoomBig")
st.caption("Football & basketball predictions. Ask in your own words.")

if not GEMINI_KEY:
    st.error("The app isn't fully set up yet: no Gemini API key found in Settings → Secrets. "
             "Add GEMINI_API_KEY there, then reload this page.")
    st.stop()


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading match data (first time takes a minute)...")
def get_data():
    return E.load_history()


@st.cache_data(ttl=12 * 3600, show_spinner="Getting today's fixtures...")
def get_fixtures(team_names):
    """Main source: The Odds API (many leagues, kickoff times, odds). Backup: football-data.co.uk.
    Cached for 12 hours because the free plan has a small monthly credit allowance."""
    import pandas as pd
    import odds_feed
    fx, info = odds_feed.load_fixtures_odds(teams=list(team_names))
    try:
        backup = E.load_fixtures()
    except Exception:
        backup = pd.DataFrame()
    if len(backup):
        fx = pd.concat([fx, backup], ignore_index=True)
        fx = fx.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    return fx, info


try:
    hist, failed = get_data()
    teams = E.team_list(hist)
    fixtures, feed_info = get_fixtures(tuple(teams))
except Exception as err:
    st.error(f"Could not load match data: {err}")
    st.stop()


# ------------------------------------------------------------- formatting --
def fmt_single_match(home, away):
    if home == away:
        return "That's the same team twice — please give me two different teams."
    models, cols = E.train_models(hist)
    res = E.analyze_match(models, cols, hist, home, away, fixtures)
    best = res["picks"][0]
    lines = [f"**{res['home']} vs {res['away']}** · {res['league']}", ""]
    if best["prob"] >= 0.60:
        odds_txt = f" (Bet365: {best['odds']:.2f})" if best["odds"] else ""
        lines.append(f"**Best pick: {best['market']}** — {best['prob']:.0%} confidence{odds_txt}")
    else:
        lines.append(f"No strong pick here — the most likely outcome ({best['market']}) "
                     f"is only {best['prob']:.0%}. I'd skip this one.")
    allp = {p["market"]: p["prob"] for p in res["picks"]}
    lines.append("")
    lines.append(f"Home win {allp.get('Home win', 0):.0%} · Draw {allp.get('Draw', 0):.0%} · "
                f"Away win {allp.get('Away win', 0):.0%}")
    goals = [f"{k} {allp[k]:.0%}" for k in ("Over 1.5 goals", "Over 2.5 goals", "Both teams to score") if k in allp]
    if goals:
        lines.append(" · ".join(goals))
    lines.append("")
    lines.append(f"**{res['home']} form (last 5):** {res['home_sum']['string']} "
                f"({res['home_sum']['W']}W {res['home_sum']['D']}D {res['home_sum']['L']}L)")
    lines.append(f"**{res['away']} form (last 5):** {res['away_sum']['string']} "
                f"({res['away_sum']['W']}W {res['away_sum']['D']}D {res['away_sum']['L']}L)")
    if not res["h2h"].empty:
        last = res["h2h"].iloc[0]
        lines.append(f"**Last meeting:** {last['Home']} {last['Score']} {last['Away']} ({last['Date']})")
    lines.append("")
    lines.append("_Statistical estimate, not a guarantee — can't see injuries or lineups yet._")
    return "\n\n".join(lines)


def fmt_accumulator(target_odds, cutoff_hour, cutoff_minute, tonight=False):
    before = (cutoff_hour, cutoff_minute or 0) if cutoff_hour is not None else None
    after = (17, 0) if tonight and before is None else None
    cands = E.daily_candidates(hist, fixtures, before=before, after=after)
    if not cands:
        msg = ("I couldn't find usable fixtures for that window. Matches only count when both teams "
               "have enough past results for me to judge them (international teams don't yet).")
        if feed_info.get("error"):
            msg += f"\n\n_Fixture feed problem: {feed_info['error']}_"
        elif feed_info.get("leagues_checked", 0) == 0:
            msg += "\n\n_The odds feed returned no leagues, so I'm only using the backup source._"
        return msg
    acc = E.build_accumulator(cands, target_odds)
    if not acc["legs"]:
        return "None of today's fixtures reached a safe enough confidence level to build a combination. Try again tomorrow."
    lines = [f"**Closest safe combination to {target_odds}x**", ""]
    lines.append(f"Combined odds: **{acc['combined_odds']}x** · "
                f"Estimated chance all legs win: **{acc['combined_prob']:.0%}**")
    if not acc["reached"]:
        lines.append(f"_I couldn't safely reach {target_odds}x today — this is the closest I'd trust. "
                    f"Stretching further would mean including much weaker picks._")
    lines.append("")
    for i, leg in enumerate(acc["legs"], 1):
        tag = " (estimated price)" if leg["estimated_odds"] else ""
        lines.append(f"{i}. **{leg['match']}** ({leg['league']}) — {leg['market']}, "
                    f"{leg['prob']:.0%} confidence, odds {leg['odds']:.2f}{tag}")
    lines.append("")
    lines.append("_Every extra leg adds risk, since all of them must win. "
                "This is the least risky way to reach that number, not a sure thing. "
                "Never stake money you can't afford to lose._")
    return "\n\n".join(lines)


def handle(text):
    req = nlu.parse_request(text, GEMINI_KEY)
    if req["sport"] == "basketball":
        return "Basketball predictions aren't built yet — football is what I can help with right now."
    if req["intent"] == "single_match":
        h, hs = E.resolve_team(req["home_team"] or "", teams)
        a, as_ = E.resolve_team(req["away_team"] or "", teams)
        if not h or not a:
            missing = req["home_team"] if not h else req["away_team"]
            sugg = hs or as_
            msg = f"I couldn't find a team called '{missing}' in the leagues I cover."
            if sugg:
                msg += " Did you mean: " + ", ".join(sugg) + "?"
            return msg
        return fmt_single_match(h, a)
    if req["intent"] == "accumulator":
        if not req["target_odds"]:
            return "What combined odds are you aiming for? For example: \"safest 20 odds from today's fixtures\"."
        return fmt_accumulator(req["target_odds"], req["cutoff_hour"], req["cutoff_minute"],
                               tonight="tonight" in text.lower())
    reply = ("I can predict a single match (\"Arsenal vs Chelsea\") or build a safe combination "
            "(\"safest 20 odds from today's fixtures\"). Try one of those.")
    if req.get("_error"):
        reply += f"\n\n_Debug info (remove later): {req['_error']}_"
    return reply


# ------------------------------------------------------------------- chat --
EXAMPLES = ["Arsenal vs Chelsea", "Safest 20 odds from today's fixtures", "Safest 15 odds before 6pm today"]

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("**Try asking:**")
    cols = st.columns(len(EXAMPLES))
    for c, ex in zip(cols, EXAMPLES):
        if c.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.session_state.messages.append({"role": "assistant", "content": handle(ex)})
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your prediction request..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = handle(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

if failed:
    st.caption("Some data files could not be downloaded: " + ", ".join(failed))
