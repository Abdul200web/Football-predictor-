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
.stApp { background: linear-gradient(180deg, #0b0f1a 0%, #111a2e 100%); }
.block-container { padding-top: 1.5rem; max-width: 760px; }
.bb-title { font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #ffb703, #fb5607);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
.bb-sub { color: #9aa4b8; margin-top: 0; margin-bottom: 1rem; }
.hint { color: #9aa4b8; margin: .5rem 0; }
[data-testid="stChatMessage"] { background: #161e33; border-radius: 14px; padding: .6rem .8rem; }
.stButton > button { border-radius: 12px; border: 1px solid #2a3552; }
</style>
<p class="bb-title">⚡ BoomBig</p>
<p class="bb-sub">Football predictions and tips. Ask in your own words, any league.</p>
""", unsafe_allow_html=True)

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


def _window(req):
    before = (req["cutoff_hour"], req["cutoff_minute"] or 0) if req.get("cutoff_hour") is not None else None
    after = (req["after_hour"], 0) if req.get("after_hour") is not None else None
    return before, after


def _no_matches(req):
    msg = ("I couldn't find matches with odds for that day, league and time window. "
           "The fixture feed only lists some leagues at a time, and matches that have already started are skipped.")
    if feed_info.get("error"):
        msg += f"\n\n_Fixture feed problem: {feed_info['error']}_"
    return msg


def _leg_line(c):
    tag = " (estimated price)" if c["estimated_odds"] else ""
    if c.get("basis") == "bookmaker odds":
        tag += " · from bookmaker odds, no match history"
    ko = f"{c['kickoff']} · " if c.get("kickoff") else ""
    return (f"**{c['match']}** ({c['league']}) — {ko}{c['market']}, "
            f"{c['prob']:.0%} confidence, odds {c['odds']:.2f}{tag}")


def fmt_tips(req):
    before, after = _window(req)
    cands = E.daily_candidates(hist, fixtures, day=req["date"] or "today", before=before, after=after,
                               league=req.get("league"), market=req.get("market"))
    if not cands:
        return _no_matches(req)
    n = max(1, min(int(req.get("num_picks") or 5), 15))
    top = sorted(cands, key=lambda c: -c["prob"])[:n]
    lines = [f"**Top {len(top)} tips**", ""]
    lines += [f"{i}. {_leg_line(c)}" for i, c in enumerate(top, 1)]
    lines += ["", "_Estimates, not guarantees. Never stake money you can't afford to lose._"]
    return "\n\n".join(lines)


def fmt_accumulator(req):
    target_odds = req["target_odds"]
    before, after = _window(req)
    cands = E.daily_candidates(hist, fixtures, day=req["date"] or "today", before=before, after=after,
                               league=req.get("league"), market=req.get("market"))
    if not cands:
        return _no_matches(req)
    acc = E.build_accumulator(cands, target_odds)
    if not acc["legs"]:
        return "None of the fixtures reached a safe enough confidence level to build a combination."
    lines = [f"**Closest safe combination to {target_odds}x**", "",
             f"Combined odds: **{acc['combined_odds']}x** · "
             f"Estimated chance all legs win: **{acc['combined_prob']:.0%}**"]
    if not acc["reached"]:
        lines.append(f"_I couldn't safely reach {target_odds}x — this is the closest I'd trust._")
    lines.append("")
    lines += [f"{i}. {_leg_line(c)}" for i, c in enumerate(acc["legs"], 1)]
    lines += ["", "_Every extra leg adds risk, since all of them must win. Never stake money you can't afford to lose._"]
    return "\n\n".join(lines)


def recent_context(n=6):
    msgs = st.session_state.get("messages", [])[-(n + 1):-1]
    return "\n".join(f"{'User' if m['role'] == 'user' else 'BoomBig'}: {m['content'][:300]}" for m in msgs)


def handle(text):
    ctx = recent_context()
    req = nlu.parse_request(text, GEMINI_KEY, ctx)
    if req["sport"] == "basketball":
        return "Basketball predictions aren't built yet. Football is what I can help with right now."
    intent = req["intent"]
    if intent == "single_match":
        h, hs = E.resolve_team(req["home_team"] or "", teams)
        a, as_ = E.resolve_team(req["away_team"] or "", teams)
        if not h or not a:
            missing = req["home_team"] if not h else req["away_team"]
            msg = f"I don't have match history for '{missing}', so I can't analyse that game properly."
            if hs or as_:
                msg += " Did you mean: " + ", ".join(hs or as_) + "?"
            return msg
        return fmt_single_match(h, a)
    if intent == "tips":
        return fmt_tips(req)
    if intent == "accumulator":
        if not req["target_odds"]:
            return "What combined odds are you aiming for? For example: \"safest 20 odds tonight\"."
        return fmt_accumulator(req)
    reply, err = nlu.chat_reply(text, GEMINI_KEY, ctx)
    if reply:
        return reply
    return "I couldn't process that just now. Please try again in a moment.\n\n_Debug: " + (err or req.get("_error", "")) + "_"


# ------------------------------------------------------------------- chat --
EXAMPLES = ["Arsenal vs Chelsea", "5 tips for tomorrow", "Safest 10 odds tonight", "Best over 2.5 games today"]

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask(text):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.spinner("Thinking..."):
        reply = handle(text)
    st.session_state.messages.append({"role": "assistant", "content": reply})


if not st.session_state.messages:
    st.markdown("<div class='hint'>Try one of these, or type anything:</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for k, ex in enumerate(EXAMPLES):
        if (c1 if k % 2 == 0 else c2).button(ex, use_container_width=True, key=f"ex{k}"):
            ask(ex)
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "⚡"):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask anything: a match, tips for a league, a safe combo..."):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="⚡"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            reply = handle(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

if st.session_state.messages:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if failed:
    st.caption("Some data files could not be downloaded: " + ", ".join(failed))
