"""
dashboard.py - ML-XIDS live SOC dashboard (Streamlit).

    streamlit run dashboard.py

Reads the files written by live_monitor.py (outputs/live/*). It never runs the model itself.
"""
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import config

DEMO_DIR = os.path.join(config.BASE_DIR, "demo_data", "live")
LOCAL_STATS = config.LIVE_STATS_JSON
LOCAL_ALERTS = config.LIVE_ALERTS_JSONL

# Cloud mirror: sync_to_cloud.py (running on the laptop next to the monitor) pushes feed.json to this branch.
FEED_REPO = os.environ.get("FEED_REPO", "Heldon-Leo/ML_XIDS_project")
FEED_BRANCH = os.environ.get("FEED_BRANCH", "live-data")
FEED_PATH = "feed.json"

st.set_page_config(page_title="ML-XIDS Live Monitor", page_icon="🛡️", layout="wide")

CYAN, RED, AMBER, GREEN, GREY = "#22d3ee", "#f87171", "#fbbf24", "#34d399", "#94a3b8"
st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; max-width: 1500px;}
.kpi {background:#111a2e; border:1px solid #1e2b45; border-radius:10px; padding:12px 16px;}
.kpi .l {color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {font-size:1.7rem; font-weight:700; line-height:1.2;}
.pill {display:inline-block; padding:2px 10px; border-radius:99px; font-size:.78rem; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- optional password (Streamlit Cloud: Settings > Secrets)
def _password_ok() -> bool:
    try:
        expected = st.secrets.get("DASH_PASSWORD")
    except Exception:          # no secrets file locally
        expected = None
    if not expected:
        return True
    if st.session_state.get("_auth"):
        return True
    pw = st.text_input("Password", type="password")
    if pw:
        if pw == expected:
            st.session_state["_auth"] = True
            st.rerun()
        st.error("Wrong password.")
    return False


if not _password_ok():
    st.stop()


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _read_jsonl(path: str, mtime: float, tail: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows[-tail:]


def load_events(path: str, tail: int = 2000) -> list:
    if not os.path.exists(path):
        return []
    return _read_jsonl(path, os.path.getmtime(path), tail)


def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:          # no secrets file locally
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_feed(url: str, token: str | None, bucket: int):
    """Download feed.json (stats + recent alert events) pushed by the laptop. Returns (stats, events) or None."""
    import urllib.request
    try:
        headers = {"User-Agent": "ml-xids-dashboard"}
        if token:                                  # GitHub API: no CDN delay (needs a read-only token in Secrets)
            headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw"})
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            feed = json.loads(r.read().decode("utf-8"))
        return feed.get("stats") or {}, feed.get("events") or []
    except Exception:
        return None


def _feed_url(token) -> str:
    override = os.environ.get("FEED_URL") or _secret("FEED_URL")
    if override:
        return override
    if token:
        return f"https://api.github.com/repos/{FEED_REPO}/contents/{FEED_PATH}?ref={FEED_BRANCH}"
    return f"https://raw.githubusercontent.com/{FEED_REPO}/{FEED_BRANCH}/{FEED_PATH}"      # CDN: up to ~5 min old


def get_data():
    """Returns (mode, stats, events). mode: LOCAL (monitor on this PC) | MIRROR (pushed from the laptop) | DEMO (saved)."""
    if os.path.exists(LOCAL_STATS):
        try:
            with open(LOCAL_STATS, encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, ValueError):
            stats = {}
        return "LOCAL", stats, load_events(LOCAL_ALERTS)
    token = _secret("GITHUB_TOKEN")
    got = fetch_feed(_feed_url(token), token, 0)
    if got and got[0]:
        return "MIRROR", got[0], got[1]
    demo_stats = os.path.join(DEMO_DIR, "live_stats.json")
    if os.path.exists(demo_stats):
        with open(demo_stats, encoding="utf-8") as f:
            return "DEMO", json.load(f), load_events(os.path.join(DEMO_DIR, "alerts.jsonl"))
    return "NONE", {}, []


def kpi(col, label, value, color="#e5edf7"):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ ML-XIDS")
    st.caption("Module 4 · ML network-flow detection")
    refresh = st.slider("Refresh every (s)", 2, 30, 5)
    show = st.multiselect("Status", ["ALERT", "REVIEW"], default=["ALERT", "REVIEW"])
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    st.divider()
    st.caption("Data: " + {"LOCAL": config.LIVE_OUTPUT_DIR, "MIRROR": "laptop feed (GitHub, branch live-data)",
                           "DEMO": "saved demo run (demo_data/live)"}.get(get_data()[0], "none"))
    st.caption("Attack classes: ALERT = confidence ≥ threshold; REVIEW = attack predicted below it.")


def _age_text(stats: dict):
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(str(stats.get("updated_utc")).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        sec = (datetime.now(timezone.utc) - t).total_seconds()
    except ValueError:
        return None, None
    return sec, (f"{sec:.0f} s" if sec < 120 else f"{sec / 60:.0f} min" if sec < 7200 else f"{sec / 3600:.1f} h")


def render():
    mode, stats, events = get_data()
    st.markdown({"LOCAL": "### Live detection feed", "MIRROR": "### Live detection feed (cloud mirror)"}.get(mode, "### Detection feed (saved demo run)"))
    if mode == "MIRROR":
        sec, txt = _age_text(stats)
        if sec is not None and sec > 600:
            st.warning(f"The laptop has not sent new results for {txt}. The monitor or sync_to_cloud.py is probably stopped; "
                       "the numbers below are the last ones received.")
        else:
            st.info("Cloud mirror: the ML-XIDS monitor (XGBoost + TreeSHAP) runs on the author's laptop and sends its "
                    "latest results here about once a minute." + (f" Last update {txt} ago." if txt else ""))
    elif mode == "DEMO":
        snap = stats.get("snapshot", {})
        st.info("Cloud demo: this page replays a saved run of the ML-XIDS live monitor (XGBoost + TreeSHAP). "
                "No live traffic is processed here."
                + (f" Snapshot: {snap.get('kept', '?')} of {snap.get('source_events', '?')} alert events, "
                   f"saved {snap.get('created_utc', '?')}." if snap else ""))
    if not stats:
        st.info("No data yet. Start the monitor: `python live_monitor.py` "
                "(demo traffic: `python simulate_traffic.py`). For the cloud mirror also run `python sync_to_cloud.py`.")
        return

    c = st.columns(6)
    kpi(c[0], "Flows processed", f"{stats.get('processed', 0):,}")
    kpi(c[1], "Alerts", f"{stats.get('alerts', 0):,}", RED)
    kpi(c[2], "Review", f"{stats.get('reviews', 0):,}", AMBER)
    kpi(c[3], "Benign", f"{stats.get('benign', 0):,}", GREEN)
    kpi(c[4], "Rejected rows", f"{stats.get('rejected', 0):,}", GREY)
    kpi(c[5], "Last batch", f"{stats.get('last_batch_latency_ms', 0):.0f} ms", CYAN)
    st.caption(f"Monitor last updated {stats.get('updated_utc')} (UTC) · "
               f"{stats.get('duplicates', 0)} duplicate rows skipped")
    st.write("")

    df = pd.DataFrame([{
        "time": e["timestamp_utc"], "status": e["status"], "class": e["predicted_class"],
        "confidence": e["confidence"], "src_ip": e.get("src_ip"), "dst_ip": e.get("dst_ip"),
        "dst_port": e.get("dst_port"), "mitre": (e.get("mitre") or {}).get("technique_id"),
        "event_id": e["event_id"],
    } for e in events])
    if df.empty:
        st.success("No alerts so far — all processed flows were classified benign.")
        return
    df = df[df["status"].isin(show) & (df["confidence"] >= min_conf)].iloc[::-1].reset_index(drop=True)

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown("**Attack classes**")
        by = df["class"].value_counts().reset_index()
        by.columns = ["class", "count"]
        st.altair_chart(
            alt.Chart(by).mark_bar(color=RED, cornerRadiusEnd=3).encode(
                y=alt.Y("class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("count:Q", title=None),
                tooltip=["class", "count"]).properties(height=max(120, 28 * len(by))),
            width="stretch")
    with right:
        st.markdown("**Alerts (newest first) — select a row for the explanation**")
        sel = st.dataframe(
            df.drop(columns=["event_id"]), hide_index=True, width="stretch", height=300,
            on_select="rerun", selection_mode="single-row", key="alert_table",
            column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")})
        rows = sel.selection.rows if sel and sel.selection else []

    if df.empty:
        return
    idx = rows[0] if rows else 0
    ev = next(e for e in events if e["event_id"] == df.loc[idx, "event_id"])
    st.divider()
    st.markdown(f"#### {ev['predicted_class']} — {ev['confidence']:.1%} "
                f"<span class='pill' style='background:{RED if ev['status']=='ALERT' else AMBER}22;"
                f"color:{RED if ev['status']=='ALERT' else AMBER}'>{ev['status']} · {ev['severity']}</span>",
                unsafe_allow_html=True)
    a, b = st.columns([1, 1.5])
    with a:
        m = ev.get("mitre") or {}
        st.write(f"**Flow:** {ev.get('src_ip')}:{ev.get('src_port')} → {ev.get('dst_ip')}:{ev.get('dst_port')} (proto {ev.get('protocol')})")
        st.write(f"**MITRE ATT&CK (best-effort):** {m.get('technique_id')} {m.get('technique_name') or ''}")
        st.write(f"**Captured:** {ev.get('capture_timestamp')}")
        dq = ev.get("data_quality", {})
        st.write(f"**Data quality:** {dq.get('imputed_features', 0)} imputed, {dq.get('out_of_range_features', 0)} features outside training range")
        if ev.get("ground_truth"):
            st.write(f"**Ground truth (replay):** {ev['ground_truth']}")
        st.write("**Top class probabilities**")
        st.altair_chart(
            alt.Chart(pd.DataFrame({"class": list(ev["class_probabilities"]), "p": list(ev["class_probabilities"].values())}))
            .mark_bar(color=CYAN).encode(y=alt.Y("class:N", sort="-x", title=None),
                                         x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), title=None)).properties(height=90),
            width="stretch")
    with b:
        st.write("**Why this prediction? (TreeSHAP)**")
        sh = pd.DataFrame(ev["shap_evidence"])
        if sh.empty:
            st.caption("No SHAP evidence stored for this event.")
        else:
            sh["label"] = sh["feature"] + " = " + sh["value"].map(lambda v: "n/a" if v is None else f"{v:.4g}")
            st.altair_chart(
                alt.Chart(sh).mark_bar(cornerRadius=2).encode(
                    y=alt.Y("label:N", sort=alt.EncodingSortField("shap_value", order="descending"), title=None, axis=alt.Axis(labelLimit=280)),
                    x=alt.X("shap_value:Q", title="SHAP value (log-odds) → toward / away from predicted class"),
                    color=alt.Color("effect:N", scale=alt.Scale(domain=["supports", "opposes"], range=[RED, CYAN]), legend=alt.Legend(title=None)),
                    tooltip=["feature", "value", "shap_value", "effect"]).properties(height=30 * len(sh) + 40),
                width="stretch")
        st.caption(ev.get("explanation") or "")
    with st.expander("Raw event JSON (the contract sent to the ontology module)"):
        st.json(ev)


if hasattr(st, "fragment"):
    st.fragment(run_every=refresh)(render)()
else:  # very old Streamlit
    render()
