"""GroundedRx demo UI (Phase 7) -- clinical B2B dashboard.

3-panel layout: patient directory + engine status | context graph + grounded plan | copilot +
evidence drawer. Custom CSS (app/styles.css) turns Streamlit into a standalone clinical app.
All underlying logic, --mock fallbacks and data models are untouched -- this is presentation only.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import math
import os
import re
import sys
import time

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import citation_faithfulness, danger_recall, has_hallucination, score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="GroundedRx", page_icon="💊", layout="wide")


# --------------------------------------------------------------------------- helpers
def load_css():
    with open(os.path.join(APP_DIR, "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    corpus = GuidelineCorpus.load(os.path.join(ROOT, "guidelines", "heart_failure.json"))
    with open(os.path.join(ROOT, "data", "eval_cases.json")) as f:
        cases = json.load(f)["cases"]
    return corpus, cases


_AVATAR_COLORS = ["#2563eb", "#0ea5e9", "#7c3aed", "#0891b2", "#4f46e5", "#0d9488"]


def avatar_color(key):
    return _AVATAR_COLORS[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(_AVATAR_COLORS)]


def esc(s):
    return _html.escape(str(s))


def card(html_inner, tight=False):
    cls = "gr-card gr-card-tight" if tight else "gr-card"
    st.markdown(f"<div class='{cls}'>{html_inner}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- context graph
_TYPE_STYLE = {
    "danger_sign": ("#dc2626", "#fef2f2"),
    "medication": ("#2563eb", "#eff6ff"),
    "lifestyle": ("#16a34a", "#f0fdf4"),
}


def short_code(p):
    if p["type"] == "danger_sign":
        return "DS" + p["id"].split("-")[-1]
    if p["type"] == "medication":
        return (p.get("drug") or p["id"])[:5]
    if p["type"] == "lifestyle":
        return "LF" + p["id"].split("-")[-1]
    return p["id"][:4]


def context_graph_html(case, corpus, retrieved):
    """Self-contained inline-SVG knowledge graph (no external deps, works offline).
    Center = patient; ring = the retrieved guideline chunks (danger red / med blue / lifestyle
    green). Visualizes context SELECTION -- the heart of the context-engineering story."""
    W, H, cx, cy, R = 720, 360, 360, 178, 132
    nodes = retrieved
    n = max(1, len(nodes))
    edges, circles = [], []
    for i, p in enumerate(nodes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + R * math.cos(ang), cy + R * math.sin(ang)
        stroke, fill = _TYPE_STYLE.get(p["type"], ("#64748b", "#f1f5f9"))
        edges.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.0f}' y2='{y:.0f}' stroke='#e2e8f0' stroke-width='1.5'/>")
        title = esc(p.get("text", ""))
        circles.append(
            f"<g><title>{title}</title>"
            f"<circle cx='{x:.0f}' cy='{y:.0f}' r='21' fill='{fill}' stroke='{stroke}' stroke-width='2'/>"
            f"<text x='{x:.0f}' y='{y+3:.0f}' text-anchor='middle' font-size='9' font-weight='700' fill='{stroke}'>{esc(short_code(p))}</text>"
            f"</g>"
        )
    init = "".join(w[0] for w in re.findall(r"\d+", case["profile"])[:1]) or "P"
    age = (re.findall(r"\d+", case["profile"]) or ["?"])[0]
    center = (
        f"<circle cx='{cx}' cy='{cy}' r='34' fill='#0f172a'/>"
        f"<text x='{cx}' y='{cy-2}' text-anchor='middle' font-size='13' font-weight='800' fill='#fff'>{esc(age)}y</text>"
        f"<text x='{cx}' y='{cy+12}' text-anchor='middle' font-size='8' fill='#cbd5e1'>PATIENT</text>"
    )
    pruned = sum(1 for p in corpus.passages if p["type"] == "distractor")
    legend = (
        "<div class='glegend'>"
        "<span><i style='background:#dc2626'></i>Danger sign</span>"
        "<span><i style='background:#2563eb'></i>Medication</span>"
        "<span><i style='background:#16a34a'></i>Lifestyle</span>"
        "</div>"
    )
    return f"""
    <style>
      .gwrap {{ font-family: system-ui,-apple-system,sans-serif; }}
      .ghead {{ background:#f1f5f9; border:1px solid #e2e8f0; border-radius:12px; padding:10px 14px;
                font-weight:700; color:#0f172a; font-size:15px; margin-bottom:12px; display:flex;
                align-items:center; justify-content:space-between; }}
      .gmeta {{ font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.06em; }}
      .glegend {{ display:flex; gap:16px; justify-content:center; margin-top:6px; font-size:12px; color:#475569; }}
      .glegend i {{ display:inline-block; width:9px; height:9px; border-radius:9999px; margin-right:5px; }}
      svg text {{ font-family: system-ui,-apple-system,sans-serif; }}
    </style>
    <div class='gwrap'>
      <div class='ghead'>🧠 Dynamic Clinical Context Graph
        <span class='gmeta'>{len(nodes)} chunks retrieved · {pruned} distractors pruned</span>
      </div>
      <svg viewBox='0 0 {W} {H}' width='100%' style='display:block'>
        {''.join(edges)}{center}{''.join(circles)}
      </svg>
      {legend}
    </div>
    """


# --------------------------------------------------------------------------- grounded copilot
def grounded_answer(query, corpus):
    """Offline grounded QA: retrieve the most relevant guideline chunks and answer with citations.
    (A real model would generate; this stays honest and cited without a network call.)"""
    toks = re.findall(r"[a-z0-9]+", query.lower())
    scored = []
    if corpus._bm25 is not None and toks:
        scores = corpus._bm25.get_scores(toks)
        scored = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    else:
        scored = range(len(corpus.passages))
    hits = [corpus.passages[i] for i in scored[:3] if corpus.passages[i]["type"] != "distractor"][:2]
    if not hits:
        return "I can only answer from the grounded guideline context, and I found no relevant passage. Please rephrase or defer to the care team.", []
    body = " ".join(f"{h['text']} [{h['id']}]" for h in hits)
    return f"Based on the grounded guidelines: {body} — This is assistive; always defer to the care team.", hits


# --------------------------------------------------------------------------- generation (cached)
@st.cache_data(show_spinner=False)
def generate(case_id, backend, base_url, model):
    corpus, cases = load_data()
    case = next(c for c in cases if c["id"] == case_id)
    if backend == "ollama":
        client = make_client(False, base_url=base_url, model=model, schema=PLAN_SCHEMA)
    else:
        client = make_client(True, schema=PLAN_SCHEMA)
    retrieved = corpus.retrieve(case, top_k=8)
    t0 = time.time()
    res = pipeline.run_case(case, corpus, client, pipeline.FULL)
    latency_ms = (time.time() - t0) * 1000
    plan = res["plan"]
    metrics = {
        "recall": danger_recall(plan, case, corpus),
        "halluc": has_hallucination(plan, case, corpus),
        "faith": citation_faithfulness(plan, corpus),
        "tokens": res["ctx_tokens"],
        "latency_ms": latency_ms,
    }
    return plan, metrics, retrieved


@st.cache_data(show_spinner=False)
def ablation_table(backend, base_url, model):
    corpus, cases = load_data()
    client = make_client(backend != "ollama", base_url=base_url, model=model, schema=PLAN_SCHEMA)
    rows = []
    for name, cfg in pipeline.additive_configs():
        res = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
        agg = score_run(res, cases, corpus)
        rows.append({"config": name, "danger recall": f"{agg['danger_recall']*100:.0f}%",
                     "halluc": f"{agg['hallucination_rate']*100:.0f}%",
                     "faithfulness": f"{agg['citation_faithfulness']*100:.0f}%",
                     "avg ctx tokens": f"{agg['avg_ctx_tokens']:.0f}"})
    return rows


# =========================================================================== APP
load_css()
corpus, cases = load_data()
st.session_state.setdefault("case_id", cases[0]["id"])
st.session_state.setdefault("chat", [])
st.session_state.setdefault("backend", "mock")

# ---- Top bar -------------------------------------------------------------------
be_label = "Gemma 4 E4B · Offline" if st.session_state.backend == "ollama" else "Mock engine · Offline"
st.markdown(f"""
<div class='gr-topbar'>
  <div class='gr-brand'>
    <div class='gr-logo'>💊</div>
    <div><div class='gr-brand-title'>GroundedRx</div>
    <div class='gr-brand-sub'>Context-engineered aftercare · grounded to source · privacy-preserving</div></div>
  </div>
  <div><span class='gr-pill gr-pill-ground'><span class='gr-live'></span>{esc(be_label)}</span></div>
</div>
""", unsafe_allow_html=True)

left, center, right = st.columns([1, 2.5, 1.5], gap="small")

# =============================================================== LEFT: directory + engine
with left:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head'>👥 Patient Directory</div>", unsafe_allow_html=True)
        for c in cases[:10]:
            age = (re.findall(r"\d+", c["profile"]) or ["?"])[0]
            regimen = "+".join(c["meds"][:2]) + ("…" if len(c["meds"]) > 2 else "")
            if st.button(f"{c['id']}  ·  {age}y  ·  {regimen}", key=f"pat_{c['id']}"):
                st.session_state.case_id = c["id"]

    with st.container(border=True):
        st.markdown("<div class='gr-label'>Inference backend</div>", unsafe_allow_html=True)
        bk = st.selectbox("Backend", ["Mock (no model)", "Ollama (gemma4:e4b)"],
                          index=0 if st.session_state.backend == "mock" else 1, label_visibility="collapsed")
        st.session_state.backend = "ollama" if bk.startswith("Ollama") else "mock"

        lat = st.session_state.get("last_latency")
        lat_txt = f"{lat:.0f} ms" if lat else "—"
        engine = "Gemma 4 E4B (q4)" if st.session_state.backend == "ollama" else "Mock stub"
        st.markdown(f"""
          <div class='gr-label' style='margin-top:14px'>⚡ Edge engine status</div>
          <div class='gr-engine'>
            <div class='gr-engine-row'><span class='k'>Engine</span><span class='v'>{esc(engine)}</span></div>
            <div class='gr-engine-row'><span class='k'>Mode</span><span class='v'><span class='gr-live'></span>Offline · on-device</span></div>
            <div class='gr-engine-row'><span class='k'>Last inference</span><span class='v'>{lat_txt}</span></div>
            <div class='gr-engine-row'><span class='k'>Model footprint</span><span class='v'>~3.8 GB</span></div>
          </div>
          <div class='gr-disclaimer'>Illustrative / synthetic guidelines. Not clinical advice. Assistive, human-in-the-loop.</div>
        """, unsafe_allow_html=True)

# =============================================================== CENTER: graph + plan
case = next(c for c in cases if c["id"] == st.session_state.case_id)
plan, metrics, retrieved = generate(st.session_state.case_id, st.session_state.backend,
                                    "http://localhost:11434/v1", "gemma4:e4b")
st.session_state["last_latency"] = metrics["latency_ms"]

with center:
    # Hero: context graph
    with st.container(border=True):
        components.html(context_graph_html(case, corpus, retrieved), height=430, scrolling=False)

    # Grounded aftercare plan
    def rows_html(items, kind):
        dot = {"danger": "gr-dot-danger", "med": "gr-dot-med", "life": "gr-dot-life"}[kind]
        out = []
        for it in items:
            cid = it.get("guideline_id", "")
            src = corpus.by_id.get(cid)
            tip = esc(src["text"]) if src else "uncited"
            cite = f"<span class='gr-cite' title='{tip}'>{esc(cid) or '—'}</span>"
            if kind == "med":
                txt = f"<span class='gr-drug'>{esc(it.get('drug',''))}</span> — {esc(it.get('instruction',''))}"
            else:
                txt = esc(it.get("text", ""))
            rowcls = "gr-row gr-row-danger" if kind == "danger" else "gr-row"
            out.append(f"<div class='{rowcls}'><span class='gr-dot {dot}'></span><div>{txt}{cite}</div></div>")
        return "".join(out) or "<div class='gr-row'>—</div>"

    mcls_h = "bad" if metrics["halluc"] else "good"
    chips = f"""
      <div class='gr-chips'>
        <div class='gr-chip good'><div class='v'>{metrics['recall']*100:.0f}%</div><div class='k'>Danger recall</div></div>
        <div class='gr-chip {mcls_h}'><div class='v'>{'YES' if metrics['halluc'] else 'None'}</div><div class='k'>Hallucination</div></div>
        <div class='gr-chip good'><div class='v'>{metrics['faith']*100:.0f}%</div><div class='k'>Citation faith</div></div>
        <div class='gr-chip'><div class='v'>{metrics['tokens']}</div><div class='k'>Ctx tokens</div></div>
      </div>"""

    card(f"""
      <div class='gr-card-head'>📋 Grounded Aftercare Plan
        <span class='gr-pill gr-pill-ground' style='margin-left:auto'>context-engineered</span></div>
      <div class='gr-label'>Patient</div>
      <div style='font-size:15px;font-weight:600;color:#0f172a;margin-bottom:2px'>{esc(case['profile'])}</div>
      <div style='font-size:12.5px;color:#64748b;margin-bottom:14px'>Prescribed: {esc(', '.join(case['meds']))}</div>
      {chips}
      <div class='gr-label' style='margin-top:18px'>🚨 Danger signs — seek care if you notice</div>
      {rows_html(plan.get('danger_signs', []), 'danger')}
      <div class='gr-label' style='margin-top:16px'>💊 Prescribed medications & guidance</div>
      {rows_html(plan.get('medications', []), 'med')}
      <div class='gr-label' style='margin-top:16px'>🥗 Lifestyle & follow-up</div>
      {rows_html(plan.get('lifestyle', []), 'life')}
    """)

# =============================================================== RIGHT: copilot + evidence
with right:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head'>💬 Clinical AI Copilot</div>"
                    "<div class='gr-disclaimer' style='margin-top:-6px;margin-bottom:10px'>"
                    "Answers are grounded in cited guideline passages.</div>", unsafe_allow_html=True)
        if not st.session_state.chat:
            st.markdown("<div class='gr-msg gr-msg-bot'>Ask me about this patient's danger signs, "
                        "medications, or lifestyle — I answer only from cited guidelines.</div>",
                        unsafe_allow_html=True)
        for role, text in st.session_state.chat[-6:]:
            cls = "gr-msg-user" if role == "user" else "gr-msg-bot"
            st.markdown(f"<div class='gr-msg {cls}'>{text}</div>", unsafe_allow_html=True)
        q = st.chat_input("Ask about danger signs, meds, lifestyle…")
        if q:
            ans, _ = grounded_answer(q, corpus)
            st.session_state.chat.append(("user", esc(q)))
            st.session_state.chat.append(("bot", ans))
            st.rerun()

    with st.container(border=True):
        st.markdown("<div class='gr-card-head'>📊 Evidence Drawer</div>"
                    "<div class='gr-disclaimer' style='margin-top:-6px;margin-bottom:8px'>"
                    "One-click proof the context engineering pays off.</div>", unsafe_allow_html=True)
        with st.expander("Ablation table (additive)"):
            st.table(ablation_table(st.session_state.backend, "http://localhost:11434/v1", "gemma4:e4b"))
        with st.expander("Efficiency frontier"):
            fp = os.path.join(ROOT, "slides", "frontier.png")
            st.image(fp) if os.path.exists(fp) else st.caption("Run eval/frontier.py to generate.")
        with st.expander("Lost-in-the-middle"):
            lp = os.path.join(ROOT, "slides", "lost_in_middle.png")
            st.image(lp) if os.path.exists(lp) else st.caption("Run eval/lost_in_middle.py to generate.")
