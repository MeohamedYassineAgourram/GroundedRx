"""GroundedRx -- dynamic clinical dashboard (profile-centric).

Left: patient directory (real names + dates). Click a patient -> their profile opens in the
center, laid out like a modern clinical dashboard (hero card + follow-up calendar + stat tiles
+ clinical-context graph + grounded aftercare plan). Right: an interactive grounded copilot.
Bottom: the context-engineering evidence as embedded, themed Altair charts + a styled table.

All clinical ground truth comes from data/eval_cases.json + the real pipeline; the patient
*identities* (names/dates/tags) are synthetic presentation only. Backend toggles Mock <-> Ollama.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app/ dir for ui_helpers
from ui_helpers import (C_BLUE, C_ENG, C_NAIVE, C_VIOLET, calendar_html, esc, graph_html,
                        grounded_answer, identity, ring_svg)
from eval import frontier as _frontier
from eval import lost_in_middle as _litm
from eval.metrics import citation_faithfulness, danger_recall, has_hallucination, score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="GroundedRx", page_icon="💊", layout="wide")


def load_css():
    with open(os.path.join(APP_DIR, "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    corpus = GuidelineCorpus.load(os.path.join(ROOT, "guidelines", "heart_failure.json"))
    import json
    with open(os.path.join(ROOT, "data", "eval_cases.json")) as f:
        cases = json.load(f)["cases"]
    return corpus, cases


# --------------------------------------------------------------------------- generation + evidence (cached)
@st.cache_data(show_spinner=False)
def generate(case_id, backend, base_url, model):
    corpus, cases = load_data()
    case = next(c for c in cases if c["id"] == case_id)
    client = make_client(backend != "ollama", base_url=base_url, model=model, schema=PLAN_SCHEMA)
    retrieved = corpus.retrieve(case, top_k=8)
    t0 = time.time()
    res = pipeline.run_case(case, corpus, client, pipeline.FULL)
    lat = (time.time() - t0) * 1000
    plan = res["plan"]
    return plan, {
        "recall": round(danger_recall(plan, case, corpus) * 100),
        "halluc": has_hallucination(plan, case, corpus),
        "faith": round(citation_faithfulness(plan, corpus) * 100),
        "tokens": res["ctx_tokens"], "latency": lat,
    }, retrieved


@st.cache_data(show_spinner=False)
def evidence(backend, base_url, model):
    corpus, cases = load_data()
    client = make_client(backend != "ollama", base_url=base_url, model=model, schema=PLAN_SCHEMA)
    abl = []
    for name, cfg in pipeline.additive_configs():
        agg = score_run([pipeline.run_case(c, corpus, client, cfg) for c in cases], cases, corpus)
        abl.append({"config": name, "recall": round(agg["danger_recall"] * 100),
                    "halluc": round(agg["hallucination_rate"] * 100),
                    "faith": round(agg["citation_faithfulness"] * 100),
                    "tokens": round(agg["avg_ctx_tokens"])})
    ex, er, ef = _frontier.sweep(pipeline.CE_FULL, cases, corpus, client)
    nx, nr, nf = _frontier.sweep(pipeline.BASELINE, cases, corpus, client)
    front = ([{"tokens": t, "recall": r, "pipeline": "Engineered"} for t, r in zip(ex, er)]
             + [{"tokens": t, "recall": r, "pipeline": "Naive RAG"} for t, r in zip(nx, nr)])
    needles = corpus.danger_signs
    hay = [p for p in corpus.passages if p["type"] == "distractor"]
    naive = _litm.recall_by_position(needles, hay, client, True, engineered=False)
    eng = _litm.recall_by_position(needles, hay, client, True, engineered=True)
    litm = ([{"pos": p, "recall": v, "pipeline": "Naive"} for p, v in zip(_litm.POSITIONS, naive)]
            + [{"pos": p, "recall": v, "pipeline": "Engineered"} for p, v in zip(_litm.POSITIONS, eng)])
    return abl, front, litm


def _axis(title):
    return alt.Axis(title=title, titleColor="#8a93ad", labelColor="#8a93ad", grid=True,
                    gridColor="#eef1f7", domain=False, tickColor="#eef1f7", titleFontSize=11, labelFontSize=10)


# =========================================================================== APP
load_css()
corpus, cases = load_data()
idx_of = {c["id"]: i for i, c in enumerate(cases)}
st.session_state.setdefault("case_id", cases[0]["id"])
st.session_state.setdefault("chat", [])
st.session_state.setdefault("backend", "mock")
st.session_state.setdefault("pending_q", None)

BASE_URL, MODEL = "http://localhost:11434/v1", "gemma4:e4b"
be = st.session_state.backend

# ---- top bar ----
be_txt = "Gemma 4 E4B · Offline" if be == "ollama" else "Mock engine · Offline"
st.markdown(f"""
<div class='gr-topbar'>
  <div class='gr-brand'><div class='gr-logo'>💊</div>
    <div><div class='gr-brand-title'>GroundedRx</div>
    <div class='gr-brand-sub'>Context-engineered aftercare · grounded to source · privacy-preserving</div></div></div>
  <div style='display:flex;align-items:center;gap:12px'>
    <span class='gr-pill gr-pill-ground'><span class='gr-live'></span>{esc(be_txt)}</span>
    <span class='gr-switch'></span></div>
</div>""", unsafe_allow_html=True)

left, center, rightc = st.columns([1, 2.7, 1.35], gap="medium")

# =============================================================== LEFT: directory + engine
with left:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head'>👥 Patients</div>", unsafe_allow_html=True)
        for c in cases[:12]:
            ide = identity(c["id"], c["profile"], tuple(c["meds"]), idx_of[c["id"]])
            label = f"{ide['name']}  ·  disch. {ide['discharged']:%d %b}"
            if st.button(label, key=f"pat_{c['id']}", use_container_width=True):
                st.session_state.case_id = c["id"]
                st.session_state.chat = []
                st.rerun()

    with st.container(border=True):
        st.markdown("<div class='gr-label'>Inference backend</div>", unsafe_allow_html=True)
        bk = st.selectbox("b", ["Mock (no model)", "Ollama (gemma4:e4b)"],
                          index=0 if be == "mock" else 1, label_visibility="collapsed")
        st.session_state.backend = "ollama" if bk.startswith("Ollama") else "mock"

# ---- run pipeline for the selected patient ----
case = next(c for c in cases if c["id"] == st.session_state.case_id)
ide = identity(case["id"], case["profile"], tuple(case["meds"]), idx_of[case["id"]])
plan, m, retrieved = generate(case["id"], be, BASE_URL, MODEL)

# =============================================================== CENTER: profile
with center:
    # hero + calendar
    hcol, ccol = st.columns([1.85, 1], gap="medium")
    with hcol:
        with st.container(border=True):
            tags = "".join(f"<span class='gr-pill gr-pill-med'>{esc(t)}</span>" for t in ide["tags"])
            st.markdown(f"""
              <div class='gr-hero'>
                <div class='gr-av gr-av-lg' style='background:{ide['color']}'>{esc(ide['initials'])}</div>
                <div>
                  <div class='name'>{esc(ide['name'])}</div>
                  <div class='role'>{ide['age']} yrs · {esc(ide['sex'])} · Heart-failure aftercare</div>
                  <div class='gr-tags'>{tags}</div>
                  <div class='blurb'>{esc(ide['blurb'])}</div>
                  <div class='gr-contact'><span class='gr-icobtn'>✉️</span><span class='gr-icobtn'>📞</span>
                    <span class='gr-icobtn'>📍</span></div>
                </div></div>""", unsafe_allow_html=True)
    with ccol:
        with st.container(border=True):
            st.markdown(calendar_html(ide["followup"], ide["followup"].day), unsafe_allow_html=True)

    # stat tiles
    with st.container(border=True):
        n_danger = len(plan.get("danger_signs", []))
        n_med = len(plan.get("medications", []))
        n_cited = sum(1 for s in plan.values() for it in s if it.get("guideline_id"))
        st.markdown(f"""
          <div class='gr-stats'>
            <div class='gr-stat'><div class='ico ico-blue'>💓</div><div><div class='val'>{ide['age']}</div><div class='cap'>Age</div></div></div>
            <div class='gr-stat'><div class='ico ico-violet'>💊</div><div><div class='val'>{n_med}</div><div class='cap'>Medications</div></div></div>
            <div class='gr-stat'><div class='ico ico-rose'>🚨</div><div><div class='val'>{n_danger}</div><div class='cap'>Danger signs</div></div></div>
            <div class='gr-stat'><div class='ico ico-green'>🔗</div><div><div class='val'>{n_cited}</div><div class='cap'>Cited sources</div></div></div>
          </div>""", unsafe_allow_html=True)

    # clinical context graph
    with st.container(border=True):
        components.html(graph_html(corpus, retrieved), height=380, scrolling=False)

    # grounded aftercare plan
    with st.container(border=True):
        save = round(100 * (1 - m["tokens"] / 2685))
        gauges = (f"<div class='gr-gauges'>"
                  f"<div class='gr-gauge'>{ring_svg(m['recall'], C_ENG, str(m['recall'])+'%')}<div class='lab'>Danger recall</div></div>"
                  f"<div class='gr-gauge'>{ring_svg(100, C_ENG if not m['halluc'] else C_NAIVE, '✓' if not m['halluc'] else '!')}<div class='lab'>Hallucination</div></div>"
                  f"<div class='gr-gauge'>{ring_svg(m['faith'], C_BLUE, str(m['faith'])+'%')}<div class='lab'>Citation faith</div></div>"
                  f"<div class='gr-gauge'>{ring_svg(save, C_VIOLET, str(m['tokens']))}<div class='lab'>Ctx tok · −{save}%</div></div></div>")
        st.markdown(f"<div class='gr-card-head'>📋 Grounded Aftercare Plan"
                    f"<span class='gr-pill gr-pill-ground' style='margin-left:auto'>context-engineered</span></div>"
                    + gauges, unsafe_allow_html=True)

        def section(title, items, kind, icon, icocls):
            dot = {"danger": "gr-dot-danger", "med": "gr-dot-med", "life": "gr-dot-life"}[kind]
            st.markdown(f"<div class='gr-label' style='margin-top:16px'>{icon} {title}</div>", unsafe_allow_html=True)
            rows = ""
            for it in items:
                cid = it.get("guideline_id", "")
                src = corpus.by_id.get(cid)
                tip = esc(src["text"]) if src else "uncited"
                cite = f"<span class='gr-cite' title=\"{tip}\">{esc(cid) or '—'}</span>"
                txt = (f"<span class='gr-drug'>{esc(it.get('drug',''))}</span> — {esc(it.get('instruction',''))}"
                       if kind == "med" else esc(it.get("text", "")))
                rc = "gr-row gr-row-danger" if kind == "danger" else "gr-row"
                rows += f"<div class='{rc}'><span class='gr-dot {dot}'></span><div>{txt}{cite}</div></div>"
            st.markdown(rows or "<div class='gr-row'>—</div>", unsafe_allow_html=True)

        section("Danger signs — seek care if you notice", plan.get("danger_signs", []), "danger", "🚨", "ico-rose")
        section("Medications & guidance", plan.get("medications", []), "med", "💊", "ico-blue")
        section("Lifestyle & follow-up", plan.get("lifestyle", []), "life", "🥗", "ico-green")

# =============================================================== RIGHT: copilot
with rightc:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head'>💬 Clinical AI Copilot</div>", unsafe_allow_html=True)
        st.markdown("<div class='gr-disclaimer' style='margin-top:-8px;margin-bottom:10px'>Grounded in cited "
                    "guideline passages · answers offline.</div>", unsafe_allow_html=True)

        # suggested question chips
        med0 = case["meds"][0] if case["meds"] else "my medication"
        suggestions = ["What danger signs need urgent care?", f"How should I take {med0}?",
                       "What lifestyle changes help?"]
        for i, sug in enumerate(suggestions):
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_q = sug

        # render chat
        if not st.session_state.chat:
            st.markdown(f"<div class='gr-botrow'><div class='gr-botav'>✦</div>"
                        f"<div class='gr-msg gr-msg-bot' style='margin:0'>Hi — ask me anything about "
                        f"{esc(ide['name'].split()[0])}'s plan. I only answer from cited guidelines.</div></div>",
                        unsafe_allow_html=True)
        for role, text in st.session_state.chat[-8:]:
            if role == "user":
                st.markdown(f"<div class='gr-msg gr-msg-user'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='gr-botrow'><div class='gr-botav'>✦</div>"
                            f"<div class='gr-msg gr-msg-bot' style='margin:0'>{text}</div></div>", unsafe_allow_html=True)

        typed = st.chat_input("Ask about danger signs, meds, lifestyle…")
        q = typed or st.session_state.pending_q
        if q:
            st.session_state.pending_q = None
            st.session_state.chat.append(("user", esc(q)))
            st.session_state.chat.append(("bot", grounded_answer(q, corpus)))
            st.rerun()

# =============================================================== BOTTOM: evidence
abl, front, litm = evidence(be, BASE_URL, MODEL)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("<div class='gr-evhead'>📊 Evidence — why context engineering wins</div>"
                "<div class='gr-disclaimer' style='margin-bottom:6px'>Reproducible from our own eval harness. "
                "The win comes from managing the context, not from a filter.</div>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3, gap="medium")

    with e1:
        st.markdown("<div class='gr-label'>Layer ablation (additive)</div>", unsafe_allow_html=True)
        order = [r["config"] for r in abl]
        df = pd.DataFrame([{"config": r["config"], "metric": k2, "value": r[k1]}
                           for r in abl for k1, k2 in [("recall", "Danger recall"), ("halluc", "Hallucination")]])
        ch = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("config:N", sort=order, axis=_axis(None)),
            y=alt.Y("value:Q", axis=_axis("%"), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["Danger recall", "Hallucination"],
                            range=[C_ENG, C_NAIVE]), legend=alt.Legend(title=None, orient="top")))
            .properties(height=210).configure_view(strokeWidth=0).configure(background="transparent"))
        st.altair_chart(ch, use_container_width=True)
        rows = "".join(f"<tr><td>{esc(r['config'])}</td><td class='g'>{r['recall']}%</td>"
                       f"<td class='{'r' if r['halluc'] else 'g'}'>{r['halluc']}%</td>"
                       f"<td class='g'>{r['faith']}%</td><td>{r['tokens']}</td></tr>" for r in abl)
        st.markdown(f"<table class='abl'><tr><th>config</th><th>recall</th><th>halluc</th><th>faith</th><th>ctx</th></tr>{rows}</table>",
                    unsafe_allow_html=True)

    with e2:
        st.markdown("<div class='gr-label'>Efficiency frontier</div>", unsafe_allow_html=True)
        df = pd.DataFrame(front)
        ch = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("tokens:Q", axis=_axis("avg context tokens")),
            y=alt.Y("recall:Q", axis=_axis("danger recall %"), scale=alt.Scale(domain=[0, 105])),
            color=alt.Color("pipeline:N", scale=alt.Scale(domain=["Engineered", "Naive RAG"],
                            range=[C_ENG, C_NAIVE]), legend=alt.Legend(title=None, orient="top")))
            .properties(height=210).configure_view(strokeWidth=0).configure(background="transparent"))
        st.altair_chart(ch, use_container_width=True)
        st.markdown("<div class='gr-disclaimer'>Engineered context reaches ~100% recall at a fraction of "
                    "naive RAG's tokens.</div>", unsafe_allow_html=True)

    with e3:
        st.markdown("<div class='gr-label'>Lost-in-the-middle (robustness)</div>", unsafe_allow_html=True)
        df = pd.DataFrame(litm)
        ch = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("pos:Q", axis=_axis("position of fact (0=start,1=end)")),
            y=alt.Y("recall:Q", axis=_axis("danger recall %"), scale=alt.Scale(domain=[0, 105])),
            color=alt.Color("pipeline:N", scale=alt.Scale(domain=["Engineered", "Naive"],
                            range=[C_ENG, C_NAIVE]), legend=alt.Legend(title=None, orient="top")))
            .properties(height=210).configure_view(strokeWidth=0).configure(background="transparent"))
        st.altair_chart(ch, use_container_width=True)
        st.markdown("<div class='gr-disclaimer'>Naive recall sags when the key fact sits mid-context; "
                    "engineered ordering keeps it flat.</div>", unsafe_allow_html=True)

st.markdown("<div class='gr-disclaimer' style='text-align:center;margin-top:10px'>⚠️ Illustrative / synthetic "
            "guidelines and patient identities. Not clinical advice. Assistive, human-in-the-loop; defers to the "
            "care team. Metrics from the mock harness pending real-model runs.</div>", unsafe_allow_html=True)
