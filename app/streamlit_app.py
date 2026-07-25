"""GroundedRx -- dynamic clinical dashboard (profile-centric).

Left: patient directory (real names + dates). Click a patient -> their profile opens in the
center, laid out like a modern clinical dashboard (hero card + follow-up calendar + stat tiles
+ grounded aftercare plan). Right: an interactive grounded copilot.
The evidence dashboard is a separate, navigable view with themed Altair charts + a styled table.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app/ dir for ui_helpers
from ui_helpers import C_BLUE, C_ENG, C_NAIVE, C_VIOLET, calendar_html, esc, grounded_answer, identity
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

# Load .env (cloud API key) so the real backend works without manual paste. Never committed.
_envf = os.path.join(ROOT, ".env")
if os.path.exists(_envf):
    for _line in open(_envf):
        _k, _, _v = _line.strip().partition("=")
        if _k and _v:
            os.environ.setdefault(_k, _v)


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
def generate(case_id, backend, base_url, model, api_key=None):
    corpus, cases = load_data()
    case = next(c for c in cases if c["id"] == case_id)
    retrieved = corpus.retrieve(case, top_k=8)
    source = "mock" if backend == "mock" else "real"
    t0 = time.time()
    try:
        client = make_client(backend == "mock", base_url=base_url, model=model, schema=PLAN_SCHEMA, api_key=api_key)
        res = pipeline.run_case(case, corpus, client, pipeline.FULL)
        if not res["plan"].get("danger_signs"):
            raise ValueError("empty plan from model")
    except Exception:
        # Robust for live demos: any cloud error/rate-limit/empty -> instant offline plan.
        res = pipeline.run_case(case, corpus, make_client(True, schema=PLAN_SCHEMA), pipeline.FULL)
        source = "offline-fallback" if backend != "mock" else "mock"
    lat = (time.time() - t0) * 1000
    plan = res["plan"]
    return plan, {
        "recall": round(danger_recall(plan, case, corpus) * 100),
        "halluc": has_hallucination(plan, case, corpus),
        "faith": round(citation_faithfulness(plan, corpus) * 100),
        "tokens": res["ctx_tokens"], "latency": lat, "source": source,
    }, retrieved


@st.cache_data(show_spinner=False)
def evidence(backend, base_url, model, api_key=None):
    # Evidence is a fixed benchmark from the eval harness -- always computed offline so the page
    # is instant and never fires hundreds of live cloud calls during a demo.
    corpus, cases = load_data()
    client = make_client(True, schema=PLAN_SCHEMA)
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
    return alt.Axis(title=title, titleColor="#98a2b3", labelColor="#98a2b3", grid=True,
                    gridColor="#e8eef8", domain=False, tickColor="#e8eef8", titleFontSize=11, labelFontSize=10)


def render_evidence_page(abl, front, litm):
    """Render the evaluation material as its own presentation-only workspace."""
    featured = abl[-1]
    best_recall = max(r["recall"] for r in abl)
    lowest_halluc = min(r["halluc"] for r in abl)
    best_faith = max(r["faith"] for r in abl)
    leanest_context = min(r["tokens"] for r in abl)

    def metric(icon, value, label, detail, tone):
        return (f"<article class='gr-evidence-kpi gr-evidence-kpi-{tone}'>"
                f"<div class='gr-evidence-kpi-icon'>{icon}</div><div>"
                f"<div class='gr-evidence-kpi-value'>{value}</div>"
                f"<div class='gr-evidence-kpi-label'>{label}</div>"
                f"<div class='gr-evidence-kpi-detail'>{detail}</div></div></article>")

    st.markdown(
        "<section class='gr-evidence-hero'>"
        "<div><div class='gr-evidence-eyebrow'>Evaluation workspace</div>"
        "<div class='gr-evidence-title'>Evidence, made easy to read.</div>"
        "<div class='gr-evidence-copy'>A clear, reproducible view of how the context pipeline affects "
        "safety, grounding, and efficiency across the evaluation harness.</div></div>"
        "<div class='gr-evidence-hero-badge'><span class='gr-live'></span>Mock harness · offline</div>"
        "</section>",
        unsafe_allow_html=True,
    )
    kpis = "".join([
        metric("🛟", f"{best_recall}%", "Best danger recall", "best evaluated stage", "blue"),
        metric("✓", f"{lowest_halluc}%", "Unsupported output", "lower is better", "slate"),
        metric("🔗", f"{best_faith}%", "Source faithfulness", "highest evaluated stage", "violet"),
        metric("⚡", f"{leanest_context:,}", "Context budget", "smallest tested stage", "rose"),
    ])
    st.markdown(f"<section class='gr-evidence-kpis'>{kpis}</section>", unsafe_allow_html=True)

    quality_col, insight_col = st.columns([1.45, 0.85], gap="medium")
    with quality_col:
        with st.container(border=True, key="evidence_quality"):
            st.markdown(
                "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Quality progression</div>"
                "<div class='gr-evidence-card-title'>Layer-by-layer ablation</div></div>"
                "<span class='gr-pill gr-pill-ground'>all stages</span></div>",
                unsafe_allow_html=True,
            )
            order = [r["config"] for r in abl]
            df = pd.DataFrame([
                {"config": r["config"], "metric": label, "value": r[key]}
                for r in abl
                for key, label in [("recall", "Danger recall"), ("halluc", "Unsupported output")]
            ])
            chart = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("config:N", sort=order, axis=_axis(None)),
                y=alt.Y("value:Q", axis=_axis("%"), scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "metric:N",
                    scale=alt.Scale(domain=["Danger recall", "Unsupported output"], range=[C_ENG, C_NAIVE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("config:N", title="Stage"),
                    alt.Tooltip("metric:N", title="Measure"),
                    alt.Tooltip("value:Q", title="Rate", format=".0f"),
                ],
            ).properties(height=270).configure_view(strokeWidth=0).configure(background="transparent"))
            st.altair_chart(chart, use_container_width=True)
            st.markdown(
                "<div class='gr-evidence-chart-note'>Each point adds one context-engineering layer, so the "
                "quality trade-offs remain visible rather than hidden in a single score.</div>",
                unsafe_allow_html=True,
            )

    with insight_col:
        with st.container(border=True, key="evidence_insight"):
            st.markdown(
                f"<aside class='gr-evidence-insight'><div class='gr-evidence-insight-kicker'>What this shows</div>"
                "<div class='gr-evidence-insight-title'>Better context, not simply more context.</div>"
                "<p>The dashboard keeps the reliability, source-grounding, and token-cost measures together "
                "so the trade-offs are easy to compare.</p><div class='gr-evidence-insight-stats'>"
                f"<div><strong>{featured['recall']}%</strong><span>danger recall</span></div>"
                f"<div><strong>{featured['faith']}%</strong><span>source faithfulness</span></div>"
                f"<div><strong>{featured['tokens']:,}</strong><span>context tokens</span></div>"
                "</div></aside>",
                unsafe_allow_html=True,
            )

    frontier_col, robustness_col = st.columns(2, gap="medium")
    with frontier_col:
        with st.container(border=True, key="evidence_frontier"):
            st.markdown(
                "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Efficiency</div>"
                "<div class='gr-evidence-card-title'>Efficiency frontier</div></div>"
                "<span class='gr-pill gr-pill-med'>tokens vs. recall</span></div>",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(front)
            chart = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("tokens:Q", axis=_axis("average context tokens")),
                y=alt.Y("recall:Q", axis=_axis("danger recall %"), scale=alt.Scale(domain=[0, 105])),
                color=alt.Color(
                    "pipeline:N",
                    scale=alt.Scale(domain=["Engineered", "Naive RAG"], range=[C_ENG, C_NAIVE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("pipeline:N", title="Pipeline"),
                    alt.Tooltip("tokens:Q", title="Context tokens", format=".0f"),
                    alt.Tooltip("recall:Q", title="Danger recall", format=".0f"),
                ],
            ).properties(height=255).configure_view(strokeWidth=0).configure(background="transparent"))
            st.altair_chart(chart, use_container_width=True)
            st.markdown("<div class='gr-evidence-chart-note'>Compare the accuracy reached at each tested "
                        "context budget.</div>", unsafe_allow_html=True)

    with robustness_col:
        with st.container(border=True, key="evidence_robustness"):
            st.markdown(
                "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Robustness</div>"
                "<div class='gr-evidence-card-title'>Lost in the middle</div></div>"
                "<span class='gr-pill gr-pill-life'>fact position</span></div>",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(litm)
            chart = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("pos:Q", axis=_axis("fact position (start → end)")),
                y=alt.Y("recall:Q", axis=_axis("danger recall %"), scale=alt.Scale(domain=[0, 105])),
                color=alt.Color(
                    "pipeline:N",
                    scale=alt.Scale(domain=["Engineered", "Naive"], range=[C_ENG, C_NAIVE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("pipeline:N", title="Pipeline"),
                    alt.Tooltip("pos:Q", title="Fact position", format=".1f"),
                    alt.Tooltip("recall:Q", title="Danger recall", format=".0f"),
                ],
            ).properties(height=255).configure_view(strokeWidth=0).configure(background="transparent"))
            st.altair_chart(chart, use_container_width=True)
            st.markdown("<div class='gr-evidence-chart-note'>Tests whether key safety facts remain available "
                        "when their position changes.</div>", unsafe_allow_html=True)

    with st.container(border=True, key="evidence_table"):
        st.markdown(
            "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Experiment details</div>"
            "<div class='gr-evidence-card-title'>All evaluated stages</div></div>"
            "<span class='gr-pill gr-pill-muted'>reproducible run</span></div>",
            unsafe_allow_html=True,
        )
        rows = "".join(
            f"<tr class='{'gr-abl-featured' if r['config'] == featured['config'] else ''}'>"
            f"<td>{esc(r['config'])}</td><td class='g'>{r['recall']}%</td>"
            f"<td class='{'r' if r['halluc'] else 'g'}'>{r['halluc']}%</td>"
            f"<td class='g'>{r['faith']}%</td><td>{r['tokens']:,}</td></tr>"
            for r in abl
        )
        st.markdown(
            "<table class='abl gr-evidence-table'><tr><th>stage</th><th>danger recall</th>"
            "<th>unsupported output</th><th>source faithfulness</th><th>context tokens</th></tr>"
            f"{rows}</table>",
            unsafe_allow_html=True,
        )


# =========================================================================== APP
load_css()
corpus, cases = load_data()
idx_of = {c["id"]: i for i, c in enumerate(cases)}
st.session_state.setdefault("case_id", cases[0]["id"])
st.session_state.setdefault("chat", [])
st.session_state.setdefault("backend", "mock")
st.session_state.setdefault("base_url", "http://localhost:11434/v1")
st.session_state.setdefault("model", "gemma4:e4b")
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("pending_q", None)
st.session_state.setdefault("view", "workspace")
visible_cases = cases[:3]
visible_case_ids = {c["id"] for c in visible_cases}
if st.session_state.case_id not in visible_case_ids:
    st.session_state.case_id = visible_cases[0]["id"]

be = st.session_state.backend
BASE_URL = st.session_state.base_url
MODEL = st.session_state.model
API_KEY = st.session_state.api_key or None

# ---- top navigation ----
be_txt = ({"mock": "Mock engine · Offline", "ollama": f"{MODEL} · Local",
           "cloud": f"{MODEL} · Cloud"}).get(be, "Mock engine · Offline")
with st.container(border=True, key="top_navigation"):
    brand_col, menu_col, status_col = st.columns([1.55, 1.15, 0.9], gap="small")
    with brand_col:
        st.markdown(
            "<div class='gr-brand'><div class='gr-logo'>💊</div><div>"
            "<div class='gr-brand-title'>GroundedRx</div>"
            "<div class='gr-brand-sub'>Context-engineered aftercare · grounded to source · privacy-preserving</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with menu_col:
        workspace_nav, evidence_nav = st.columns(2, gap="small")
        with workspace_nav:
            if st.button("Care workspace", key="nav_workspace", use_container_width=True,
                         type="primary" if st.session_state.view == "workspace" else "secondary"):
                st.session_state.view = "workspace"
                st.rerun()
        with evidence_nav:
            if st.button("Evidence", key="nav_evidence", use_container_width=True,
                         type="primary" if st.session_state.view == "evidence" else "secondary"):
                st.session_state.view = "evidence"
                st.rerun()
    with status_col:
        st.markdown(
            f"<div class='gr-nav-status'><span class='gr-pill gr-pill-ground'><span class='gr-live'></span>"
            f"{esc(be_txt)}</span><span class='gr-switch'></span></div>",
            unsafe_allow_html=True,
        )

if st.session_state.view == "evidence":
    abl, front, litm = evidence(be, BASE_URL, MODEL, API_KEY)
    render_evidence_page(abl, front, litm)
    st.markdown("<div class='gr-disclaimer gr-page-disclaimer'>⚠️ Illustrative / synthetic guidelines and patient "
                "identities. Not clinical advice. Assistive, human-in-the-loop; defers to the care team. Metrics "
                "are from the mock harness pending real-model runs.</div>", unsafe_allow_html=True)
    st.stop()

left, center, rightc = st.columns([0.85, 3.0, 1.3], gap="medium")

# =============================================================== LEFT: directory + engine
with left:
    with st.container(border=True, key="patient_directory"):
        st.markdown("<div class='gr-card-head'>👥 Patients</div>", unsafe_allow_html=True)
        for c in visible_cases:
            ide = identity(c["id"], c["profile"], tuple(c["meds"]), idx_of[c["id"]])
            selected = c["id"] == st.session_state.case_id
            label = (f"**{ide['name']}**  \n"
                     f"{ide['age']} · {ide['sex']}  \n"
                     "Heart-failure aftercare")
            if selected:
                label += f"  \nFollow-up · {ide['followup']:%b %d}"
            if st.button(label, key=f"pat_{c['id']}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.case_id = c["id"]
                st.session_state.chat = []
                st.rerun()

    with st.container(border=True):
        st.markdown("<div class='gr-label'>Inference backend</div>", unsafe_allow_html=True)
        opts = ["Mock (no model)", "Ollama (local)", "Cloud (OpenAI-compatible)"]
        cur = {"mock": 0, "ollama": 1, "cloud": 2}.get(be, 0)
        bk = st.selectbox("b", opts, index=cur, label_visibility="collapsed")
        if bk.startswith("Mock"):
            st.session_state.backend = "mock"
        elif bk.startswith("Ollama"):
            st.session_state.backend = "ollama"
            st.session_state.base_url = "http://localhost:11434/v1"
            st.session_state.model = st.text_input("Model", st.session_state.get("model", "gemma4:e4b"))
        else:
            st.session_state.backend = "cloud"
            _dfl_url = ("https://generativelanguage.googleapis.com/v1beta/openai/"
                        if "11434" in st.session_state.base_url else st.session_state.base_url)
            _dfl_model = ("models/gemma-4-26b-a4b-it" if st.session_state.model == "gemma4:e4b"
                          else st.session_state.model)
            st.session_state.base_url = st.text_input("Base URL", _dfl_url)
            st.session_state.model = st.text_input("Model id", _dfl_model)
            st.session_state.api_key = st.text_input("API key (or set in .env)",
                                                     st.session_state.get("api_key", ""), type="password")
            st.caption("Google AI Studio Gemma 4 (26B-A4B / 31B). Key auto-loaded from .env. "
                       "First generation is slow (thinking model); falls back to offline if unavailable.")

# ---- run pipeline for the selected patient ----
case = next(c for c in cases if c["id"] == st.session_state.case_id)
ide = identity(case["id"], case["profile"], tuple(case["meds"]), idx_of[case["id"]])
plan, _metrics, _retrieved = generate(case["id"], be, BASE_URL, MODEL, API_KEY)

# =============================================================== CENTER: profile
with center:
    # hero + calendar
    hcol, ccol = st.columns([1.85, 1], gap="medium")
    with hcol:
        with st.container(border=True):
            tags = "".join(f"<span class='gr-pill gr-pill-med'>{esc(t)}</span>" for t in ide["tags"])
            st.markdown(f"""
              <div class='gr-hero gr-profile-head'>
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

    # patient-facing aftercare plan
    with st.container(border=True, key="patient_plan"):
        st.markdown(
            "<div class='gr-card-head'>📋 Your Aftercare Plan"
            "<span class='gr-pill gr-pill-ground' style='margin-left:auto'>your guide</span></div>"
            "<div class='gr-plan-intro'>Take this one step at a time. These are the key things to "
            "watch for, take, and do before your planned follow-up.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='gr-plan-guide'><div class='gr-plan-guide-label'>Your next step</div>"
            f"<div class='gr-plan-guide-title'>Planned follow-up · {ide['followup']:%b %d}</div>"
            "<div class='gr-plan-guide-copy'>Keep this plan nearby and ask your care team if anything is unclear.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        def section(title, copy, items, kind, icon, step):
            dot = {"danger": "gr-dot-danger", "med": "gr-dot-med", "life": "gr-dot-life"}[kind]
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
            content = rows or "<div class='gr-row gr-row-empty'>No additional guidance is listed here.</div>"
            st.markdown(
                f"<section class='gr-plan-section gr-plan-section-{kind}'>"
                f"<div class='gr-plan-section-head'><span class='gr-plan-step'>{step}</span><div>"
                f"<div class='gr-plan-title'>{icon} {title}</div><div class='gr-plan-copy'>{copy}</div>"
                f"</div></div><div class='gr-plan-rows'>{content}</div></section>",
                unsafe_allow_html=True,
            )

        section("When to get help", "These changes may mean you need to contact your care team.",
                plan.get("danger_signs", []), "danger", "🚨", "1")
        section("Your medicines", "Use these instructions alongside the labels from your care team.",
                plan.get("medications", []), "med", "💊", "2")
        section("Everyday care & follow-up", "Small steps to support your recovery before your next visit.",
                plan.get("lifestyle", []), "life", "🥗", "3")

        reviewed = st.checkbox("I've reviewed this plan", key=f"reviewed_plan_{case['id']}")
        note = ("✓ Marked as reviewed in this browser. This does not change your care."
                if reviewed else "This is a personal reminder only — it does not update your care team.")
        st.markdown(f"<div class='gr-plan-note'>{note}</div>", unsafe_allow_html=True)

# =============================================================== RIGHT: copilot
with rightc:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head gr-copilot-head'>💬 Clinical AI Copilot</div>", unsafe_allow_html=True)
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

st.markdown(
    "<div class='gr-disclaimer gr-page-disclaimer'>⚠️ Illustrative / synthetic guidelines and patient identities. "
    "Not clinical advice. Assistive, human-in-the-loop; defers to the care team.</div>",
    unsafe_allow_html=True,
)
