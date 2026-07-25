"""GroundedRx demo UI (Phase 7).

Left:   patient list from eval_cases.json.
Center: the grounded aftercare plan -- danger-signs (red), meds, lifestyle -- each line with
        its citation id; a badge notes it was generated offline on a 4B model.
Bottom: headline numbers + the ablation table and the frontier / lost-in-middle PNGs.

Demo move: airplane mode on -> generate live (Mock or local Ollama) -> point at the citations
-> flip to the evidence artifacts. Works today against the Mock backend; switch the sidebar to
Ollama once gemma4:e4b is pulled -- no code change.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import citation_faithfulness, danger_recall, has_hallucination, score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA, validate_plan
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="GroundedRx", page_icon="💊", layout="wide")


@st.cache_data
def load_data():
    corpus = GuidelineCorpus.load(os.path.join(ROOT, "guidelines", "heart_failure.json"))
    import json

    with open(os.path.join(ROOT, "data", "eval_cases.json")) as f:
        cases = json.load(f)["cases"]
    return corpus, cases


corpus, cases = load_data()

# ---- Sidebar: backend + patient ------------------------------------------------------------
st.sidebar.title("GroundedRx")
st.sidebar.caption("Context-engineered aftercare for a 4B SLM")

backend = st.sidebar.radio("Backend", ["Mock (no model)", "Ollama (gemma4:e4b)"], index=0)
if backend.startswith("Ollama"):
    base_url = st.sidebar.text_input("base_url", "http://localhost:11434/v1")
    model_id = st.sidebar.text_input("model", "gemma4:e4b")
    client = make_client(False, base_url=base_url, model=model_id, schema=PLAN_SCHEMA)
    model_badge = f"{model_id} · offline"
else:
    client = make_client(True, schema=PLAN_SCHEMA)
    model_badge = "mock 4B · offline"

case_id = st.sidebar.selectbox("Patient", [c["id"] for c in cases],
                               format_func=lambda cid: f"{cid} — {next(c for c in cases if c['id']==cid)['profile'][:34]}")
case = next(c for c in cases if c["id"] == case_id)

compare = st.sidebar.checkbox("Show baseline (un-engineered) side-by-side", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Illustrative / synthetic guidelines. Not clinical advice. "
                   "Assistive, human-in-the-loop; defers to the care team.")

# ---- Header --------------------------------------------------------------------------------
st.title("💊 GroundedRx")
st.markdown("**Context engineering, not parameters, makes a 4B model safe.** "
            "Every claim is grounded to a cited guideline passage; danger-signs are never omitted.")


def render_plan(plan, title, tokens):
    st.subheader(title)
    st.caption(f"🔒 generated {model_badge}  ·  ~{tokens} context tokens")
    ok, errs = validate_plan(plan)
    st.caption(("✅ schema-valid" if ok else "⚠️ schema issues: " + "; ".join(errs[:2])))

    st.markdown("**🚨 Danger signs — seek care if you notice:**")
    if not plan.get("danger_signs"):
        st.error("No danger signs listed (safety failure).")
    for d in plan.get("danger_signs", []):
        cite = d.get("guideline_id") or "—"
        st.markdown(f"<div style='color:#b23a48'>• {d.get('text','')} "
                    f"<code>[{cite}]</code></div>", unsafe_allow_html=True)

    st.markdown("**💊 Medications:**")
    for m in plan.get("medications", []):
        cite = m.get("guideline_id") or "—"
        st.markdown(f"• **{m.get('drug','')}** — {m.get('instruction','')} `[{cite}]`")

    st.markdown("**🏃 Lifestyle:**")
    for l in plan.get("lifestyle", []):
        cite = l.get("guideline_id") or "—"
        st.markdown(f"• {l.get('text','')} `[{cite}]`")


# ---- Generate ------------------------------------------------------------------------------
st.markdown(f"### Patient {case['id']} — {case['profile']}")
st.caption(f"Prescribed: {', '.join(case['meds'])}  ·  ground-truth traps that must NOT appear: "
           f"{', '.join(case['forbidden_terms'])}")

if st.button("Generate grounded aftercare plan", type="primary"):
    with st.spinner("Running retrieve → compress → schema → verify ..."):
        full = pipeline.run_case(case, corpus, client, pipeline.FULL)
        cols = st.columns(2) if compare else [st.container()]
        with cols[0]:
            render_plan(full["plan"], "✅ GroundedRx (engineered)", full["ctx_tokens"])
            r = danger_recall(full["plan"], case, corpus)
            h = has_hallucination(full["plan"], case, corpus)
            f = citation_faithfulness(full["plan"], corpus)
            st.success(f"danger recall {r*100:.0f}%  ·  hallucination {'YES' if h else 'no'}  ·  "
                       f"citation faithfulness {f*100:.0f}%")
        if compare:
            with cols[1]:
                base = pipeline.run_case(case, corpus, client, pipeline.BASELINE)
                render_plan(base["plan"], "❌ Baseline (naive prompt)", base["ctx_tokens"])
                r = danger_recall(base["plan"], case, corpus)
                h = has_hallucination(base["plan"], case, corpus)
                f = citation_faithfulness(base["plan"], corpus)
                st.error(f"danger recall {r*100:.0f}%  ·  hallucination {'YES' if h else 'no'}  ·  "
                         f"citation faithfulness {f*100:.0f}%")

# ---- Evidence ------------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 The evidence (why context engineering wins)"):
    if st.button("Run ablation over all 30 cases"):
        with st.spinner("Scoring baseline → FULL and leave-one-out ..."):
            rows = []
            for name, cfg in pipeline.additive_configs():
                res = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
                agg = score_run(res, cases, corpus)
                rows.append({"config": name, "danger_recall": f"{agg['danger_recall']*100:.0f}%",
                             "halluc": f"{agg['hallucination_rate']*100:.0f}%",
                             "faithfulness": f"{agg['citation_faithfulness']*100:.0f}%",
                             "avg_ctx_tokens": f"{agg['avg_ctx_tokens']:.0f}"})
            st.table(rows)
    c1, c2 = st.columns(2)
    fp = os.path.join(ROOT, "slides", "frontier.png")
    lp = os.path.join(ROOT, "slides", "lost_in_middle.png")
    if os.path.exists(fp):
        c1.image(fp, caption="Efficiency frontier: near-max accuracy at a fraction of naive tokens")
    if os.path.exists(lp):
        c2.image(lp, caption="Lost-in-the-middle: engineered context is position-robust")
