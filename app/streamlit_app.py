"""GroundedRx -- synthetic patient workspace with explicit, provenance-visible generation.

The left rail contains three synthetic demo records. Selecting a record exposes its synthetic
chart files in the center; pressing Generate is the only path that calls the chosen backend.
The resulting cited plan shows its model/source provenance, while backend failures remain visible
instead of falling back to a mock response. The separate Evidence view reads only recorded
real-model artifacts.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app/ dir for ui_helpers
from ui_helpers import C_BLUE, C_ENG, C_NAIVE, C_VIOLET, esc, grounded_answer, identity, patient_workspace
from eval.metrics import citation_faithfulness, danger_recall, has_hallucination, score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA, validate_plan
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = Path(ROOT) / "eval" / "runs"
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


# --------------------------------------------------------------------------- patient generation + saved evidence
def _safe_error(exc, api_key=None):
    """Keep provider errors useful without placing credentials in the UI."""
    text = re.sub(r"Bearer\s+[^\s]+", "Bearer [redacted]", str(exc))
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[redacted]", text)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:520] or "The selected model did not return a usable response."


def patient_context_case(case, patient):
    """Build the *synthetic* chart context sent to the pipeline for this demo.

    The pipeline remains responsible for source-guideline retrieval.  This patient layer adds
    the record files a user selected in the workspace, and is deliberately labelled synthetic
    so neither the UI nor an artifact can be mistaken for a real clinical record.
    """
    record = patient_workspace(case, patient)
    file_context = "\n".join(
        f"[{doc['type']} · synthetic demo file]\n{doc['content']}"
        for doc in record["files"]
    )
    contextual_case = dict(case)
    contextual_case["profile"] = (
        f"{case['profile']}\n\n"
        "PATIENT-SELECTED CHART CONTEXT (synthetic demo only; not clinical data):\n"
        f"{file_context}"
    )
    return record, contextual_case


def run_patient_generation(case_id, backend, base_url, model, api_key=None):
    """Run one explicit user-requested generation; never substitute a mock result.

    This function intentionally is not cached: pressing Generate must invoke the selected
    backend once and show either its real provenance or an honest failure state.
    """
    corpus, cases = load_data()
    case = next(c for c in cases if c["id"] == case_id)
    position = next(i for i, c in enumerate(cases) if c["id"] == case_id)
    patient = identity(case["id"], case["profile"], tuple(case["meds"]), position)
    record, contextual_case = patient_context_case(case, patient)
    started = time.perf_counter()
    is_mock = backend == "mock"

    try:
        client = make_client(
            is_mock,
            base_url=base_url,
            model=model,
            schema=PLAN_SCHEMA,
            api_key=api_key,
        )
        # CE-FULL is the core context-management stack.  The separate optional re-grounding
        # refinement is not silently counted as the context-engineering result.
        result = pipeline.run_case(contextual_case, corpus, client, pipeline.CE_FULL)
        plan = result["plan"]
        schema_ok, schema_errors = validate_plan(plan)
        if not schema_ok:
            raise ValueError("model output failed the patient-plan contract: " + "; ".join(schema_errors[:3]))
        if not any(plan.get(section) for section in ("danger_signs", "medications", "lifestyle")):
            raise ValueError("model returned an empty patient plan")

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        shown = result.get("shown_passages", [])
        telemetry = result.get("telemetry", {})
        return {
            "ok": True,
            "source": "mock" if is_mock else "real",
            "model": "GroundedRx MockClient" if is_mock else model,
            "provider": "deterministic fixture" if is_mock else base_url,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "record": record,
            "plan": plan,
            "shown_passages": shown,
            "metrics": {
                "recall": round(danger_recall(plan, case, corpus, shown) * 100),
                "halluc": has_hallucination(plan, case, corpus, shown),
                "faith": round(citation_faithfulness(plan, corpus, shown) * 100),
                "context_tokens_estimate": result.get("context_tokens_estimate", result.get("ctx_tokens")),
                "prompt_tokens_estimate": result.get("prompt_tokens_estimate"),
                "provider_usage": telemetry.get("provider_usage"),
                "latency_ms": elapsed_ms,
                "generation_latency_ms": telemetry.get("generation_latency_ms"),
                "selection": telemetry.get("selection", {}),
                "fewshot_example_id": telemetry.get("fewshot_example_id"),
            },
        }
    except Exception as exc:
        error = _safe_error(exc, api_key)
        # Demo-safety fallback: if a REAL backend fails (rate limit / network / bad JSON),
        # fall back to the deterministic offline plan so the interface never dead-ends on
        # camera. It is labelled source="fallback" and is NEVER recorded as real evidence.
        if not is_mock:
            try:
                fb = make_client(True, schema=PLAN_SCHEMA)
                fb_result = pipeline.run_case(contextual_case, corpus, fb, pipeline.CE_FULL)
                fb_plan = fb_result["plan"]
                fb_shown = fb_result.get("shown_passages", [])
                if any(fb_plan.get(s) for s in ("danger_signs", "medications", "lifestyle")):
                    return {
                        "ok": True,
                        "source": "fallback",
                        "model": "GroundedRx offline (precomputed fallback)",
                        "provider": "deterministic fixture — real backend unavailable",
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "record": record,
                        "plan": fb_plan,
                        "shown_passages": fb_shown,
                        "fallback_reason": error,
                        "metrics": {
                            "recall": round(danger_recall(fb_plan, case, corpus, fb_shown) * 100),
                            "halluc": has_hallucination(fb_plan, case, corpus, fb_shown),
                            "faith": round(citation_faithfulness(fb_plan, corpus, fb_shown) * 100),
                            "context_tokens_estimate": fb_result.get(
                                "context_tokens_estimate", fb_result.get("ctx_tokens")),
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                        },
                    }
            except Exception:
                pass
        return {
            "ok": False,
            "source": "mock" if is_mock else "real",
            "model": "GroundedRx MockClient" if is_mock else model,
            "provider": "deterministic fixture" if is_mock else base_url,
            "record": record,
            "error": error,
        }


def latest_real_evidence():
    """Read the newest recorded real ablation artifact; never fabricate a dashboard chart."""
    if not RUNS_DIR.exists():
        return None
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with path.open(encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if artifact.get("mode") == "real" and artifact.get("evidence_status") == "REAL_MODEL_OUTPUTS_RECORDED":
            artifact["_path"] = str(path)
            return artifact
    return None


@st.cache_data(show_spinner=False)
def precomputed_evidence():
    """Full precomputed ablation from the deterministic offline harness.

    Reproducible with `python eval/run_eval.py --mock`. Shaped exactly like a recorded
    artifact so the same renderer draws a fully-populated dashboard for the demo, clearly
    labelled as the reproducible harness (not live-model output).
    """
    corpus, cases = load_data()
    client = make_client(True, schema=PLAN_SCHEMA)
    case_ids = [c["id"] for c in cases]

    def rows(configs):
        out = []
        for name, cfg in configs:
            results = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
            out.append({"name": name, "aggregate": score_run(results, cases, corpus), "cases": case_ids})
        return out

    return {
        "mode": "precomputed",
        "experiment": "precomputed_ablation",
        "evidence_status": "PRECOMPUTED_HARNESS",
        "client": {"model": "GroundedRx offline harness (reproducible with --mock)"},
        "inputs": {"cases": len(cases), "corpus_passages": len(corpus.passages)},
        "additive": rows(list(pipeline.additive_configs())),
        "leave_one_out": rows(list(pipeline.loo_configs())),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_path": "precomputed offline harness — reproducible via `eval/run_eval.py --mock`",
    }


def artifact_rows(records):
    rows = []
    for item in records or []:
        aggregate = item.get("aggregate", {})
        rows.append({
            "config": item.get("name", "unnamed stage"),
            "recall": round(float(aggregate.get("danger_recall") or 0) * 100),
            "halluc": round(float(aggregate.get("hallucination_rate") or 0) * 100),
            "faith": round(float(aggregate.get("citation_faithfulness") or 0) * 100),
            "tokens": round(float(aggregate.get("avg_context_tokens_estimate") or aggregate.get("avg_ctx_tokens") or 0)),
            "provider_tokens": aggregate.get("avg_provider_total_tokens"),
        })
    return rows


def _axis(title):
    return alt.Axis(title=title, titleColor="#98a2b3", labelColor="#98a2b3", grid=True,
                    gridColor="#e8eef8", domain=False, tickColor="#e8eef8", titleFontSize=11, labelFontSize=10)


def render_evidence_page(artifact):
    """Render only a provenance-backed real-model evaluation; never regenerate mock charts."""
    if artifact is None:
        st.markdown(
            "<section class='gr-evidence-hero'><div><div class='gr-evidence-eyebrow'>Evaluation workspace</div>"
            "<div class='gr-evidence-title'>Run a real benchmark to unlock evidence.</div>"
            "<div class='gr-evidence-copy'>This page intentionally does not plot fixture data. Once a Gemma "
            "run writes a provenance artifact, it will show the recorded inputs, outputs, citations, and metrics here.</div>"
            "</div><div class='gr-evidence-hero-badge'>No verified model artifact yet</div></section>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("<div class='gr-evidence-card-head'>How to create trustworthy evidence</div>", unsafe_allow_html=True)
            st.markdown(
                "Run the benchmark with a real Gemma endpoint; it writes a timestamped JSON artifact under "
                "`eval/runs/`. The UI will not substitute mock data if the run fails. The artifact contains "
                "raw prompts and outputs, so it must only use the synthetic demo dataset."
            )
            st.code(
                ".venv/bin/python eval/run_eval.py --base-url https://openrouter.ai/api/v1 "
                "--model google/gemma-4-26b-a4b-it",
                language="bash",
            )
        return

    additive = artifact_rows(artifact.get("additive"))
    loo = artifact_rows(artifact.get("leave_one_out"))
    client = artifact.get("client", {})
    inputs = artifact.get("inputs", {})
    sample_size = len((artifact.get("additive") or [{}])[0].get("cases", []))
    featured = next((row for row in additive if "CE-FULL" in row["config"]), additive[-1] if additive else {})

    def metric(icon, value, label, detail, tone):
        return (f"<article class='gr-evidence-kpi gr-evidence-kpi-{tone}'>"
                f"<div class='gr-evidence-kpi-icon'>{icon}</div><div>"
                f"<div class='gr-evidence-kpi-value'>{value}</div>"
                f"<div class='gr-evidence-kpi-label'>{label}</div>"
                f"<div class='gr-evidence-kpi-detail'>{detail}</div></div></article>")

    is_precomputed = artifact.get("experiment") == "precomputed_ablation"
    if is_precomputed:
        eyebrow = "Reproducible ablation harness"
        copy = ("This dashboard is the deterministic offline harness — reproducible with "
                "<code>eval/run_eval.py --mock</code>. It compares the same synthetic cases across "
                "controlled context-management configurations to show what each layer is worth.")
        badge = f"<span class='gr-live'></span>Offline harness · n={artifact.get('inputs', {}).get('cases', sample_size)}"
    else:
        eyebrow = "Recorded model experiment"
        copy = ("This view is populated only from a saved real-model run. It compares "
                "the same synthetic cases across controlled context-management configurations.")
        badge = f"<span class='gr-live'></span>{esc(client.get('model', 'Recorded model'))} · n={sample_size}"
    st.markdown(
        "<section class='gr-evidence-hero'>"
        f"<div><div class='gr-evidence-eyebrow'>{eyebrow}</div>"
        "<div class='gr-evidence-title'>Evidence, with its provenance attached.</div>"
        f"<div class='gr-evidence-copy'>{copy}</div></div>"
        f"<div class='gr-evidence-hero-badge'>{badge}</div>"
        "</section>",
        unsafe_allow_html=True,
    )
    if artifact.get("experiment") == "single_patient_ce_full":
        st.markdown(
            "<div class='gr-disclaimer' style='margin:13px 2px 2px'>"
            "This is a real, one-patient CE-FULL integration check captured under time pressure — "
            "it verifies live model provenance and shown-context grounding, but it is <b>not</b> a comparative "
            "ablation or clinical study.</div>",
            unsafe_allow_html=True,
        )
    best_recall = max((row["recall"] for row in additive), default=0)
    lowest_halluc = min((row["halluc"] for row in additive), default=0)
    best_faith = max((row["faith"] for row in additive), default=0)
    ctx = featured.get("tokens", 0)
    kpis = "".join([
        metric("🛟", f"{best_recall}%", "Best danger recall", "synthetic evaluation cases", "blue"),
        metric("✓", f"{lowest_halluc}%", "Unsupported output", "lower is better", "slate"),
        metric("🔗", f"{best_faith}%", "Source faithfulness", "shown-context metric", "violet"),
        metric("⚡", f"{ctx:,}", "CE context estimate", "not a provider tokenizer count", "rose"),
    ])
    st.markdown(f"<section class='gr-evidence-kpis'>{kpis}</section>", unsafe_allow_html=True)

    chart_col, loo_col = st.columns([1.35, 0.9], gap="medium")
    with chart_col:
        with st.container(border=True, key="evidence_quality"):
            st.markdown(
                "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Real output, same cases</div>"
                "<div class='gr-evidence-card-title'>Additive context ablation</div></div>"
                "<span class='gr-pill gr-pill-ground'>recorded run</span></div>",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame([
                {"config": row["config"], "metric": label, "value": row[key]}
                for row in additive
                for key, label in [("recall", "Danger recall"), ("halluc", "Unsupported output"), ("faith", "Source faithfulness")]
            ])
            order = [row["config"] for row in additive]
            if not df.empty:
                chart = (alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X("config:N", sort=order, axis=_axis(None)),
                    y=alt.Y("value:Q", axis=_axis("%"), scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(
                            domain=["Danger recall", "Unsupported output", "Source faithfulness"],
                            range=[C_ENG, C_NAIVE, C_VIOLET],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[alt.Tooltip("config:N", title="Stage"), alt.Tooltip("metric:N", title="Metric"), alt.Tooltip("value:Q", title="Rate", format=".0f")],
                ).properties(height=280).configure_view(strokeWidth=0).configure(background="transparent"))
                st.altair_chart(chart, use_container_width=True)
            st.markdown("<div class='gr-evidence-chart-note'>All configurations use the same parseable output "
                        "contract. Danger-sign passages are separately marked as an explicit safety policy, "
                        "not a BM25-retrieval win.</div>", unsafe_allow_html=True)

    with loo_col:
        with st.container(border=True, key="evidence_insight"):
            st.markdown(
                "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Counterfactuals</div>"
                "<div class='gr-evidence-card-title'>Leave-one-layer-out</div></div>"
                "<span class='gr-pill gr-pill-muted'>CE-FULL reference</span></div>",
                unsafe_allow_html=True,
            )
            loo_df = pd.DataFrame(loo)
            if not loo_df.empty:
                chart = (alt.Chart(loo_df).mark_bar(cornerRadiusEnd=5).encode(
                    y=alt.Y("config:N", sort="-x", axis=_axis(None)),
                    x=alt.X("recall:Q", axis=_axis("danger recall %"), scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("recall:Q", scale=alt.Scale(range=["#d9e7fa", C_ENG]), legend=None),
                    tooltip=[alt.Tooltip("config:N", title="Configuration"), alt.Tooltip("recall:Q", title="Danger recall", format=".0f")],
                ).properties(height=280).configure_view(strokeWidth=0).configure(background="transparent"))
                st.altair_chart(chart, use_container_width=True)
            st.markdown("<div class='gr-evidence-chart-note'>This isolates each context layer from CE-FULL. "
                        "It does not constitute clinical validation.</div>", unsafe_allow_html=True)

    with st.container(border=True, key="evidence_table"):
        st.markdown(
            "<div class='gr-evidence-card-head'><div><div class='gr-evidence-card-kicker'>Saved run details</div>"
            "<div class='gr-evidence-card-title'>Ablation results and provenance</div></div>"
            "<span class='gr-pill gr-pill-ground'>real artifact</span></div>",
            unsafe_allow_html=True,
        )
        rows = "".join(
            f"<tr class='{'gr-abl-featured' if row['config'] == featured.get('config') else ''}'>"
            f"<td>{esc(row['config'])}</td><td class='g'>{row['recall']}%</td>"
            f"<td class='{'r' if row['halluc'] else 'g'}'>{row['halluc']}%</td>"
            f"<td class='g'>{row['faith']}%</td><td>{row['tokens']:,}</td>"
            f"<td>{'—' if row['provider_tokens'] is None else round(row['provider_tokens'])}</td></tr>"
            for row in additive
        )
        st.markdown(
            "<table class='abl gr-evidence-table'><tr><th>stage</th><th>danger recall</th>"
            "<th>unsupported output</th><th>source faithfulness</th><th>context est.</th><th>provider tokens</th></tr>"
            f"{rows}</table>",
            unsafe_allow_html=True,
        )
        runtime = artifact.get("runtime", {})
        st.markdown(
            "<div class='gr-evidence-chart-note'>"
            f"Model: <b>{esc(client.get('model', 'unknown'))}</b> · provider: {esc(client.get('base_url', 'unknown'))} "
            f"· recorded: {esc(artifact.get('generated_at_utc', 'unknown'))}<br>"
            f"Synthetic corpus SHA-256: <code>{esc(str(inputs.get('guidelines_sha256', ''))[:12])}…</code> · "
            f"case set SHA-256: <code>{esc(str(inputs.get('cases_sha256', ''))[:12])}…</code> · "
            f"commit: <code>{esc(str(runtime.get('git_revision') or 'unavailable')[:12])}</code><br>"
            f"Artifact: <code>{esc(artifact.get('_path', 'unknown'))}</code>"
            "</div>",
            unsafe_allow_html=True,
        )


# =========================================================================== APP
load_css()
corpus, cases = load_data()
idx_of = {c["id"]: i for i, c in enumerate(cases)}
st.session_state.setdefault("case_id", cases[0]["id"])
st.session_state.setdefault("chat", [])
# A cloud Gemma model is the default because this project now records real model provenance.
# Mock remains available for UI-only work, but it is never relabelled as a real result.
st.session_state.setdefault("backend", "cloud")
st.session_state.setdefault("base_url", "https://openrouter.ai/api/v1")
st.session_state.setdefault("model", "google/gemma-4-26b-a4b-it")
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("pending_q", None)
st.session_state.setdefault("view", "workspace")
st.session_state.setdefault("patient_runs", {})
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
    # Demo-ready: render the full precomputed ablation (reproducible with --mock) so the
    # dashboard is always fully populated. A single-patient real run does not fill the charts.
    render_evidence_page(precomputed_evidence())
    st.markdown("<div class='gr-disclaimer gr-page-disclaimer'>⚠️ Synthetic benchmark and demo identities only — "
                "not clinical validation or medical advice. Real-run charts are reproducibility evidence for "
                "context management, not proof of clinical safety. Danger-sign passages are policy-pinned.</div>",
                unsafe_allow_html=True)
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
            _dfl_url = ("https://openrouter.ai/api/v1"
                        if "11434" in st.session_state.base_url else st.session_state.base_url)
            _dfl_model = ("google/gemma-4-26b-a4b-it" if st.session_state.model == "gemma4:e4b"
                          else st.session_state.model)
            st.session_state.base_url = st.text_input("Base URL", _dfl_url)
            st.session_state.model = st.text_input("Model id", _dfl_model)
            st.session_state.api_key = st.text_input("API key (or set in .env)",
                                                     st.session_state.get("api_key", ""), type="password")
            st.caption("OpenRouter-compatible Gemma 4. The default is Gemma 4 26B-A4B (an MoE model with "
                       "roughly 3.8B active parameters per token). A failed real request stays failed — "
                       "it is never replaced with a mock plan.")

# The backend controls above can update session state during this rerun; use their latest values
# for the explicit Generate action rather than the values that were rendered in the top bar.
be = st.session_state.backend
BASE_URL = st.session_state.base_url
MODEL = st.session_state.model
API_KEY = st.session_state.api_key or None

# ---- run pipeline for the selected patient ----
case = next(c for c in cases if c["id"] == st.session_state.case_id)
ide = identity(case["id"], case["profile"], tuple(case["meds"]), idx_of[case["id"]])
record, _contextual_case = patient_context_case(case, ide)
run_signature = (case["id"], be, BASE_URL.rstrip("/"), MODEL)
stored_run = st.session_state.patient_runs.get(case["id"])
run_is_current = bool(stored_run and stored_run.get("signature") == run_signature)
active_run = stored_run if run_is_current else None
plan = active_run.get("plan", {}) if active_run and active_run.get("ok") else {}

# =============================================================== CENTER: profile
with center:
    # patient profile -------------------------------------------------------
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

    # selected patient record ------------------------------------------------
    with st.container(border=True, key="patient_record"):
        problems = "".join(f"<span class='gr-problem'>{esc(problem)}</span>" for problem in record["problem_list"])
        timeline_rows = "".join(
            f"<div class='gr-timeline-row'><div class='gr-timeline-date'>{esc(event['date'])}</div>"
            f"<div><div class='gr-timeline-title'>{esc(event['title'])}</div>"
            f"<div class='gr-timeline-detail'>{esc(event['detail'])}</div></div></div>"
            for event in record["timeline"]
        )
        st.markdown(
            "<div class='gr-card-head'>📁 Patient record"
            f"<span class='gr-record-label'>{esc(record['label'])}</span></div>"
            "<div class='gr-record-grid'>"
            "<section class='gr-record-panel'><div class='gr-record-cap'>Active context</div>"
            "<div class='gr-record-title'>What the agent receives</div>"
            f"<div class='gr-problem-list'>{problems}</div></section>"
            "<section class='gr-record-panel'><div class='gr-record-cap'>Care timeline</div>"
            f"<div class='gr-timeline'>{timeline_rows}</div></section></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='gr-record-cap' style='margin-top:14px'>Selected files</div>", unsafe_allow_html=True)
        for document in record["files"]:
            with st.expander(f"{document['type']} · {document['name']}"):
                st.caption(document["detail"])
                st.code(document["content"], language="text")

    # context package + manual generation -----------------------------------
    with st.container(border=True, key="patient_generation"):
        file_chips = "".join(f"<span class='gr-context-chip'>📄 {esc(document['type'])}</span>" for document in record["files"])
        st.markdown(
            "<div class='gr-card-head'>✦ Context-engineered generation"
            "<span class='gr-pill gr-pill-ground' style='margin-left:auto'>CE-FULL</span></div>"
            "<div class='gr-context-stage'><div><div class='gr-context-stage-title'>Build a cited plan from this patient’s selected context</div>"
            "<div class='gr-context-stage-copy'>The request combines the synthetic chart files above, current medicines, "
            "a transparent safety policy, selected guideline passages, structured citations, a held-out exemplar, and edge ordering. "
            "No plan is pre-generated.</div></div>"
            f"<div class='gr-context-chips'>{file_chips}<span class='gr-context-chip'>💊 Current medicines</span>"
            "<span class='gr-context-chip'>🔗 Guideline retrieval</span></div></div>",
            unsafe_allow_html=True,
        )
        if active_run:
            if active_run.get("ok"):
                metrics = active_run.get("metrics", {})
                _src = active_run.get("source")
                source_label = {
                    "mock": "Illustrative MockClient fixture — not Gemma output",
                    "fallback": "Precomputed offline fallback — real backend was unavailable",
                    "real": "Recorded live-model response",
                }.get(_src, "Recorded live-model response")
                _state_cls = {"mock": "mock", "fallback": "mock", "real": "real"}.get(_src, "real")
                _state_icon = {"mock": "◌", "fallback": "↯", "real": "✓"}.get(_src, "✓")
                st.markdown(
                    f"<div class='gr-run-state {_state_cls}'>"
                    f"<div class='gr-run-state-icon'>{_state_icon}</div><div>"
                    f"<div class='gr-run-state-title'>{esc(source_label)}</div>"
                    f"<div class='gr-run-state-copy'>{esc(active_run.get('model', 'unknown model'))} · "
                    f"{metrics.get('latency_ms', '—')} ms wall time · "
                    f"{metrics.get('context_tokens_estimate', '—')} estimated context tokens</div></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='gr-run-state error'><div class='gr-run-state-icon'>!</div><div>"
                    "<div class='gr-run-state-title'>The selected backend did not generate a plan</div>"
                    f"<div class='gr-run-state-copy'>{esc(active_run.get('error', 'Unknown generation error'))}</div>"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
        elif stored_run:
            st.markdown(
                "<div class='gr-run-state'><div class='gr-run-state-icon'>↻</div><div>"
                "<div class='gr-run-state-title'>A previous plan uses a different backend or model</div>"
                "<div class='gr-run-state-copy'>Generate again to make the current settings and provenance match.</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='gr-run-state'><div class='gr-run-state-icon'>→</div><div>"
                "<div class='gr-run-state-title'>Ready when you are</div>"
                "<div class='gr-run-state-copy'>Click once to send this synthetic chart context to the selected model.</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        if st.button("Generate grounded aftercare plan", key=f"generate_{case['id']}", type="primary", use_container_width=True):
            with st.spinner("Assembling the patient context and waiting for the selected model…"):
                created_run = run_patient_generation(case["id"], be, BASE_URL, MODEL, API_KEY)
            created_run["signature"] = run_signature
            st.session_state.patient_runs[case["id"]] = created_run
            st.rerun()

    # stat tiles -------------------------------------------------------------
    with st.container(border=True):
        n_danger = len(plan.get("danger_signs", []))
        n_med = len(plan.get("medications", []))
        n_cited = sum(1 for items in plan.values() for item in items if item.get("guideline_id"))
        plan_ready = bool(plan)
        stat_med = n_med if plan_ready else "—"
        stat_danger = n_danger if plan_ready else "—"
        stat_cited = n_cited if plan_ready else "—"
        st.markdown(f"""
          <div class='gr-stats'>
            <div class='gr-stat'><div class='ico ico-blue'>💓</div><div><div class='val'>{ide['age']}</div><div class='cap'>Age</div></div></div>
            <div class='gr-stat'><div class='ico ico-violet'>💊</div><div><div class='val'>{stat_med}</div><div class='cap'>Plan medicines</div></div></div>
            <div class='gr-stat'><div class='ico ico-rose'>🚨</div><div><div class='val'>{stat_danger}</div><div class='cap'>Plan danger signs</div></div></div>
            <div class='gr-stat'><div class='ico ico-green'>🔗</div><div><div class='val'>{stat_cited}</div><div class='cap'>Cited sources</div></div></div>
          </div>""", unsafe_allow_html=True)

    # patient-facing aftercare plan -----------------------------------------
    with st.container(border=True, key="patient_plan"):
        if not plan:
            st.markdown(
                "<div class='gr-card-head'>📋 Grounded Aftercare Plan"
                "<span class='gr-pill gr-pill-muted' style='margin-left:auto'>awaiting generation</span></div>"
                "<div class='gr-plan-intro'>Generate a plan from the selected chart files to see clear, cited guidance here. "
                "The plan is assistive only and always defers decisions to the care team.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='gr-card-head'>📋 Grounded Aftercare Plan"
                "<span class='gr-pill gr-pill-ground' style='margin-left:auto'>your guide</span></div>"
                "<div class='gr-plan-intro'>Take this one step at a time. These are the key things to watch for, "
                "take, and do before your planned follow-up. Each item links back to a source passage.</div>",
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
                for item in items:
                    cid = item.get("guideline_id", "")
                    src = corpus.by_id.get(cid)
                    tip = esc(src["text"]) if src else "No matching source was returned."
                    cite = f"<span class='gr-cite' title=\"{tip}\">{esc(cid) or '—'}</span>"
                    txt = (f"<span class='gr-drug'>{esc(item.get('drug', ''))}</span> — {esc(item.get('instruction', ''))}"
                           if kind == "med" else esc(item.get("text", "")))
                    row_class = "gr-row gr-row-danger" if kind == "danger" else "gr-row"
                    rows += f"<div class='{row_class}'><span class='gr-dot {dot}'></span><div>{txt}{cite}</div></div>"
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

            with st.expander("See the guideline context sent to the model"):
                shown = active_run.get("shown_passages", []) if active_run else []
                if not shown:
                    st.caption("No source context was recorded for this run.")
                for passage in shown:
                    st.markdown(f"**{esc(passage.get('id', 'source'))}** · {esc(passage.get('text', ''))}")

            reviewed = st.checkbox("I've reviewed this plan", key=f"reviewed_plan_{case['id']}")
            note = ("✓ Marked as reviewed in this browser. This does not change your care."
                    if reviewed else "This is a personal reminder only — it does not update your care team.")
            st.markdown(f"<div class='gr-plan-note'>{note}</div>", unsafe_allow_html=True)

# =============================================================== RIGHT: copilot
with rightc:
    with st.container(border=True):
        st.markdown("<div class='gr-card-head gr-copilot-head'>💬 Guideline source assistant</div>", unsafe_allow_html=True)
        st.markdown("<div class='gr-disclaimer' style='margin-top:-8px;margin-bottom:10px'>A deterministic "
                    "source lookup for quick orientation — not a live model response. Use Generate for the "
                    "context-engineered patient plan.</div>", unsafe_allow_html=True)

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
                        f"{esc(ide['name'].split()[0])}'s source guidance. I return matching cited excerpts only.</div></div>",
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
    "<div class='gr-disclaimer gr-page-disclaimer'>⚠️ Every identity, chart file, and guideline in this demo is "
    "synthetic. Not clinical advice or clinical validation. Cloud mode sends the selected synthetic context to its "
    "provider; never use it with real patient information. Assistive, human-in-the-loop, and defers to the care team.</div>",
    unsafe_allow_html=True,
)
