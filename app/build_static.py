"""Build a browser-only GroundedRx snapshot from a recorded *real* model run.

The exporter never calls a model.  It intentionally refuses mock artifacts so a standalone
page cannot turn a deterministic fixture into apparent Gemma evidence.  The saved artifact
contains prompts and raw responses; use only the repository's synthetic benchmark data.

    python app/build_static.py --artifact eval/runs/<real-ablation>.json
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ui_helpers import identity, patient_workspace


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_real_artifact(path: str) -> tuple[Path, dict]:
    artifact_path = Path(path).expanduser().resolve()
    try:
        with artifact_path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read artifact: {exc}") from exc
    if artifact.get("mode") != "real" or artifact.get("evidence_status") != "REAL_MODEL_OUTPUTS_RECORDED":
        raise SystemExit(
            "Refusing to export: --artifact must be a saved real-model provenance artifact, "
            "not a mock fixture."
        )
    if not artifact.get("additive"):
        raise SystemExit("Refusing to export: artifact has no additive-ablation records.")
    return artifact_path, artifact


def config_rows(records):
    rows = []
    for record in records or []:
        aggregate = record.get("aggregate", {})
        rows.append({
            "name": record.get("name", "unnamed stage"),
            "recall": round(float(aggregate.get("danger_recall") or 0) * 100),
            "unsupported": round(float(aggregate.get("hallucination_rate") or 0) * 100),
            "faith": round(float(aggregate.get("citation_faithfulness") or 0) * 100),
            "context_estimate": round(float(aggregate.get("avg_context_tokens_estimate") or aggregate.get("avg_ctx_tokens") or 0)),
            "provider_tokens": aggregate.get("avg_provider_total_tokens"),
        })
    return rows


def select_ce_record(artifact):
    additive = artifact.get("additive", [])
    for record in additive:
        if "CE-FULL" in record.get("name", "") and "reground" not in record.get("name", "").lower():
            return record
    return additive[-1]


def snapshot_payload(artifact_path: Path, artifact: dict) -> dict:
    ce_record = select_ce_record(artifact)
    patients = {}
    order = []
    for index, entry in enumerate(ce_record.get("cases", [])[:3]):
        case = entry.get("case_input", {})
        # A live-record artifact includes the expanded prompt profile plus the original
        # presentation profile. Keep the standalone card readable while preserving the exact
        # expanded profile in the artifact itself.
        presentation_case = dict(case)
        presentation_case["profile"] = case.get("presentation_profile", case.get("profile", ""))
        case_id = case.get("id", f"case-{index + 1}")
        display = identity(case_id, presentation_case.get("profile", ""), tuple(case.get("meds", [])), index)
        record = patient_workspace(presentation_case, display)
        patients[case_id] = {
            "identity": {
                "name": display["name"], "age": display["age"], "sex": display["sex"],
                "followup": display["followup"].strftime("%b %d"), "initials": display["initials"],
                "color": display["color"], "tags": display["tags"], "blurb": display["blurb"],
            },
            "profile": presentation_case.get("profile", ""),
            "meds": case.get("meds", []),
            "record": record,
            "plan": entry.get("parsed_plan", {}),
            "sources": entry.get("context", {}).get("shown_passages", []),
            "metrics": entry.get("metrics", {}),
        }
        order.append(case_id)
    client = artifact.get("client", {})
    return {
        "patients": patients,
        "order": order,
        "additive": config_rows(artifact.get("additive")),
        "leave_one_out": config_rows(artifact.get("leave_one_out")),
        "provenance": {
            "model": client.get("model", "unknown"),
            "provider": client.get("base_url", "unknown"),
            "recorded_at": artifact.get("generated_at_utc", "unknown"),
            "sample_size": len(ce_record.get("cases", [])),
            "artifact_name": artifact_path.name,
            "artifact_sha256": sha256_file(artifact_path),
            "guidelines_sha256": artifact.get("inputs", {}).get("guidelines_sha256", ""),
            "cases_sha256": artifact.get("inputs", {}).get("cases_sha256", ""),
            "git_revision": artifact.get("runtime", {}).get("git_revision") or "unavailable",
        },
    }


def script_json(value) -> str:
    """Make a JSON script payload safe even if a model ever emitted a closing script tag."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build(artifact_path: str, output: str | None = None) -> Path:
    path, artifact = load_real_artifact(artifact_path)
    payload = snapshot_payload(path, artifact)
    css = (ROOT / "app" / "styles.css").read_text(encoding="utf-8")
    page = TEMPLATE.replace("/*CSS*/", css).replace("__PAYLOAD__", script_json(payload))
    target = Path(output).expanduser().resolve() if output else ROOT / "GroundedRx.html"
    target.write_text(page, encoding="utf-8")
    return target


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GroundedRx — recorded evidence snapshot</title>
<style>
/*CSS*/
body { margin:0; padding:18px 24px 40px; }
.static-grid { display:grid; grid-template-columns:.9fr 2.4fr 1.2fr; gap:16px; align-items:start; }
.static-panel { min-width:0; }
.patient-button { width:100%; margin:0 0 11px; padding:14px 15px; border:0; border-radius:17px; background:#fff; box-shadow:0 12px 24px -20px rgba(45,68,99,.48); color:#687385; cursor:pointer; font:500 12px/1.55 var(--font); text-align:left; }
.patient-button:hover,.patient-button.active { background:#edf4ff; box-shadow:0 15px 27px -20px rgba(30,111,245,.44); }
.patient-button strong { display:block; color:var(--ink); font-size:15px; font-weight:780; }
.patient-button small { display:block; color:var(--body); margin-top:2px; }
.static-files,.static-sources { display:grid; gap:8px; margin-top:12px; }
.static-file,.static-source { padding:10px 11px; border-radius:12px; background:#f7f9fc; box-shadow:0 9px 18px -18px rgba(45,68,99,.35); color:var(--body); font-size:12px; line-height:1.45; }
.static-file b,.static-source b { color:var(--ink); }
.static-source code { color:#34699f; font-size:10.5px; }
.static-search { width:100%; box-sizing:border-box; margin-top:9px; padding:10px 11px; border:0; border-radius:11px; background:#f7f9fc; color:var(--ink); font:12px var(--font); outline:none; }
.static-search:focus { box-shadow:0 0 0 3px rgba(30,111,245,.17); }
.static-answer { margin-top:10px; color:var(--body); font-size:12px; line-height:1.55; }
.evidence-grid-static { display:grid; grid-template-columns:1.25fr .75fr; gap:16px; margin-top:16px; }
.metric-bar { display:grid; grid-template-columns:minmax(115px,.9fr) minmax(70px,2fr) 42px; gap:10px; align-items:center; margin:11px 0; }
.metric-bar-label { overflow:hidden; color:var(--body); font-size:11px; text-overflow:ellipsis; white-space:nowrap; }
.metric-track { overflow:hidden; height:9px; border-radius:999px; background:#e9eef7; }
.metric-fill { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#61a0fb,#1e6ff5); }
.metric-value { color:var(--ink); font-size:12px; font-weight:750; text-align:right; }
.evidence-table-static { width:100%; border-collapse:collapse; font-size:12px; }
.evidence-table-static th,.evidence-table-static td { padding:10px 8px; border-bottom:1px solid #eef1f6; text-align:left; color:var(--body); }
.evidence-table-static th { color:var(--muted); font-size:10px; letter-spacing:.05em; text-transform:uppercase; }
.evidence-table-static td:first-child { color:var(--ink); font-weight:720; }
.provenance-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin-top:12px; }
.provenance-grid div { padding:10px 11px; border-radius:12px; background:#f7f9fc; color:var(--body); font-size:11px; line-height:1.45; }
.provenance-grid b { display:block; color:var(--ink); font-size:11px; }
@media (max-width:1100px) { .static-grid,.evidence-grid-static { grid-template-columns:1fr; } }
</style></head><body>
<header class="gr-topbar">
  <div class="gr-brand"><div class="gr-logo">💊</div><div><div class="gr-brand-title">GroundedRx</div>
    <div class="gr-brand-sub">Synthetic benchmark · context-engineered aftercare · provenance attached</div></div></div>
  <nav class="gr-nav-tabs" aria-label="Primary navigation">
    <button type="button" class="gr-nav-link active" data-view="care" onclick="showView('care')">Care workspace</button>
    <button type="button" class="gr-nav-link" data-view="evidence" onclick="showView('evidence')">Evidence</button>
  </nav>
  <span class="gr-pill gr-pill-ground">Static evidence snapshot · no live inference</span>
</header>
<main id="care-view">
  <div class="static-grid">
    <aside class="static-panel"><section class="gr-card"><div class="gr-card-head">👥 Patients</div><div id="patients"></div></section>
      <section class="gr-card"><div class="gr-label">Snapshot status</div><div class="gr-engine-row"><span class="k">Model</span><span class="v" id="model-label"></span></div>
      <div class="gr-engine-row"><span class="k">Mode</span><span class="v">Recorded run · no live call</span></div>
      <div class="gr-disclaimer">All identities, files, and guidelines in this page are synthetic. Do not use cloud inference with real patient information.</div></section></aside>
    <section class="static-panel"><article class="gr-card" id="patient-workspace"></article></section>
    <aside class="static-panel"><section class="gr-card gr-copilot-card"><div class="gr-card-head">💬 Guideline lookup</div>
      <div class="gr-disclaimer">Static source search only — not model inference.</div><input class="static-search" id="search" placeholder="Find a cited guideline excerpt" oninput="searchSources()"><div id="search-answer" class="static-answer"></div></section></aside>
  </div>
</main>
<main id="evidence-view" hidden>
  <section class="gr-evidence-hero"><div><div class="gr-evidence-eyebrow">Recorded model experiment</div><div class="gr-evidence-title">Evidence, with provenance attached.</div>
    <div class="gr-evidence-copy">These figures are read from a saved real-model artifact. Read the model and sample size beside them: a single CE row is a live integration check, not a comparison or clinical validation.</div></div>
    <div class="gr-evidence-hero-badge" id="evidence-badge"></div></section>
  <div class="evidence-grid-static"><section class="gr-card"><div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Additive ablation</div><div class="gr-evidence-card-title">Danger recall by context stage</div></div><span class="gr-pill gr-pill-ground">recorded output</span></div><div id="bars"></div></section>
    <aside class="gr-card"><div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Run provenance</div><div class="gr-evidence-card-title">Reproduce this snapshot</div></div></div><div class="provenance-grid" id="provenance"></div></aside></div>
  <section class="gr-card"><div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Ablation details</div><div class="gr-evidence-card-title">Recorded real-model metrics</div></div><span class="gr-pill gr-pill-muted">synthetic n only</span></div><div id="table"></div></section>
  <p class="gr-disclaimer gr-page-disclaimer">⚠️ Danger-sign passages are explicitly safety-pinned and reported separately from BM25 retrieval. Automated grounding metrics are conservative checks against shown synthetic context; they do not replace clinician review or clinical validation.</p>
</main>
<script>
const DATA = __PAYLOAD__;
let activeId = DATA.order[0];
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const sourceFor = id => (DATA.patients[activeId].sources || []).find(source => source.id === id);
function citation(item){ const id = item.guideline_id || ''; const source = sourceFor(id); return `<span class="gr-cite" title="${esc(source?.text || 'No matching shown source')}">${esc(id || '—')}</span>`; }
function planRows(items, kind){
  const dot = {danger:'gr-dot-danger',med:'gr-dot-med',life:'gr-dot-life'}[kind];
  if(!items?.length) return '<div class="gr-row gr-row-empty">No item returned for this section.</div>';
  return items.map(item => { const text = kind === 'med' ? `<span class="gr-drug">${esc(item.drug)}</span> — ${esc(item.instruction)}` : esc(item.text); const extra = kind === 'danger' ? ' gr-row-danger' : ''; return `<div class="gr-row${extra}"><span class="gr-dot ${dot}"></span><div>${text}${citation(item)}</div></div>`; }).join('');
}
function planSection(title, description, items, kind, icon, step){ return `<section class="gr-plan-section gr-plan-section-${kind}"><div class="gr-plan-section-head"><span class="gr-plan-step">${step}</span><div><div class="gr-plan-title">${icon} ${title}</div><div class="gr-plan-copy">${description}</div></div></div><div class="gr-plan-rows">${planRows(items,kind)}</div></section>`; }
function renderPatients(){
  const holder = document.getElementById('patients'); holder.innerHTML = '';
  DATA.order.forEach(id => { const patient = DATA.patients[id].identity; const button = document.createElement('button'); button.className = 'patient-button' + (id === activeId ? ' active' : ''); button.innerHTML = `<strong>${esc(patient.name)}</strong><small>${esc(patient.age)} · ${esc(patient.sex)}</small><small>Heart-failure aftercare</small>`; button.onclick = () => { activeId = id; renderCare(); }; holder.appendChild(button); });
}
function renderCare(){
  const item = DATA.patients[activeId]; const person = item.identity; const record = item.record; const plan = item.plan || {}; const sources = item.sources || [];
  const tags = person.tags.map(tag => `<span class="gr-pill gr-pill-med">${esc(tag)}</span>`).join('');
  const problems = record.problem_list.map(problem => `<span class="gr-problem">${esc(problem)}</span>`).join('');
  const timeline = record.timeline.map(event => `<div class="gr-timeline-row"><div class="gr-timeline-date">${esc(event.date)}</div><div><div class="gr-timeline-title">${esc(event.title)}</div><div class="gr-timeline-detail">${esc(event.detail)}</div></div></div>`).join('');
  const files = record.files.map(file => `<div class="static-file"><b>📄 ${esc(file.type)}</b><br>${esc(file.name)}<br><span>${esc(file.detail)}</span></div>`).join('');
  const sourceCards = sources.map(source => `<div class="static-source"><b><code>${esc(source.id)}</code></b><br>${esc(source.text)}</div>`).join('') || '<div class="static-source">No shown-context record is available.</div>';
  document.getElementById('patient-workspace').innerHTML = `<div class="gr-hero gr-profile-head"><div class="gr-av gr-av-lg" style="background:${esc(person.color)}">${esc(person.initials)}</div><div><div class="name">${esc(person.name)}</div><div class="role">${esc(person.age)} yrs · ${esc(person.sex)} · Heart-failure aftercare</div><div class="gr-tags">${tags}</div><div class="blurb">${esc(person.blurb)}</div></div></div><div class="gr-card-head" style="margin-top:17px">📁 Patient record <span class="gr-record-label">Synthetic demo record · not clinical data</span></div><div class="gr-record-grid"><section class="gr-record-panel"><div class="gr-record-cap">Active context</div><div class="gr-record-title">What the recorded run received</div><div class="gr-problem-list">${problems}</div></section><section class="gr-record-panel"><div class="gr-record-cap">Care timeline</div><div class="gr-timeline">${timeline}</div></section></div><div class="gr-record-cap" style="margin-top:14px">Selected files</div><div class="static-files">${files}</div><div class="gr-card-head" style="margin-top:18px">📋 Recorded Grounded Aftercare Plan <span class="gr-pill gr-pill-ground" style="margin-left:auto">real-run snapshot</span></div><div class="gr-plan-intro">This is a saved model response from the artifact, not a live call. It is assistive only and defers to the care team.</div><div class="gr-plan-guide"><div class="gr-plan-guide-label">Planned follow-up</div><div class="gr-plan-guide-title">${esc(person.followup)}</div><div class="gr-plan-guide-copy">Every returned item has its recorded shown-context citation beside it.</div></div>${planSection('When to get help','These changes may mean you need to contact your care team.',plan.danger_signs,'danger','🚨','1')}${planSection('Your medicines','Use these instructions alongside labels from your care team.',plan.medications,'med','💊','2')}${planSection('Everyday care & follow-up','Small steps to support recovery before the next visit.',plan.lifestyle,'life','🥗','3')}<details style="margin-top:14px"><summary>Shown guideline context (${sources.length} passages)</summary><div class="static-sources">${sourceCards}</div></details>`;
  document.getElementById('model-label').textContent = DATA.provenance.model;
  document.getElementById('search').value = ''; document.getElementById('search-answer').textContent = 'Choose a patient, then search the recorded source excerpts.'; renderPatients();
}
function searchSources(){ const q = document.getElementById('search').value.toLowerCase().trim(); const output = document.getElementById('search-answer'); if(!q){ output.textContent = 'Choose a patient, then search the recorded source excerpts.'; return; } const tokens = q.match(/[a-z0-9]+/g) || []; const hits = (DATA.patients[activeId].sources || []).filter(source => tokens.some(token => String(source.text).toLowerCase().includes(token))).slice(0,2); output.innerHTML = hits.length ? hits.map(hit => `<div class="static-source"><b><code>${esc(hit.id)}</code></b><br>${esc(hit.text)}</div>`).join('') : 'No matching excerpt in this patient’s recorded context.'; }
function renderEvidence(){
  const rows = DATA.additive; const p = DATA.provenance; document.getElementById('evidence-badge').textContent = `${p.model} · n=${p.sample_size}`;
  document.getElementById('bars').innerHTML = rows.map(row => `<div class="metric-bar"><div class="metric-bar-label" title="${esc(row.name)}">${esc(row.name)}</div><div class="metric-track"><span class="metric-fill" style="width:${Math.max(0,Math.min(100,row.recall))}%"></span></div><div class="metric-value">${row.recall}%</div></div>`).join('');
  const fields = [['Model',p.model],['Provider',p.provider],['Recorded',p.recorded_at],['Sample size',`n=${p.sample_size} synthetic cases`],['Artifact SHA-256',p.artifact_sha256],['Git revision',p.git_revision],['Guidelines SHA-256',p.guidelines_sha256],['Cases SHA-256',p.cases_sha256]];
  document.getElementById('provenance').innerHTML = fields.map(([label,value]) => `<div><b>${esc(label)}</b>${esc(value)}</div>`).join('');
  document.getElementById('table').innerHTML = `<table class="evidence-table-static"><thead><tr><th>Stage</th><th>Danger recall</th><th>Unsupported output</th><th>Source faithfulness</th><th>Context est.</th><th>Provider tokens</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${esc(row.name)}</td><td>${row.recall}%</td><td>${row.unsupported}%</td><td>${row.faith}%</td><td>${row.context_estimate.toLocaleString()}</td><td>${row.provider_tokens == null ? '—' : Math.round(row.provider_tokens)}</td></tr>`).join('')}</tbody></table>`;
}
function showView(view){ const care = view === 'care'; document.getElementById('care-view').hidden = !care; document.getElementById('evidence-view').hidden = care; document.querySelectorAll('.gr-nav-link').forEach(button => button.classList.toggle('active',button.dataset.view === view)); if(care) renderCare(); else renderEvidence(); }
renderCare(); renderEvidence();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="path to a real-model eval provenance JSON")
    parser.add_argument("--output", default=None, help="output HTML path (default: GroundedRx.html)")
    args = parser.parse_args()
    target = build(args.artifact, args.output)
    print(f"wrote real-evidence snapshot -> {target}")


if __name__ == "__main__":
    main()
