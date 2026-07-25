"""Generate a single self-contained GroundedRx.html (open directly in a browser -- no server).

Bakes in, from the REAL pipeline (mock backend): three patient examples, their grounded
aftercare plans, the corpus (for a client-side grounded copilot), the additive ablation table,
and the frontier / lost-in-the-middle charts (embedded as base64 data URIs). All logic/data
come from src/ and data/ -- this only renders them.

    python app/build_static.py      # -> writes GroundedRx.html at the repo root
"""
from __future__ import annotations

import base64
import html as H
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.metrics import score_run
from app.ui_helpers import identity as display_identity
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

esc = H.escape


def build():
    corpus = GuidelineCorpus.load(os.path.join(ROOT, "guidelines", "heart_failure.json"))
    cases = json.load(open(os.path.join(ROOT, "data", "eval_cases.json")))["cases"]
    example_cases = cases[:3]
    client = make_client(True, schema=PLAN_SCHEMA)

    data = {}
    for idx, c in enumerate(example_cases):
        res = pipeline.run_case(c, corpus, client, pipeline.FULL)
        plan = res["plan"]
        display = display_identity(c["id"], c["profile"], tuple(c["meds"]), idx)
        data[c["id"]] = {
            "profile": c["profile"], "meds": c["meds"], "forbidden": c["forbidden_terms"],
            "patient": {"name": display["name"], "age": display["age"], "sex": display["sex"],
                        "followup": display["followup"].strftime("%b %d")},
            "plan": plan,
        }

    # corpus map for copilot tooltips + client-side grounded retrieval (real passages only)
    corpus_map = {p["id"]: {"type": p["type"], "drug": p.get("drug", ""), "text": p["text"]}
                  for p in corpus.passages}
    real_passages = [{"id": p["id"], "type": p["type"], "text": p["text"]}
                     for p in corpus.passages if p["type"] != "distractor"]

    # additive ablation table
    ablation = []
    for name, cfg in pipeline.additive_configs():
        agg = score_run([pipeline.run_case(c, corpus, client, cfg) for c in cases], cases, corpus)
        ablation.append({"config": name, "recall": round(agg["danger_recall"] * 100),
                         "halluc": round(agg["hallucination_rate"] * 100),
                         "faith": round(agg["citation_faithfulness"] * 100),
                         "tokens": round(agg["avg_ctx_tokens"])})

    def data_uri(path):
        if not os.path.exists(path):
            return ""
        return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

    frontier_uri = data_uri(os.path.join(ROOT, "slides", "frontier.png"))
    litm_uri = data_uri(os.path.join(ROOT, "slides", "lost_in_middle.png"))
    css = open(os.path.join(ROOT, "app", "styles.css")).read()

    payload = {"cases": data, "order": [c["id"] for c in example_cases], "corpus": corpus_map,
               "real": real_passages, "ablation": ablation}

    html = TEMPLATE.replace("/*CSS*/", css) \
                   .replace("__DATA__", json.dumps(payload)) \
                   .replace("__FRONTIER__", frontier_uri) \
                   .replace("__LITM__", litm_uri)
    out = os.path.join(ROOT, "GroundedRx.html")
    with open(out, "w") as f:
        f.write(html)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(data)} patients, self-contained)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GroundedRx — context-engineered aftercare</title>
<style>
/*CSS*/
/* standalone layout (no Streamlit) */
body { margin:0; padding:18px 24px 40px; }
.grid { display:grid; grid-template-columns:.85fr 2.8fr 1.3fr; gap:16px; align-items:start; }
@media (max-width:1100px){ .grid{ grid-template-columns:1fr; } }
 .pat-row{ display:flex; flex-direction:column; align-items:flex-start; width:100%; min-height:104px; box-sizing:border-box; text-align:left; background:#fff; border:none; border-radius:18px;
  padding:15px 16px; margin-bottom:12px; color:#687385; font-weight:500; cursor:pointer; transition:all .16s; box-shadow:0 12px 24px -20px rgba(45,68,99,.48); }
 .pat-row:hover{ border-color:transparent; background:#f8fbff; color:#687385; transform:translateY(-1px);
  box-shadow:0 10px 20px -18px rgba(30,111,245,.46); }
 .pat-row.active{ min-height:126px; border:none; background:#edf4ff; color:#687385; font-weight:500;
  box-shadow:0 15px 27px -20px rgba(30,111,245,.44); }
 .pat-name{ display:block; margin-bottom:3px; color:#171b24; font-size:15px; font-weight:780; letter-spacing:-.02em; }
 .pat-meta,.pat-reason{ display:block; color:#687385; font-size:12.5px; line-height:1.72; }
 .pat-followup{ display:none; width:100%; box-sizing:border-box; margin-top:10px; padding:8px 10px; border:none; border-radius:11px; background:#fff; box-shadow:0 8px 16px -14px rgba(30,111,245,.35); color:#4f78a0; font-size:11.5px; font-weight:700; }
 .pat-row.active .pat-followup{ display:block; }
.copilot-input{ display:flex; gap:8px; margin-top:10px; }
.copilot-input input{ flex:1; border:1px solid var(--border); border-radius:10px; padding:9px 12px; font-size:13.5px;
  font-family:var(--font); color:var(--body); outline:none; }
 .copilot-input input:focus{ border-color:#9bc2fa; }
 .copilot-input button{ background:var(--primary); color:#fff; border:none; border-radius:10px; padding:0 14px;
  font-weight:700; cursor:pointer; }
</style></head>
<body>
<div class="gr-topbar">
  <div class="gr-brand"><div class="gr-logo">💊</div>
    <div><div class="gr-brand-title">GroundedRx</div>
    <div class="gr-brand-sub">Context-engineered aftercare · grounded to source · privacy-preserving</div></div></div>
  <nav class="gr-nav-tabs" aria-label="Primary navigation">
    <button type="button" class="gr-nav-link active" data-view="workspace" aria-current="page"
      onclick="showView('workspace')">Care workspace</button>
    <button type="button" class="gr-nav-link" data-view="evidence" aria-current="false"
      onclick="showView('evidence')">Evidence</button>
  </nav>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="gr-pill gr-pill-ground"><span class="gr-live"></span>Gemma 4 E4B · Offline</span>
    <span class="gr-switch" title="On-device inference"></span>
  </div>
</div>
<main id="workspace-view" class="gr-view">
<div class="grid">
  <div>
    <div class="gr-card"><div class="gr-card-head">👥 Patient Directory</div><div id="directory"></div></div>
    <div class="gr-card"><div class="gr-label">⚡ Edge engine status</div>
      <div class="gr-engine">
        <div class="gr-engine-row"><span class="k">Engine</span><span class="v">Gemma 4 E4B (q4)</span></div>
        <div class="gr-engine-row"><span class="k">Mode</span><span class="v"><span class="gr-live"></span>Offline · on-device</span></div>
        <div class="gr-engine-row"><span class="k">Latency</span><span class="v">~42 ms</span></div>
        <div class="gr-engine-row"><span class="k">Model footprint</span><span class="v">~3.8 GB</span></div></div>
      <div class="gr-disclaimer">Illustrative / synthetic guidelines. Not clinical advice. Assistive, human-in-the-loop; defers to the care team. Metrics shown are from the mock harness pending real-model runs.</div></div>
  </div>
  <div>
    <div class="gr-card" id="plan"></div>
  </div>
  <div>
    <div class="gr-card gr-copilot-card"><div class="gr-card-head">💬 Clinical AI Copilot</div>
      <div class="gr-disclaimer" style="margin-top:-6px;margin-bottom:10px">Answers are grounded in cited guideline passages (client-side retrieval, fully offline).</div>
      <div id="chat"></div>
      <div class="copilot-input"><input id="q" placeholder="Ask about danger signs, meds, lifestyle…"
        onkeydown="if(event.key==='Enter')ask()"><button onclick="ask()">Ask</button></div></div>
  </div>
</div>
</main>
<main id="evidence-view" class="gr-view gr-evidence-page" hidden aria-labelledby="evidence-title">
  <section class="gr-evidence-hero">
    <div><div class="gr-evidence-eyebrow">Evaluation workspace</div>
      <div id="evidence-title" class="gr-evidence-title">Evidence, made easy to read.</div>
      <div class="gr-evidence-copy">A clear, reproducible view of how the context pipeline affects safety,
        grounding, and efficiency across the evaluation harness.</div></div>
    <div class="gr-evidence-hero-badge"><span class="gr-live"></span>Mock harness · offline</div>
  </section>
  <section id="evidence-kpis" class="gr-evidence-kpis" aria-label="Evaluation summary"></section>
  <div class="gr-evidence-grid">
    <section class="gr-card">
      <div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Quality progression</div>
        <div class="gr-evidence-card-title">Layer-by-layer ablation</div></div>
        <span class="gr-pill gr-pill-ground">all stages</span></div>
      <div id="evidence-quality" class="gr-evidence-bars"></div>
      <div class="gr-evidence-chart-note">Each row adds one context-engineering layer, keeping the quality
        trade-offs visible rather than hiding them in a single score.</div>
    </section>
    <aside id="evidence-takeaway" class="gr-evidence-insight"></aside>
  </div>
  <div class="gr-evidence-grid gr-evidence-grid-two">
    <section class="gr-card">
      <div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Efficiency</div>
        <div class="gr-evidence-card-title">Efficiency frontier</div></div>
        <span class="gr-pill gr-pill-med">tokens vs. recall</span></div>
      <img class="gr-evidence-image" src="__FRONTIER__"
        alt="Efficiency frontier comparing context tokens and accuracy for engineered and naive RAG">
      <div class="gr-evidence-chart-note">Compare the accuracy reached at each tested context budget.</div>
    </section>
    <section class="gr-card">
      <div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Robustness</div>
        <div class="gr-evidence-card-title">Lost in the middle</div></div>
        <span class="gr-pill gr-pill-life">fact position</span></div>
      <img class="gr-evidence-image" src="__LITM__"
        alt="Lost-in-the-middle chart comparing recall by fact position">
      <div class="gr-evidence-chart-note">Tests whether key safety facts remain available when their position changes.</div>
    </section>
  </div>
  <section class="gr-card">
    <div class="gr-evidence-card-head"><div><div class="gr-evidence-card-kicker">Experiment details</div>
      <div class="gr-evidence-card-title">All evaluated stages</div></div>
      <span class="gr-pill gr-pill-muted">reproducible run</span></div>
    <div id="ev-table"></div>
  </section>
  <div class="gr-disclaimer gr-page-disclaimer">⚠️ Illustrative / synthetic guidelines and patient identities.
    Not clinical advice. Assistive, human-in-the-loop; defers to the care team. Metrics are from the mock harness
    pending real-model runs.</div>
</main>
<script>
const D = __DATA__;
let active = D.order[0];
const byId = D.corpus;
let reviewed = {};

function directory(){
  const el = document.getElementById('directory'); el.innerHTML='';
  D.order.forEach(id=>{
    const c = D.cases[id];
    const patient = c.patient;
    const div = document.createElement('div');
    div.className = 'pat-row' + (id===active?' active':'');
    const name = document.createElement('span'); name.className = 'pat-name'; name.textContent = patient.name;
    const meta = document.createElement('span'); meta.className = 'pat-meta'; meta.textContent = `${patient.age} · ${patient.sex}`;
    const reason = document.createElement('span'); reason.className = 'pat-reason'; reason.textContent = 'Heart-failure aftercare';
    const followup = document.createElement('span'); followup.className = 'pat-followup'; followup.textContent = `Follow-up · ${patient.followup}`;
    div.append(name, meta, reason, followup);
    div.onclick = ()=>{ active=id; render(); };
    el.appendChild(div);
  });
}
function rows(items, kind){
  const dot = {danger:'gr-dot-danger', med:'gr-dot-med', life:'gr-dot-life'}[kind];
  if(!items.length) return "<div class='gr-row gr-row-empty'>No additional guidance is listed here.</div>";
  return items.map(it=>{
    const cid = it.guideline_id||'';
    const src = byId[cid];
    const tip = src ? src.text.replace(/'/g,'&#39;').replace(/"/g,'&quot;') : 'uncited';
    const cite = `<span class='gr-cite' title="${tip}">${cid||'—'}</span>`;
    const txt = kind==='med'
      ? `<span class='gr-drug'>${it.drug||''}</span> — ${it.instruction||''}`
      : (it.text||'');
    const rc = kind==='danger' ? 'gr-row gr-row-danger' : 'gr-row';
    return `<div class='${rc}'><span class='gr-dot ${dot}'></span><div>${txt}${cite}</div></div>`;
  }).join('');
}
function planSection(title, copy, items, kind, icon, step){
  return "<section class='gr-plan-section gr-plan-section-" + kind + "'>"
    + "<div class='gr-plan-section-head'><span class='gr-plan-step'>" + step + "</span><div>"
    + "<div class='gr-plan-title'>" + icon + " " + title + "</div>"
    + "<div class='gr-plan-copy'>" + copy + "</div></div></div>"
    + "<div class='gr-plan-rows'>" + rows(items, kind) + "</div></section>";
}
function reviewPlan(){
  reviewed[active] = !reviewed[active];
  plan();
}
function plan(){
  const c = D.cases[active];
  const done = Boolean(reviewed[active]);
  const reviewLabel = done ? "✓ Plan reviewed" : "I've reviewed this plan";
  const reviewNote = done
    ? "Marked as reviewed in this browser. This does not change your care."
    : "This is a personal reminder only — it does not update your care team.";
  document.getElementById('plan').innerHTML =
    "<div class='gr-card-head'>📋 Your Aftercare Plan"
      + "<span class='gr-pill gr-pill-ground' style='margin-left:auto'>your guide</span></div>"
    + "<div class='gr-plan-intro'>Take this one step at a time. These are the key things to watch for, take, and do before your planned follow-up.</div>"
    + "<div class='gr-plan-guide'><div class='gr-plan-guide-label'>Your next step</div>"
      + "<div class='gr-plan-guide-title'>Planned follow-up · " + c.patient.followup + "</div>"
      + "<div class='gr-plan-guide-copy'>Keep this plan nearby and ask your care team if anything is unclear.</div></div>"
    + planSection("When to get help", "These changes may mean you need to contact your care team.",
      c.plan.danger_signs, "danger", "🚨", "1")
    + planSection("Your medicines", "Use these instructions alongside the labels from your care team.",
      c.plan.medications, "med", "💊", "2")
    + planSection("Everyday care & follow-up", "Small steps to support your recovery before your next visit.",
      c.plan.lifestyle, "life", "🥗", "3")
    + "<button type='button' class='gr-review-btn' aria-pressed='" + done + "' onclick='reviewPlan()'>"
      + reviewLabel + "</button>"
    + "<div class='gr-plan-note'>" + reviewNote + "</div>";
}

let chat = [];
function renderChat(){
  const el = document.getElementById('chat');
  if(!chat.length){ el.innerHTML = "<div class='gr-msg gr-msg-bot'>Ask me about this patient's danger signs, medications, or lifestyle — I answer only from cited guidelines.</div>"; return; }
  el.innerHTML = chat.slice(-6).map(m=>`<div class='gr-msg ${m.r==='user'?'gr-msg-user':'gr-msg-bot'}'>${m.t}</div>`).join('');
}
function ask(){
  const inp = document.getElementById('q'); const q = inp.value.trim(); if(!q) return;
  const toks = (q.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>3);
  const scored = D.real.map(p=>{
    const pt = new Set((p.text.toLowerCase().match(/[a-z0-9]+/g)||[]));
    let s=0; toks.forEach(t=>{ if(pt.has(t)) s++; });
    return {p, s};
  }).sort((a,b)=>b.s-a.s);
  const hits = scored.filter(x=>x.s>0).slice(0,2).map(x=>x.p);
  let ans;
  if(!hits.length){ ans = "I can only answer from the grounded guideline context, and I found no relevant passage. Please rephrase or defer to the care team."; }
  else { ans = "Based on the grounded guidelines: " + hits.map(h=>`${h.text} <span class='gr-cite'>${h.id}</span>`).join(' ') + " — This is assistive; always defer to the care team."; }
  chat.push({r:'user', t:q}); chat.push({r:'bot', t:ans}); inp.value=''; renderChat();
  document.getElementById('chat').scrollTop = 1e9;
}
function evidenceKpi(icon, value, label, detail, tone){
  return "<article class='gr-evidence-kpi gr-evidence-kpi-" + tone + "'>"
    + "<div class='gr-evidence-kpi-icon'>" + icon + "</div><div>"
    + "<div class='gr-evidence-kpi-value'>" + value + "</div>"
    + "<div class='gr-evidence-kpi-label'>" + label + "</div>"
    + "<div class='gr-evidence-kpi-detail'>" + detail + "</div></div></article>";
}
function renderEvidence(){
  const stages = D.ablation;
  const featured = stages[stages.length - 1];
  const bestRecall = Math.max.apply(null, stages.map(r=>r.recall));
  const lowestHalluc = Math.min.apply(null, stages.map(r=>r.halluc));
  const bestFaith = Math.max.apply(null, stages.map(r=>r.faith));
  const leanestContext = Math.min.apply(null, stages.map(r=>r.tokens));
  document.getElementById('evidence-kpis').innerHTML = [
    evidenceKpi("🛟", bestRecall + "%", "Best danger recall", "best evaluated stage", "blue"),
    evidenceKpi("✓", lowestHalluc + "%", "Unsupported output", "lower is better", "slate"),
    evidenceKpi("🔗", bestFaith + "%", "Source faithfulness", "highest evaluated stage", "violet"),
    evidenceKpi("⚡", leanestContext.toLocaleString(), "Context budget", "smallest tested stage", "rose")
  ].join('');
  document.getElementById('evidence-quality').innerHTML = stages.map(r =>
    "<div class='gr-evidence-bar-row'><div class='gr-evidence-bar-label' title='" + r.config + "'>"
      + r.config + "</div><div class='gr-evidence-bar-track'><span class='gr-evidence-bar-fill' style='width:"
      + r.recall + "%'></span></div><div class='gr-evidence-bar-value'>" + r.recall + "%</div></div>"
  ).join('');
  document.getElementById('evidence-takeaway').innerHTML =
    "<div class='gr-evidence-insight-kicker'>What this shows</div>"
    + "<div class='gr-evidence-insight-title'>Better context, not simply more context.</div>"
    + "<p>The dashboard keeps reliability, source grounding, and token cost together so the trade-offs are easy to compare.</p>"
    + "<div class='gr-evidence-insight-stats'><div><strong>" + featured.recall + "%</strong><span>danger recall</span></div>"
    + "<div><strong>" + featured.faith + "%</strong><span>source faithfulness</span></div>"
    + "<div><strong>" + featured.tokens.toLocaleString() + "</strong><span>context tokens</span></div></div>";
  const rows = stages.map(r => {
    const featuredRow = r.config === featured.config ? " gr-abl-featured" : "";
    const hallucClass = r.halluc > 0 ? "r" : "g";
    return "<tr class='" + featuredRow.trim() + "'><td>" + r.config + "</td><td class='g'>" + r.recall
      + "%</td><td class='" + hallucClass + "'>" + r.halluc + "%</td><td class='g'>" + r.faith
      + "%</td><td>" + r.tokens.toLocaleString() + "</td></tr>";
  }).join('');
  document.getElementById('ev-table').innerHTML =
    "<table class='abl gr-evidence-table'><tr><th>stage</th><th>danger recall</th><th>unsupported output</th>"
    + "<th>source faithfulness</th><th>context tokens</th></tr>" + rows + "</table>";
}
function viewFromHash(){
  return typeof window !== 'undefined' && window.location.hash === '#evidence' ? 'evidence' : 'workspace';
}
function showView(view, updateHash){
  const next = view === 'evidence' ? 'evidence' : 'workspace';
  const workspace = document.getElementById('workspace-view');
  const evidence = document.getElementById('evidence-view');
  workspace.hidden = next !== 'workspace';
  evidence.hidden = next !== 'evidence';
  workspace.setAttribute('aria-hidden', String(next !== 'workspace'));
  evidence.setAttribute('aria-hidden', String(next !== 'evidence'));
  document.querySelectorAll('.gr-nav-link').forEach(button => {
    const activeView = button.dataset.view === next;
    button.classList.toggle('active', activeView);
    button.setAttribute('aria-current', activeView ? 'page' : 'false');
  });
  if(updateHash !== false && typeof window !== 'undefined'){
    try { window.history.replaceState(null, '', '#' + next); }
    catch (error) { window.location.hash = next; }
    if(window.scrollTo) window.scrollTo(0, 0);
  }
}
function render(){ directory(); plan(); }
render(); renderChat(); renderEvidence(); showView(viewFromHash(), false);
if(typeof window !== 'undefined') window.addEventListener('hashchange', () => showView(viewFromHash(), false));
</script>
</body></html>"""


if __name__ == "__main__":
    build()
