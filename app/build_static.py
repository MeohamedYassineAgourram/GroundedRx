"""Generate a single self-contained GroundedRx.html (open directly in a browser -- no server).

Bakes in, from the REAL pipeline (mock backend): every patient case, its grounded aftercare
plan + metrics, a per-case context graph (inline SVG), the corpus (for a client-side grounded
copilot), the additive ablation table, and the frontier / lost-in-the-middle charts (embedded
as base64 data URIs). All logic/data come from src/ and data/ -- this only renders them.

    python app/build_static.py      # -> writes GroundedRx.html at the repo root
"""
from __future__ import annotations

import base64
import html as H
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.metrics import citation_faithfulness, danger_recall, has_hallucination, score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

esc = H.escape
_TS = {"danger_sign": ("#dc2626", "#fef2f2"), "medication": ("#2563eb", "#eff6ff"),
       "lifestyle": ("#16a34a", "#f0fdf4")}


def short(p):
    if p["type"] == "danger_sign":
        return "DS" + p["id"].split("-")[-1]
    if p["type"] == "medication":
        return (p.get("drug") or p["id"])[:5]
    if p["type"] == "lifestyle":
        return "LF" + p["id"].split("-")[-1]
    return p["id"][:4]


def graph_svg(case, corpus, nodes):
    W, Hh, cx, cy, R = 720, 360, 360, 178, 132
    n = max(1, len(nodes))
    edges, circles = [], []
    for i, p in enumerate(nodes):
        a = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        s, f = _TS.get(p["type"], ("#64748b", "#f1f5f9"))
        edges.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.0f}' y2='{y:.0f}' stroke='#e2e8f0' stroke-width='1.5'/>")
        circles.append(
            f"<g><title>{esc(p.get('text',''))}</title>"
            f"<circle cx='{x:.0f}' cy='{y:.0f}' r='21' fill='{f}' stroke='{s}' stroke-width='2'/>"
            f"<text x='{x:.0f}' y='{y+3:.0f}' text-anchor='middle' font-size='9' font-weight='700' fill='{s}'>{esc(short(p))}</text></g>")
    age = (re.findall(r"\d+", case["profile"]) or ["?"])[0]
    center = (f"<circle cx='{cx}' cy='{cy}' r='34' fill='#0f172a'/>"
              f"<text x='{cx}' y='{cy-2}' text-anchor='middle' font-size='13' font-weight='800' fill='#fff'>{age}y</text>"
              f"<text x='{cx}' y='{cy+12}' text-anchor='middle' font-size='8' fill='#cbd5e1'>PATIENT</text>")
    pruned = sum(1 for p in corpus.passages if p["type"] == "distractor")
    head = (f"<div class='ghead'>🧠 Dynamic Clinical Context Graph"
            f"<span class='gmeta'>{len(nodes)} chunks retrieved · {pruned} distractors pruned</span></div>")
    legend = ("<div class='glegend'>"
              "<span><i style='background:#dc2626'></i>Danger sign</span>"
              "<span><i style='background:#2563eb'></i>Medication</span>"
              "<span><i style='background:#16a34a'></i>Lifestyle</span></div>")
    return (head + f"<svg viewBox='0 0 {W} {Hh}' width='100%' style='display:block'>"
            + "".join(edges) + center + "".join(circles) + "</svg>" + legend)


def build():
    corpus = GuidelineCorpus.load(os.path.join(ROOT, "guidelines", "heart_failure.json"))
    cases = json.load(open(os.path.join(ROOT, "data", "eval_cases.json")))["cases"]
    client = make_client(True, schema=PLAN_SCHEMA)

    data = {}
    for c in cases:
        res = pipeline.run_case(c, corpus, client, pipeline.FULL)
        plan = res["plan"]
        # graph nodes = the sources the plan actually cites (grounded provenance) + all danger signs
        cited = []
        seen = set()
        for section in ("danger_signs", "medications", "lifestyle"):
            for it in plan.get(section, []):
                pid = it.get("guideline_id")
                if pid and pid in corpus.by_id and pid not in seen:
                    seen.add(pid)
                    cited.append(corpus.by_id[pid])
        for d in corpus.danger_signs:
            if d["id"] not in seen:
                seen.add(d["id"])
                cited.append(d)
        data[c["id"]] = {
            "profile": c["profile"], "meds": c["meds"], "forbidden": c["forbidden_terms"],
            "plan": plan,
            "metrics": {
                "recall": round(danger_recall(plan, c, corpus) * 100),
                "halluc": has_hallucination(plan, c, corpus),
                "faith": round(citation_faithfulness(plan, corpus) * 100),
                "tokens": res["ctx_tokens"],
            },
            "svg": graph_svg(c, corpus, cited),
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

    payload = {"cases": data, "order": [c["id"] for c in cases], "corpus": corpus_map,
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
.grid { display:grid; grid-template-columns:1fr 2.5fr 1.5fr; gap:16px; align-items:start; }
@media (max-width:1100px){ .grid{ grid-template-columns:1fr; } }
.pat-row{ width:100%; text-align:left; background:#fff; border:1px solid var(--border); border-radius:12px;
  padding:10px 12px; margin-bottom:8px; color:var(--body); font-weight:500; cursor:pointer; transition:all .15s; font-size:13px; }
.pat-row:hover{ border-color:#bfdbfe; background:#f8fbff; color:var(--ink); transform:translateY(-1px);
  box-shadow:0 6px 14px -8px rgba(37,99,235,.4); }
.pat-row.active{ border-color:#2563eb; background:#eff6ff; color:#1d4ed8; font-weight:600; }
.ghead{ background:#f1f5f9; border:1px solid #e2e8f0; border-radius:12px; padding:10px 14px; font-weight:700;
  color:#0f172a; font-size:15px; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; }
.gmeta{ font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.06em; }
.glegend{ display:flex; gap:16px; justify-content:center; margin-top:6px; font-size:12px; color:#475569; }
.glegend i{ display:inline-block; width:9px; height:9px; border-radius:9999px; margin-right:5px; }
svg text{ font-family:system-ui,-apple-system,sans-serif; }
.copilot-input{ display:flex; gap:8px; margin-top:10px; }
.copilot-input input{ flex:1; border:1px solid var(--border); border-radius:10px; padding:9px 12px; font-size:13.5px;
  font-family:var(--font); color:var(--body); outline:none; }
.copilot-input input:focus{ border-color:#93c5fd; }
.copilot-input button{ background:var(--primary); color:#fff; border:none; border-radius:10px; padding:0 14px;
  font-weight:700; cursor:pointer; }
.ev-btns{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
.ev-btn{ background:#f1f5f9; border:1px solid var(--border); color:#475569; border-radius:9999px; padding:5px 13px;
  font-size:12px; font-weight:600; cursor:pointer; }
.ev-btn.active{ background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8; }
.ev-panel{ display:none; } .ev-panel.show{ display:block; }
table.abl{ width:100%; border-collapse:collapse; font-size:12px; }
table.abl th{ text-align:left; color:var(--muted); text-transform:uppercase; font-size:10px; letter-spacing:.05em;
  padding:6px 8px; border-bottom:1px solid var(--border); }
table.abl td{ padding:7px 8px; border-bottom:1px dashed #eef2f7; color:var(--body); }
table.abl td.g{ color:#047857; font-weight:700; } table.abl td.r{ color:#dc2626; font-weight:700; }
.ev-panel img{ width:100%; border-radius:12px; border:1px solid var(--border); }
</style></head>
<body>
<div class="gr-topbar">
  <div class="gr-brand"><div class="gr-logo">💊</div>
    <div><div class="gr-brand-title">GroundedRx</div>
    <div class="gr-brand-sub">Context-engineered aftercare · grounded to source · privacy-preserving</div></div></div>
  <div><span class="gr-pill gr-pill-ground"><span class="gr-live"></span>Gemma 4 E4B · Offline demo</span></div>
</div>
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
    <div class="gr-card"><div id="graph"></div></div>
    <div class="gr-card" id="plan"></div>
  </div>
  <div>
    <div class="gr-card"><div class="gr-card-head">💬 Clinical AI Copilot</div>
      <div class="gr-disclaimer" style="margin-top:-6px;margin-bottom:10px">Answers are grounded in cited guideline passages (client-side retrieval, fully offline).</div>
      <div id="chat"></div>
      <div class="copilot-input"><input id="q" placeholder="Ask about danger signs, meds, lifestyle…"
        onkeydown="if(event.key==='Enter')ask()"><button onclick="ask()">Ask</button></div></div>
    <div class="gr-card"><div class="gr-card-head">📊 Evidence Drawer</div>
      <div class="gr-disclaimer" style="margin-top:-6px;margin-bottom:8px">One-click proof the context engineering pays off.</div>
      <div class="ev-btns">
        <button class="ev-btn active" onclick="evShow('abl',this)">Ablation table</button>
        <button class="ev-btn" onclick="evShow('front',this)">Efficiency frontier</button>
        <button class="ev-btn" onclick="evShow('litm',this)">Lost-in-the-middle</button></div>
      <div id="ev-abl" class="ev-panel show"></div>
      <div id="ev-front" class="ev-panel"><img src="__FRONTIER__" alt="frontier"></div>
      <div id="ev-litm" class="ev-panel"><img src="__LITM__" alt="lost in the middle"></div></div>
  </div>
</div>
<script>
const D = __DATA__;
let active = D.order[0];
const byId = D.corpus;

function directory(){
  const el = document.getElementById('directory'); el.innerHTML='';
  D.order.forEach(id=>{
    const c = D.cases[id];
    const age = (c.profile.match(/\d+/)||['?'])[0];
    const reg = c.meds.slice(0,2).join('+') + (c.meds.length>2?'…':'');
    const div = document.createElement('div');
    div.className = 'pat-row' + (id===active?' active':'');
    div.textContent = `${id}  ·  ${age}y  ·  ${reg}`;
    div.onclick = ()=>{ active=id; render(); };
    el.appendChild(div);
  });
}
function rows(items, kind){
  const dot = {danger:'gr-dot-danger', med:'gr-dot-med', life:'gr-dot-life'}[kind];
  if(!items.length) return "<div class='gr-row'>—</div>";
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
function plan(){
  const c = D.cases[active], m = c.metrics;
  const hcls = m.halluc?'bad':'good';
  document.getElementById('plan').innerHTML = `
    <div class='gr-card-head'>📋 Grounded Aftercare Plan
      <span class='gr-pill gr-pill-ground' style='margin-left:auto'>context-engineered</span></div>
    <div class='gr-label'>Patient</div>
    <div style='font-size:15px;font-weight:600;color:#0f172a'>${c.profile}</div>
    <div style='font-size:12.5px;color:#64748b;margin-bottom:14px'>Prescribed: ${c.meds.join(', ')}</div>
    <div class='gr-chips'>
      <div class='gr-chip good'><div class='v'>${m.recall}%</div><div class='k'>Danger recall</div></div>
      <div class='gr-chip ${hcls}'><div class='v'>${m.halluc?'YES':'None'}</div><div class='k'>Hallucination</div></div>
      <div class='gr-chip good'><div class='v'>${m.faith}%</div><div class='k'>Citation faith</div></div>
      <div class='gr-chip'><div class='v'>${m.tokens}</div><div class='k'>Ctx tokens</div></div></div>
    <div class='gr-label' style='margin-top:18px'>🚨 Danger signs — seek care if you notice</div>${rows(c.plan.danger_signs,'danger')}
    <div class='gr-label' style='margin-top:16px'>💊 Prescribed medications & guidance</div>${rows(c.plan.medications,'med')}
    <div class='gr-label' style='margin-top:16px'>🥗 Lifestyle & follow-up</div>${rows(c.plan.lifestyle,'life')}`;
}
function graph(){ document.getElementById('graph').innerHTML = D.cases[active].svg; }

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
function evShow(which, btn){
  document.querySelectorAll('.ev-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active');
  document.querySelectorAll('.ev-panel').forEach(p=>p.classList.remove('show'));
  document.getElementById('ev-'+which).classList.add('show');
}
function ablation(){
  const rows = D.ablation.map(r=>`<tr><td>${r.config}</td>
    <td class='g'>${r.recall}%</td><td class='${r.halluc>0?'r':'g'}'>${r.halluc}%</td>
    <td class='g'>${r.faith}%</td><td>${r.tokens}</td></tr>`).join('');
  document.getElementById('ev-abl').innerHTML =
    `<table class='abl'><tr><th>config</th><th>recall</th><th>halluc</th><th>faith</th><th>ctx tok</th></tr>${rows}</table>
     <div class='gr-disclaimer' style='margin-top:8px'>Context engineering alone (through +fewshot) drives the win; re-grounding is an optional final refinement.</div>`;
}
function render(){ directory(); graph(); plan(); }
render(); renderChat(); ablation();
</script>
</body></html>"""


if __name__ == "__main__":
    build()
