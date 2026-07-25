"""Pure UI helpers (no Streamlit import) shared by the app and the static preview.
Keeps presentation logic testable/renderable outside a running Streamlit server."""
from __future__ import annotations

import calendar as _cal
import datetime as _dt
import hashlib
import html as _html
import math
import re

C_ENG, C_NAIVE, C_BLUE, C_VIOLET = "#34d399", "#f43f5e", "#4f7cff", "#a78bfa"
AV_COLORS = ["#5a86ff", "#7c6cf0", "#0ea5e9", "#0d9488", "#8b5cf6", "#2563eb", "#0891b2", "#6366f1"]

NAMES = [
    ("James Carter", "M"), ("Angelica Monica", "F"), ("Robert Hughes", "M"), ("Maria Alvarez", "F"),
    ("David Okafor", "M"), ("Linda Chen", "F"), ("Michael Brahmani", "M"), ("Sofia Rossi", "F"),
    ("William Turner", "M"), ("Fatima Nasser", "F"), ("Daniel Leon", "M"), ("Grace Park", "F"),
    ("Henry Alhernaym", "M"), ("Olivia Bennett", "F"), ("Samuel Adebayo", "M"), ("Emma Novak", "F"),
    ("Thomas Fischer", "M"), ("Aisha Khan", "F"), ("Charles Dupont", "M"), ("Isabella Costa", "F"),
    ("George Papadakis", "M"), ("Mei Ling", "F"), ("Ahmed Farouk", "M"), ("Hannah Schmidt", "F"),
    ("Victor Moreau", "M"), ("Priya Nair", "F"), ("Frank Weber", "M"), ("Elena Petrova", "F"),
    ("Omar Haddad", "M"), ("Ruth Mensah", "F"),
]


def esc(s):
    return _html.escape(str(s))


def av_color(key):
    return AV_COLORS[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(AV_COLORS)]


def initials(name):
    p = name.split()
    return (p[0][0] + (p[-1][0] if len(p) > 1 else "")).upper()


def identity(case_id, profile, meds, idx):
    """Synthetic but stable patient identity (presentation only)."""
    name, sex = NAMES[idx % len(NAMES)]
    age = int((re.findall(r"\d+", profile) or ["70"])[0])
    base = _dt.date(2026, 7, 22)
    admitted = base - _dt.timedelta(days=(idx * 3) % 60 + 5)
    discharged = admitted + _dt.timedelta(days=4 + (idx % 5))
    followup = discharged + _dt.timedelta(days=14)
    tags = ["HFrEF", "Cardiology"]
    low = profile.lower()
    if "diabetes" in low: tags.append("Type 2 Diabetes")
    if "kidney" in low: tags.append("CKD")
    if "atrial" in low: tags.append("Atrial Fibrillation")
    if "hypertension" in low: tags.append("Hypertension")
    if "lives alone" in low: tags.append("Lives alone")
    sexw = "Female" if sex == "F" else "Male"
    blurb = (f"{age}-year-old {sexw.lower()} patient with {'newly diagnosed ' if 'new' in low else ''}"
             f"heart failure with reduced ejection fraction. Admitted {admitted:%d %b %Y}, discharged "
             f"{discharged:%d %b %Y} on {', '.join(meds)}. This aftercare plan was generated on-device "
             f"and cites every instruction to a source guideline; assistive and deferring to the care team.")
    return {"name": name, "sex": sexw, "age": age, "admitted": admitted, "discharged": discharged,
            "followup": followup, "tags": tags, "blurb": blurb, "initials": initials(name),
            "color": av_color(name)}


def ring_svg(pct, color, center, size=62):
    r = 25.0
    c = 2 * math.pi * r
    off = c * (1 - max(0, min(100, pct)) / 100)
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 62 62'>"
            f"<circle cx='31' cy='31' r='25' fill='none' stroke='#edf0f7' stroke-width='7'/>"
            f"<circle cx='31' cy='31' r='25' fill='none' stroke='{color}' stroke-width='7' stroke-linecap='round' "
            f"stroke-dasharray='{c:.1f}' stroke-dashoffset='{off:.1f}' transform='rotate(-90 31 31)'/>"
            f"<text x='31' y='36' text-anchor='middle' font-size='14' font-weight='800' fill='#1c2340'>{esc(center)}</text></svg>")


def calendar_html(d, highlight):
    cal = _cal.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(d.year, d.month)
    head = "".join(f"<th>{x}</th>" for x in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    rows = ""
    for w in weeks:
        cells = ""
        for day in w:
            if day == 0:
                cells += "<td class='dim'></td>"
            elif day == highlight:
                cells += f"<td class='hl'>{day}</td>"
            else:
                cells += f"<td>{day}</td>"
        rows += f"<tr>{cells}</tr>"
    return (f"<div class='gr-cal'><div class='cap'><span class='mo'>{d:%B %Y}</span>"
            f"<span class='gr-pill gr-pill-muted'>Next follow-up · {highlight}</span></div>"
            f"<table><tr>{head}</tr>{rows}</table></div>")


_TS = {"danger_sign": ("#f43f5e", "#fff1f2"), "medication": ("#4f7cff", "#eef4ff"), "lifestyle": ("#22c55e", "#ecfdf3")}


def short(p):
    if p["type"] == "danger_sign": return "DS" + p["id"].split("-")[-1]
    if p["type"] == "medication": return (p.get("drug") or p["id"])[:5]
    if p["type"] == "lifestyle": return "LF" + p["id"].split("-")[-1]
    return p["id"][:4]


def graph_html(corpus, nodes):
    W, Hh, cx, cy, R = 720, 320, 360, 158, 118
    n = max(1, len(nodes)); edges = []; circles = []
    for i, p in enumerate(nodes):
        a = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        s, f = _TS.get(p["type"], ("#64748b", "#f1f5f9"))
        edges.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.0f}' y2='{y:.0f}' stroke='#e5e9f5' stroke-width='1.5'/>")
        circles.append(f"<g><title>{esc(p.get('text',''))}</title>"
                       f"<circle cx='{x:.0f}' cy='{y:.0f}' r='20' fill='{f}' stroke='{s}' stroke-width='2'/>"
                       f"<text x='{x:.0f}' y='{y+3:.0f}' text-anchor='middle' font-size='9' font-weight='700' fill='{s}'>{esc(short(p))}</text></g>")
    center = (f"<circle cx='{cx}' cy='{cy}' r='32' fill='#1c2340'/>"
              f"<text x='{cx}' y='{cy+4}' text-anchor='middle' font-size='11' font-weight='800' fill='#fff'>PATIENT</text>")
    pruned = sum(1 for p in corpus.passages if p["type"] == "distractor")
    return f"""<style>body{{margin:0;font-family:system-ui,-apple-system,sans-serif}}
      .h{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
      .h b{{color:#1c2340;font-size:14px}} .h span{{font-size:10.5px;font-weight:600;color:#8a93ad;text-transform:uppercase;letter-spacing:.06em}}
      .lg{{display:flex;gap:14px;justify-content:center;margin-top:2px;font-size:11.5px;color:#495276}}
      .lg i{{display:inline-block;width:9px;height:9px;border-radius:9999px;margin-right:5px}}</style>
      <div class='h'><b>🧠 Clinical context graph</b><span>{len(nodes)} chunks retrieved · {pruned} pruned</span></div>
      <svg viewBox='0 0 {W} {Hh}' width='100%' style='display:block'>{''.join(edges)}{center}{''.join(circles)}</svg>
      <div class='lg'><span><i style='background:#f43f5e'></i>Danger sign</span><span><i style='background:#4f7cff'></i>Medication</span><span><i style='background:#22c55e'></i>Lifestyle</span></div>"""


def grounded_answer(query, corpus):
    toks = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3]
    real = [p for p in corpus.passages if p["type"] != "distractor"]
    scored = []
    for p in real:
        pt = set(re.findall(r"[a-z0-9]+", p["text"].lower()))
        scored.append((sum(1 for t in toks if t in pt), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [p for s, p in scored if s > 0][:2]
    if not hits:
        return "I can only answer from the grounded guideline context and found no relevant passage. Please rephrase, or defer to the care team."
    body = " ".join(f"{h['text']} <span class='gr-cite'>{h['id']}</span>" for h in hits)
    return f"Based on the grounded guidelines: {body} <br><span style='color:#8a93ad'>Assistive — always defer to the care team.</span>"
