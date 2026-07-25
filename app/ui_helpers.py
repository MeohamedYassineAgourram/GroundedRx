"""Pure UI helpers (no Streamlit import) shared by the app and the static preview.
Keeps presentation logic testable/renderable outside a running Streamlit server."""
from __future__ import annotations

import datetime as _dt
import hashlib
import html as _html
import re

C_ENG, C_NAIVE, C_BLUE, C_VIOLET = "#1f6feb", "#f04438", "#4d8df5", "#7c5cfc"
AV_COLORS = ["#4d8df5", "#6b8fb8", "#5f9bea", "#537da9", "#8b82ad", "#4775a8", "#5b89be", "#7587ad"]

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
    blurb = (f"Synthetic demo profile: {age}-year-old {sexw.lower()} patient with "
             f"{'newly diagnosed ' if 'new' in low else ''}heart failure with reduced ejection fraction. "
             f"The workspace contains a sample admission timeline, listed medicines, and selected source files. "
             f"Generate a cited aftercare draft only after reviewing that context; it remains assistive and defers "
             f"to the care team.")
    return {"name": name, "sex": sexw, "age": age, "admitted": admitted, "discharged": discharged,
            "followup": followup, "tags": tags, "blurb": blurb, "initials": initials(name),
            "color": av_color(name)}


def patient_workspace(case, patient):
    """Return a clearly-labelled synthetic chart assembled from the demo case.

    The benchmark intentionally contains synthetic, non-identifying data.  This helper turns
    that source data into the files a clinician would review before asking the model for an
    aftercare brief; it never claims to be a real patient record.
    """
    meds = case.get("meds", [])
    med_list = ", ".join(meds) if meds else "No medicines listed"
    profile = case.get("profile", "Heart-failure aftercare case")
    concerns = []
    profile_lower = profile.lower()
    if "diabetes" in profile_lower:
        concerns.append("Type 2 diabetes")
    if "lives alone" in profile_lower:
        concerns.append("Lives alone / support check")
    if "hospital" in profile_lower or "fluid overload" in profile_lower:
        concerns.append("Recent fluid-overload admission")
    if not concerns:
        concerns.append("Heart-failure follow-up")

    admitted = patient["admitted"]
    discharged = patient["discharged"]
    return {
        "label": "Synthetic demo record · not clinical data",
        "problem_list": ["Heart failure with reduced ejection fraction", *concerns],
        "timeline": [
            {"date": admitted.strftime("%b %d, %Y"), "title": "Hospital admission", "detail": profile},
            {"date": discharged.strftime("%b %d, %Y"), "title": "Discharge medication reconciliation", "detail": med_list},
            {"date": patient["followup"].strftime("%b %d, %Y"), "title": "Planned cardiology follow-up", "detail": "Review symptoms, medicines, and self-care plan."},
        ],
        "files": [
            {
                "type": "Discharge summary",
                "name": f"{case['id'].lower()}_discharge_summary.pdf",
                "detail": "Synthetic summary of admission, diagnosis, and planned follow-up.",
                "content": (
                    f"Admission: {admitted:%d %b %Y}. Discharge: {discharged:%d %b %Y}. "
                    f"Primary context: {profile}"
                ),
            },
            {
                "type": "Medication reconciliation",
                "name": f"{case['id'].lower()}_medications.csv",
                "detail": "Current medicines supplied to the context-engineering pipeline.",
                "content": med_list,
            },
            {
                "type": "Follow-up note",
                "name": f"{case['id'].lower()}_care_team_note.txt",
                "detail": "Synthetic care-team note for the demo; no real patient information.",
                "content": (
                    "Use the grounded aftercare workflow to explain only cited guidance and "
                    "defer decisions to the care team."
                ),
            },
        ],
    }


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
    return f"Based on the grounded guidelines: {body} <br><span style='color:#98a2b3'>Assistive — always defer to the care team.</span>"
