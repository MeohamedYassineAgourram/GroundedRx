"""Prompt builders (baseline vs engineered) + the locked JSON schema.

Layers touched here:
  - A *shared* JSON envelope keeps every evaluation arm parseable.  This is deliberately
    held constant so a baseline cannot lose merely because free-form prose was parsed as
    JSON.  Layer 3 adds citation binding and grounded output constraints on top of that
    common envelope.
  - Layer 3 (locked schema): PLAN_SCHEMA, enforced via guided/structured decoding.
  - Layer 4 (dynamic few-shot): nearest gold example injected per case.
  - Layer 6 (ordering/budgeting): safety-critical facts placed at the EDGES of context.
"""
from __future__ import annotations

# --- Layer 3: the locked output schema --------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "danger_signs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "guideline_id": {"type": "string"},
                },
                "required": ["text", "guideline_id"],
            },
        },
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string"},
                    "instruction": {"type": "string"},
                    "guideline_id": {"type": "string"},
                },
                "required": ["drug", "instruction", "guideline_id"],
            },
        },
        "lifestyle": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "guideline_id": {"type": "string"},
                },
                "required": ["text", "guideline_id"],
            },
        },
    },
    "required": ["danger_signs", "medications", "lifestyle"],
}

# Keep output shape constant across the ablation.  Without this, a free-form baseline is
# parsed as an empty plan while the engineered arm gets structured decoding, which measures
# a parser mismatch rather than context engineering.  Empty guideline ids are permitted in
# the baseline; the ``schema`` layer is what requires them to be meaningful source bindings.
COMMON_OUTPUT_CONTRACT = (
    "Return exactly one JSON object and no Markdown. It must have arrays named "
    "danger_signs, medications, and lifestyle. Each danger sign/lifestyle item must contain "
    "text and guideline_id; each medication item must contain drug, instruction, and "
    "guideline_id. Use an empty string for guideline_id when no source is cited. "
    "Use [] for a section with no applicable items."
)

BASELINE_SYSTEM = (
    "You are a helpful assistant. Write clear discharge instructions for the patient, "
    "covering danger signs to watch for, their medications, and lifestyle advice. "
    "Do not claim certainty about facts that are not in the supplied reference material."
)

ENGINEERED_SYSTEM = (
    "You are a clinical aftercare assistant. You may ONLY state facts present in the provided "
    "CONTEXT passages. Every claim MUST cite its guideline_id. Return JSON matching the schema "
    "and nothing else. `danger_signs` must include EVERY danger-sign passage present in the "
    "context and must never be empty. `lifestyle` must include EVERY lifestyle passage present "
    "in the context. Include every medication passage that matches the patient's listed "
    "medications, preserving any timing or caution in that passage. Do NOT add any medication "
    "that is not in the patient's listed medications. If a fact is not in the context, omit it. "
    "This is assistive and defers to the care team."
)


def build_system(cfg):
    # Retrieval/compression should change only the supplied context.  Turning on retrieval must
    # not silently add a stronger grounding instruction, otherwise its ablation is confounded.
    # The schema layer owns citation binding + source-only generation constraints.
    base = ENGINEERED_SYSTEM if (cfg.schema or cfg.reground) else BASELINE_SYSTEM
    return base + "\n\nOUTPUT CONTRACT: " + COMMON_OUTPUT_CONTRACT


def order_context_pairs(pairs, enabled):
    """Layer 6: place safety-critical (danger-sign) passages at the edges, not buried."""
    if not enabled:
        return pairs
    danger = [(p, t) for (p, t) in pairs if p["type"] == "danger_sign"]
    other = [(p, t) for (p, t) in pairs if p["type"] != "danger_sign"]
    if not danger:
        return pairs
    head = danger[: (len(danger) + 1) // 2]
    tail = danger[(len(danger) + 1) // 2 :]
    return head + other + tail


def build_context_block(compressed_pairs, cfg):
    pairs = order_context_pairs(compressed_pairs, cfg.ordering)
    lines = []
    for p, text in pairs:
        lines.append(render_context_passage(p, text))
    return "\n".join(lines)


def render_context_passage(passage, text):
    """The exact serialized representation that is counted and shown to the model."""
    return f"[{passage['id']}] ({passage['type']}) {text}"


def build_fewshot_block(example):
    """Layer 4: one *held-out training* example rendered as an input->output demo."""
    if not example:
        return ""
    import json

    return (
        f"EXAMPLE {example.get('id', 'TRAINING-EXAMPLE')} (held-out training memory; "
        "format to imitate, not patient facts to copy):\n"
        f"PATIENT: {example['profile']} MEDS: {', '.join(example['meds'])}\n"
        f"OUTPUT: {json.dumps(example['output'])}\n\n"
    )


def build_user(case, context_block, fewshot_block, cfg):
    parts = []
    if fewshot_block:
        parts.append(fewshot_block)
    # Keep the framing stable across arms. The baseline receives the raw corpus; engineered
    # arms receive selected/compressed/ordered passages. The label itself is not a treatment.
    parts.append("CONTEXT passages:\n" + context_block + "\n")
    parts.append(f"PATIENT: {case['profile']}\nMEDICATIONS: {', '.join(case['meds'])}\n")
    if cfg.schema:
        parts.append(
            "Produce the aftercare plan using the OUTPUT CONTRACT. Every item needs a non-empty "
            "guideline_id from the CONTEXT, and the cited passage must support the item."
        )
    else:
        parts.append(
            "Produce the aftercare plan using the OUTPUT CONTRACT. Citations are optional in this "
            "baseline arm; leave guideline_id empty rather than guessing a source."
        )
    return "\n".join(parts)


def validate_plan(plan):
    """Lightweight check that a plan conforms to PLAN_SCHEMA. Returns (ok, [errors]).

    Mirrors the locked schema (Layer 3): required keys, danger_signs non-empty, each item
    carries its required fields including a guideline_id. Used for the Phase 3 DONE check
    ('schema-valid JSON for a case, per config') without pulling in a jsonschema dependency.
    """
    errors = []
    if not isinstance(plan, dict):
        return False, ["plan is not an object"]
    for key in ("danger_signs", "medications", "lifestyle"):
        if key not in plan or not isinstance(plan[key], list):
            errors.append(f"missing/invalid array: {key}")
    if isinstance(plan.get("danger_signs"), list) and len(plan["danger_signs"]) < 1:
        errors.append("danger_signs must be non-empty (minItems 1)")
    fields = {
        "danger_signs": ("text", "guideline_id"),
        "medications": ("drug", "instruction", "guideline_id"),
        "lifestyle": ("text", "guideline_id"),
    }
    for key, req in fields.items():
        for i, item in enumerate(plan.get(key, []) or []):
            if not isinstance(item, dict):
                errors.append(f"{key}[{i}] is not an object")
                continue
            for f in req:
                if f not in item:
                    errors.append(f"{key}[{i}] missing field: {f}")
    return (len(errors) == 0), errors
