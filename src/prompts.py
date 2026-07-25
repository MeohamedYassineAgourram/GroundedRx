"""Prompt builders (baseline vs engineered) + the locked JSON schema + verify prompt.

Layers touched here:
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

BASELINE_SYSTEM = (
    "You are a helpful assistant. Write clear discharge instructions for the patient, "
    "covering danger signs to watch for, their medications, and lifestyle advice."
)

ENGINEERED_SYSTEM = (
    "You are a clinical aftercare assistant. You may ONLY state facts present in the provided "
    "CONTEXT passages. Every claim MUST cite its guideline_id. Return JSON matching the schema "
    "and nothing else. `danger_signs` must include EVERY danger-sign passage present in the "
    "context and must never be empty. Do NOT add any medication that is not in the patient's "
    "listed medications. If a fact is not in the context, omit it. This is assistive and defers "
    "to the care team."
)


def build_system(cfg):
    return ENGINEERED_SYSTEM if (cfg.schema or cfg.retrieval or cfg.verify) else BASELINE_SYSTEM


def _order_for_robustness(pairs, enabled):
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
    pairs = _order_for_robustness(compressed_pairs, cfg.ordering)
    lines = []
    for p, text in pairs:
        lines.append(f"[{p['id']}] ({p['type']}) {text}")
    return "\n".join(lines)


def build_fewshot_block(example):
    """Layer 4: one nearest gold example rendered as an input->output demo."""
    if not example:
        return ""
    import json

    return (
        "EXAMPLE (format to imitate exactly):\n"
        f"PATIENT: {example['profile']} MEDS: {', '.join(example['meds'])}\n"
        f"OUTPUT: {json.dumps(example['output'])}\n\n"
    )


def build_user(case, context_block, fewshot_block, cfg):
    parts = []
    if fewshot_block:
        parts.append(fewshot_block)
    if cfg.retrieval or cfg.compression or cfg.schema:
        parts.append("CONTEXT passages:\n" + context_block + "\n")
    else:
        # baseline: raw dump, no citation instruction
        parts.append("Reference material:\n" + context_block + "\n")
    parts.append(f"PATIENT: {case['profile']}\nMEDICATIONS: {', '.join(case['meds'])}\n")
    if cfg.schema:
        parts.append(
            "Produce the aftercare plan as JSON with keys danger_signs, medications, lifestyle. "
            "Every item needs a guideline_id from the CONTEXT."
        )
    else:
        parts.append("Write the aftercare plan.")
    return "\n".join(parts)


VERIFY_SYSTEM = (
    "You are a strict fact-checker. Given a claim and its cited passage, answer with exactly "
    "one word: SUPPORTED or UNSUPPORTED."
)
