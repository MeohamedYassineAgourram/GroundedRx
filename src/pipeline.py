"""The context-engineering pipeline: retrieve -> compress -> structure -> (re-ground).

Every layer is a toggle (PipelineConfig) so the ablation can isolate it. ``run_case`` returns
the final plan plus auditable prompt/context telemetry.  Token figures are explicitly labelled
as estimates unless the provider reports exact usage; they are never presented as tokenizer
ground truth.

Design note (context-engineering track): the four CE layers -- retrieval, compression,
schema, few-shot -- plus ordering do the work by MANAGING THE CONTEXT. `reground` is an
optional multi-step self-correction pass (regenerate from the managed context), reported
separately; it is not a filter and is not where the win comes from.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace

from . import prompts
from .compress import compress_context
from .reground import reground_plan


@dataclass
class PipelineConfig:
    retrieval: bool = False          # Layer 1: external knowledge retrieval / context selection
    compression: bool = False        # Layer 2: distil each chunk to its high-signal core
    schema: bool = False             # Layer 3: citation-binding / grounded output constraints
    fewshot: bool = False            # Layer 4: dynamic exemplar (memory) retrieved per case
    ordering: bool = False           # Layer 6: edge-placement of safety facts (position robustness)
    reground: bool = False           # optional multi-step self-correction (regenerate from context)
    top_k: int = 8
    token_budget: int | None = None  # hard cap on *estimated serialized context* tokens

    def label(self):
        on = [n for n in ("retrieval", "compression", "schema", "fewshot", "ordering", "reground")
              if getattr(self, n)]
        return "+".join(on) if on else "baseline"

    def as_dict(self):
        return asdict(self)


BASELINE = PipelineConfig()
# CE_FULL = all context-engineering layers, WITHOUT the optional re-grounding pass. This is the
# config that proves context engineering ALONE drives the win.
CE_FULL = PipelineConfig(retrieval=True, compression=True, schema=True, fewshot=True, ordering=True)
# FULL = CE_FULL plus the optional multi-step re-grounding refinement.
FULL = replace(CE_FULL, reground=True)

# Additive sequence: baseline -> +retrieval -> +compression -> +schema -> +fewshot (= CE core).
# ordering's value is position-robustness (shown in lost_in_middle), so it rides in CE_FULL/FULL
# rather than the standard additive table; reground is shown as a separate optional row.
ADDITIVE_STEPS = ["retrieval", "compression", "schema", "fewshot"]

# Leave-one-out is computed from CE_FULL (re-grounding OFF) so each CE layer is isolated cleanly,
# without the multi-step pass masking its contribution.  Ordering is included: its effect should
# be demonstrated both in the LITM experiment and in the same LOO protocol as other layers.
LOO_LAYERS = ["retrieval", "compression", "schema", "fewshot", "ordering"]


def est_tokens(text):
    """Conservative, model-agnostic token *estimate* for pre-generation budgeting.

    Exact token counts require the specific serving tokenizer and can differ across Gemma
    endpoints.  We therefore retain provider-reported token usage separately and use this
    deterministic estimate only for fair, reproducible context-budget sweeps.
    """
    text = str(text or "")
    if not text:
        return 0
    lexical_units = len(re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE))
    # English BPE tokenization is usually bounded by character density and lexical pieces.
    return max(1, math.ceil(len(text) / 4), math.ceil(lexical_units * 1.12))


FEWSHOT_MEMORY = (
    # These are intentionally separate, hand-reviewed training exemplars—not rows from
    # data/eval_cases.json.  The selector below uses only patient-visible attributes and never
    # the evaluation answer key, which prevents test-set leakage while making the example real
    # dynamic retrieval rather than a single hard-coded prompt fragment.
    {
        "id": "TRAIN-HF-01",
        "split": "train",
        "profile": "68-year-old with chronic heart failure and recent fluid retention.",
        "meds": ["furosemide", "lisinopril"],
        "output": {
            "danger_signs": [{"text": "Rapid weight gain over 2 to 3 days needs urgent clinician contact.", "guideline_id": "HF-DS-01"}],
            "medications": [{"drug": "furosemide", "instruction": "Take it in the morning.", "guideline_id": "HF-MED-furosemide"}],
            "lifestyle": [{"text": "Limit dietary sodium as advised by the care team.", "guideline_id": "HF-LIFE-01"}],
        },
    },
    {
        "id": "TRAIN-HF-02",
        "split": "train",
        "profile": "76-year-old with heart failure taking a beta-blocker and SGLT2 inhibitor.",
        "meds": ["carvedilol", "dapagliflozin"],
        "output": {
            "danger_signs": [{"text": "New or worsening shortness of breath at rest needs urgent care.", "guideline_id": "HF-DS-02"}],
            "medications": [{"drug": "carvedilol", "instruction": "Take with food and do not stop suddenly.", "guideline_id": "HF-MED-carvedilol"}],
            "lifestyle": [{"text": "Record a morning weight each day.", "guideline_id": "HF-LIFE-02"}],
        },
    },
    {
        "id": "TRAIN-HF-03",
        "split": "train",
        "profile": "59-year-old with HFrEF using a potassium-sparing diuretic.",
        "meds": ["spironolactone"],
        "output": {
            "danger_signs": [{"text": "Increasing ankle, leg, or abdominal swelling should be reported promptly.", "guideline_id": "HF-DS-03"}],
            "medications": [{"drug": "spironolactone", "instruction": "Avoid potassium supplements and salt substitutes.", "guideline_id": "HF-MED-spironolactone"}],
            "lifestyle": [{"text": "Follow the fluid-intake limit set by the care team.", "guideline_id": "HF-LIFE-03"}],
        },
    },
)


def select_fewshot_example(case, memory=FEWSHOT_MEMORY):
    """Retrieve the nearest held-out training exemplar using only non-gold case features."""
    case_meds = {str(m).lower() for m in case.get("meds", [])}
    case_terms = set(re.findall(r"[a-z0-9]+", str(case.get("profile", "")).lower()))
    candidates = [
        ex for ex in memory
        if ex.get("split") == "train" and ex.get("source_case_id") != case.get("id")
    ]
    if not candidates:
        return None

    def score(example):
        ex_meds = {str(m).lower() for m in example.get("meds", [])}
        union = case_meds | ex_meds
        med_jaccard = len(case_meds & ex_meds) / len(union) if union else 0.0
        ex_terms = set(re.findall(r"[a-z0-9]+", example.get("profile", "").lower()))
        # Medication match is the clinically useful retrieval signal; profile overlap breaks
        # ties without looking at required_danger_ids or any other evaluation label.
        profile_overlap = len(case_terms & ex_terms) / max(1, len(case_terms | ex_terms))
        return (3.0 * med_jaccard) + profile_overlap

    return max(candidates, key=lambda ex: (score(ex), ex["id"]))


def _pair_tokens(pair):
    passage, text = pair
    return est_tokens(prompts.render_context_passage(passage, text))


def _apply_budget(pairs, budget):
    """Apply a hard cap to the serialized-context *estimate*.

    Safety-pinned passages are intentionally never pruned.  If the requested budget cannot
    fit that explicit policy block, raising is more honest than silently exceeding the cap.
    """
    if budget is None:
        return pairs
    danger = [(p, t) for (p, t) in pairs if p["type"] == "danger_sign"]
    other = [(p, t) for (p, t) in pairs if p["type"] != "danger_sign"]
    kept = list(danger)  # safety facts are never budgeted out
    used = sum(_pair_tokens(pair) for pair in kept)
    if used > budget:
        raise ValueError(
            f"Context budget {budget} cannot fit the {used}-token estimated safety-pinned block. "
            "Increase --token-budget or use a compact, policy-approved safety representation."
        )
    for p, t in other:
        c = _pair_tokens((p, t))
        if used + c > budget:
            continue
        kept.append((p, t))
        used += c
    return kept


def _usage_totals(events):
    """Aggregate provider-reported usage only; absent usage remains ``None`` rather than guessed."""
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {}
    for field in fields:
        values = [e.get("usage", {}).get(field) for e in events if isinstance(e.get("usage"), dict)]
        numeric = [v for v in values if isinstance(v, (int, float))]
        totals[field] = sum(numeric) if numeric else None
    return totals


def run_case(case, corpus, client, cfg):
    """Run one case through the toggled pipeline and return auditable generation telemetry."""
    generation_cursor = client.generation_cursor() if hasattr(client, "generation_cursor") else 0

    # --- Layer 1: retrieval / context selection --------------------------------
    if cfg.retrieval:
        passages, selection = corpus.retrieve_with_metadata(case, top_k=cfg.top_k)
    else:
        passages = corpus.dump_all()
        selection = {
            "mode": "raw_corpus_dump",
            "retrieved_ids": [p["id"] for p in passages],
            "safety_pinned_ids": [],
            "medication_pinned_ids": [],
        }

    # --- Layer 2: compression --------------------------------------------------
    pairs = compress_context(passages, enabled=cfg.compression)

    # --- Layer 6: token budgeting (frontier sweeps set token_budget) -----------
    pairs = _apply_budget(pairs, cfg.token_budget)
    if cfg.token_budget is not None:
        # Budgeting retains danger signs through the same explicit safety policy in every arm,
        # including the raw baseline. Record it so a low-budget frontier point is never
        # misdescribed as pure retrieval output.
        pinned = {p["id"] for p, _ in pairs if p.get("type") == "danger_sign"}
        selection["safety_pinned_ids"] = sorted(set(selection.get("safety_pinned_ids", [])) | pinned)
        selection["budget_safety_policy"] = "danger-sign passages retained before optional context"

    # Apply ordering before both rendering and telemetry; the reported sequence is exactly what
    # the model saw, not an un-ordered approximation.
    ordered_pairs = prompts.order_context_pairs(pairs, cfg.ordering)
    shown = [{**p, "text": t} for p, t in ordered_pairs]

    # --- Layer 4: dynamic few-shot --------------------------------------------
    fewshot_example = select_fewshot_example(case) if cfg.fewshot else None
    fewshot_block = prompts.build_fewshot_block(fewshot_example) if fewshot_example else ""

    # --- Assemble prompts -------------------------------------------------------
    context_block = prompts.build_context_block(ordered_pairs, replace(cfg, ordering=False))
    system_prompt = prompts.build_system(cfg)
    user_prompt = prompts.build_user(case, context_block, fewshot_block, cfg)
    context_tokens_estimate = est_tokens(context_block)
    prompt_tokens_estimate = est_tokens(system_prompt + user_prompt)

    # --- Layer 3: generation (schema-guided when enabled) ----------------------
    plan = client.generate_plan(system_prompt, user_prompt, shown, cfg, cfg.fewshot, case)

    # --- Optional multi-step re-grounding (regenerate from the managed context) --
    plan = reground_plan(plan, corpus, shown, client, system_prompt, user_prompt, cfg, case,
                         enabled=cfg.reground)

    generation_events = (
        client.generation_events_since(generation_cursor)
        if hasattr(client, "generation_events_since") else []
    )
    usage = _usage_totals(generation_events)
    telemetry = {
        "context_tokens_estimate": context_tokens_estimate,
        "prompt_tokens_estimate": prompt_tokens_estimate,
        "provider_usage": usage,
        "generation_count": len(generation_events),
        "generation_latency_ms": sum(float(e.get("latency_ms", 0) or 0) for e in generation_events),
        "selection": selection,
        "fewshot_example_id": fewshot_example.get("id") if fewshot_example else None,
        "fewshot_split": fewshot_example.get("split") if fewshot_example else None,
    }
    return {
        "plan": plan,
        # Legacy key retained for the dashboard.  It is an estimated serialized-context count,
        # not provider-reported total tokens; consumers should prefer telemetry's explicit fields.
        "ctx_tokens": context_tokens_estimate,
        "context_tokens_estimate": context_tokens_estimate,
        "prompt_tokens_estimate": prompt_tokens_estimate,
        "n_passages": len(shown),
        "config": cfg.label(),
        "config_values": cfg.as_dict(),
        "shown_passages": shown,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generation_events": generation_events,
        "telemetry": telemetry,
    }


def additive_configs():
    """Yield (name, cfg) for the additive ablation: the CE layers, then the optional refinement.

    Ends at CE_FULL (context engineering alone), then adds the multi-step re-grounding as a
    clearly-separate final row so its (small) marginal value is visible but not conflated."""
    yield ("baseline", BASELINE)
    acc = {}
    for step in ADDITIVE_STEPS:
        acc[step] = True
        yield ("+" + step, PipelineConfig(**acc))
    yield ("+ordering (=CE-FULL)", CE_FULL)
    yield ("+reground (optional)", FULL)


def loo_configs():
    """Yield (name, cfg) for leave-one-out over the CE layers, computed from CE_FULL."""
    yield ("CE-FULL", CE_FULL)
    for layer in LOO_LAYERS:
        yield ("CE-FULL -" + layer, replace(CE_FULL, **{layer: False}))
