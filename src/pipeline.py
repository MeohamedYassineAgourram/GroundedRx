"""The context-engineering pipeline: retrieve -> compress -> generate(schema) -> verify.

Every layer is a toggle (PipelineConfig) so the ablation can isolate it. run_case returns
the final plan plus the assembled prompt (for token accounting) so metrics can measure the
context-token budget each config actually spent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from . import prompts
from .compress import compress_context
from .verify import verify_plan


@dataclass
class PipelineConfig:
    retrieval: bool = False
    compression: bool = False
    schema: bool = False
    fewshot: bool = False
    verify: bool = False
    ordering: bool = False          # Layer 6: edge-placement of safety facts
    top_k: int = 8
    token_budget: int | None = None  # Layer 6: hard cap on context tokens (frontier)

    def label(self):
        on = [n for n in ("retrieval", "compression", "schema", "fewshot", "verify", "ordering") if getattr(self, n)]
        return "+".join(on) if on else "baseline"


BASELINE = PipelineConfig()
FULL = PipelineConfig(retrieval=True, compression=True, schema=True, fewshot=True, verify=True, ordering=True)

# Additive sequence (Phase 7a): baseline -> +retrieval -> ... -> FULL.
ADDITIVE_STEPS = ["retrieval", "compression", "schema", "fewshot", "verify"]

# Leave-one-out layers (Phase 7a): FULL minus each.
LOO_LAYERS = ["retrieval", "compression", "schema", "fewshot", "verify"]


def est_tokens(text):
    """Rough token estimate (len/4) -- good enough for budget comparisons."""
    return len(text) // 4


def _gold_example():
    """Dynamic few-shot source (Layer 4). One canonical gold plan to imitate.

    Phase 4 can expand this into a small bank and pick the nearest by med overlap.
    """
    return {
        "profile": "70-year-old with heart failure, on diuretic therapy.",
        "meds": ["furosemide", "lisinopril"],
        "output": {
            "danger_signs": [{"text": "Rapid weight gain over 2-3 days.", "guideline_id": "HF-DS-01"}],
            "medications": [
                {"drug": "furosemide", "instruction": "Take in the morning.", "guideline_id": "HF-MED-furosemide"}
            ],
            "lifestyle": [{"text": "Limit dietary sodium.", "guideline_id": "HF-LIFE-01"}],
        },
    }


def _apply_budget(pairs, budget):
    """Layer 6 budgeting: keep danger-signs, then fill remaining token budget."""
    if budget is None:
        return pairs
    danger = [(p, t) for (p, t) in pairs if p["type"] == "danger_sign"]
    other = [(p, t) for (p, t) in pairs if p["type"] != "danger_sign"]
    kept = list(danger)  # safety facts are never budgeted out
    used = sum(est_tokens(t) for _, t in kept)
    for p, t in other:
        c = est_tokens(t)
        if used + c > budget:
            continue
        kept.append((p, t))
        used += c
    return kept


def run_case(case, corpus, client, cfg):
    """Run one case through the (toggled) pipeline. Returns dict with plan + telemetry."""
    # --- Layer 1: retrieval / context selection --------------------------------
    passages = corpus.retrieve(case, top_k=cfg.top_k) if cfg.retrieval else corpus.dump_all()

    # --- Layer 2: compression --------------------------------------------------
    pairs = compress_context(passages, enabled=cfg.compression)

    # --- Layer 6: token budgeting (frontier sweeps set token_budget) -----------
    pairs = _apply_budget(pairs, cfg.token_budget)

    # context passages actually shown to the model (for the mock + telemetry)
    shown = [{**p, "text": t} for p, t in pairs]

    # --- Layer 4: dynamic few-shot --------------------------------------------
    fewshot_block = prompts.build_fewshot_block(_gold_example()) if cfg.fewshot else ""

    # --- Assemble prompts ------------------------------------------------------
    context_block = prompts.build_context_block(pairs, cfg)
    system_prompt = prompts.build_system(cfg)
    user_prompt = prompts.build_user(case, context_block, fewshot_block, cfg)
    ctx_tokens = est_tokens(system_prompt + user_prompt)

    # --- Layer 3: generation (schema-guided when enabled) ----------------------
    plan = client.generate_plan(system_prompt, user_prompt, shown, cfg, cfg.fewshot, case)

    # --- Layer 5: verification -------------------------------------------------
    plan = verify_plan(plan, corpus, client, enabled=cfg.verify)

    return {"plan": plan, "ctx_tokens": ctx_tokens, "n_passages": len(shown), "config": cfg.label()}


def additive_configs():
    """Yield (name, cfg) for the additive ablation."""
    yield ("baseline", BASELINE)
    acc = {}
    for step in ADDITIVE_STEPS:
        acc[step] = True
        # ordering rides along with the final full config only; keep additive to the 5 core layers
        yield ("+" + step, PipelineConfig(**acc))


def loo_configs():
    """Yield (name, cfg) for the leave-one-out ablation (FULL minus each layer)."""
    yield ("FULL", FULL)
    for layer in LOO_LAYERS:
        cfg = replace(FULL, **{layer: False})
        yield ("FULL -" + layer, cfg)
