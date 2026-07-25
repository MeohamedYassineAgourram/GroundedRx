"""The context-engineering pipeline: retrieve -> compress -> structure -> (re-ground).

Every layer is a toggle (PipelineConfig) so the ablation can isolate it. run_case returns
the final plan plus the assembled prompt (for token accounting) so metrics can measure the
context-token budget each config actually spent.

Design note (context-engineering track): the four CE layers -- retrieval, compression,
schema, few-shot -- plus ordering do the work by MANAGING THE CONTEXT. `reground` is an
optional multi-step self-correction pass (regenerate from the managed context), reported
separately; it is not a filter and is not where the win comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from . import prompts
from .compress import compress_context
from .reground import reground_plan


@dataclass
class PipelineConfig:
    retrieval: bool = False          # Layer 1: external knowledge retrieval / context selection
    compression: bool = False        # Layer 2: distil each chunk to its high-signal core
    schema: bool = False             # Layer 3: locked, citation-per-claim output structure
    fewshot: bool = False            # Layer 4: dynamic exemplar (memory) retrieved per case
    ordering: bool = False           # Layer 6: edge-placement of safety facts (position robustness)
    reground: bool = False           # optional multi-step self-correction (regenerate from context)
    top_k: int = 8
    token_budget: int | None = None  # Layer 6: hard cap on context tokens (frontier sweeps)

    def label(self):
        on = [n for n in ("retrieval", "compression", "schema", "fewshot", "ordering", "reground")
              if getattr(self, n)]
        return "+".join(on) if on else "baseline"


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
# without the multi-step pass masking its contribution.
LOO_LAYERS = ["retrieval", "compression", "schema", "fewshot"]


def est_tokens(text):
    """Rough token estimate (len/4) -- good enough for budget comparisons."""
    return len(text) // 4


def _gold_example():
    """Dynamic few-shot source (Layer 4): one canonical grounded plan to imitate.
    A fuller build picks the nearest exemplar per case by med overlap (dynamic memory)."""
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
    """Layer 6 budgeting: keep danger-signs, then fill the remaining token budget."""
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

    # --- Assemble prompts (Layer 6 ordering applied inside build_context_block) --
    context_block = prompts.build_context_block(pairs, cfg)
    system_prompt = prompts.build_system(cfg)
    user_prompt = prompts.build_user(case, context_block, fewshot_block, cfg)
    ctx_tokens = est_tokens(system_prompt + user_prompt)

    # --- Layer 3: generation (schema-guided when enabled) ----------------------
    plan = client.generate_plan(system_prompt, user_prompt, shown, cfg, cfg.fewshot, case)

    # --- Optional multi-step re-grounding (regenerate from the managed context) --
    plan = reground_plan(plan, corpus, shown, client, system_prompt, user_prompt, cfg, case,
                         enabled=cfg.reground)

    return {"plan": plan, "ctx_tokens": ctx_tokens, "n_passages": len(shown), "config": cfg.label()}


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
