"""Grounded self-correction (optional multi-step layer).

This is NOT a post-hoc filter that deletes claims. It is a second pass of context engineering:
we detect claims that the *current context does not ground*, then re-compose the managed context
(re-emphasising the cited passages) and ask the model to REGENERATE the plan strictly from it.
This is "multi-step execution" over a managed information ecosystem -- the model produces a
better answer because we managed its context across steps, not because we censored its output.

The heavy lifting is done by the upstream context-engineering layers (retrieval, grounding
instructions, citation schema, dynamic exemplar); this pass only closes the small residual a 4B
model still leaves. It is reported separately from the core CE ablation for exactly that reason.
"""
from __future__ import annotations

import re

# Generic instruction words carry no grounding signal -- a claim that only shares "take/daily"
# with a passage is NOT grounded by it. Keying on distinctive terms stops a mis-cited drug
# (e.g. digoxin citing the furosemide passage) from being falsely judged grounded.
_STOP = {"take", "taken", "daily", "morning", "night", "report", "avoid", "prescribed", "care",
         "team", "continue", "before", "after", "dose", "times", "your", "with", "food", "when",
         "this", "that", "from", "into", "each", "used", "using", "help", "keep", "make", "over"}


def _tok(text):
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 3 and w not in _STOP}


def _grounded(claim_text, cid, corpus):
    """A claim is grounded iff it cites a real passage whose text actually supports it,
    measured on distinctive (non-generic) terms."""
    passage = corpus.by_id.get(cid)
    if passage is None:
        return False
    claim_terms = _tok(claim_text)
    if not claim_terms:
        return False
    passage_terms = _tok(passage["text"] + " " + passage.get("drug", "") + " " + passage.get("concept", ""))
    return len(claim_terms & passage_terms) / len(claim_terms) >= 0.3


def _plan_is_grounded(plan, corpus):
    for c in plan.get("danger_signs", []):
        if not _grounded(c.get("text", ""), c.get("guideline_id", ""), corpus):
            return False
    for c in plan.get("medications", []):
        if not _grounded(f"{c.get('drug','')} {c.get('instruction','')}", c.get("guideline_id", ""), corpus):
            return False
    for c in plan.get("lifestyle", []):
        if not _grounded(c.get("text", ""), c.get("guideline_id", ""), corpus):
            return False
    return True


def reground_plan(plan, corpus, context_passages, client, system_prompt, user_prompt, cfg, case, enabled=True):
    """If any claim is not grounded by the managed context, regenerate the plan strictly from
    that context (a second, context-managed generation). Returns the (re)generated plan."""
    if not enabled:
        return plan
    if _plan_is_grounded(plan, corpus):
        return plan  # context already did the job -- no correction needed
    return client.generate_plan(system_prompt, user_prompt, context_passages, cfg, cfg.fewshot, case, strict=True)
