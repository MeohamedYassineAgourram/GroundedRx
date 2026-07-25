"""Verification pass (Layer 5): drop unsupported claims -> hallucination toward 0.

For each claim, look up its cited passage and ask the client whether the passage supports
the claim. Claims that cite a non-existent guideline_id, or that the client marks
UNSUPPORTED, are dropped before the final output. This is the backstop that removes the
invented drug the base model likes to add.
"""
from __future__ import annotations


def verify_plan(plan, corpus, client, enabled=True):
    if not enabled:
        return plan

    def _keep(claim, claim_text):
        cid = claim.get("guideline_id", "")
        passage = corpus.by_id.get(cid)
        if passage is None:
            return False  # cites nothing real -> drop
        return client.verify_claim(claim_text, passage["text"])

    return {
        "danger_signs": [c for c in plan.get("danger_signs", []) if _keep(c, c.get("text", ""))],
        "medications": [
            c
            for c in plan.get("medications", [])
            if _keep(c, f"{c.get('drug','')}: {c.get('instruction','')}")
        ],
        "lifestyle": [c for c in plan.get("lifestyle", []) if _keep(c, c.get("text", ""))],
    }
