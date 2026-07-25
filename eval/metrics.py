"""Metrics (Phase 6). Scored automatically per case, then averaged.

  danger_recall        = |found danger ids  intersect required| / |required|   (safety, target 1.0)
  hallucination_rate   = fraction of cases mentioning an unlisted med OR a forbidden term (target 0)
  citation_faithfulness= fraction of claims whose cited guideline_id exists AND whose key terms
                         appear in that chunk                                   (grounding quality)
  avg_ctx_tokens       = mean assembled-prompt tokens                           (efficiency)

Danger recall is measured by matching each output danger-sign back to the guideline passage it
best overlaps -- so it scores completeness whether or not the model cited an id (baseline won't).
"""
from __future__ import annotations

import re


def _tok(text):
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 3}


def _best_danger_id(claim_text, corpus, threshold=0.34):
    """Map an output danger-sign text to the required danger id it best matches."""
    claim_terms = _tok(claim_text)
    if not claim_terms:
        return None
    best_id, best_score = None, 0.0
    for p in corpus.danger_signs:
        pt = _tok(p["text"] + " " + p.get("concept", ""))
        if not pt:
            continue
        score = len(claim_terms & pt) / len(claim_terms)
        if score > best_score:
            best_id, best_score = p["id"], score
    return best_id if best_score >= threshold else None


def danger_recall(plan, case, corpus):
    required = set(case["required_danger_ids"])
    if not required:
        return 1.0
    found = set()
    for c in plan.get("danger_signs", []):
        cid = c.get("guideline_id") or _best_danger_id(c.get("text", ""), corpus)
        if cid in required:
            found.add(cid)
    return len(found & required) / len(required)


def has_hallucination(plan, case, corpus):
    allowed = {d.lower() for d in case.get("allowed_med_drugs", [])}
    forbidden = {t.lower() for t in case.get("forbidden_terms", [])}

    # Any medication whose drug is not in the allowed list.
    for c in plan.get("medications", []):
        drug = str(c.get("drug", "")).lower().strip()
        if drug and drug not in allowed:
            return True

    # Any forbidden term appearing anywhere in the output text.
    blob = " ".join(
        [c.get("text", "") for c in plan.get("danger_signs", [])]
        + [f"{c.get('drug','')} {c.get('instruction','')}" for c in plan.get("medications", [])]
        + [c.get("text", "") for c in plan.get("lifestyle", [])]
    ).lower()
    return any(term in blob for term in forbidden)


def citation_faithfulness(plan, corpus):
    claims = []
    for c in plan.get("danger_signs", []):
        claims.append((c.get("guideline_id", ""), c.get("text", "")))
    for c in plan.get("medications", []):
        claims.append((c.get("guideline_id", ""), f"{c.get('drug','')} {c.get('instruction','')}"))
    for c in plan.get("lifestyle", []):
        claims.append((c.get("guideline_id", ""), c.get("text", "")))
    if not claims:
        return 0.0
    faithful = 0
    for cid, text in claims:
        passage = corpus.by_id.get(cid)
        if passage is None:
            continue  # uncited or bogus id -> not faithful
        claim_terms = _tok(text)
        passage_terms = _tok(passage["text"] + " " + passage.get("drug", "") + " " + passage.get("concept", ""))
        if claim_terms and len(claim_terms & passage_terms) / len(claim_terms) >= 0.25:
            faithful += 1
    return faithful / len(claims)


def score_run(results, cases, corpus):
    """results: list of pipeline.run_case outputs aligned with `cases`. Returns aggregate dict."""
    n = len(cases)
    recall = sum(danger_recall(r["plan"], c, corpus) for r, c in zip(results, cases)) / n
    halluc = sum(1 for r, c in zip(results, cases) if has_hallucination(r["plan"], c, corpus)) / n
    faith = sum(citation_faithfulness(r["plan"], corpus) for r in results) / n
    tokens = sum(r["ctx_tokens"] for r in results) / n
    return {
        "danger_recall": recall,
        "hallucination_rate": halluc,
        "citation_faithfulness": faith,
        "avg_ctx_tokens": tokens,
    }
