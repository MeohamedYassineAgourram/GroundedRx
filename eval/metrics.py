"""Grounded evaluation metrics.

All grounding checks are performed against the passages *actually shown to the model*, not
against the whole corpus.  This prevents a plan from receiving credit for citing a correct
document that was never present in its context.

Reported safety metrics:

``danger_recall``
    Required danger-sign content correctly expressed and supported by shown context.
``hallucination_rate``
    Fraction of cases with at least one unsupported/unsafe claim (including an unlisted drug,
    an unsupported medication instruction, or a forbidden term).
``unsupported_claim_rate``
    Claim-level counterpart of ``hallucination_rate``.
``citation_faithfulness``
    A cited id must be in shown context *and* support the associated claim.

These deterministic checks are intentionally conservative.  They complement, rather than
replace, clinician review of a real medical evaluation set.
"""
from __future__ import annotations

import re


# Keep clinically meaningful timing and quantity tokens (morning, nightly, daily, etc.) out of
# the stop-list: otherwise an incorrect dose/timing could look falsely grounded through generic
# words such as "take" or "continue".
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "care", "team",
    "should", "would", "could", "their", "they", "them", "then", "when", "where", "while",
    "about", "after", "before", "each", "every", "also", "have", "been", "were", "will",
    "take", "taken", "continue", "report", "avoid", "prescribed", "using", "used", "help",
    "keep", "make", "need", "needs", "urgent", "clinician", "patient",
}
_NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:\.\d+)?")


def _tok(text):
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(word) > 2 and word not in _STOP and not word.isdigit()
    }


def _numbers(text):
    return set(_NUMBER_RE.findall(str(text).lower()))


def _passage_text(passage):
    return " ".join(
        str(passage.get(key, ""))
        for key in ("text", "concept", "drug")
    )


def _support_score(claim_text, passage):
    """Return a conservative lexical support score plus an exact-number compatibility flag."""
    claim_terms = _tok(claim_text)
    if not claim_terms:
        return 0.0, False
    passage_terms = _tok(_passage_text(passage))
    overlap = len(claim_terms & passage_terms) / len(claim_terms)
    numbers_ok = _numbers(claim_text) <= _numbers(_passage_text(passage))
    return overlap, numbers_ok


def _claim_supported(claim_text, passage, *, drug=None, threshold=0.55):
    if not isinstance(passage, dict):
        return False
    if drug:
        # Medication identity is non-negotiable: a claim about digoxin cannot be grounded by a
        # furosemide passage simply because both say "take daily".
        passage_drug = str(passage.get("drug", "")).strip().lower()
        if str(drug).strip().lower() != passage_drug:
            return False
    score, numbers_ok = _support_score(claim_text, passage)
    return numbers_ok and score >= threshold


def _shown_by_id(shown_passages, corpus):
    # Backward-compatible fallback for callers that score a hand-built plan.  Pipeline results
    # always include ``shown_passages``; evaluation artifacts expose that fact explicitly.
    passages = shown_passages if shown_passages is not None else corpus.passages
    return {p.get("id"): p for p in passages if isinstance(p, dict) and p.get("id")}


def _best_supported_danger_id(claim_text, required_ids, shown_by_id, corpus, threshold=0.45):
    best_id, best_score = None, 0.0
    for cid in required_ids:
        passage = shown_by_id.get(cid)
        if passage is None:
            continue
        score, numbers_ok = _support_score(claim_text, passage)
        if numbers_ok and score > best_score:
            best_id, best_score = cid, score
    return best_id if best_score >= threshold else None


def danger_recall(plan, case, corpus, shown_passages=None):
    """Required danger-sign recall, grounded to the shown context.

    Citation ids are not trusted on their own.  A claim must semantically match the required
    passage; a wrong citation is handled separately by citation faithfulness.  This still lets
    the citation-free baseline be measured fairly for content recall.
    """
    required = set(case.get("required_danger_ids", []))
    if not required:
        return 1.0
    shown_by_id = _shown_by_id(shown_passages, corpus)
    found = set()
    for claim in plan.get("danger_signs", []) or []:
        if not isinstance(claim, dict):
            continue
        matched = _best_supported_danger_id(claim.get("text", ""), required, shown_by_id, corpus)
        if matched:
            found.add(matched)
    return len(found) / len(required)


def _claim_records(plan):
    """Yield (section, source_id, claim_text, medication_drug_or_none)."""
    for claim in plan.get("danger_signs", []) or []:
        if isinstance(claim, dict):
            yield "danger_signs", claim.get("guideline_id", ""), claim.get("text", ""), None
    for claim in plan.get("medications", []) or []:
        if isinstance(claim, dict):
            drug = str(claim.get("drug", ""))
            yield "medications", claim.get("guideline_id", ""), f"{drug} {claim.get('instruction', '')}", drug
    for claim in plan.get("lifestyle", []) or []:
        if isinstance(claim, dict):
            yield "lifestyle", claim.get("guideline_id", ""), claim.get("text", ""), None


def _candidate_passages(section, drug, shown_by_id):
    passages = list(shown_by_id.values())
    if section == "medications":
        return [p for p in passages if p.get("type") == "medication" and p.get("drug", "").lower() == str(drug).lower()]
    if section == "danger_signs":
        return [p for p in passages if p.get("type") == "danger_sign"]
    if section == "lifestyle":
        return [p for p in passages if p.get("type") == "lifestyle"]
    return []


def _is_grounded_record(section, cid, text, drug, shown_by_id):
    """Check an explicit citation when present, otherwise seek a shown compatible passage."""
    if cid:
        return _claim_supported(text, shown_by_id.get(cid), drug=drug)
    return any(_claim_supported(text, passage, drug=drug, threshold=0.50)
               for passage in _candidate_passages(section, drug, shown_by_id))


def unsupported_claims(plan, case, corpus, shown_passages=None):
    """Return machine-readable reasons for claims that cannot be justified by shown context."""
    shown_by_id = _shown_by_id(shown_passages, corpus)
    allowed_drugs = {str(d).lower() for d in case.get("allowed_med_drugs", case.get("meds", []))}
    forbidden = {str(t).lower() for t in case.get("forbidden_terms", [])}
    reasons = []

    for section, cid, text, drug in _claim_records(plan):
        blob = str(text).lower()
        if any(term in blob for term in forbidden):
            reasons.append({"section": section, "claim": text, "reason": "forbidden_term"})
            continue
        if section == "medications" and (not drug or drug.lower() not in allowed_drugs):
            reasons.append({"section": section, "claim": text, "reason": "unlisted_medication"})
            continue
        if not _is_grounded_record(section, cid, text, drug, shown_by_id):
            reasons.append({"section": section, "claim": text, "reason": "not_supported_by_shown_context"})
    return reasons


def has_hallucination(plan, case, corpus, shown_passages=None):
    return bool(unsupported_claims(plan, case, corpus, shown_passages))


def citation_faithfulness(plan, corpus, shown_passages=None):
    """Fraction of output claims with a non-empty, shown, semantically supporting citation."""
    shown_by_id = _shown_by_id(shown_passages, corpus)
    claims = list(_claim_records(plan))
    if not claims:
        return 0.0
    faithful = sum(
        1 for section, cid, text, drug in claims
        if bool(cid) and _claim_supported(text, shown_by_id.get(cid), drug=drug)
    )
    return faithful / len(claims)


def citation_coverage(plan):
    claims = list(_claim_records(plan))
    return (sum(1 for _, cid, _, _ in claims if cid) / len(claims)) if claims else 0.0


def score_case(result, case, corpus):
    shown = result.get("shown_passages")
    plan = result.get("plan", {})
    unsupported = unsupported_claims(plan, case, corpus, shown)
    claims = list(_claim_records(plan))
    return {
        "case_id": case.get("id"),
        "danger_recall": danger_recall(plan, case, corpus, shown),
        "has_hallucination": bool(unsupported),
        "unsupported_claim_rate": len(unsupported) / len(claims) if claims else 0.0,
        "citation_faithfulness": citation_faithfulness(plan, corpus, shown),
        "citation_coverage": citation_coverage(plan),
        "unsupported_claims": unsupported,
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _mean_present(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return _mean(values)


def score_run(results, cases, corpus):
    """Score aligned pipeline results and expose both estimated and provider-reported tokens."""
    if len(results) != len(cases):
        raise ValueError(f"results/cases length mismatch: {len(results)} != {len(cases)}")
    case_scores = [score_case(result, case, corpus) for result, case in zip(results, cases)]
    provider_usage = [result.get("telemetry", {}).get("provider_usage", {}) or {} for result in results]
    context_tokens = [result.get("context_tokens_estimate", result.get("ctx_tokens", 0)) for result in results]
    prompt_tokens = [result.get("prompt_tokens_estimate") for result in results]
    return {
        "danger_recall": _mean(score["danger_recall"] for score in case_scores),
        "hallucination_rate": _mean(1.0 if score["has_hallucination"] else 0.0 for score in case_scores),
        "unsupported_claim_rate": _mean(score["unsupported_claim_rate"] for score in case_scores),
        "citation_faithfulness": _mean(score["citation_faithfulness"] for score in case_scores),
        "citation_coverage": _mean(score["citation_coverage"] for score in case_scores),
        # Legacy key retained for current UI/eval table consumers.  It is explicitly estimated.
        "avg_ctx_tokens": _mean(context_tokens),
        "avg_context_tokens_estimate": _mean(context_tokens),
        "avg_prompt_tokens_estimate": _mean_present(prompt_tokens),
        "avg_provider_prompt_tokens": _mean_present(usage.get("prompt_tokens") for usage in provider_usage),
        "avg_provider_completion_tokens": _mean_present(usage.get("completion_tokens") for usage in provider_usage),
        "avg_provider_total_tokens": _mean_present(usage.get("total_tokens") for usage in provider_usage),
        "avg_generation_count": _mean(result.get("telemetry", {}).get("generation_count", 0) for result in results),
        "case_scores": case_scores,
    }
