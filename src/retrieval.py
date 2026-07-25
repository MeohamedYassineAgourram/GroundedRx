"""BM25 retrieval over guideline chunks (Layer 1: retrieval / context selection).

Safety rule: danger-sign and foundational self-care chunks are included through explicit
*clinical safety policies*, not presented as BM25 discoveries.  Telemetry keeps policy-pinned
and retrieval-selected passages separate so evaluation claims do not confuse a safety scaffold
with retriever recall.
When retrieval is OFF, the full corpus is dumped (raw, distractor-padded) -- this is the
naive-RAG baseline whose token cost the frontier undercuts.
"""
from __future__ import annotations

import json
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # keep --mock importable before deps are installed
    BM25Okapi = None


def _tok(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


class GuidelineCorpus:
    def __init__(self, passages):
        self.passages = passages
        self.by_id = {p["id"]: p for p in passages}
        self._bm25 = None
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi([_tok(p["text"] + " " + p.get("concept", "") + " " + p.get("drug", "")) for p in passages])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["passages"])

    @property
    def danger_signs(self):
        return [p for p in self.passages if p["type"] == "danger_sign"]

    def retrieve_with_metadata(self, case, top_k=8):
        """Return ``(passages, selection_metadata)`` for one patient.

        ``retrieved_ids`` contains only BM25-selected, non-distractor passages. Exact
        medication records, condition-wide danger signs, and the compact foundational
        self-care checklist are policy-pinned from patient/condition data, never from
        ``required_danger_ids`` (the evaluation key).
        """
        selected = {}
        medication_pinned_ids = []
        retrieved_ids = []
        safety_pinned_ids = []
        self_care_policy_ids = []

        # 1. Exact medication passages for the patient's prescribed drugs. This is a transparent
        # patient-record safety policy, not a retrieval-quality measurement.
        for p in self.passages:
            if p["type"] == "medication" and p.get("drug") in case.get("meds", []):
                selected[p["id"]] = p
                medication_pinned_ids.append(p["id"])

        # 2. BM25 top-k over the query (adds lifestyle + any relevant extras).
        if self._bm25 is not None:
            query = _tok(case.get("profile", "") + " " + " ".join(case.get("meds", [])))
            scores = self._bm25.get_scores(query)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in ranked[:top_k]:
                p = self.passages[i]
                if p["type"] != "distractor":  # prune distractors; that's the point
                    selected[p["id"]] = p
                    retrieved_ids.append(p["id"])

        # 3. Explicit condition-wide safety policy. These are tagged separately so downstream
        # metrics/artifacts can say "safety-pinned plan recall", never "BM25 retrieved it".
        # It deliberately never reads case.required_danger_ids (the ground-truth label).
        for p in self.danger_signs:
            selected[p["id"]] = p
            safety_pinned_ids.append(p["id"])

        # 4. Foundation self-care policy.  Sodium guidance, daily weights, and any prescribed
        # fluid limit are useful for every heart-failure aftercare conversation, yet generic
        # wording means a sparse patient query often will not rank them with BM25.  Pinning this
        # tiny, transparent checklist prevents an otherwise polished plan from omitting the
        # everyday actions patients can actually follow; it is not reported as retrieval gain.
        for p in self.passages:
            if p["type"] == "lifestyle":
                selected[p["id"]] = p
                self_care_policy_ids.append(p["id"])

        return list(selected.values()), {
            "mode": "bm25_plus_explicit_safety_policy",
            "top_k": top_k,
            "retrieved_ids": list(dict.fromkeys(retrieved_ids)),
            "medication_pinned_ids": medication_pinned_ids,
            "safety_pinned_ids": safety_pinned_ids,
            "self_care_policy_ids": self_care_policy_ids,
            "shown_ids": list(selected),
            "note": (
                "Danger-sign and foundational self-care passages are policy-pinned for patient "
                "safety/understandability and must not be interpreted as retrieval discovery."
            ),
        }

    def retrieve(self, case, top_k=8):
        """Backward-compatible passage-only retrieval API."""
        passages, _ = self.retrieve_with_metadata(case, top_k=top_k)
        return passages

    def dump_all(self):
        """Naive baseline: the entire raw corpus, distractors and all."""
        return list(self.passages)
