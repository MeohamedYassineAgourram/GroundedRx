"""BM25 retrieval over guideline chunks (Layer 1: retrieval / context selection).

Safety rule: danger-sign chunks are ALWAYS included and never pruned.
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

    def retrieve(self, case, top_k=8):
        """Return relevant passages for a case. Danger-signs always included.

        Query = patient profile + prescribed meds. We also force-include the exact
        medication passages for the patient's drugs (safety/completeness), then top up
        with BM25 hits, and ALWAYS append every danger-sign passage.
        """
        selected = {}

        # 1. Exact medication passages for the patient's drugs.
        for p in self.passages:
            if p["type"] == "medication" and p.get("drug") in case.get("meds", []):
                selected[p["id"]] = p

        # 2. BM25 top-k over the query (adds lifestyle + any relevant extras).
        if self._bm25 is not None:
            query = _tok(case.get("profile", "") + " " + " ".join(case.get("meds", [])))
            scores = self._bm25.get_scores(query)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in ranked[:top_k]:
                p = self.passages[i]
                if p["type"] != "distractor":  # prune distractors; that's the point
                    selected[p["id"]] = p

        # 3. Danger-signs ALWAYS included -- safety-critical, never pruned.
        for p in self.danger_signs:
            selected[p["id"]] = p

        return list(selected.values())

    def dump_all(self):
        """Naive baseline: the entire raw corpus, distractors and all."""
        return list(self.passages)
