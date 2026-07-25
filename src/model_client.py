"""OpenAI-compatible model client (Ollama/vLLM) plus a MockClient.

Both clients implement the same two methods used by the pipeline:

    generate_plan(system_prompt, user_prompt, context_passages, cfg, fewshot_on) -> dict
    verify_claim(claim_text, passage_text) -> bool   # True = SUPPORTED

The real client ignores `context_passages`/`fewshot_on` (those are already baked into the
prompts); they are passed so the MockClient can *simulate* a groundable-but-buggy model
without any network. The mock deliberately drops danger-signs and invents a drug when
un-engineered, and behaves when the engineering layers are on -- so the ablation harness
prints a sensible table before the model is warm. Never break --mock.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _frac(s: str) -> float:
    """Deterministic pseudo-random value in [0, 1) for reproducible mock behavior."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Mock client -- no network. Simulates a small model that is groundable but buggy.
# ---------------------------------------------------------------------------
class MockClient:
    HALLUCINATED_DRUG = "digoxin"  # an unlisted med the "model" likes to invent

    def generate_plan(self, system_prompt, user_prompt, context_passages, cfg, fewshot_on, case):
        danger = [p for p in context_passages if p["type"] == "danger_sign"]
        meds = [p for p in context_passages if p["type"] == "medication"]
        life = [p for p in context_passages if p["type"] == "lifestyle"]

        plan = {"danger_signs": [], "medications": [], "lifestyle": []}

        # --- Danger signs: completeness is driven by the schema layer. ---------
        # schema (guided decoding) forces a complete, cited danger_signs array.
        # few-shot alone recovers *most* completeness but not all and stays uncited.
        # un-engineered: drops ~half, uncited.
        n = len(danger)
        if cfg.schema:
            kept = danger
        elif fewshot_on:
            kept = danger[: max(1, round(0.75 * n))]
        else:
            kept = danger[: max(1, n // 2)]
        for p in kept:
            cid = p["id"] if cfg.schema else ""
            plan["danger_signs"].append({"text": p["text"], "guideline_id": cid})

        # --- Medications: only those actually in context. ---------------------
        for p in meds:
            cid = p["id"] if cfg.schema else ""
            plan["medications"].append(
                {"drug": p["drug"], "instruction": "Take as prescribed by your care team.", "guideline_id": cid}
            )

        # --- HALLUCINATION: always invent an unlisted med with a bogus citation.
        # Verification (which checks the citation) is what removes it.
        plan["medications"].append(
            {
                "drug": self.HALLUCINATED_DRUG,
                "instruction": "Take 0.25 mg daily.",
                "guideline_id": "FAKE-DIG-01" if cfg.schema else "",
            }
        )

        # --- Distractor leakage: without retrieval, a forbidden term from the raw
        # dump gets parroted in as a med. Retrieval prunes distractors -> it vanishes.
        if not cfg.retrieval:
            for term in case.get("forbidden_terms", []):
                if any(term in p["text"].lower() for p in context_passages):
                    plan["medications"].append(
                        {"drug": term, "instruction": "Continue as before.", "guideline_id": ""}
                    )
                    break

        # --- Lifestyle -------------------------------------------------------
        for p in life:
            cid = p["id"] if cfg.schema else ""
            plan["lifestyle"].append({"text": p["text"], "guideline_id": cid})

        # --- Few-shot (Layer 4): without a format exemplar, the model's citations
        # drift -- it drops the guideline_id on ~35% of otherwise-cited claims.
        # With verification on, those uncited claims are later discarded, so a
        # missing few-shot visibly costs recall/faithfulness. This is why the layer
        # earns its place in leave-one-out.
        if cfg.schema and not fewshot_on:
            for section in ("danger_signs", "medications", "lifestyle"):
                for i, claim in enumerate(plan[section]):
                    if claim.get("guideline_id") and _frac(f"{case['id']}:{section}:{i}") < 0.35:
                        claim["guideline_id"] = ""

        return plan

    def verify_claim(self, claim_text, passage_text):
        """SUPPORTED iff the passage exists and shares key terms with the claim."""
        if not passage_text:
            return False
        claim_terms = {w for w in _tokenize(claim_text) if len(w) > 3}
        passage_terms = set(_tokenize(passage_text))
        if not claim_terms:
            return False
        overlap = len(claim_terms & passage_terms) / len(claim_terms)
        return overlap >= 0.3


# ---------------------------------------------------------------------------
# Real client -- OpenAI-compatible /v1/chat/completions (Ollama or vLLM).
# ---------------------------------------------------------------------------
class OpenAIClient:
    def __init__(self, base_url, model, schema=None, timeout=120):
        import requests  # local import so --mock has zero deps

        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.schema = schema
        self.timeout = timeout

    def _chat(self, system_prompt, user_prompt, response_format=None, max_tokens=1024):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            # Ollama accepts a JSON schema here; vLLM uses guided_json (see below).
            payload["response_format"] = response_format
        r = self._requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate_plan(self, system_prompt, user_prompt, context_passages, cfg, fewshot_on, case):
        response_format = None
        if cfg.schema and self.schema is not None:
            # Ollama-style structured output; vLLM would use extra_body={"guided_json": schema}.
            response_format = {"type": "json_schema", "json_schema": {"name": "plan", "schema": self.schema}}
        content = self._chat(system_prompt, user_prompt, response_format=response_format)
        try:
            return _coerce_plan(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            # JSON drift with schema off: salvage what we can, else empty plan.
            return _coerce_plan(_extract_json(content))

    def verify_claim(self, claim_text, passage_text):
        if not passage_text:
            return False
        sys = "You are a strict fact-checker. Answer with exactly one word: SUPPORTED or UNSUPPORTED."
        usr = f"CLAIM:\n{claim_text}\n\nSOURCE PASSAGE:\n{passage_text}\n\nIs the claim fully supported by the passage?"
        ans = self._chat(sys, usr, max_tokens=8).strip().upper()
        return ans.startswith("SUPPORTED")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tokenize(text):
    return [w.strip(".,;:()").lower() for w in str(text).split()]


def _extract_json(content):
    if not content:
        return {}
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _coerce_plan(obj):
    if not isinstance(obj, dict):
        return {"danger_signs": [], "medications": [], "lifestyle": []}
    return {
        "danger_signs": obj.get("danger_signs", []) or [],
        "medications": obj.get("medications", []) or [],
        "lifestyle": obj.get("lifestyle", []) or [],
    }


def make_client(mock, base_url=None, model=None, schema=None):
    if mock:
        return MockClient()
    if not base_url or not model:
        raise ValueError("Real client requires --base-url and --model")
    return OpenAIClient(base_url=base_url, model=model, schema=schema)
