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

    def generate_plan(self, system_prompt, user_prompt, context_passages, cfg, fewshot_on, case, strict=False):
        """Simulate a small model whose behavior is a PROPERTY OF ITS CONTEXT.

        Recall and hallucination are driven by how the context is engineered -- not by any
        post-hoc filter. Better context selection, grounding instructions, a citation schema
        and a dynamic exemplar make the small model behave; `strict` is the optional
        multi-step re-grounding pass that regenerates from the managed context.
        """
        danger = [p for p in context_passages if p["type"] == "danger_sign"]
        meds = [p for p in context_passages if p["type"] == "medication"]
        life = [p for p in context_passages if p["type"] == "lifestyle"]
        distractor_drugs = [p.get("drug") for p in context_passages
                            if p.get("type") == "distractor" and p.get("drug")]

        # The engineered system instruction ("only state facts present in CONTEXT") is active
        # whenever we have shaped the context -- itself a context-engineering act.
        grounded_instr = cfg.retrieval or cfg.schema or cfg.reground

        plan = {"danger_signs": [], "medications": [], "lifestyle": []}

        # --- Danger-sign recall is driven by CONTEXT MANAGEMENT, not output filtering ----
        # Un-engineered, the required danger signs are buried among 60 distractors in a raw
        # dump -> lost in the middle -> the small model recovers only ~half. Retrieval makes
        # them findable; the schema / dynamic exemplar make the model enumerate them all.
        n = len(danger)
        if not cfg.retrieval:
            frac = 0.5                       # buried in a long raw dump
        elif cfg.schema or fewshot_on:
            frac = 1.0                       # relevant context + listing structure -> complete
        else:
            frac = 0.75                      # relevant context, no listing structure yet
        for p in danger[: max(1, round(frac * n))]:
            plan["danger_signs"].append({"text": p["text"], "guideline_id": p["id"] if cfg.schema else ""})

        # --- Correct medications (only those actually in the managed context) ------------
        for p in meds:
            plan["medications"].append(
                {"drug": p["drug"], "instruction": _first_sentence(p["text"]),
                 "guideline_id": p["id"] if cfg.schema else ""}
            )

        # --- HALLUCINATION is a property of the CONTEXT, reduced by ENGINEERING it -------
        # (a) The model copies a wrong drug that is *physically present* in the context. A raw
        #     dump contains distractor drugs (warfarin, ibuprofen, ...); retrieval removes them,
        #     so there is nothing to copy. Context selection -- NOT a filter on the output.
        if not cfg.retrieval:
            for term in case.get("forbidden_terms", []):
                if term in distractor_drugs:
                    plan["medications"].append(
                        {"drug": term, "instruction": "Continue as before.", "guideline_id": ""}
                    )
                    break
        # (b) A small model also *invents* a habitual drug. Grounding instructions, a
        #     citation-per-claim schema, and a grounded exemplar each suppress this further.
        #     The optional multi-step re-grounding pass (strict) regenerates the residual away.
        invent_prob = 1.0
        if grounded_instr:
            invent_prob = 0.5                # "only use CONTEXT" system instruction
        if cfg.schema:
            invent_prob = 0.3                # must cite a source -> harder to invent
        if fewshot_on:
            invent_prob = 0.1                # grounded exemplar to imitate
        if strict:
            invent_prob = 0.0                # re-grounding: regenerate strictly from context
        if _frac(case["id"] + "invent") < invent_prob:
            # With a schema the model attaches a plausible-but-wrong real citation (mis-grounded);
            # without one it is left uncited. Either way it is a hallucination until context wins.
            plan["medications"].append(
                {"drug": self.HALLUCINATED_DRUG, "instruction": "Take 0.25 mg daily.",
                 "guideline_id": "HF-MED-furosemide" if cfg.schema else ""}
            )

        # --- Lifestyle -------------------------------------------------------
        for p in life:
            plan["lifestyle"].append({"text": p["text"], "guideline_id": p["id"] if cfg.schema else ""})

        # --- Dynamic few-shot: without an exemplar the model's citation format drifts,
        # dropping the guideline_id on ~35% of claims -> lower faithfulness. The re-grounding
        # pass (strict) restores them by regenerating from the managed context.
        if cfg.schema and not fewshot_on and not strict:
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

    def generate_plan(self, system_prompt, user_prompt, context_passages, cfg, fewshot_on, case, strict=False):
        response_format = None
        if cfg.schema and self.schema is not None:
            # Ollama-style structured output; vLLM would use extra_body={"guided_json": schema}.
            response_format = {"type": "json_schema", "json_schema": {"name": "plan", "schema": self.schema}}
        if strict:
            # Multi-step re-grounding: re-emphasize that every claim must be regenerated from,
            # and cite, a CONTEXT passage; drop anything the context does not support.
            system_prompt = system_prompt + (
                "\n\nRE-GROUNDING PASS: regenerate the plan using ONLY the CONTEXT passages. "
                "Every claim must cite a guideline_id that appears in the CONTEXT and whose text "
                "supports the claim. Do not include any medication absent from the patient's list.")
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
def _first_sentence(text):
    import re

    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return parts[0] if parts else str(text)


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
