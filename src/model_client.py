"""OpenAI-compatible model client (Ollama/vLLM) plus a MockClient.

Both clients implement the same two methods used by the pipeline:

    generate_plan(system_prompt, user_prompt, context_passages, cfg, fewshot_on) -> dict
    verify_claim(claim_text, passage_text) -> bool   # True = SUPPORTED

The real client ignores `context_passages`/`fewshot_on` (those are already baked into the
prompts); they are passed so the MockClient can *simulate* a groundable-but-buggy model
without any network.  Every generation is recorded in-memory with raw output and provider
usage metadata.  ``pipeline.run_case`` persists those records when an evaluation is run.

The mock is deliberately deterministic and illustrative only.  It is useful for exercising
the harness, never as evidence about a Gemma model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass


class _RecordingClient:
    """Small shared provenance interface without changing the plan-returning API.

    The UI expects ``generate_plan`` to return a plain plan dict, so event records live beside
    that return value.  They intentionally contain no Authorization header or API key.
    """

    is_mock = False

    def __init__(self):
        self._generation_events = []

    def generation_cursor(self):
        return len(self._generation_events)

    def generation_events_since(self, cursor):
        return list(self._generation_events[cursor:])

    def _record_generation(self, event):
        self._generation_events.append(event)

    def client_provenance(self):
        return {"mode": "mock" if self.is_mock else "real"}


def _frac(s: str) -> float:
    """Deterministic pseudo-random value in [0, 1) for reproducible mock behavior."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Mock client -- no network. Simulates a small model that is groundable but buggy.
# ---------------------------------------------------------------------------
class MockClient(_RecordingClient):
    HALLUCINATED_DRUG = "digoxin"  # an unlisted med the "model" likes to invent
    is_mock = True

    def __init__(self):
        super().__init__()

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
        # (b) A small model also *invents* a habitual drug. The schema/citation-binding
        #     instruction, and then a grounded exemplar, suppress this further. Retrieval alone
        #     must not secretly receive this stronger instruction or its ablation is confounded.
        #     The optional multi-step re-grounding pass (strict) regenerates the residual away.
        invent_prob = 1.0
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

        # Keep the deterministic raw response so mock runs exercise the same provenance
        # plumbing as real runs.  It is explicitly labelled mock by ``client_provenance``.
        self._record_generation({
            "provider": "deterministic-mock",
            "model": "GroundedRx MockClient",
            "strict": bool(strict),
            "raw_response": json.dumps(plan, ensure_ascii=False, sort_keys=True),
            "parsed_plan": plan,
            "parse_status": "mock_structured",
            "usage": None,
            "latency_ms": 0,
        })
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
class OpenAIClient(_RecordingClient):
    def __init__(self, base_url, model, schema=None, timeout=120, api_key=None, extra_headers=None,
                 provider_preferences=None):
        import requests  # local import so --mock has zero deps

        super().__init__()
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.schema = schema
        self.timeout = timeout
        # API key for cloud providers (OpenRouter / Google AI Studio / Together / Groq / ...).
        # Local Ollama needs none. Falls back to env vars so keys never live in code.
        self.api_key = (
            api_key
            or os.getenv("GROUNDEDRX_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.extra_headers = extra_headers or {}
        # OpenRouter can silently route to a provider that does not honour requested generation
        # parameters unless these are pinned.  Enable its documented strict routing defaults
        # automatically for that endpoint; other OpenAI-compatible servers receive no extra key.
        self.provider_preferences = provider_preferences
        if self.provider_preferences is None and "openrouter.ai" in self.base_url.lower():
            self.provider_preferences = {"require_parameters": True, "allow_fallbacks": False}
        self._last_chat_metadata = None

    def client_provenance(self):
        # Keep endpoint/model identity for reproducibility, but never write credentials.
        return {
            "mode": "real",
            "provider": "openai-compatible",
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout,
            "provider_preferences": self.provider_preferences,
        }

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

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
        if self.provider_preferences:
            payload["provider"] = self.provider_preferences
        last = None
        format_fallback = False
        for attempt in range(5):
            started = time.perf_counter()
            r = self._requests.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers(), timeout=self.timeout
            )
            if r.status_code in (429, 500, 502, 503, 504):  # rate-limited / transient -> back off
                last = r
                time.sleep(min(2 ** attempt * 1.5, 30))
                continue
            # Some OpenAI-compatible gateways expose Gemma but not JSON-schema decoding. The
            # shared prompt contract is still valid JSON guidance, so retry once without the
            # transport feature and record that structured decoding was unavailable.
            error_text = r.text.lower() if r.status_code == 400 else ""
            if (r.status_code == 400 and "response_format" in payload and not format_fallback
                    and any(marker in error_text for marker in ("response_format", "json_schema", "structured output"))):
                payload.pop("response_format", None)
                format_fallback = True
                continue
            r.raise_for_status()
            data = r.json()
            message = data["choices"][0]["message"]
            content = _message_content(message.get("content"))
            self._last_chat_metadata = {
                "response_id": data.get("id"),
                "response_model": data.get("model", self.model),
                "created": data.get("created"),
                "usage": _normalise_usage(data.get("usage")),
                "http_status": r.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "attempt": attempt + 1,
                "structured_output_requested": response_format is not None,
                "structured_output_enforced": response_format is not None and not format_fallback,
            }
            return content
        last.raise_for_status()  # exhausted retries

    def generate_plan(self, system_prompt, user_prompt, context_passages, cfg, fewshot_on, case, strict=False):
        # A common envelope is intentionally enabled for *every* evaluation arm.  The baseline
        # and engineered prompts therefore differ in context management/citation constraints,
        # not in whether a prose response happens to survive JSON parsing.
        response_format = None
        if self.schema is not None:
            # Ollama-style structured output; vLLM would use extra_body={"guided_json": schema}.
            response_format = {"type": "json_schema", "json_schema": {"name": "plan", "schema": self.schema}}
        if strict:
            # Multi-step re-grounding: re-emphasize that every claim must be regenerated from,
            # and cite, a CONTEXT passage; drop anything the context does not support.
            system_prompt = system_prompt + (
                "\n\nRE-GROUNDING PASS: regenerate the plan using ONLY the CONTEXT passages. "
                "Every claim must cite a guideline_id that appears in the CONTEXT and whose text "
                "supports the claim. Do not include any medication absent from the patient's list.")
        # Large budget: reasoning models (e.g. Gemma 4 thinking mode, which can't be disabled)
        # spend hidden tokens before emitting JSON; too small a cap truncates the plan.
        content = self._chat(system_prompt, user_prompt, response_format=response_format, max_tokens=6144)
        parse_status = "valid_json"
        try:
            plan = _coerce_plan(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            # JSON drift with schema off: salvage what we can, else empty plan.
            extracted = _extract_json(content)
            parse_status = "extracted_json" if extracted else "invalid_json"
            plan = _coerce_plan(extracted)
        event = {
            "provider": "openai-compatible",
            "model": self.model,
            "strict": bool(strict),
            "raw_response": content,
            "parsed_plan": plan,
            "parse_status": parse_status,
        }
        event.update(self._last_chat_metadata or {})
        self._record_generation(event)
        return plan

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
def _message_content(content):
    """Normalize OpenAI-compatible content variants to text.

    Some gateways return content as typed parts rather than a string.  Keeping this local
    avoids falsely recording an empty/invalid response when the model did answer.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _normalise_usage(usage):
    """Keep only numeric provider usage fields, if the gateway reports them."""
    if not isinstance(usage, dict):
        return None
    out = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for canonical, keys in aliases.items():
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                out[canonical] = int(value)
                break
    # Preserve provider-specific breakdowns only when they are simple scalar values.
    for key, value in usage.items():
        if key not in out and isinstance(value, (int, float)):
            out[key] = int(value)
    return out or None


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

    def _items(key):
        # Drop malformed array members rather than allowing a string/list to crash metrics.
        value = obj.get(key, []) or []
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    return {
        "danger_signs": _items("danger_signs"),
        "medications": _items("medications"),
        "lifestyle": _items("lifestyle"),
    }


def make_client(mock, base_url=None, model=None, schema=None, api_key=None, extra_headers=None,
                provider_preferences=None):
    if mock:
        return MockClient()
    if not base_url or not model:
        raise ValueError("Real client requires --base-url and --model")
    return OpenAIClient(base_url=base_url, model=model, schema=schema,
                        api_key=api_key, extra_headers=extra_headers,
                        provider_preferences=provider_preferences)
