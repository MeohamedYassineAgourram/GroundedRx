# Is GroundedRx actually doing context engineering?

**Track definition (given by the track owner):**
> Context engineering is the practice of designing, structuring, and managing the entire
> information ecosystem provided to an LLM — moving beyond static prompt wording to optimize
> dynamic memory, external knowledge retrieval, and tool outputs during multi-step execution.
> It encompasses system instructions, short-term conversational state, and real-time data
> integration.

**Goal for this track:** make a *small* model perform much better than larger models by
**managing the context** — *not* by filtering information out of the output.

---

## The correction we made

Our first design leaned on a **verification filter**: the model generated a plan, then a pass
**deleted** any unsupported claim. That drove hallucination to 0 — but the win came from a
*post-hoc filter*, not from context engineering. Our own leave-one-out proved it: remove the
filter and hallucination went straight back to 100%. That is exactly what the definition excludes.

**What changed:**
1. **Hallucination reduction now comes from managing the context**, not from deleting output.
   A small model hallucinates for two reasons, and we address both by *engineering the context*:
   - it **copies** wrong drugs that are physically present in a bloated context → **retrieval**
     removes the distractor drugs, so there is nothing to copy (context *selection*, not a filter);
   - it **invents** a habitual drug → **grounding system instructions**, a **citation-per-claim
     schema**, and a **dynamic grounded exemplar** progressively suppress invention.
2. The old filter was replaced by an **optional multi-step re-grounding** pass: when the current
   context does not ground a claim, we **re-compose the managed context and regenerate** from it
   (multi-step execution over a managed information ecosystem). It is reported **separately** and
   only closes a small residual — it is explicitly *not* where the win comes from.

---

## Layer-by-layer mapping to the definition

| Layer | What it does | Definition clause it satisfies | CE? |
|---|---|---|---|
| **Retrieval / context selection** | pull relevant guideline chunks; policy-pin danger signs and a small foundational self-care checklist, each reported separately from BM25 | *external knowledge retrieval*; *structuring the information ecosystem* | ✅ core |
| **Compression / distillation** | reduce each chunk to its high-signal clinical core while retaining medication timing and cautions | *designing & managing* the context under a token budget | ✅ core |
| **Locked schema (citation per claim)** | require every claim to name a source; guided decoding | *system instructions*; structuring the model's grounding | ✅ core |
| **Dynamic few-shot** | retrieve the nearest grounded exemplar per case | *dynamic memory* | ✅ core |
| **Ordering / budgeting** | place safety-critical facts at context edges; cap token budget | *managing* the context for position-robustness | ✅ core |
| **Re-grounding (multi-step)** | regenerate from the re-composed managed context when a claim isn't grounded | *tool outputs during multi-step execution* | ✅ optional refinement |

Nothing in the core claim is a post-hoc filter that deletes information. Retrieval and
compression select and condense inputs; the evaluator records the exact shown passages so a
claim cannot receive credit for a source it never saw.

---

## Evaluation protocol and evidence status

The repository includes a deterministic `MockClient` to exercise the interface and evaluator.
Its behavior is intentionally programmed by configuration; **mock output is an illustrative
fixture, never Gemma evidence**. No mock percentage belongs in a pitch or a static export.

Real runs use the same JSON output contract in every arm and save a timestamped provenance
artifact under `eval/runs/` containing:

- exact model / endpoint, provider-reported token usage and latency;
- code revision plus SHA-256 hashes for the synthetic corpus and case set;
- prompts, selected/ordered passages, policy-pinning telemetry, dynamic-example identity, raw
  model responses, parsed plan, and per-case metrics;
- additive and leave-one-out configurations, with optional re-grounding reported separately.

The app’s Evidence page reads only artifacts with
`evidence_status = REAL_MODEL_OUTPUTS_RECORDED`. Before such an artifact exists, it deliberately
shows no chart. The static exporter also refuses mock artifacts.

### What the automatic metrics mean

| Metric | Check | Important limitation |
|---|---|---|
| Danger recall | required danger content matches a passage shown to the model | synthetic requirement set, not a patient-outcome measure |
| Unsupported output | unlisted medication, forbidden term, or claim not supported by shown context | lexical/deterministic support checks still need clinician review |
| Citation faithfulness | non-empty cited ID is shown and supports the particular claim | source support is not clinical correctness |
| Context / provider tokens | deterministic context estimate plus provider-reported usage where available | endpoint tokenizers and hidden reasoning vary |

### Honest judge framing

> “This is a reproducible synthetic benchmark for grounding and context management. The chart
> file, selected context, raw response, citations, and real-run artifact are inspectable. It is
> not clinical validation, and safety-pinned passages are a transparent product policy rather
> than a retrieval-quality result.”

Only make a “small beats big” claim after recorded runs of both models on the same case selection
are available. `google/gemma-4-26b-a4b-it` is an MoE model with roughly 3.8B active parameters
per token; do not describe it as a dense 4B model.
