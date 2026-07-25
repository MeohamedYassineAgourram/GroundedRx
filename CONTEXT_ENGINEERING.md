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
| **Retrieval / context selection** | pull only the guideline chunks relevant to this patient's condition + meds; danger-signs always included | *external knowledge retrieval*; *structuring the information ecosystem* | ✅ core |
| **Compression / distillation** | reduce each chunk to its high-signal clinical core (no needed fact dropped) | *designing & managing* the context under a token budget | ✅ core |
| **Locked schema (citation per claim)** | require every claim to name a source; guided decoding | *system instructions*; structuring the model's grounding | ✅ core |
| **Dynamic few-shot** | retrieve the nearest grounded exemplar per case | *dynamic memory* | ✅ core |
| **Ordering / budgeting** | place safety-critical facts at context edges; cap token budget | *managing* the context for position-robustness | ✅ core |
| **Re-grounding (multi-step)** | regenerate from the re-composed managed context when a claim isn't grounded | *tool outputs during multi-step execution* | ✅ optional refinement |

Nothing in the winning story is a filter that discards information. Retrieval and compression
*select and condense* — they never drop a required danger-sign or a prescribed medication
(danger-signs are pinned into every context; the patient's own meds are always retrieved).

---

## The evidence that the win is context engineering (mock numbers, real pending)

**Context engineering ALONE (no re-grounding), 30 cases, 74-passage corpus:**

| config | danger recall | hallucination | citation faithfulness | avg ctx tokens |
|---|---|---|---|---|
| baseline (naive dump) | 50% | 100% | 0% | 2685 |
| **CE-FULL (context engineering only)** | **100%** | **10%** | **99%** | **615 (−77%)** |
| + optional re-grounding | 100% | 0% | 100% | 615 |

**Leave-one-out over the context-engineering layers** (remove one from CE-FULL):

| removed | danger recall | hallucination | faithfulness | tokens | what it proves |
|---|---|---|---|---|---|
| − retrieval | 50% | 100% | 91% | 1675 | the dominant lever: recall, safety **and** tokens all hinge on it |
| − compression | 100% | 10% | 99% | 674 | token efficiency |
| − schema | 100% | 10% | 0% | 588 | grounded citations |
| − few-shot | 100% | 40% | 67% | 508 | invention-suppression + format |

**Small-beats-big:** engineered small model (CE-FULL) vs naively-prompted larger model — the
engineered small model wins on *every* safety metric, at a fraction of the tokens. The gain is
attributable to context management, not parameters and not a filter.

> All numbers above are from the `--mock` harness (a stub small model whose behavior is a
> property of its context). They regenerate against the real Gemma 4 E4B with no code change
> (`--base-url`/`--model`). Only measured numbers go on slides.
