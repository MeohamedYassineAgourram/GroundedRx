# GroundedRx — pitch outline (2 min, 4 slides)

**One-liner:** Context engineering — not parameters — makes a small language model produce safe,
fully-grounded patient aftercare. We treat the context window as a scarce resource and prove,
with an ablation, that every layer earns its place.

---

### Slide 1 — Thesis
> A small model + engineered context ≥ a bigger model prompted naively — and it's private,
> offline, and every claim is cited.
- Domain: heart-failure discharge / medication instructions (illustrative, synthetic guidelines).
- Safety framing: assistive, human-in-the-loop, defers to the care team.

### Slide 2 — The failure (why it's hard)
- A naively-prompted model, given a bloated raw context (14 real passages + **60 distractors**):
  - **misses danger-signs** (lost in the middle),
  - **invents medications** (copies distractor drugs / hallucinates),
  - **cites nothing**.
- Show a real bad baseline output.

### Slide 3 — The evidence (the win) — from `eval/run_eval.py`
Context engineering ALONE (no post-hoc filter) on the 30-case harness:

| | danger recall | hallucination | citation faithfulness | ctx tokens |
|---|---|---|---|---|
| baseline (naive dump) | 50% | 100% | 0% | 2685 |
| **context-engineered** | **100%** | **10%** | **~100%** | **615 (−77%)** |

- **Leave-one-out**: `retrieval` is the dominant lever (removing it: recall→50%, halluc→100%,
  tokens→1675). Schema owns citations; few-shot owns invention-suppression.
- **Efficiency frontier**: near-max accuracy at a fraction of naive RAG's tokens.
- **Lost-in-the-middle**: naive sags to ~8% mid-context; engineered ordering stays flat.
- The optional multi-step re-grounding closes the last 10%→0% — reported separately; **the win
  is context management, not a filter** (see `CONTEXT_ENGINEERING.md`).

### Slide 4 — Live demo + responsibility
- Dynamic dashboard: pick a patient → grounded aftercare plan, every line cited; context graph
  shows retrieved chunks vs pruned distractors; grounded copilot.
- Runs offline (privacy-preserving) and **live on the real open Gemma 4** model.
- Honest, sourced, human-in-the-loop; illustrative synthetic data.

---

## Model note (be transparent)
- Intended SLM = **Gemma 4 E4B** (~4B). It is **not available on Google AI Studio's API**
  (only `gemma-4-26b-a4b-it` — a MoE with ~3.8B **active** params — and `gemma-4-31b-it`).
- Live demo therefore runs on **`gemma-4-26b-a4b-it`** (efficient MoE, ~3.8B active) as the SLM
  proxy; the dense **31B** is the naive big baseline. The context-engineering thesis is
  model-agnostic — the ablation is the proof.
- The evaluation **harness numbers** above are reproducible offline (`--mock`) and via the real
  model with the CLI (`run_eval.py --base-url … --model … --api-key …`); the app's evidence
  panel uses the harness so it stays instant and offline during the pitch.

> Every number on a slide must be reproducible by our own code.
