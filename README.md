# GroundedRx

**Context engineering, not parameters, makes a 4B SLM safe.**

GroundedRx is a context-engineering pipeline that turns a small **Gemma 4 E4B** model into
one that produces safe, fully-grounded patient discharge / medication instructions — every
claim cited, danger-signs never omitted, hallucinations driven to ~0 — while treating the
context window as a scarce resource and *proving with numbers* that the context engineering
is what did it.

> ⚠️ **Illustrative / synthetic guideline content. Not clinical advice.** Assistive,
> human-in-the-loop; defers to the care team. Privacy-preserving (runs offline on a 4B model).

## The evidence that wins (Phases 5–6)

1. **Additive ablation** — turn layers on one at a time; safety metrics climb.
2. **Leave-one-out ablation** — remove each layer from the full stack; show its marginal value.
3. **Efficiency frontier** — accuracy vs context-token budget (the signature artifact).
4. **Small-beats-big** — engineered E4B ≥ naively-prompted Gemma 4 12B.
5. **Lost-in-the-middle** — engineered context stays robust to fact position.

## Environment

- Apple M5, 24 GB, macOS (arm64). **No CUDA → Ollama serving only** (vLLM not applicable).
- Python **3.12** (Homebrew). Build a venv with it.
- Models: `gemma4:e4b` (SLM under test), `gemma4:12b` (big baseline).

## See the platform (no server)

Open **`GroundedRx.html`** directly in any browser (double-click). It is fully self-contained —
all 30 patients, grounded plans, context graphs, a client-side grounded copilot, and the
evidence artifacts (ablation table + charts) are baked in; nothing is fetched, so it works
offline / in airplane mode. Regenerate it from the pipeline any time:

```bash
python app/build_static.py     # -> GroundedRx.html
```

For the live, model-backed app: `streamlit run app/streamlit_app.py`.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The harness runs WITHOUT a model via a stub that misbehaves un-engineered and behaves engineered:
python eval/run_eval.py --mock

# Real model (after: brew install ollama; ollama serve; ollama pull gemma4:e4b):
python eval/run_eval.py --base-url http://localhost:11434/v1 --model gemma4:e4b
```

## Layout
See `GroundedRx_BUILD_PLAN.md` Section 3. Every context-engineering layer is a toggle so the
ablation can isolate it: retrieval, compression, locked schema, dynamic few-shot, verification,
ordering/budgeting.
