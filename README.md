# GroundedRx

GroundedRx is a synthetic, reproducible context-engineering prototype for cited heart-failure
aftercare drafts. It tests whether a carefully managed context helps an efficient Gemma model
produce more complete, source-linked output than the same model given a raw, distractor-heavy
context.

It is not a clinical product, clinical validation, or a source of real patient data.

## What is real vs. illustrative

- The patient identities, chart files, guidelines, and evaluation cases are synthetic.
- `MockClient` is a deterministic UI/test fixture only. Its numbers are explicitly illustrative
  and must never be presented as Gemma results.
- A real run uses the selected OpenAI-compatible provider, records the model response, prompts,
  shown context, provider usage, configuration, dataset hashes, and metrics in `eval/runs/`.
- The Evidence page and static export read only recorded **real-model** artifacts. If no artifact
  exists, they deliberately show no benchmark chart.

## Context-engineering stack

1. Patient-specific guideline selection, with explicit safety and foundational self-care policy
   passages tracked separately from BM25 retrieval.
2. Extractive compression that retains medication timing and cautions.
3. Citation-bound structured output.
4. A dynamically selected, held-out training exemplar.
5. Edge ordering for position robustness.
6. Optional, separately reported re-grounding pass.

The real evaluation uses the same parseable JSON envelope in every arm. Metrics check claims
against only the passages shown to the model; they are automated synthetic-benchmark checks, not
clinical safety claims.

## Run the app

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The workspace shows three synthetic patient records. Open the selected files, then press
**Generate grounded aftercare plan**. The action builds context from that record and the
guideline pipeline; it displays the chosen model, latency, usage, and cited passages. A failed
cloud call remains an error — it is never replaced with a mock answer.

The default cloud configuration is OpenRouter-compatible:

```text
base URL: https://openrouter.ai/api/v1
model:    google/gemma-4-26b-a4b-it
```

Set a credential locally in an ignored `.env` file, for example as `GROUNDEDRX_API_KEY`. Cloud
mode sends the selected synthetic context to that provider. Never send real patient information
to an unapproved endpoint.

## Produce real evidence

Run a bounded pilot first. The command below selects six varied synthetic profiles and medication
sets, writes a provenance artifact, and labels the sample size in the output:

```bash
.venv/bin/python eval/run_eval.py \
  --base-url https://openrouter.ai/api/v1 \
  --model google/gemma-4-26b-a4b-it \
  --case-ids HF-01,HF-07,HF-13,HF-19,HF-25,HF-26
```

For a full 30-case experiment, omit `--case-ids`. Review the real artifact before making a
claim; it includes the raw model outputs and should stay out of source control (`eval/runs/` is
ignored).

To exercise the UI and evaluator without a model, run:

```bash
.venv/bin/python eval/run_eval.py --mock
```

That is a fixture check, not evidence about Gemma.

## Export a static snapshot

The standalone page is intentionally generated only from a recorded real artifact:

```bash
.venv/bin/python app/build_static.py --artifact eval/runs/<real-ablation>.json
```

The resulting `GroundedRx.html` is a read-only evidence snapshot: it has no live model call,
contains three synthetic demo records, and includes model/run provenance. It omits unrecorded
frontier and lost-in-the-middle charts rather than presenting mock figures.

## Model framing

`google/gemma-4-26b-a4b-it` is a mixture-of-experts model with roughly 3.8B active parameters
per token, not a dense 4B model. A dense `google/gemma-4-31b-it` comparison is meaningful only
after both models have recorded real artifacts; GroundedRx does not claim that a smaller model
beats it until those results exist.

See [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md) for the methodology and caveats.
