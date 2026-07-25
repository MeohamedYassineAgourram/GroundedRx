# GroundedRx — pitch outline (2 min, 4 slides)

**One-liner:** GroundedRx treats context as a product surface: it assembles a patient-selected
synthetic chart, source passages, citation constraints, and dynamic memory so a Gemma model can
produce an inspectable aftercare draft.

## Slide 1 — The problem

> Small models fail less because we add clever wording and more because we give them the right
> information, in the right form, at the right time.

- Domain: illustrative heart-failure aftercare with a synthetic benchmark.
- Raw context is distractor-heavy; a patient needs clear, sourced next steps.
- GroundedRx is assistive and human-in-the-loop — not a medical device or clinical validation.

## Slide 2 — The context system

Show the live workspace:

1. Select one of three clearly labelled synthetic records.
2. Open the discharge summary, medication reconciliation, and follow-up note.
3. Press **Generate grounded aftercare plan**.
4. Show the exact context ingredients: patient record, current medicines, explicit safety and
   self-care policy, guideline selection, compression, citation schema, held-out exemplar, and
   edge ordering.
5. Show the cited plan and the exact guideline passages the model received.

Emphasize: the deterministic source lookup is labelled as lookup-only; the Generate action is
the actual model call. A provider error remains visible rather than falling back to a mock plan.

## Slide 3 — The evidence

Open the Evidence tab and show the saved provenance artifact, not a hand-made chart.

- Exact Gemma model ID, endpoint, date, sample size, code revision, corpus/case hashes.
- Same synthetic cases across the baseline, additive layers, and leave-one-out configurations.
- Raw response, parsed JSON, shown passages, citations, and provider-reported token usage are
  all stored in `eval/runs/`.
- Read the recorded values from the page; do **not** substitute mock percentages.
- Explain that danger signs and foundational self-care are transparent policy-pinned context,
  not a BM25-retrieval claim.

Suggested sentence:

> “The claim here is reproducible grounding on a synthetic benchmark. You can inspect every
> piece of context the model saw and reproduce this run from the artifact.”

## Slide 4 — Responsibility and next step

- Every identity, file, guideline, and evaluation case is synthetic.
- Cloud mode sends the selected context to the configured provider; only a verified local
  deployment can be described as offline.
- Automated source checks are not clinician validation.
- Next: clinician-reviewed cases, approved deployment, and a real same-case comparison against
  `google/gemma-4-31b-it` before making a size-based claim.

## Model note

- Live demo model: `google/gemma-4-26b-a4b-it`, a mixture-of-experts model with roughly 3.8B
  active parameters per token.
- Do not call it a dense “4B” model.
- Do not say “small beats big” until both models have recorded real outputs for the same selected
  case set.
