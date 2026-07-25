# GroundedRx — pitch outline (Phase 8 stub, 4 slides / 2 min)

1. **Thesis (1 line):** context engineering, not parameters, makes a 4B SLM safe.
2. **The failure:** baseline Gemma 4 E4B misses **X%** of danger-signs, invents instructions
   (show a real bad output from `run_eval.py`).
3. **The evidence (the win):** leave-one-out ablation (each layer earns its place) + efficiency
   frontier (near-max accuracy at **N% fewer tokens**) + **4B ≥ 12B** + flat lost-in-middle curve.
4. **Live demo + responsibility:** airplane-mode cited plan; assistive, human-in-the-loop,
   defers to clinician; illustrative/synthetic guidelines.

> Every number on a slide must be reproducible by our own code.

_Fill X% / N% from real `eval/` output in Phase 8. Do not hardcode placeholders on slides._
