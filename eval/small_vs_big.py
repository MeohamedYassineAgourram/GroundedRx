"""Small-beats-big (Phase 6c).

Compares an ENGINEERED small model (Gemma 4 E4B, FULL config) against a NAIVELY-prompted
larger model (Gemma 4 12B, baseline config) on the same cases. Headline: our 4B + context
engineering >= a ~3x larger model prompted naively -- the win came from context, not size.

    # mock (both sides use the stub; proves the harness + shows the FULL-vs-baseline gap)
    python eval/small_vs_big.py --mock

    # real: engineered E4B vs naive 12B
    python eval/small_vs_big.py \
        --small-url http://localhost:11434/v1 --small-model gemma4:e4b \
        --big-url   http://localhost:11434/v1 --big-model   gemma4:12b
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_run
from eval.run_eval import load_cases
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cfg, cases, corpus, client):
    results = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
    return score_run(results, cases, corpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--small-url", default=None)
    ap.add_argument("--small-model", default=None)
    ap.add_argument("--big-url", default=None)
    ap.add_argument("--big-model", default=None)
    ap.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "eval_cases.json"))
    args = ap.parse_args()

    corpus = GuidelineCorpus.load(args.guidelines)
    cases = load_cases(args.cases)

    if args.mock:
        small = big = make_client(True, schema=PLAN_SCHEMA)
        small_name, big_name = "E4B (mock)", "12B (mock)"
    else:
        if not (args.small_url and args.small_model and args.big_url and args.big_model):
            ap.error("real mode needs --small-url/--small-model and --big-url/--big-model")
        small = make_client(False, base_url=args.small_url, model=args.small_model, schema=PLAN_SCHEMA)
        big = make_client(False, base_url=args.big_url, model=args.big_model, schema=PLAN_SCHEMA)
        small_name, big_name = args.small_model, args.big_model

    # Engineered small uses CE-FULL (context engineering ALONE, no re-grounding filter) so the
    # comparison proves context management -- not a post-hoc pass -- is what beats the bigger model.
    eng = _run(pipeline.CE_FULL, cases, corpus, small)   # engineered small
    naive = _run(pipeline.BASELINE, cases, corpus, big)  # naive big

    print(f"\nSMALL-BEATS-BIG  |  {len(cases)} cases\n")
    print(f"{'metric':>22}  {'ENGINEERED '+small_name:>24}  {'NAIVE '+big_name:>24}")
    rows = [
        ("danger recall", "danger_recall", True),
        ("hallucination rate", "hallucination_rate", False),
        ("citation faithfulness", "citation_faithfulness", True),
        ("avg ctx tokens", "avg_ctx_tokens", None),
    ]
    for label, key, higher_better in rows:
        e, n = eng[key], naive[key]
        if key == "avg_ctx_tokens":
            print(f"{label:>22}  {e:>24.0f}  {n:>24.0f}")
        else:
            print(f"{label:>22}  {e*100:>23.0f}%  {n*100:>23.0f}%")

    if args.mock:
        print("\nSmall-vs-big verdict: NOT APPLICABLE — deterministic mock output is not a model comparison.")
        print("(Mock mode only exercises the evaluation harness. Run two recorded real endpoints/models.)")
    else:
        wins = (eng["danger_recall"] >= naive["danger_recall"]
                and eng["hallucination_rate"] <= naive["hallucination_rate"]
                and eng["citation_faithfulness"] >= naive["citation_faithfulness"])
        verdict = "YES" if wins else "NO"
        print(f"\nEngineered small model matches or beats naive larger model on every safety metric: {verdict}")


if __name__ == "__main__":
    main()
