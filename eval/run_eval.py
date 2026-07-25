"""THE TABLE (Phase 5). Additive AND leave-one-out ablations.

    python eval/run_eval.py --mock
    python eval/run_eval.py --base-url http://localhost:11434/v1 --model gemma4:e4b

--mock uses a stub model (no network) that drops danger-signs and invents a drug when
un-engineered, and behaves when engineered -- so this prints a sensible table before the
model is warm. Never break --mock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_run
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cases(path):
    with open(path) as f:
        return json.load(f)["cases"]


def run_config(cfg, cases, corpus, client):
    results = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
    agg = score_run(results, cases, corpus)
    return agg


def _fmt_table(rows, baseline_tokens):
    header = ["config", "danger_recall", "halluc_rate", "citation_faith", "avg_ctx_tok", "Δtok% vs base"]
    widths = [max(len(h), 16) for h in header]
    widths[0] = max(widths[0], max(len(r[0]) for r in rows))
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for name, agg in rows:
        dtok = (agg["avg_ctx_tokens"] - baseline_tokens) / baseline_tokens * 100 if baseline_tokens else 0.0
        cells = [
            name,
            f"{agg['danger_recall']*100:.0f}%",
            f"{agg['hallucination_rate']*100:.0f}%",
            f"{agg['citation_faithfulness']*100:.0f}%",
            f"{agg['avg_ctx_tokens']:.0f}",
            f"{dtok:+.0f}%",
        ]
        out.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the no-network stub model")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)")
    ap.add_argument("--model", default=None, help="model id (e.g. gemma4:e4b)")
    ap.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "eval_cases.json"))
    ap.add_argument("--json-out", default=None, help="optional path to dump raw results as JSON")
    args = ap.parse_args()

    if not args.mock and (not args.base_url or not args.model):
        ap.error("provide --mock, or both --base-url and --model")

    corpus = GuidelineCorpus.load(args.guidelines)
    cases = load_cases(args.cases)
    client = make_client(args.mock, base_url=args.base_url, model=args.model, schema=PLAN_SCHEMA)

    mode = "MOCK (stub model)" if args.mock else f"REAL ({args.model} @ {args.base_url})"
    print(f"\nGroundedRx ablation  |  {mode}  |  {len(cases)} cases, {len(corpus.passages)} corpus passages\n")

    # --- Additive ----------------------------------------------------------
    additive_rows = [(name, run_config(cfg, cases, corpus, client)) for name, cfg in pipeline.additive_configs()]
    base_tokens = additive_rows[0][1]["avg_ctx_tokens"]
    print("=== ADDITIVE ABLATION (turn layers on one at a time) ===")
    print(_fmt_table(additive_rows, base_tokens))

    # --- Leave-one-out -----------------------------------------------------
    loo_rows = [(name, run_config(cfg, cases, corpus, client)) for name, cfg in pipeline.loo_configs()]
    full_tokens = loo_rows[0][1]["avg_ctx_tokens"]
    print("\n=== LEAVE-ONE-OUT ABLATION (remove each layer from FULL) ===")
    print(_fmt_table(loo_rows, base_tokens))
    print("\nRead LOO as: how much each layer was worth (removing it should hurt some metric).")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"additive": {n: a for n, a in additive_rows}, "loo": {n: a for n, a in loo_rows}}, f, indent=2)
        print(f"\nRaw results -> {args.json_out}")


if __name__ == "__main__":
    main()
