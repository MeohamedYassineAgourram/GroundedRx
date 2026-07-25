"""Efficiency frontier (Phase 6b) -- the signature CE artifact.

Sweeps the context-token budget and plots accuracy (danger recall & citation faithfulness)
vs the average context tokens actually spent, for two pipelines:

  - engineered  = CE_FULL config, token budget swept (retrieval + compression + ordering)
  - naive RAG   = baseline config (raw dump), same budget sweep (just truncates the dump)

The money finding: the engineered pipeline reaches near-max accuracy far to the LEFT (cheap)
of naive RAG, which stays inaccurate no matter how many tokens it spends. Saves frontier.png.

    python eval/frontier.py --mock
    python eval/frontier.py --base-url http://localhost:11434/v1 --model gemma4:e4b
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_run
from eval.run_eval import load_cases
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 250 is the smallest current estimated budget that can fit the explicit safety-pinned block.
# Lower budgets fail loudly rather than silently exceeding a claimed hard cap.
BUDGETS = [250, 300, 400, 600, 900, None]  # None = no cap (dump-all for naive)


def _point(cfg, cases, corpus, client):
    results = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
    agg = score_run(results, cases, corpus)
    return agg


def sweep(base_cfg, cases, corpus, client):
    xs, recall, faith = [], [], []
    for b in BUDGETS:
        cfg = replace(base_cfg, token_budget=b)
        agg = _point(cfg, cases, corpus, client)
        xs.append(agg["avg_ctx_tokens"])
        recall.append(agg["danger_recall"] * 100)
        faith.append(agg["citation_faithfulness"] * 100)
    return xs, recall, faith


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "eval_cases.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "slides", "frontier.png"))
    args = ap.parse_args()
    if not args.mock and (not args.base_url or not args.model):
        ap.error("provide --mock, or both --base-url and --model")

    corpus = GuidelineCorpus.load(args.guidelines)
    cases = load_cases(args.cases)
    client = make_client(args.mock, base_url=args.base_url, model=args.model, schema=PLAN_SCHEMA)

    # Do not use optional re-grounding here: it can add a second model generation, so comparing
    # it as if it had one-pass cost would be misleading. The frontier is CE_FULL only.
    eng_x, eng_r, eng_f = sweep(pipeline.CE_FULL, cases, corpus, client)
    naive_x, naive_r, naive_f = sweep(pipeline.BASELINE, cases, corpus, client)

    print("\nEFFICIENCY FRONTIER (accuracy vs avg context tokens)\n")
    print(f"{'budget':>8}  {'engineered: tok/recall/faith':>34}  {'naive: tok/recall/faith':>30}")
    for i, b in enumerate(BUDGETS):
        blab = "dump-all" if b is None else str(b)
        print(f"{blab:>8}  {eng_x[i]:>8.0f} {eng_r[i]:>6.0f}% {eng_f[i]:>6.0f}%          "
              f"{naive_x[i]:>8.0f} {naive_r[i]:>6.0f}% {naive_f[i]:>6.0f}%")

    # Headline numbers.
    best_eng = min((x for x, r in zip(eng_x, eng_r) if r >= 99), default=eng_x[-1])
    naive_best_r = max(naive_r)
    print(f"\nEngineered reaches ~100% danger recall at ~{best_eng:.0f} ctx tokens.")
    print(f"Naive RAG tops out at {naive_best_r:.0f}% recall even at {max(naive_x):.0f} tokens.")

    _plot(eng_x, eng_r, eng_f, naive_x, naive_r, naive_f, args.out, mock=args.mock)
    print(f"\nCurve saved -> {args.out}")


def _plot(eng_x, eng_r, eng_f, naive_x, naive_r, naive_f, out, mock):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping PNG)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
    ax.set_facecolor("#fbfcff")
    ax.plot(eng_x, eng_r, "o-", color="#1f6feb", lw=2.4, label="Engineered — danger recall")
    ax.plot(eng_x, eng_f, "o--", color="#7aa9f7", lw=1.8, label="Engineered — citation faithfulness")
    ax.plot(naive_x, naive_r, "s-", color="#dc6875", lw=2.2, label="Naive RAG — danger recall")
    ax.plot(naive_x, naive_f, "s--", color="#e9adb6", lw=1.6, label="Naive RAG — citation faithfulness")
    ax.set_xlabel("Average context tokens (lower = cheaper)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 105)
    title = "GroundedRx efficiency frontier"
    if mock:
        title += "  (MOCK — illustrative)"
    ax.set_title(title, color="#172033", fontweight="bold")
    ax.legend(fontsize=8, loc="center right", frameon=False)
    ax.grid(color="#e8eef8", alpha=1)
    ax.tick_params(colors="#718096")
    ax.xaxis.label.set_color("#637084")
    ax.yaxis.label.set_color("#637084")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dfe7f2")
    ax.spines["bottom"].set_color("#dfe7f2")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
