"""Lost-in-the-middle (Phase 6d) -- position robustness.

A required danger-sign "needle" is embedded at fractional positions from start (0.0) to end
(1.0) of a long distractor-padded "haystack" context, and we measure danger recall by position:

  - naive      = raw context in that order (the needle sits where it was placed)
  - engineered = our Layer-6 ordering moves safety-critical (danger-sign) facts to the EDGES,
                 so the needle is never buried

Naive recall sags in the middle; engineered stays flat. Saves lost_in_middle.png.

    python eval/lost_in_middle.py --mock
    python eval/lost_in_middle.py --base-url http://localhost:11434/v1 --model gemma4:e4b
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_client import make_client
from src.prompts import ENGINEERED_SYSTEM, PLAN_SCHEMA
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSITIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _hash01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _build_context(needle, haystack, frac, engineered):
    """Insert `needle` into `haystack` at fractional position `frac`.
    Engineered ordering relocates the danger-sign needle to the front edge."""
    idx = round(frac * len(haystack))
    ctx = haystack[:idx] + [needle] + haystack[idx:]
    if engineered:
        # Layer 6: pull danger-sign passages to the edges (front here).
        ctx = [p for p in ctx if p["type"] == "danger_sign"] + [p for p in ctx if p["type"] != "danger_sign"]
    return ctx


def _mock_found(needle, ctx, engineered, seed=""):
    """Position-sensitive stub reader: recovers facts near the edges, misses the buried middle.
    Averaged over needles/trials this yields the classic U-shaped 'lost in the middle' curve; the
    engineered reorder puts the needle at an edge, so it is found. Real models show the same
    effect for real -- this stub just lets us render the artifact before the weights are warm."""
    pos = ctx.index(needle) / max(1, len(ctx) - 1)
    # U-shape: high at edges (pos~0 or ~1), low in the middle.
    p_found = 1.0 - 0.9 * (1.0 - (2 * pos - 1) ** 2)
    return _hash01(f"{needle['id']}:{pos:.3f}:{engineered}:{seed}") < p_found


def _real_found(needle, ctx, client, engineered):
    """Send the haystack context to the model and check whether the needle fact is reported."""
    context_block = "\n".join(f"[{p['id']}] ({p['type']}) {p['text']}" for p in ctx)
    user = ("CONTEXT passages:\n" + context_block +
            "\n\nList EVERY danger sign present in the CONTEXT as JSON danger_signs with guideline_id.")
    from src.pipeline import PipelineConfig
    cfg = PipelineConfig(schema=True)
    plan = client.generate_plan(ENGINEERED_SYSTEM, user, ctx, cfg, False, {"id": "LITM"})
    needle_terms = {w for w in needle["text"].lower().split() if len(w) > 4}
    for c in plan.get("danger_signs", []):
        if c.get("guideline_id") == needle["id"]:
            return True
        txt = c.get("text", "").lower()
        if needle_terms and len(needle_terms & set(txt.split())) / len(needle_terms) >= 0.4:
            return True
    return False


def recall_by_position(needles, haystack, client, mock, engineered, trials=8):
    """Average recall over needles x shuffled haystacks. More trials -> smoother curve.
    Real mode uses a single trial per needle (each is a model call -> keep it cheap)."""
    if not mock:
        trials = 1
    curve = []
    for frac in POSITIONS:
        hits = total = 0
        for t in range(trials):
            hay = list(haystack)
            random.Random(f"{frac}:{t}").shuffle(hay)
            for needle in needles:
                ctx = _build_context(needle, hay, frac, engineered)
                found = _mock_found(needle, ctx, engineered, seed=str(t)) if mock else _real_found(needle, ctx, client, engineered)
                hits += 1 if found else 0
                total += 1
        curve.append(100.0 * hits / total)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "slides", "lost_in_middle.png"))
    args = ap.parse_args()
    if not args.mock and (not args.base_url or not args.model):
        ap.error("provide --mock, or both --base-url and --model")

    corpus = GuidelineCorpus.load(args.guidelines)
    client = make_client(args.mock, base_url=args.base_url, model=args.model, schema=PLAN_SCHEMA)

    needles = corpus.danger_signs  # each danger sign, in turn, is the needle
    haystack = [p for p in corpus.passages if p["type"] == "distractor"]  # 60 distractors

    naive = recall_by_position(needles, haystack, client, args.mock, engineered=False)
    eng = recall_by_position(needles, haystack, client, args.mock, engineered=True)

    print(f"\nLOST-IN-THE-MIDDLE  |  needle=danger-sign, haystack={len(haystack)} distractors\n")
    print(f"{'position':>9}  {'naive recall':>13}  {'engineered recall':>18}")
    for i, f in enumerate(POSITIONS):
        print(f"{f:>9.1f}  {naive[i]:>12.0f}%  {eng[i]:>17.0f}%")
    mid = POSITIONS.index(0.5)
    print(f"\nNaive recall at the MIDDLE (pos 0.5): {naive[mid]:.0f}%  vs edges ~{naive[0]:.0f}%.")
    print(f"Engineered recall stays ~{min(eng):.0f}-{max(eng):.0f}% across all positions (safety facts kept at edges).")

    _plot(naive, eng, args.out, mock=args.mock)
    print(f"\nCurve saved -> {args.out}")


def _plot(naive, eng, out, mock):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping PNG)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
    ax.set_facecolor("#fbfcff")
    ax.plot(POSITIONS, naive, "s-", color="#dc6875", lw=2.2, label="Naive (raw dump)")
    ax.plot(POSITIONS, eng, "o-", color="#1f6feb", lw=2.4, label="Engineered (ordering to edges)")
    ax.set_xlabel("Position of required danger-sign in context (0=start, 1=end)")
    ax.set_ylabel("Danger-sign recall (%)")
    ax.set_ylim(-5, 105)
    title = "Lost-in-the-middle: position robustness"
    if mock:
        title += "  (MOCK — illustrative)"
    ax.set_title(title, color="#172033", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
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
