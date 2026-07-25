"""Reproducible additive and leave-one-out ablations.

    python eval/run_eval.py --mock
    python eval/run_eval.py --base-url http://localhost:11434/v1 --model gemma4:e4b

``--mock`` exercises the harness with a deterministic stub.  It is labelled illustrative and
must never be presented as model evidence.  Real runs save a provenance artifact containing raw
outputs, prompts, shown context, model usage, input hashes, and per-case metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_run
from eval.provenance import build_config_record, run_metadata, write_artifact
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA, validate_plan
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cases(path):
    with open(path) as f:
        return json.load(f)["cases"]


def run_config(cfg, cases, corpus, client, include_results=False):
    results = [pipeline.run_case(c, corpus, client, cfg) for c in cases]
    agg = score_run(results, cases, corpus)
    return (agg, results) if include_results else agg


def _fmt_table(rows, baseline_tokens):
    header = [
        "config", "danger_recall", "unsafe_case", "unsupported_claim", "citation_faith",
        "ctx_tok_est", "Δctx% vs base",
    ]
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
            f"{agg['unsupported_claim_rate']*100:.0f}%",
            f"{agg['citation_faithfulness']*100:.0f}%",
            f"{agg['avg_ctx_tokens']:.0f}",
            f"{dtok:+.0f}%",
        ]
        out.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(out)


def _count_claims(plan):
    return sum(len(plan.get(k, [])) for k in ("danger_signs", "medications", "lifestyle"))


def _inspect_case(case_id, cases, corpus, client):
    """One case through baseline vs context-engineered (CE-FULL): schema-valid per config,
    and the effect of the optional multi-step re-grounding shown separately."""
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        print(f"case {case_id} not found. available: {', '.join(c['id'] for c in cases[:8])} ...")
        return
    print(f"CASE {case['id']}: {case['profile']}")
    print(f"  meds={case['meds']}  forbidden={case['forbidden_terms']}\n")

    for name, cfg in (("baseline", pipeline.BASELINE), ("context-engineered (CE-FULL)", pipeline.CE_FULL)):
        r = pipeline.run_case(case, corpus, client, cfg)
        ok, errs = validate_plan(r["plan"])
        print(f"--- {name} ({cfg.label()}) ---  ctx_tokens={r['ctx_tokens']}  schema_valid={ok}")
        if errs:
            print("    schema errors:", "; ".join(errs[:4]))
        for k in ("danger_signs", "medications", "lifestyle"):
            for item in r["plan"].get(k, []):
                label = item.get("drug") or item.get("text", "")[:60]
                print(f"    [{k[:6]:6}] {str(label)[:60]:60} cite={item.get('guideline_id','') or '—'}")
        print()

    # Show the optional multi-step re-grounding refinement (CE-FULL vs FULL).
    before = pipeline.run_case(case, corpus, client, pipeline.CE_FULL)["plan"]
    after = pipeline.run_case(case, corpus, client, pipeline.FULL)["plan"]
    delta = _count_claims(before) - _count_claims(after)
    print(f"OPTIONAL RE-GROUNDING (multi-step): {_count_claims(before)} claims -> {_count_claims(after)} "
          f"({'no change needed' if delta == 0 else f'{delta} residual claim(s) regenerated away'}). "
          f"The heavy lifting was already done by context engineering.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the no-network stub model")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)")
    ap.add_argument("--model", default=None, help="model id (e.g. gemma4:e4b, google/gemma-4-31b-it)")
    ap.add_argument("--api-key", default=None, help="cloud API key (else uses $GROUNDEDRX_API_KEY / $OPENAI_API_KEY)")
    ap.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "eval_cases.json"))
    ap.add_argument("--case-limit", type=int, default=None,
                    help="run only the first N cases (useful for a bounded real-model pilot; artifact records N)")
    ap.add_argument("--case-ids", default=None,
                    help="comma-separated case ids for a stratified subset; recorded in the provenance artifact")
    ap.add_argument("--json-out", default=None, help="optional path to dump raw results as JSON")
    ap.add_argument(
        "--artifacts-dir", default=os.path.join(ROOT, "eval", "runs"),
        help="directory for timestamped provenance JSON (required for real runs; contains raw prompts/outputs)",
    )
    ap.add_argument("--no-artifacts", action="store_true", help="skip artifact writing (allowed only for --mock)")
    ap.add_argument("--inspect", metavar="CASE_ID", default=None,
                    help="print one case: baseline vs context-engineered, + optional re-grounding effect")
    args = ap.parse_args()

    if not args.mock and (not args.base_url or not args.model):
        ap.error("provide --mock, or both --base-url and --model")
    if not args.mock and args.no_artifacts:
        ap.error("real evaluations must save a provenance artifact; omit --no-artifacts")

    corpus = GuidelineCorpus.load(args.guidelines)
    cases = load_cases(args.cases)
    if args.case_ids:
        requested_ids = [case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()]
        by_id = {case.get("id"): case for case in cases}
        missing = [case_id for case_id in requested_ids if case_id not in by_id]
        if missing:
            ap.error(f"unknown --case-ids: {', '.join(missing)}")
        cases = [by_id[case_id] for case_id in requested_ids]
    if args.case_limit is not None:
        if args.case_limit < 1:
            ap.error("--case-limit must be at least 1")
        cases = cases[:args.case_limit]
        if not cases:
            ap.error("--case-limit selected no cases")
    client = make_client(args.mock, base_url=args.base_url, model=args.model, schema=PLAN_SCHEMA,
                         api_key=args.api_key)

    mode = "MOCK — ILLUSTRATIVE, NOT MODEL EVIDENCE" if args.mock else f"REAL ({args.model} @ {args.base_url})"
    print(f"\nGroundedRx ablation  |  {mode}  |  {len(cases)} cases, {len(corpus.passages)} corpus passages")
    print("Safety note: danger-sign policy passages are pinned explicitly and recorded separately from BM25 retrieval.\n")

    if args.inspect:
        _inspect_case(args.inspect, cases, corpus, client)
        return

    # --- Additive ----------------------------------------------------------
    additive_runs = []
    for name, cfg in pipeline.additive_configs():
        aggregate, results = run_config(cfg, cases, corpus, client, include_results=True)
        additive_runs.append((name, cfg, aggregate, results))
    additive_rows = [(name, aggregate) for name, _, aggregate, _ in additive_runs]
    base_tokens = additive_rows[0][1]["avg_ctx_tokens"]
    print("=== ADDITIVE ABLATION (context-engineering layers, one at a time) ===")
    print(_fmt_table(additive_rows, base_tokens))
    ce_full = next(a for n, a in additive_rows if "CE-FULL" in n)
    base = additive_rows[0][1]
    summary_prefix = "Illustrative mock delta" if args.mock else "Observed real-model delta"
    print(f"\n{summary_prefix} (CE-FULL, no re-grounding): danger recall {base['danger_recall']*100:.0f}%"
          f"->{ce_full['danger_recall']*100:.0f}%, unsafe cases {base['hallucination_rate']*100:.0f}%"
          f"->{ce_full['hallucination_rate']*100:.0f}%, faithfulness {base['citation_faithfulness']*100:.0f}%"
          f"->{ce_full['citation_faithfulness']*100:.0f}% at {(ce_full['avg_ctx_tokens']-base_tokens)/base_tokens*100:+.0f}% estimated context tokens.")

    # --- Leave-one-out -----------------------------------------------------
    loo_runs = []
    for name, cfg in pipeline.loo_configs():
        aggregate, results = run_config(cfg, cases, corpus, client, include_results=True)
        loo_runs.append((name, cfg, aggregate, results))
    loo_rows = [(name, aggregate) for name, _, aggregate, _ in loo_runs]
    print("\n=== LEAVE-ONE-OUT ABLATION (remove each CE layer from CE-FULL) ===")
    print(_fmt_table(loo_rows, base_tokens))
    print("\nRead LOO as: how much each context-engineering layer was worth (removing it hurts some metric).")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"additive": {n: a for n, a in additive_rows}, "loo": {n: a for n, a in loo_rows}}, f, indent=2)
        print(f"\nRaw results -> {args.json_out}")

    if not args.no_artifacts:
        provenance = run_metadata(
            client=client,
            guidelines_path=args.guidelines,
            cases_path=args.cases,
            mode="mock" if args.mock else "real",
        )
        provenance["experiment"] = "additive_and_leave_one_out_ablation"
        provenance["case_selection"] = {
            "count": len(cases),
            "case_ids": [case.get("id") for case in cases],
            "limit_requested": args.case_limit,
            "ids_requested": args.case_ids,
        }
        provenance["additive"] = [
            build_config_record(name, cfg, cases, results, aggregate)
            for name, cfg, aggregate, results in additive_runs
        ]
        provenance["leave_one_out"] = [
            build_config_record(name, cfg, cases, results, aggregate)
            for name, cfg, aggregate, results in loo_runs
        ]
        artifact = write_artifact(args.artifacts_dir, provenance, stem="ablation")
        print(f"\nProvenance artifact -> {artifact}")
        if args.mock:
            print("Artifact status: ILLUSTRATIVE_MOCK_NOT_MODEL_EVIDENCE.")


if __name__ == "__main__":
    main()
