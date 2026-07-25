"""Record one fast, provenance-backed real CE-FULL patient run.

This is intentionally a live integration check, not an ablation or clinical study.  It is useful
for a time-boxed demo because it stores the exact shown context, raw response, citations, metric
check, provider usage, and input hashes without running the slow raw-context baseline.

    python eval/record_patient_run.py --base-url https://openrouter.ai/api/v1 \
      --model google/gemma-4-26b-a4b-it --case-id HF-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_run
from eval.provenance import build_config_record, run_metadata, write_artifact
from app.ui_helpers import identity, patient_workspace
from src import pipeline
from src.model_client import make_client
from src.prompts import PLAN_SCHEMA, validate_plan
from src.retrieval import GuidelineCorpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cases(path):
    with open(path) as handle:
        return json.load(handle)["cases"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--case-id", default="HF-01")
    parser.add_argument("--guidelines", default=os.path.join(ROOT, "guidelines", "heart_failure.json"))
    parser.add_argument("--cases", default=os.path.join(ROOT, "data", "eval_cases.json"))
    parser.add_argument("--artifacts-dir", default=os.path.join(ROOT, "eval", "runs"))
    args = parser.parse_args()

    corpus = GuidelineCorpus.load(args.guidelines)
    all_cases = load_cases(args.cases)
    case = next((item for item in all_cases if item.get("id") == args.case_id), None)
    if case is None:
        parser.error(f"unknown case id: {args.case_id}")
    # Mirror the live workspace: attach the selected synthetic chart-file text to the model's
    # patient context.  It is retained in the artifact for an inspector to verify.
    index = next(i for i, item in enumerate(all_cases) if item.get("id") == case["id"])
    display = identity(case["id"], case["profile"], tuple(case.get("meds", [])), index)
    chart = patient_workspace(case, display)
    contextual_case = dict(case)
    contextual_case["presentation_profile"] = case["profile"]
    contextual_case["selected_chart_files"] = chart["files"]
    contextual_case["profile"] = (
        f"{case['profile']}\n\nPATIENT-SELECTED CHART CONTEXT (synthetic demo only; not clinical data):\n"
        + "\n".join(f"[{doc['type']} · synthetic demo file]\n{doc['content']}" for doc in chart["files"])
    )
    client = make_client(False, base_url=args.base_url, model=args.model, schema=PLAN_SCHEMA, api_key=args.api_key)
    result = pipeline.run_case(contextual_case, corpus, client, pipeline.CE_FULL)
    valid, errors = validate_plan(result["plan"])
    if not valid:
        raise SystemExit("Real model returned an invalid plan: " + "; ".join(errors))
    aggregate = score_run([result], [contextual_case], corpus)
    provenance = run_metadata(client=client, guidelines_path=args.guidelines, cases_path=args.cases, mode="real")
    provenance.update({
        "experiment": "single_patient_ce_full",
        "case_selection": {"count": 1, "case_ids": [case["id"]], "purpose": "time-boxed live integration check, not comparative evaluation"},
        "additive": [build_config_record("CE-FULL (single recorded patient)", pipeline.CE_FULL, [contextual_case], [result], aggregate)],
        "leave_one_out": [],
    })
    artifact = write_artifact(args.artifacts_dir, provenance, stem="patient-run")
    print(json.dumps({
        "artifact": artifact,
        "model": args.model,
        "case_id": case["id"],
        "danger_recall": aggregate["danger_recall"],
        "unsupported_output": aggregate["hallucination_rate"],
        "citation_faithfulness": aggregate["citation_faithfulness"],
        "provider_usage": result.get("telemetry", {}).get("provider_usage"),
    }, indent=2))


if __name__ == "__main__":
    main()
