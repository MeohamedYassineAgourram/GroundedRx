"""Run-artifact helpers for reproducible, inspectable evaluation evidence.

An aggregate table alone is not evidence.  A saved artifact carries model identity, input
corpus/case hashes, exact prompts/context shown, raw model output, parsed plans, and metric
breakdowns.  API keys are deliberately never accepted or serialized here.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run_metadata(*, client, guidelines_path, cases_path, mode, command=None):
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": "groundedrx-eval-run/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            "ILLUSTRATIVE_MOCK_NOT_MODEL_EVIDENCE" if mode == "mock" else "REAL_MODEL_OUTPUTS_RECORDED"
        ),
        "mode": mode,
        "client": client.client_provenance() if hasattr(client, "client_provenance") else {"mode": mode},
        "inputs": {
            "guidelines_path": os.path.abspath(guidelines_path),
            "guidelines_sha256": sha256_file(guidelines_path),
            "cases_path": os.path.abspath(cases_path),
            "cases_sha256": sha256_file(cases_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": _git_revision(root),
            "command": command or list(sys.argv),
        },
        "privacy_note": (
            "This artifact contains prompts, context, and raw model outputs. Do not store real "
            "patient-identifying information in a shared repository or unencrypted location."
        ),
    }


def case_record(case, result, case_score):
    """Make an inspectable per-case record; no credentials are present in pipeline telemetry."""
    return {
        "case_id": case.get("id"),
        "case_input": case,
        "config": result.get("config_values"),
        "context": {
            "shown_passages": result.get("shown_passages", []),
            "selection": result.get("telemetry", {}).get("selection", {}),
        },
        "fewshot": {
            "example_id": result.get("telemetry", {}).get("fewshot_example_id"),
            "split": result.get("telemetry", {}).get("fewshot_split"),
        },
        "prompts": {
            "system": result.get("system_prompt"),
            "user": result.get("user_prompt"),
        },
        "raw_generations": result.get("generation_events", []),
        "parsed_plan": result.get("plan", {}),
        "telemetry": result.get("telemetry", {}),
        "metrics": case_score,
    }


def build_config_record(name, cfg, cases, results, aggregate):
    case_scores = aggregate.get("case_scores", [])
    return {
        "name": name,
        "config": cfg.as_dict() if hasattr(cfg, "as_dict") else str(cfg),
        "aggregate": aggregate,
        "cases": [
            case_record(case, result, score)
            for case, result, score in zip(cases, results, case_scores)
        ],
    }


def write_artifact(run_dir, payload, stem="ablation"):
    """Atomically write a JSON run artifact and return its absolute path."""
    folder = Path(run_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = sha256_json(payload)[:10]
    target = folder / f"{stamp}_{stem}_{suffix}.json"
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return str(target)
