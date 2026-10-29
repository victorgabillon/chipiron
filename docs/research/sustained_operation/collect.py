"""Summarize retained experiment observations; this script never changes a workspace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.atomic_activation.measure import disk


def numbers(pattern: str, text: str) -> list[float]:
    """Extract one numeric capture from existing worker logs."""
    return [float(v) for v in re.findall(pattern, text)]


def main() -> None:
    """Keep raw evidence separate from concise derived measurements."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    workspace = root / "workspace"
    paths = MorpionBootstrapPaths.from_work_dir(workspace)
    state = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(state.metadata)
    model, _ = committed_evaluator(workspace, state)
    budget = json.loads((root / "budget.json").read_text())
    report = {
        "root": str(root),
        "budget": budget,
        "head": state.generation,
        "model": model,
        "disk": disk(workspace),
        "calls": json.loads((root / "calls.json").read_text()),
    }
    report["capsules"] = [
        {
            "generation": r["generation"],
            "checkpoint": disk(workspace / r["checkpoint_path"]),
            "science": disk((workspace / r["tree_snapshot_path"]).parent),
            "accepted_dataset": r.get("accepted_dataset"),
            "accepted_training": r.get("accepted_training"),
        }
        for r in records
    ]
    workers = {}
    for role in ("growth", "dataset", "training", "reevaluation"):
        log = workspace / "logs" / f"{role}.log"
        text = log.read_text() if log.exists() else ""
        workers[role] = {
            "ownership_held_s": numbers(
                r"\[ownership\] released held_seconds=([\d.]+)", text
            ),
            "ownership_wait_s": numbers(
                r"\[ownership\] acquired wait_seconds=([\d.]+)", text
            ),
            "contention_events": len(re.findall(r"\[ownership\] contention", text)),
            "lock_failures": len(re.findall(r"BlockingIOError:", text)),
            "growth_steps": numbers(r"\[growth\] done steps=(\d+)", text),
            "new_nodes": numbers(r"\[growth\] done steps=\d+ nodes_added=(\d+)", text),
            "useful_growth_s": numbers(
                r"\[growth\] cycle_done elapsed=([\d.]+)s", text
            ),
            "restore_s": numbers(r"\[runtime\] restore_done elapsed=([\d.]+)s", text),
            "checkpoint_s": numbers(
                r"\[checkpoint\] save_done .*?elapsed=([\d.]+)s", text
            ),
            "training_compute_s": numbers(
                r"\[train-stream\] done elapsed=([\d.]+)s", text
            ),
            "reevaluation_publications": len(
                re.findall(r"\[reevaluation-patch\] create_done", text)
            ),
            "reevaluation_inference_s": numbers(
                r"\[reevaluation-progress\].*?elapsed=([\d.]+)s", text
            ),
        }
    report["workers"] = workers
    report["sessions"] = []
    for path in sorted(root.glob("*.groups.json")):
        observed = json.loads(path.read_text())
        session = {
            "label": path.name,
            **{k: v for k, v in observed.items() if k != "processes"},
        }
        session["sampled_cpu_s"] = sum(
            p["cpu_s"] for p in observed["processes"].values()
        )
        first_source, accepted = {}, {}
        for event in observed.get("authority_observations", []):
            elapsed = event["elapsed_s"]
            first_source.setdefault(event["generation"], elapsed)
            for result in event["results"]:
                generation = result["generation"]
                for kind in ("accepted_dataset", "accepted_training"):
                    if kind in result:
                        accepted.setdefault(
                            (generation, kind),
                            {
                                "generation": generation,
                                "kind": kind,
                                "elapsed_s": elapsed,
                                "source_latency_s": elapsed
                                - first_source.get(generation, 0),
                                "reference": result[kind],
                            },
                        )
        session["accepted_latencies"] = list(accepted.values())
        for item in session["accepted_latencies"]:
            dataset = accepted.get((item["generation"], "accepted_dataset"))
            if item["kind"] == "accepted_training" and dataset is not None:
                item["latency_after_dataset_s"] = (
                    item["elapsed_s"] - dataset["elapsed_s"]
                )
        report["sessions"].append(session)
    report["limitations"] = [
        "Sampled CPU can miss exits; GNU time measures waited children separately.",
        "RSS sums count shared pages; peaks and acceptance latencies have 200ms sampling plus observer cost.",
        "Log step/node totals include computations abandoned by shutdown; committed authority determines durable progress.",
        "Historical capsule totals may share model inodes; per-capsule allocations are not additive.",
    ]
    with options.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
