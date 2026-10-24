"""Explicit serial two-cycle experiment recipe, not a pipeline coordinator.

Run from the pinned checkout: python -m docs.research.e2e_bootstrap.reproduce NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state


def main() -> None:
    """Execute the explicit experiment and stop at its first failed assertion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--endgame", action="store_true")
    options = parser.parse_args()
    destination = options.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "workspace"
    paths = MorpionBootstrapPaths.from_work_dir(root)
    calls = []

    def run(stage: str, *extra: object) -> dict:
        label = f"{len(calls):02d}-{stage}"
        command = [
            sys.executable,
            "-m",
            "docs.research.e2e_bootstrap.worker",
            stage,
            str(root),
            "--label",
            label,
            *map(str, extra),
        ]
        started = time.perf_counter()
        with (destination / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        calls.append({
            "command": command,
            "exit_code": result.returncode,
            "process_wall_s": time.perf_counter() - started,
        })
        (destination / "processes.json").write_text(json.dumps(calls, indent=2) + "\n")
        assert result.returncode == 0, (
            label,
            (destination / f"{label}.log").read_text()[-4000:],
        )
        return json.loads((destination / "evidence" / f"{label}.json").read_text())

    def continuation() -> dict:
        predicted = run("predict-growth")
        run("growth")
        checked = run("check")
        oracle = json.loads(Path(predicted["oracle"]).read_text())
        actual = json.loads(
            (destination / "evidence" / checked["label"] / "payload.json").read_text()
        )
        assert actual == oracle["payload"], "complete continuation checkpoint mismatch"
        assert checked["scientific_hashes"] == oracle["science"]
        return checked

    initialized = run("init", *(["--endgame"] if options.endgame else []))
    checked = run("check")
    m1 = initialized["after"]["model"]
    if options.endgame:
        assert checked["certificates"]["terminal"] > 0
        assert checked["certificates"]["nonterminal_forced"] > 0
        assert checked["root_science"]["direct"] == 42
        assert checked["root_science"]["tree"] == 35
        assert (
            checked["root_science"]["contract"]["certainties"]["direct"] == "ESTIMATE"
        )
        assert checked["root_science"]["contract"]["certainties"]["tree"] == "FORCED"
    continuation()
    activations = []
    for era in (2, 3):
        current = load_bootstrap_run_state(paths.run_state_path)
        generation = current.generation
        run("reevaluation")
        pending = run("check")["pending"]
        run("dataset", "--generation", generation)
        run("restart")
        checked = run("check")
        original_model = checked["after"]["model"]
        trained = run("training", "--generation", generation)
        assert trained["after"]["model"] == original_model
        run("restart")
        run("check")
        request = destination / f"M{era}-request.json"
        run("prepare", "--generation", generation, "--request", request)
        if era == 2:
            # Generated test evidence only. Preserve this exact real accepted state
            # for interrupted/invalid producer scenarios; no synthesized acceptance.
            shutil.copytree(root, destination / "pre-M2")
        activated = run("activate", "--request", request)
        run("restart")
        checked = run("check")
        candidate = json.loads(request.read_text())["candidate"]
        assert activated["after"]["model"] == candidate
        assert activated["after"]["generation"] == generation + 2
        assert (
            activated["after"]["workspace_epoch"]
            == initialized["after"]["workspace_epoch"]
        )
        receipt = checked["receipts"][pending["id"]]
        assert receipt["content_sha256"] == pending["sha256"]
        assert receipt["committed_generation"] == generation + 1
        after = load_bootstrap_run_state(paths.run_state_path)
        assert (
            after.metadata["atomic_reevaluation"]["accepted_cursor"]
            == pending["envelope"]["next_cursor"]
        )
        retry = run("activate", "--request", request)
        assert retry["before"] == retry["after"]
        activations.append(candidate)
        continuation()
    run("reevaluation")
    final = run("check")
    assert final["pending"]["envelope"]["model"] == activations[-1]
    assert len({m["sha256"] for m in [m1, *activations]}) == 3
    # Retrying M2 after M3 must not promote it again.
    retry = run("activate", "--request", destination / "M2-request.json")
    assert retry["before"] == retry["after"]
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "models": [m1, *activations],
                "generations": publication_records(
                    load_bootstrap_run_state(paths.run_state_path).metadata
                ),
                "processes": len(calls),
                "final": final,
                "parent_pid": os.getpid(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
