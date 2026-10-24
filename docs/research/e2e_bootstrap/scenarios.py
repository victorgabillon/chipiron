"""Focused stale/invalid process tests on copies of a genuinely accepted M2 input."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    reevaluation_worker,
)
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from docs.research.e2e_bootstrap.worker import arguments, authority


def paused(root: Path) -> None:
    """Pause a real producer after computation, before its existing final fence."""
    original = reevaluation_worker.proposal_patch

    def pause(*args: Any, **kwargs: Any) -> Any:
        proposal = original(*args, **kwargs)
        (root.parent / "arrived").write_text("M1 computation complete\n")
        deadline = time.monotonic() + 120
        while not (root.parent / "release").exists():
            assert time.monotonic() < deadline, "parent did not release producer"
            time.sleep(0.05)
        return proposal

    with patch.object(reevaluation_worker, "proposal_patch", pause):
        try:
            reevaluation_worker.run_morpion_reevaluation_worker_once(
                arguments(root),
                max_nodes_per_patch=7,
                use_streaming_sharded_reader=True,
                allow_scientific_v2=True,
            )
        except ValueError as error:
            assert "Model changed" in str(error), str(error)
            (root.parent / "producer-refusal.json").write_text(
                json.dumps({"error": str(error)}) + "\n"
            )
        else:
            pytest.fail("M1 producer was admitted after M2")


def main() -> None:
    """Exercise refusal and late publication using separate real processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--producer", action="store_true")
    options = parser.parse_args()
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    if options.producer:
        paused(options.source.resolve())
        return
    source, destination = options.source.resolve(), options.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    request = json.loads((source / "M2-request.json").read_text())
    evidence = {}
    for case in ("stale", "invalid", "late"):
        root = destination / case / "workspace"
        shutil.copytree(source / "pre-M2", root)
        paths = MorpionBootstrapPaths.from_work_dir(root)
        before = authority(paths)
        if case == "late":
            old_projection = paths.pipeline_active_model_path.read_bytes()
            # Preserve original pending bytes as test evidence; only the cloned,
            # unpublished proposal is moved. No committed artifact is changed.
            paths.pipeline_reevaluation_patch_path.rename(
                root.parent / "original-pending.json"
            )
            with (root.parent / "producer.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "docs.research.e2e_bootstrap.scenarios",
                        str(root),
                        "--producer",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 60
                    while not (root.parent / "arrived").exists():
                        assert process.poll() is None, (
                            root.parent / "producer.log"
                        ).read_text()
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                    request_path = root.parent / "request.json"
                    request_path.write_text(json.dumps(request))
                    with (root.parent / "activation.log").open("w") as activation_log:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "docs.research.e2e_bootstrap.worker",
                                "activate",
                                str(root),
                                "--label",
                                "activation",
                                "--request",
                                str(request_path),
                            ],
                            stdout=activation_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    after = authority(paths)
                    (root.parent / "release").touch()
                    assert process.wait(timeout=60) == 0
                    assert authority(paths) == after
                    assert not paths.pipeline_reevaluation_patch_path.exists()
                    paths.pipeline_active_model_path.write_bytes(old_projection)
                    with (root.parent / "stale-projection-restart.log").open(
                        "w"
                    ) as restart_log:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "docs.research.e2e_bootstrap.worker",
                                "restart",
                                str(root),
                                "--label",
                                "stale-projection",
                            ],
                            stdout=restart_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    assert authority(paths) == after
                    evidence["stale_projection"] = {
                        "authority_unchanged": True,
                        "cold_restored_model": after["model"],
                    }
                    evidence[case] = {
                        "before": before,
                        "after": after,
                        "refusal": json.loads(
                            (root.parent / "producer-refusal.json").read_text()
                        ),
                        "producer_pid": process.pid,
                    }
                finally:
                    (root.parent / "release").touch()
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            continue
        attempted = json.loads(json.dumps(request))
        if case == "stale":
            attempted["expected_head"] = "deliberately-stale-synthetic-head"
        else:
            paths.pipeline_reevaluation_patch_path.write_text("{}")
        try:
            activate_model(paths, attempted)
        except (ValueError, KeyError, TypeError) as error:
            assert authority(paths) == before
            evidence[case] = {
                "error_type": type(error).__name__,
                "error": str(error),
                "authority_unchanged": True,
            }
        else:
            pytest.fail(f"{case} was accepted")
    (destination / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
