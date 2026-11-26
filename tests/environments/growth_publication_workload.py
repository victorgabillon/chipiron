"""Process barriers inside real search; no replacement scientific result."""

import json
import os
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import generation_publication as capsules
from chipiron.environments.morpion.bootstrap import launcher
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    growth_receipt_ownership,
    growth_tree_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.learning.torch_runtime import (
    configure_torch_process_threads,
    torch_execution_profile,
)

if TYPE_CHECKING:
    from pathlib import Path


def cold_restore(args: Any, evidence: Path) -> None:
    """Reload authority and restore its actual checkpoint, without a new cycle."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    with growth_tree_ownership(paths), growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        producer = publication_records(state.metadata)[-1]["scientific_identity"][
            "producer"
        ]["identity"]
        runner = AnemoneMorpionSearchRunner(
            AnemoneMorpionSearchRunnerArgs(
                runtime_checkpoint_format=args.runtime_checkpoint_format,
                random_seed=4,
                search_args=default_search_args(rollout=args.search.rollout),
            )
        )
        session = GrowthPatchReceiptSession(
            paths,
            runner,
            durable_generation_publication=True,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=True,
        )
        try:
            state = session.prepare(state, args)
            model, _ = committed_evaluator(paths.work_dir, state)
            assert model is not None and session.publication is not None
            model_path = session.publication.model_pin(
                state, model["path"], model["sha256"]
            )
            runner.load_or_create(
                paths.work_dir / state.latest_runtime_checkpoint_path, model_path
            )
            assert runner.current_tree_size() == state.tree_size_at_last_save
            (evidence / "c72-cold-state.json").write_bytes(
                paths.run_state_path.read_bytes()
            )
            (evidence / "c72-cold-runtime.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "generation": state.generation,
                        "checkpoint": state.latest_runtime_checkpoint_path,
                        "evaluator": model,
                        "restored_nodes": runner.current_tree_size(),
                    },
                    indent=2,
                )
            )
        finally:
            session.discard_runtime()
            assert session.publication is not None
            session.publication.close()


def worker(args: Any, evidence: Path, stage: str, entered: Any, release: Any) -> None:
    """Run a stock entrypoint in its own process group with explicit test events."""
    os.setsid()
    with (evidence / (stage + ".log")).open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        if stage == "cold":
            configure_torch_process_threads(intra_op=1, inter_op=1)
            with torch_execution_profile("strict_math"):
                cold_restore(args, evidence)
            return
        original_grow = AnemoneMorpionSearchRunner.grow
        original_publish = capsules.publish_directory_if_absent

        def relocate(staging: Path, final: Path, work_dir: Path) -> None:
            if final.parent.name != "objects":
                original_publish(staging, final, work_dir)
                return
            (evidence / "c72-relocation-before.json").write_text(
                json.dumps({"staging": str(staging), "final": str(final)})
            )
            entered.set()
            assert release.wait(120), "Collector did not release real rename"
            release.clear()
            original_publish(staging, final, work_dir)
            (evidence / "c72-relocation-after.json").write_text(
                json.dumps({
                    "staging_exists": staging.exists(),
                    "final_exists": final.exists(),
                })
            )
            entered.set()
            assert release.wait(120), "Collector did not finish its scan"

        def grow(self: Any, max_growth_steps: int) -> None:
            runtime = self._require_runtime()
            original_step = type(runtime).step
            paused = False

            def step(instance: Any, *a: Any, **kw: Any) -> Any:
                nonlocal paused
                result = original_step(instance, *a, **kw)
                if instance is runtime and not paused:
                    paused = True
                    (evidence / "c72-growth-paused.json").write_text(
                        json.dumps({
                            "pid": os.getpid(),
                            "at_ns": time.monotonic_ns(),
                            "nodes_after_real_step": self.current_tree_size(),
                        })
                    )
                    entered.set()
                    assert release.wait(120), "Controller did not release real Growth"
                return result

            # The real Anemone step has completed; the real grow loop is still
            # on the stack and has neither exported nor published its result.
            with patch.object(type(runtime), "step", step):
                original_grow(self, max_growth_steps)

        arguments = [
            "--work-dir",
            str(args.work_dir),
            "--scientific",
            "--pipeline-stage",
            "growth" if stage == "growth-relocate" else stage,
            "--persistent-worker",
            "--worker-max-jobs",
            "1",
            "--worker-poll-seconds",
            "1",
            "--max-cycles",
            "1",
            "--no-print-startup-summary",
        ]
        if stage == "growth-relocate":
            with patch.object(capsules, "publish_directory_if_absent", relocate):
                result = launcher.main(arguments)
        elif stage == "growth":
            with patch.object(AnemoneMorpionSearchRunner, "grow", grow):
                result = launcher.main(arguments)
        else:
            result = launcher.main(arguments)
        assert result == 0
        (evidence / ("c72-" + stage + "-done.json")).write_text(
            json.dumps({
                "pid": os.getpid(),
                "at_ns": time.monotonic_ns(),
                "exit_code": result,
            })
        )
