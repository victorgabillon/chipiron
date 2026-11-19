"""Real child workers; test controls pause Growth while it owns its native lease."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap import launcher
from chipiron.environments.morpion.bootstrap.pipeline import stages


def main() -> int:
    """Run the stock persistent dispatcher with bounded timing/observation hooks."""
    root, signals, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if stage == "growth":
        owned = stages._run_pipeline_growth_stage_owned

        def paused(*args: Any, **kwargs: Any) -> Any:
            # Called INSIDE the real Growth flock, before real search/publication.
            (signals / "growth-owned").write_text("held")
            deadline = time.monotonic() + 60
            while not (signals / "release-growth").exists():
                if time.monotonic() >= deadline:
                    message = "Test controller did not release Growth"
                    raise TimeoutError(message)
                time.sleep(0.02)
            return owned(*args, **kwargs)

        stages._run_pipeline_growth_stage_owned = paused
    elif stage == "training_worker":
        training = launcher.run_next_pipeline_training_stage_once

        def observed_training(*args: Any, **kwargs: Any) -> Any:
            result = training(*args, **kwargs)
            with (signals / "training-results.jsonl").open("a") as stream:
                stream.write(json.dumps(asdict(result)) + "\n")
            if result.reason == "ownership_contended":
                (signals / "training-contended").write_text("no-work")
                # Controller releases Growth and publishes real Dataset input.
                # The next stock persistent poll must reload that new state.
                deadline = time.monotonic() + 60
                while not (signals / "resume-training").exists():
                    if time.monotonic() >= deadline:
                        message = "Test controller did not resume Training"
                        raise TimeoutError(message)
                    time.sleep(0.02)
            return result

        launcher.run_next_pipeline_training_stage_once = observed_training
    else:
        dataset = launcher.run_next_pipeline_dataset_stage_once

        def observed(*args: Any, **kwargs: Any) -> Any:
            result = dataset(*args, **kwargs)
            with (signals / "dataset-results.jsonl").open("a") as stream:
                stream.write(json.dumps(asdict(result)) + "\n")
            if result.reason == "ownership_contended":
                (signals / "dataset-contended").write_text("no-work")
            return result

        launcher.run_next_pipeline_dataset_stage_once = observed
    return launcher.main([
        "--work-dir",
        str(root),
        "--scientific",
        "--pipeline-stage",
        stage,
        "--persistent-worker",
        "--worker-max-jobs",
        {"growth": "1", "training_worker": "2"}.get(stage, "3"),
        "--worker-poll-seconds",
        "1",
        "--max-cycles",
        "1",
        "--no-print-startup-summary",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
