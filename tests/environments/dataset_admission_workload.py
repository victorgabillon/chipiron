"""Real child workers; test controls pause Growth while it owns its native lease."""

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap import launcher
from chipiron.environments.morpion.bootstrap.pipeline import stages

if TYPE_CHECKING:
    from collections.abc import Iterator


def main() -> int:
    """Run the stock persistent dispatcher with bounded timing/observation hooks."""
    root, signals, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    publication_overlap = sys.argv[4:] == ["publication"]
    if publication_overlap and stage == "dataset_worker":
        from chipiron.environments.morpion.bootstrap import generation_publication_io

        original_ownership = generation_publication_io.growth_receipt_ownership

        @contextmanager
        def timed_ownership(*args: Any, **kwargs: Any) -> Iterator[None]:
            bounded = kwargs.get("wait_until") is not None
            if bounded:
                (signals / "acceptance-waiting").touch()
            start = time.monotonic()
            with original_ownership(*args, **kwargs):
                acquired = time.monotonic()
                yield
            if bounded:
                (signals / "acceptance-wait.json").write_text(
                    json.dumps({
                        "wait_s": acquired - start,
                        "deadline": kwargs["wait_until"],
                    })
                )

        generation_publication_io.growth_receipt_ownership = timed_ownership
        publish = stages.publish_dataset

        def published(*args: Any, **kwargs: Any) -> Any:
            reference = publish(*args, **kwargs)
            with (signals / "published.jsonl").open("a") as stream:
                stream.write(json.dumps(reference) + "\n")
            deadline = time.monotonic() + 60
            while not (signals / "growth-owned").exists():
                if time.monotonic() >= deadline:
                    message = "Test Growth did not acquire publication ownership"
                    raise TimeoutError(message)
                time.sleep(0.02)
            return reference

        stages.publish_dataset = published
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
        "1"
        if publication_overlap
        else {"growth": "1", "training_worker": "2"}.get(stage, "3"),
        "--worker-poll-seconds",
        "1",
        "--max-cycles",
        "1",
        "--no-print-startup-summary",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
