"""Owned child entrypoint for real prepared Dataset crash and consumer fixtures."""

import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation as shared,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation_builder as builder,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset


def main() -> None:
    """Fault only this generated fixture; no external process or historical input."""
    args, generation, action, point = pickle.loads(Path(sys.argv[1]).read_bytes())
    if action == "hold":
        from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
            MorpionBootstrapPaths,
        )
        from chipiron.environments.morpion.bootstrap.control import (
            load_bootstrap_control,
        )
        from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
            load_pipeline_manifest,
        )

        paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
        manifest = load_pipeline_manifest(
            paths.pipeline_manifest_path_for_generation(generation)
        )
        config = training.configuration(
            args,
            args.resolved_evaluators_config(),
            load_bootstrap_control(paths.control_path),
        )
        with training.admit_training(
            paths, manifest, config, ttl_seconds=60, owner=None
        ) as attempt:
            assert isinstance(attempt, training.TrainingAttempt)
            with prepared.borrow(paths, args, attempt.admission) as borrowed:
                assert borrowed is not None
                os.write(int(os.environ["J41_READY_FD"]), b"R")
                assert os.read(int(os.environ["J41_RELEASE_FD"]), 1) == b"G"
        return
    if action == "consume":

        def unexpected(*args: Any, **kwargs: Any) -> Any:
            message = "Warm consumer copied rows or rebuilt native tensors"
            raise AssertionError(message)

        if point == "announce-admission":
            original_borrow = prepared.borrow

            @contextmanager
            def announced(*args: Any, **kwargs: Any) -> Any:
                os.write(int(os.environ["J41_CONSUMER_FD"]), b"T")
                with original_borrow(*args, **kwargs) as borrowed:
                    yield borrowed

            prepared.borrow = announced
        training.copy_artifact = unexpected
        builder.build_shared_bundle = unexpected
        run_pipeline_training_stage(args, generation=generation)
    else:
        sync = prepared.sync_artifact
        publish = shared.publish_directory_if_absent
        save = prepared._save

        def after_copy(path: Path, *args: Any, **kwargs: Any) -> Any:
            result = sync(path, *args, **kwargs)
            if path.name == "rows.jsonl":
                if point == "copy":
                    os._exit(73)
                if point == "hold-copy":
                    os.write(int(os.environ["J41_READY_FD"]), b"R")
                    assert os.read(int(os.environ["J41_RELEASE_FD"]), 1) == b"G"
            return result

        def after_native(*args: Any, **kwargs: Any) -> Any:
            result = publish(*args, **kwargs)
            if point == "native":
                os._exit(73)
            return result

        def after_ready(*args: Any, **kwargs: Any) -> Any:
            result = save(*args, **kwargs)
            if point == "ready" and args[3]["phase"] == "COMPLETE":
                os._exit(73)
            return result

        prepared.sync_artifact = after_copy
        shared.publish_directory_if_absent = after_native
        prepared._save = after_ready
        run_dataset(args, generation=generation)


if __name__ == "__main__":
    main()
