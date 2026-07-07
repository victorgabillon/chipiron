"""Module for test learn nn from scratch and fixed boards."""
# pylint: disable=duplicate-code

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")


def _make_config(*, tmp_path: Path) -> Any:
    """Build a local-only config for the scratch learning test."""
    from chipiron.environments.chess.players.evaluators.boardevaluators.datasets.datasets import (
        DataSetArgs,
    )
    from chipiron.players.move_selector.random_args import RandomSelectorArgs
    from chipiron.players.player_args import PlayerArgs
    from chipiron.scripts.chipiron_args import ImplementationArgs
    from chipiron.scripts.learn_from_scratch_value_and_fixed_boards.learn_from_scratch_value_and_fixed_boards import (
        LearnNNFromScratchScriptArgs,
    )
    from chipiron.scripts.learn_nn_supervised.training_args import (
        SupervisedTrainingArgs,
    )
    from chipiron.scripts.script_args import BaseScriptArgs

    out_dir = tmp_path / "learn_from_scratch_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = str((Path(__file__).parent / "small_dataset.pi").resolve())

    return LearnNNFromScratchScriptArgs(
        epochs_number_with_respect_to_evaluating_player=1,
        number_of_evaluating_player_per_loop=1,
        number_of_gradient_descent_per_loop=0,
        nn_trainer_args=SupervisedTrainingArgs(
            reuse_existing_model=False,
            specific_saving_folder=str(out_dir),
            saving_intermediate_copy=False,
        ),
        dataset_args=DataSetArgs(
            train_file_name=dataset_file,
            test_file_name=dataset_file,
        ),
        evaluating_player_args=PlayerArgs(
            name="Random",
            main_move_selector=RandomSelectorArgs(),
            oracle_play=False,
        ),
        base_script_args=BaseScriptArgs(testing=True),
        implementation_args=ImplementationArgs(),
    )


def test_learn_nn_from_scratch_and_fixed_boards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test learn nn from scratch and fixed boards."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")
    _install_learning_script_observability_stubs(monkeypatch)
    from chipiron.scripts.learn_from_scratch_value_and_fixed_boards.learn_from_scratch_value_and_fixed_boards import (
        LearnNNFromScratchScript,
    )

    config = _make_config(tmp_path=tmp_path)

    script_object = LearnNNFromScratchScript(base_script=_FakeBaseScript(config))

    script_object.run()
    script_object.terminate()


class _FakeBaseScript:
    """Tiny base-script adapter avoiding the parser/factory stack."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.terminated = False

    def initiate(self, experiment_output_folder: str | None = None) -> Any:
        """Return prebuilt args and fill the experiment output path."""
        if self.args.base_script_args.experiment_output_folder is None:
            self.args.base_script_args.experiment_output_folder = (
                experiment_output_folder
            )
        return self.args

    def terminate(self) -> None:
        """Mark termination for the test adapter."""
        self.terminated = True


class _NoopRun:
    """No-op context manager matching mlflow.start_run."""

    def __enter__(self) -> _NoopRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type, exc, traceback


def _install_learning_script_observability_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install tiny mlflow and torchinfo stubs for dependency-light script import."""
    mlflow_module = types.ModuleType("mlflow")
    torchinfo_module = types.ModuleType("torchinfo")

    mlflow_module.set_tracking_uri = lambda uri: None
    mlflow_module.log_params = lambda params: None
    mlflow_module.log_artifact = lambda path: None
    mlflow_module.start_run = lambda: _NoopRun()
    torchinfo_module.summary = lambda model: f"summary({type(model).__name__})"

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torchinfo", torchinfo_module)
