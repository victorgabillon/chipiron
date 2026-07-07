"""Tests for the supervised chess neural-network learning script."""

import inspect
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
import torch


def create_tiny_model_bundle(
    tmp_path: Path,
    *,
    bundle_name: str = "model_bundle",
    input_representation: str = "piece_difference",
    weights_file: str = "weights.pt",
) -> Path:
    """Create a tiny local model bundle for offline tests."""
    bundle_dir = tmp_path / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "architecture.yaml").write_text(
        textwrap.dedent(
            """\
            model_output_type:
              point_of_view: player_to_move
            model_type_args:
              list_of_activation_functions:
              - hyperbolic_tangent
              number_neurons_per_layer:
              - 5
              - 1
              type: multi_layer_perceptron
            """
        ),
        encoding="utf-8",
    )
    (bundle_dir / "chipiron_nn.yaml").write_text(
        textwrap.dedent(
            f"""\
            version: 1
            game_kind: chess
            input_representation: {input_representation}
            """
        ),
        encoding="utf-8",
    )
    torch.save({"dummy": True}, bundle_dir / weights_file)
    return bundle_dir


def _make_config(*, saving_root: Path) -> Any:
    """Build a local-only config for the supervised learning test."""
    from chipiron.environments.chess.players.evaluators.boardevaluators.datasets.datasets import (
        DataSetArgs,
    )
    from chipiron.environments.types import GameKind
    from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
        ModelInputRepresentationType,
    )
    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        LearnNNScriptArgs,
    )
    from chipiron.scripts.learn_nn_supervised.training_args import (
        GameInputArgs,
        NNTrainerArgs,
    )
    from chipiron.scripts.script_args import BaseScriptArgs

    dataset_file = str((Path(__file__).parent / "small_dataset.pi").resolve())
    return LearnNNScriptArgs(
        nn_trainer_args=NNTrainerArgs(
            reuse_existing_model=False,
            specific_saving_folder=str(saving_root / "piece_difference"),
            game_input=GameInputArgs(
                game_kind=GameKind.CHESS,
                representation=ModelInputRepresentationType.PIECE_DIFFERENCE,
            ),
            epochs_number=1,
            saving_interval=10_000,
            saving_intermediate_copy=False,
        ),
        dataset_args=DataSetArgs(
            train_file_name=dataset_file,
            test_file_name=dataset_file,
        ),
        base_script_args=BaseScriptArgs(testing=True),
    )


def test_learn_nn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test learn nn."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")
    _install_learning_script_observability_stubs(monkeypatch)
    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        LearnNNScript,
    )

    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    saving_root = tmp_path / "learn_nn_supervised_outputs"
    (saving_root / "piece_difference").mkdir(parents=True, exist_ok=True)
    config = _make_config(saving_root=saving_root)

    try:
        script_object = LearnNNScript(base_script=_FakeBaseScript(config))
        script_object.run()
        script_object.terminate()
    finally:
        torch.cuda.is_available = original_is_available


def test_supervised_learning_script_no_longer_uses_legacy_trainer_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active supervised script should not construct NNPytorchTrainer."""
    pytest.importorskip("atomheart")
    pytest.importorskip("coral")
    _install_learning_script_observability_stubs(monkeypatch)
    from chipiron.scripts.learn_nn_supervised import learn_nn_from_supervised_datasets

    source = inspect.getsource(learn_nn_from_supervised_datasets)

    assert "create_nn_trainer" not in source
    assert "NNPytorchTrainer" not in source


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

    def __enter__(self) -> "_NoopRun":
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
    mlflow_pytorch_module = types.ModuleType("mlflow.pytorch")
    mlflow_models_module = types.ModuleType("mlflow.models")
    mlflow_signature_module = types.ModuleType("mlflow.models.signature")
    torchinfo_module = types.ModuleType("torchinfo")

    mlflow_module.set_tracking_uri = lambda uri: None
    mlflow_module.log_metric = lambda *args, **kwargs: None
    mlflow_module.log_params = lambda params: None
    mlflow_module.log_artifact = lambda path: None
    mlflow_module.start_run = lambda: _NoopRun()
    mlflow_pytorch_module.log_model = lambda *args, **kwargs: None
    mlflow_pytorch_module.get_default_conda_env = dict
    mlflow_module.pytorch = mlflow_pytorch_module

    class _ModelSignature:
        """Placeholder signature type used by the supervised script."""

    mlflow_signature_module.ModelSignature = _ModelSignature
    mlflow_signature_module.infer_signature = lambda *args, **kwargs: _ModelSignature()
    mlflow_models_module.signature = mlflow_signature_module
    torchinfo_module.summary = lambda model: f"summary({type(model).__name__})"

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", mlflow_pytorch_module)
    monkeypatch.setitem(sys.modules, "mlflow.models", mlflow_models_module)
    monkeypatch.setitem(sys.modules, "mlflow.models.signature", mlflow_signature_module)
    monkeypatch.setitem(sys.modules, "torchinfo", torchinfo_module)
