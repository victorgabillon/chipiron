import textwrap
from importlib import import_module
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("parsley")
pytest.importorskip("torch")
pytest.importorskip("PySide6")
import torch
from parsley import make_partial_dataclass_with_optional_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHIPIRON_SRC = _REPO_ROOT / "src"
_ATOMHEART_SRC = _REPO_ROOT.parent / "atomheart" / "src"
_ANEMONE_SRC = _REPO_ROOT.parent / "anemone" / "src"

if str(_CHIPIRON_SRC) not in sys.path:
    sys.path.insert(0, str(_CHIPIRON_SRC))
if str(_ATOMHEART_SRC) not in sys.path:
    sys.path.insert(0, str(_ATOMHEART_SRC))
if str(_ANEMONE_SRC) not in sys.path:
    sys.path.insert(0, str(_ANEMONE_SRC))

_stale_atomheart_modules = [
    module_name
    for module_name, module in sys.modules.items()
    if module_name == "atomheart" or module_name.startswith("atomheart.")
    if not hasattr(module, "__file__")
]
for module_name in _stale_atomheart_modules:
    del sys.modules[module_name]

import_module("atomheart")

_stale_anemone_modules = [
    module_name
    for module_name, module in sys.modules.items()
    if module_name == "anemone" or module_name.startswith("anemone.")
    if not hasattr(module, "__file__")
]
for module_name in _stale_anemone_modules:
    del sys.modules[module_name]

import_module("anemone")

_stale_chipiron_modules = [
    module_name
    for module_name, module in sys.modules.items()
    if module_name == "chipiron" or module_name.startswith("chipiron.")
    if not hasattr(module, "__file__")
]
for module_name in _stale_chipiron_modules:
    del sys.modules[module_name]

import_module("chipiron")


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


def _make_config(*, tmp_path: Path, saving_root: Path) -> Any:
    """Build a local-only config for the supervised learning test."""
    from chipiron.environments.chess.players.evaluators.boardevaluators.datasets.datasets import (
        DataSetArgs,
    )
    from chipiron.environments.types import GameKind
    from chipiron.learningprocesses.nn_trainer.factory import (
        GameInputArgs,
        NNTrainerArgs,
    )
    from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
        ModelInputRepresentationType,
    )
    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        LearnNNScriptArgs,
    )
    from chipiron.scripts.script_args import BaseScriptArgs

    PartialOpLearnNNScriptArgs = make_partial_dataclass_with_optional_paths(
        cls=LearnNNScriptArgs
    )
    PartialOpNNTrainerArgs = make_partial_dataclass_with_optional_paths(
        cls=NNTrainerArgs
    )
    PartialOpDataSetArgs = make_partial_dataclass_with_optional_paths(cls=DataSetArgs)
    PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(
        cls=BaseScriptArgs
    )
    PartialOpGameInputArgs = make_partial_dataclass_with_optional_paths(
        cls=GameInputArgs
    )

    dataset_file = str((Path(__file__).parent / "small_dataset.pi").resolve())
    bundle_dir = create_tiny_model_bundle(
        tmp_path,
        bundle_name="supervised_model_bundle",
        input_representation=ModelInputRepresentationType.PIECE_DIFFERENCE.value,
    )
    return PartialOpLearnNNScriptArgs(
        nn_trainer_args=PartialOpNNTrainerArgs(
            reuse_existing_model=False,
            specific_saving_folder=str(saving_root / "piece_difference"),
            neural_network_architecture_args_path_to_yaml_file=str(
                bundle_dir / "architecture.yaml"
            ),
            game_input=PartialOpGameInputArgs(
                game_kind=GameKind.CHESS,
                representation=ModelInputRepresentationType.PIECE_DIFFERENCE,
            ),
        ),
        dataset_args=PartialOpDataSetArgs(
            train_file_name=dataset_file,
            test_file_name=dataset_file,
        ),
        base_script_args=PartialOpBaseScriptArgs(testing=True),
    )


def test_learn_nn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test learn nn."""
    from chipiron import scripts
    import chipiron.scripts.factory as script_factory_module
    from chipiron.scripts.factory import create_script

    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    monkeypatch.setattr(
        script_factory_module,
        "get_package_root_path",
        lambda package_name: str(_REPO_ROOT / "src" / package_name),
    )
    saving_root = tmp_path / "learn_nn_supervised_outputs"
    saving_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(tmp_path=tmp_path, saving_root=saving_root)

    try:
        script_object: scripts.IScript = create_script(
            script_type=scripts.ScriptType.LEARN_NN,
            extra_args=config,
            should_parse_command_line_arguments=False,
        )
        script_object.run()
        script_object.terminate()
    finally:
        torch.cuda.is_available = original_is_available
