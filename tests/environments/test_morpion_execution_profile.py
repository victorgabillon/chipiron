"""Real train/eval/continuation and complete-bundle contracts for strict_math."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from chipiron.environments.morpion.bootstrap.candidate_executor import (
    CandidateExecutionError,
    execute_candidates,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    train_morpion_regressor,
    train_morpion_regressor_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    continuation as lifecycle,
)
from chipiron.learning.torch_runtime import (
    TORCH_EXECUTION_PROFILE_KEY,
    TorchExecutionProfileError,
    torch_execution_profile,
    torch_execution_profile_info,
    torch_execution_profile_metadata,
)
from tests.environments.test_morpion_continual_training import rows_file


@pytest.mark.fast
@pytest.mark.parametrize("streaming", [False, True])
def test_profile_covers_training_validation_and_real_adam_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """Existing trainers retain complete states and the same process profile."""
    rows = rows_file(tmp_path)
    args = MorpionTrainingArgs(
        rows,
        tmp_path / "first",
        device="cpu",
        num_epochs=1,
        batch_size=4,
        continuation=lifecycle.MorpionTrainingContinuation("linear", 71),
    )
    observed = []
    original = lifecycle.build_morpion_regressor

    def build(model_args: MorpionRegressorArgs) -> torch.nn.Module:
        model = original(model_args)
        model.register_forward_pre_hook(
            lambda module, _inputs: observed.append((
                module.training,
                torch_execution_profile_info(),
            ))
        )
        return model

    monkeypatch.setattr(lifecycle, "build_morpion_regressor", build)

    def train(training_args: MorpionTrainingArgs) -> object:
        if streaming:
            return train_morpion_regressor_streaming(
                MorpionStreamingTrainingArgs(training_args, row_chunk_size=5)
            )
        return train_morpion_regressor(training_args)

    with torch_execution_profile("strict_math") as info:
        train(args)
        assert {mode for mode, _ in observed} == {False, True}
        assert all(observation == info for _, observation in observed)
        parent = Path(args.output_dir)
        model, model_args, manifest = load_morpion_model_bundle(parent)
        assert manifest.metadata[TORCH_EXECUTION_PROFILE_KEY] == info.to_metadata()
        checkpoint = torch.load(parent / "param.pt", weights_only=True)
        assert checkpoint.keys() == model.state_dict().keys()
        assert all(
            torch.equal(value, checkpoint[name])
            for name, value in model.state_dict().items()
        )
        first_optimizer = torch.load(parent / "optimizer.pt", weights_only=True)
        assert first_optimizer["state"]
        next_args = replace(
            args,
            output_dir=tmp_path / "second",
            continuation=lifecycle.MorpionTrainingContinuation(
                "linear",
                72,
                str(parent),
                artifact_digest(parent),
                file_digest(parent / "param.pt"),
            ),
        )
        train(next_args)
        _, _, resumed = load_morpion_model_bundle(next_args.output_dir)
        assert resumed.metadata[TORCH_EXECUTION_PROFILE_KEY] == info.to_metadata()
        assert resumed.metadata["continuation"]["optimizer_state_loaded"]
        assert resumed.metadata["continuation"]["loaded_previous_weights"]
    with pytest.raises(TorchExecutionProfileError):
        train(replace(next_args, output_dir=tmp_path / "rejected"))
    assert not (tmp_path / "rejected" / "param.pt").exists()
    with pytest.raises(TorchExecutionProfileError):
        model(torch.zeros((1, model_args.input_dim)))


@pytest.mark.fast
def test_existing_bundle_and_model_args_remain_unchanged(tmp_path: Path) -> None:
    """Opt-in only changes sidecar provenance, never state keys or model args."""
    args = MorpionRegressorArgs()
    model = build_morpion_regressor(args)
    existing = tmp_path / "existing"
    strict = tmp_path / "strict"
    save_morpion_model_bundle(
        model, existing, model_args=args, metadata={"fixture": True}
    )
    _, _, before = load_morpion_model_bundle(existing)
    assert TORCH_EXECUTION_PROFILE_KEY not in before.metadata
    with torch_execution_profile("strict_math"):
        save_morpion_model_bundle(
            model, strict, model_args=args, metadata={"fixture": True}
        )
        loaded, _, after = load_morpion_model_bundle(strict)
        assert {
            k: v for k, v in after.metadata.items() if k != TORCH_EXECUTION_PROFILE_KEY
        } == before.metadata
        assert (
            torch_execution_profile_metadata()[TORCH_EXECUTION_PROFILE_KEY]
            == after.metadata[TORCH_EXECUTION_PROFILE_KEY]
        )
        assert loaded.state_dict().keys() == model.state_dict().keys()
    assert (existing / "morpion_regressor_args.json").read_bytes() == (
        strict / "morpion_regressor_args.json"
    ).read_bytes()
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
    with pytest.raises(TorchExecutionProfileError):
        load_morpion_model_bundle(strict)
    with pytest.raises(TorchExecutionProfileError):
        save_morpion_model_bundle(model, tmp_path / "downgrade", model_args=args)
    assert not (tmp_path / "downgrade").exists()
    # Corrupt evidence is not accepted even inside the requested profile.
    manifest_path = strict / "morpion_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["metadata"][TORCH_EXECUTION_PROFILE_KEY]["warn_only"] = True
    manifest_path.write_text(json.dumps(payload))
    with (
        torch_execution_profile("strict_math"),
        pytest.raises(TorchExecutionProfileError),
    ):
        load_morpion_model_bundle(strict)


@pytest.mark.fast
def test_strict_continuation_cannot_adopt_historical_lineage(tmp_path: Path) -> None:
    """Enabling strict mode mid-lineage requires a distinct scientific run."""
    args = MorpionTrainingArgs(
        rows_file(tmp_path),
        tmp_path / "first",
        num_epochs=1,
        device="cpu",
        continuation=lifecycle.MorpionTrainingContinuation("linear", 1),
    )
    train_morpion_regressor(args)
    parent = Path(args.output_dir)
    continuation = lifecycle.MorpionTrainingContinuation(
        "linear",
        2,
        str(parent),
        artifact_digest(parent),
        file_digest(parent / "param.pt"),
    )
    with (
        torch_execution_profile("strict_math"),
        pytest.raises(TorchExecutionProfileError),
    ):
        train_morpion_regressor(
            replace(args, output_dir=tmp_path / "rejected", continuation=continuation)
        )
    assert not (tmp_path / "rejected" / "param.pt").exists()


@pytest.mark.fast
def test_spawned_candidates_reject_profile_before_starting_work(tmp_path: Path) -> None:
    """Do not silently drop a parent's opt-in in fresh CPU worker processes."""
    with (
        torch_execution_profile("strict_math"),
        pytest.raises(CandidateExecutionError, match="separate profile qualification"),
    ):
        execute_candidates([], None, tmp_path / "unused")  # type: ignore[arg-type]
    assert not (tmp_path / "unused").exists()
