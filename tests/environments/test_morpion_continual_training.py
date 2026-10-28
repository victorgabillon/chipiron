"""Independent tensor and optimizer oracles for continuing candidate training."""

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
from atomheart.games.morpion import MorpionDynamics, initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    train_morpion_regressor,
    train_morpion_regressor_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
    weights_digest,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split import (
    canonical_identity,
    prepare_state_split,
)


def rows_file(root: Path, *, reverse: bool = False, extra: int = 0) -> Path:
    """Different legal states with a target independent of the model's predictions."""
    dynamics = MorpionDynamics()
    state = initial_state()
    codec = MorpionStateCheckpointCodec()
    rows = []
    for index in range(16 + extra):
        rows.append(
            MorpionSupervisedRow(
                str(index),
                codec.dump_state_ref(state),
                float(20 + index),
                False,
                False,
                index,
                metadata={"target_source": "test_external_target"},
            )
        )
        state = dynamics.step(state, dynamics.all_legal_actions(state)[0]).next_state
    if reverse:
        rows.reverse()
    path = root / "rows.jsonl"
    save_morpion_supervised_rows_streaming(rows=iter(rows), metadata={}, path=path)
    return path


@pytest.mark.fast
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_three_rounds_load_real_parent_before_first_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    kind: str,
) -> None:
    """Would fail if a trainer built fresh weights and proceeded directly to updates."""
    rows = rows_file(tmp_path)
    parent = None
    expected_parameters = None
    cumulative = 0
    initial_identity = None
    expected_optimizer_state = None
    original_step = torch.optim.Adam.step
    for generation in range(1, 4):
        output = tmp_path / f"round-{generation}"
        continuation = MorpionTrainingContinuation(
            "candidate",
            111 + generation,
            parent_bundle=None if parent is None else str(parent),
            parent_bundle_sha256=None if parent is None else artifact_digest(parent),
            parent_weights_sha256=None
            if parent is None
            else file_digest(parent / "param.pt"),
        )
        args = MorpionTrainingArgs(
            rows,
            output,
            model_kind=kind,
            hidden_sizes=(4,) if kind == "mlp" else None,
            num_epochs=1,
            batch_size=4,
            device="cpu",
            stable_validation_split=True,
            continuation=continuation,
        )
        first = True

        def checked_step(
            optimizer: torch.optim.Adam,
            *args: object,
            expected_parameters: list[torch.Tensor] | None = expected_parameters,
            cumulative: int = cumulative,
            expected_optimizer_state: list[dict] | None = expected_optimizer_state,
            **kwargs: object,
        ) -> object:
            nonlocal first
            if first and expected_parameters is not None:
                actual = optimizer.param_groups[0]["params"]
                assert expected_optimizer_state is not None
                for value, expected, previous_state in zip(
                    actual, expected_parameters, expected_optimizer_state, strict=True
                ):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    assert int(optimizer.state[value]["step"].item()) == cumulative
                    for moment in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[value][moment],
                            previous_state[moment],
                            rtol=0,
                            atol=0,
                        )
                first = False
            return original_step(optimizer, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(torch.optim.Adam, "step", checked_step)
            if streaming:
                model, _ = train_morpion_regressor_streaming(
                    MorpionStreamingTrainingArgs(args, row_chunk_size=5)
                )
            else:
                model, _ = train_morpion_regressor(args)
        _, _, manifest = load_morpion_model_bundle(output)
        evidence = manifest.metadata["continuation"]
        if parent is not None:
            assert not first
            previous, _, _ = load_morpion_model_bundle(parent)
            assert evidence["starting_weights_sha256"] == weights_digest(previous)
        assert evidence["final_weights_sha256"] == weights_digest(model)
        assert evidence["loaded_previous_weights"] == (parent is not None)
        assert evidence["optimizer_steps"] > 0
        cumulative += evidence["optimizer_steps"]
        assert evidence["cumulative_optimizer_steps"] == cumulative
        if initial_identity is None:
            initial_identity = evidence["initial_bundle_sha256"]
            assert artifact_digest(output / "initial") == initial_identity
        assert evidence["initial_bundle_sha256"] == initial_identity
        expected_parameters = [p.detach().clone() for p in model.parameters()]
        checkpoint = torch.load(
            output / "optimizer.pt", map_location="cpu", weights_only=True
        )
        expected_optimizer_state = [
            checkpoint["state"][index]
            for index in checkpoint["param_groups"][0]["params"]
        ]
        parent = output


@pytest.mark.fast
@pytest.mark.parametrize("damage", ["corrupt", "missing", "architecture", "candidate"])
def test_parent_failure_never_falls_back_to_initialization(
    tmp_path: Path, damage: str
) -> None:
    """Even a complete but incompatible parent must fail before another update."""
    rows = rows_file(tmp_path)
    output = tmp_path / "first"
    args = MorpionTrainingArgs(
        rows,
        output,
        num_epochs=1,
        device="cpu",
        continuation=MorpionTrainingContinuation("a", 3),
    )
    train_morpion_regressor(args)
    continuation = MorpionTrainingContinuation(
        "b" if damage == "candidate" else "a",
        4,
        str(output),
        artifact_digest(output),
        file_digest(output / "param.pt"),
    )
    if damage == "corrupt":
        (output / "param.pt").write_bytes(b"corrupt")
    if damage == "missing":
        (output / "param.pt").unlink()
    next_args = replace(
        args,
        output_dir=tmp_path / "second",
        continuation=continuation,
        model_kind="mlp" if damage == "architecture" else "linear",
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        train_morpion_regressor(next_args)
    assert not (tmp_path / "second" / "param.pt").exists()


@pytest.mark.fast
def test_canonical_partition_survives_reordering_and_population_growth(
    tmp_path: Path,
) -> None:
    """State membership survives changed node IDs, row order and dataset size."""
    from chipiron.environments.morpion.learning import load_morpion_supervised_rows

    assignments = []
    for name, reverse, extra in [("first", False, 0), ("second", True, 4)]:
        path = rows_file(tmp_path / name, reverse=reverse, extra=extra)
        args, evidence = prepare_state_split(
            MorpionTrainingArgs(path, tmp_path / "unused", stable_validation_split=True)
        )
        assert evidence["train_count"] > 0 and evidence["validation_count"] > 0
        rows = load_morpion_supervised_rows(path).rows
        assignments.append({
            canonical_identity(row): index in args.validation_indices
            for index, row in enumerate(rows)
        })
    assert all(assignments[1][key] == value for key, value in assignments[0].items())


@pytest.mark.fast
def test_d4_equivalent_states_share_partition_identity(tmp_path: Path) -> None:
    """Different encoded orientations must not enter opposing train/validation partitions."""
    from chipiron.environments.morpion.learning import (
        decode_morpion_state_ref_payload,
        load_morpion_supervised_rows,
    )
    from chipiron.environments.morpion.symmetry import transform_morpion_state
    from chipiron.environments.morpion.types import MorpionDynamics as ChipironDynamics

    row = load_morpion_supervised_rows(rows_file(tmp_path)).rows[-1]
    state = ChipironDynamics().wrap_atomheart_state(
        decode_morpion_state_ref_payload(row.state_ref_payload)
    )
    reflected = transform_morpion_state(state, 1)
    payload = MorpionStateCheckpointCodec().dump_state_ref(
        reflected.to_atomheart_state()
    )
    assert payload != row.state_ref_payload
    assert canonical_identity(row) == canonical_identity(
        replace(row, node_id="different-id", state_ref_payload=payload)
    )


@pytest.mark.fast
def test_explicit_optimizer_reset_preserves_model_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resetting Adam is a separate, accurately recorded policy from resetting weights."""
    rows = rows_file(tmp_path)
    first = tmp_path / "first"
    args = MorpionTrainingArgs(
        rows,
        first,
        num_epochs=1,
        device="cpu",
        continuation=MorpionTrainingContinuation("candidate", 2),
    )
    model, _ = train_morpion_regressor(args)
    expected = [p.detach().clone() for p in model.parameters()]
    continuation = MorpionTrainingContinuation(
        "candidate",
        3,
        str(first),
        artifact_digest(first),
        file_digest(first / "param.pt"),
        optimizer_state_policy="reset_each_round",
    )
    original = torch.optim.Adam.step
    seen = False

    def inspect(optimizer: torch.optim.Adam, *args: object, **kwargs: object) -> object:
        nonlocal seen
        if not seen:
            assert not optimizer.state
            for actual, value in zip(
                optimizer.param_groups[0]["params"], expected, strict=True
            ):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            seen = True
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", inspect)
    second = tmp_path / "second"
    train_morpion_regressor(replace(args, output_dir=second, continuation=continuation))
    _, _, manifest = load_morpion_model_bundle(second)
    assert seen
    assert manifest.metadata["continuation"]["loaded_previous_weights"]
    assert not manifest.metadata["continuation"]["optimizer_state_loaded"]
