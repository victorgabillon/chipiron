"""Independent scientific oracle and full source/row parity after real activation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, get_args

import pytest
import torch
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    load_morpion_evaluator_from_model_bundle,
)
from docs.research.model_activation.probe import reference
from docs.research.model_activation.probe import state as values
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_value_certainty import endgame

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", ["dag", "endgame", "partial"])
def test_real_activation_independent_oracle_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Actual accepted weights; independent max oracle, exact checkpoint and JSONL."""
    late = mode == "endgame"
    local_leaderboard(monkeypatch, tmp_path)
    with monkeypatch.context() as context:
        if late:
            context.setattr(
                runner_module, "initial_state", lambda: endgame.__wrapped__()[0]
            )
        args, paths, _ = setup_activation(
            tmp_path / "workspace",
            count=76 if mode == "dag" else 2,
            branch_limit=1 if mode == "partial" else 100000,
            seed_score=42,
        )
    before = load_bootstrap_run_state(paths.run_state_path)
    request = prepare_model_activation_request(paths, training_generation=1)
    live, effective = _runner(paths, before)
    _, physical = committed_evaluator(paths.work_dir, before)
    live.load_or_create(
        paths.work_dir / before.latest_runtime_checkpoint_path, physical, effective
    )
    old = load_closed_scientific_snapshot(
        paths.work_dir / before.latest_tree_snapshot_path
    )
    if late:
        assert old.value_contracts["0"]["certainties"] == {
            "direct": "ESTIMATE",
            "tree": "FORCED",
            "effective": "FORCED",
            "target": "FORCED",
        }
        assert old.snapshot.nodes[0].direct_value_scalar == 42
        assert old.snapshot.nodes[-1].direct_value_scalar == 35
    with torch.random.fork_rng(devices=[]):
        evaluator = load_morpion_evaluator_from_model_bundle(
            paths.work_dir / request["candidate"]["path"]
        )
    expected_values = reference(live, evaluator)
    live._runtime.refresh_with_evaluator(evaluator, scope="all", batch_size=64)
    assert values(live) == expected_values
    expected_payload = checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(live._runtime, state_codec=live._state_codec)
    )
    producer = before.metadata["durable_generation_publication_v1"]["generations"][-1][
        "scientific_identity"
    ]["producer"]["identity"]
    expected_science = live.export_closed_scientific_snapshot(
        tmp_path / "oracle-science",
        generation=before.generation + 1,
        producer={**producer, "evaluator": request["candidate"]["sha256"]},
        committed_source=paths.work_dir / before.latest_tree_snapshot_path,
        previous_producer=producer,
    )
    live.discard_uncommitted_runtime()

    after = activate_model(paths, request)
    restored, effective = _runner(paths, after)
    _, physical = committed_evaluator(paths.work_dir, after)
    restored.load_or_create(
        paths.work_dir / after.latest_runtime_checkpoint_path, physical, effective
    )
    assert values(restored) == expected_values
    assert (
        checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(
                restored._runtime, state_codec=restored._state_codec
            )
        )
        == expected_payload
    )
    restored.discard_uncommitted_runtime()
    source = paths.work_dir / after.latest_tree_snapshot_path
    assert {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()} == {
        p.name: p.read_bytes() for p in expected_science.parent.iterdir() if p.is_file()
    }
    full = load_closed_scientific_snapshot(source)
    assert [n.state_ref_payload for n in full.snapshot.nodes] == [
        n.state_ref_payload for n in old.snapshot.nodes
    ]
    assert [(n.parent_ids, n.child_ids) for n in full.snapshot.nodes] == [
        (n.parent_ids, n.child_ids) for n in old.snapshot.nodes
    ]
    assert (source.parent / "records.jsonl").read_bytes().splitlines()[1:] == (
        paths.work_dir / before.latest_tree_snapshot_path
    ).parent.joinpath("records.jsonl").read_bytes().splitlines()[1:]
    paired = 0
    with ShardedTrainingGenerationView(
        source, cache_dir=tmp_path / "index", allow_closed_snapshot=True
    ) as view:
        for policy in get_args(PvFamilyTargetPolicy):
            for backup in (False, True):
                for exact in (False, True):
                    options = replace(
                        args,
                        max_rows=None,
                        dataset_family_target_policy=policy,
                        use_backed_up_value=backup,
                        require_exact_or_terminal=exact,
                    )
                    if policy.startswith("pv_exact_then_"):
                        with pytest.raises(ValueError, match="gates pv_exact_then"):
                            extract_rows_from_training_snapshot(
                                args=options,
                                snapshot=full.snapshot,
                                generation=after.generation,
                                value_contracts=full.value_contracts,
                                source_provenance=full.provenance,
                            )
                        with (
                            pytest.raises(ValueError, match="gates pv_exact_then"),
                            disk_dataset_context(IndexedDatasetSource(view), options),
                        ):
                            pytest.fail("unsupported exact-family propagation")
                        continue
                    expected = extract_rows_from_training_snapshot(
                        args=options,
                        snapshot=full.snapshot,
                        generation=after.generation,
                        value_contracts=full.value_contracts,
                        source_provenance=full.provenance,
                    )
                    left, right = tmp_path / "full.jsonl", tmp_path / "indexed.jsonl"
                    save_morpion_supervised_rows_streaming(
                        rows=expected.rows, metadata=expected.metadata, path=left
                    )
                    with disk_dataset_context(
                        IndexedDatasetSource(view), options
                    ) as context:
                        stream = context.stream(after.generation)
                        assert stream.metadata == expected.metadata
                        save_morpion_supervised_rows_streaming(
                            rows=stream.rows, metadata=stream.metadata, path=right
                        )
                    assert left.read_bytes() == right.read_bytes()
                    paired += len(expected.rows)
                    if late:
                        assert expected.rows[-1].target_value == 35
                        assert (
                            expected.rows[-1].metadata["target_confidence"][
                                "training_target_certainty"
                            ]
                            == "TERMINAL"
                        )
                        if not backup and exact:
                            assert len(expected.rows) == 1
    (tmp_path / "parity.txt").write_text(
        f"{paired} paired rows; 20 enabled cases, 16 gated cases\n"
    )


@pytest.mark.parametrize("damage", ["direct_forced", "terminal_score", "tree_proof"])
def test_activation_preflights_certificates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """A damaged restored certificate is rejected before M2 loading or any commit."""
    from valanga.evaluations import Certainty, Value

    local_leaderboard(monkeypatch, tmp_path)
    with monkeypatch.context() as context:
        context.setattr(
            runner_module, "initial_state", lambda: endgame.__wrapped__()[0]
        )
        _, paths, _ = setup_activation(tmp_path / "workspace", count=2, seed_score=42)
    request = prepare_model_activation_request(paths, training_generation=1)
    authority = paths.run_state_path.read_bytes()
    original = runner_module.AnemoneMorpionSearchRunner.load_or_create

    def corrupted(runner: object, *args: object, **kwargs: object) -> None:
        original(runner, *args, **kwargs)
        nodes = list(runner.iter_profile_nodes())
        if damage == "direct_forced":
            nodes[0].tree_evaluation.direct_value = Value(
                score=42, certainty=Certainty.FORCED
            )
        elif damage == "terminal_score":
            evaluation = nodes[-1].tree_evaluation
            evaluation.direct_value = replace(evaluation.direct_value, score=999)
        else:
            evaluation = nodes[0].tree_evaluation
            evaluation.tree_value = replace(evaluation.tree_value, score=999)

    def forbidden(*_: object, **__: object) -> None:
        pytest.fail("M2 loading preceded certificate verification")

    monkeypatch.setattr(
        runner_module.AnemoneMorpionSearchRunner, "load_or_create", corrupted
    )
    monkeypatch.setattr(
        runner_module.AnemoneMorpionSearchRunner, "refresh_model_estimates", forbidden
    )
    with pytest.raises(ValueError, match=r"(?i)(certificate|terminal)"):
        activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == authority
