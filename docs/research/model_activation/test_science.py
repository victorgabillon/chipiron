"""Post-refresh scientific source parity; every existing policy stays explicit."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
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
from docs.research.model_activation.probe import bounded_probe
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    neural_evaluator,
)

if TYPE_CHECKING:
    from pathlib import Path


CASES = [("legacy_v1", p, False, False) for p in get_args(PvFamilyTargetPolicy)] + [
    ("selected_value_v2", p, backup, exact)
    for p in get_args(PvFamilyTargetPolicy)
    for backup in (False, True)
    for exact in (False, True)
]


@pytest.mark.parametrize("confidence,policy,backup,exact", CASES)
def test_refreshed_endgame_full_and_bounded_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confidence: str,
    policy: str,
    backup: bool,
    exact: bool,
) -> None:
    """Keep every row byte, confidence, metadata, source and existing gate visible."""
    instance = make_runner(monkeypatch, state=endgame.__wrapped__()[0])
    bounded_probe(instance, neural_evaluator(42), 1)
    path = instance.export_closed_scientific_snapshot(
        tmp_path / "science",
        generation=1,
        producer={
            "revision": "activation-research",
            "evaluator": "constant-neural-42",
            "config_sha256": "0" * 64,
        },
    )
    oracle = load_closed_scientific_snapshot(path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        use_backed_up_value=backup,
        require_exact_or_terminal=exact,
        dataset_family_target_policy=policy,
        dataset_confidence_policy=confidence,
    )
    assert oracle.value_contracts["0"]["certainties"]["direct"] == "ESTIMATE"
    assert oracle.value_contracts["0"]["certainties"]["tree"] == "FORCED"
    if confidence == "selected_value_v2" and policy.startswith("pv_exact_then_"):
        with pytest.raises(ValueError, match="gates pv_exact_then"):
            extract_rows_from_training_snapshot(
                args=args,
                snapshot=oracle.snapshot,
                generation=1,
                value_contracts=oracle.value_contracts,
                source_provenance=oracle.provenance,
            )
        with (
            ShardedTrainingGenerationView(
                path, cache_dir=tmp_path / "index", allow_closed_snapshot=True
            ) as view,
            pytest.raises(ValueError, match="gates pv_exact_then"),
            disk_dataset_context(IndexedDatasetSource(view), args),
        ):
            pytest.fail("Gated policy was admitted")
        return
    expected = extract_rows_from_training_snapshot(
        args=args,
        snapshot=oracle.snapshot,
        generation=1,
        value_contracts=oracle.value_contracts,
        source_provenance=oracle.provenance,
    )
    left, right = tmp_path / "reference.jsonl", tmp_path / "indexed.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=expected.rows, metadata=expected.metadata, path=left
    )
    with (
        ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "index", allow_closed_snapshot=True
        ) as view,
        disk_dataset_context(IndexedDatasetSource(view), args) as context,
    ):
        actual = context.stream(1)
        assert actual.metadata == expected.metadata
        save_morpion_supervised_rows_streaming(
            rows=actual.rows, metadata=actual.metadata, path=right
        )
    assert left.read_bytes() == right.read_bytes()
    if confidence == "selected_value_v2":
        terminal = expected.rows[-1]
        assert terminal.target_value == 35 and terminal.is_terminal
        assert (
            terminal.metadata["target_confidence"]["training_target_certainty"]
            == "TERMINAL"
        )
        if not backup and exact:
            assert len(expected.rows) == 1
        if not backup and not exact and policy == "none":
            assert expected.rows[0].target_value == 42
            assert (
                expected.rows[0].metadata["target_confidence"][
                    "training_target_certainty"
                ]
                == "ESTIMATE"
            )
