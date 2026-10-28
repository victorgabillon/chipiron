"""Entire accepted JSONL bytes against the independent full scientific reference."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, get_args

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from tests.environments import test_morpion_durable_scientific_v2 as scientific
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_durable_closed_science import closed_adapter

if TYPE_CHECKING:
    from pathlib import Path

CASES = [("legacy_v1", p, False, False) for p in get_args(PvFamilyTargetPolicy)] + [
    ("selected_value_v2", p, backup, exact)
    for p in get_args(PvFamilyTargetPolicy)
    for backup in (False, True)
    for exact in (False, True)
]


@pytest.mark.parametrize("confidence,policy,backup,exact", CASES)
def test_real_endgame_jsonl_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confidence: str,
    policy: str,
    backup: bool,
    exact: bool,
) -> None:
    """No normalization: all row bytes, targets, counters, provenance and certainty match."""
    from chipiron.environments.morpion.bootstrap import (
        scientific_publication_initialization as genesis,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    closed_adapter(monkeypatch)
    monkeypatch.setattr(
        genesis,
        "initialize_scientific_v2_publication",
        partial(
            genesis.initialize_scientific_v2_publication,
            atomic_reevaluation=True,
            fenced_dataset=True,
        ),
    )
    _source, args, oracle = scientific.committed_endgame.__wrapped__(
        SimpleNamespace(mktemp=lambda _: tmp_path / "work")
    )
    args = replace(
        args,
        dataset_confidence_policy=confidence,
        dataset_family_target_policy=policy,
        use_backed_up_value=backup,
        require_exact_or_terminal=exact,
    )
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    before = paths.run_state_path.read_bytes()
    local_leaderboard(monkeypatch, tmp_path)
    if confidence == "selected_value_v2" and policy.startswith("pv_exact_then_"):
        with pytest.raises(ValueError, match="gates pv_exact_then"):
            run_dataset(args)
        assert paths.run_state_path.read_bytes() == before
        return
    expected = extract_rows_from_training_snapshot(
        args=args,
        snapshot=oracle.snapshot,
        generation=1,
        value_contracts=oracle.value_contracts,
        source_provenance=oracle.provenance,
    )
    reference = tmp_path / "reference.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=expected.rows, metadata=expected.metadata, path=reference
    )
    monkeypatch.setattr(
        stages,
        "_load_training_snapshot_for_generation",
        lambda **_: pytest.fail("Full snapshot in bounded production Dataset"),
    )
    completed = run_dataset(args)
    assert (paths.work_dir / completed.rows_path).read_bytes() == reference.read_bytes()
    assert completed.metadata["dataset_rows"] == len(expected.rows)
    if confidence == "selected_value_v2":
        assert oracle.value_contracts["0"]["certainties"]["direct"] == "ESTIMATE"
        assert oracle.value_contracts["0"]["certainties"]["tree"] == "FORCED"
        terminal = expected.rows[-1]
        assert (
            terminal.target_value == 35 and terminal.is_exact and terminal.is_terminal
        )
        assert (
            terminal.metadata["target_confidence"]["training_target_certainty"]
            == "TERMINAL"
        )
        if not backup and not exact and policy == "none":
            assert expected.rows[0].target_value == 42
            assert (
                expected.rows[0].metadata["target_confidence"][
                    "training_target_certainty"
                ]
                == "ESTIMATE"
            )
    after = load_bootstrap_run_state(paths.run_state_path)
    assert after.metadata["atomic_reevaluation"]["accepted_cursor"] is None
