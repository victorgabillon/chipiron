"""Reproduce the certainty/target matrix against an explicitly selected source.

Run from the repository with PYTHONPATH=SELECTED_CHIPIRON_SRC:PINNED_ANEMONE_SRC:.
Requires the test dependencies; fixtures reuse the public-runtime regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.node_evaluation.common import canonical_value

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.learning import morpion_supervised_rows_to_dict
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    model_patch,
    values,
)


def dataset_matrix(runner: Any, root: Path) -> list[dict[str, Any]]:
    """Hash every serialized row field and metadata; also expose target decisions."""
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    results = []
    for policy in get_args(PvFamilyTargetPolicy):
        for backup in (True, False):
            for blend in (0.0, 0.25, 1.0):
                for exact in (True, False):
                    args = MorpionBootstrapArgs(
                        work_dir=root,
                        dataset_family_target_policy=policy,
                        dataset_family_prediction_blend=blend,
                        use_backed_up_value=backup,
                        require_exact_or_terminal=exact,
                    )
                    data = extract_rows_from_training_snapshot(
                        args=args, snapshot=snapshot, generation=1
                    )
                    encoded = json.dumps(
                        morpion_supervised_rows_to_dict(data), sort_keys=True
                    ).encode()
                    results.append({
                        "policy": policy,
                        "use_backup": backup,
                        "blend": blend,
                        "exact_only": exact,
                        "full_output_sha256": hashlib.sha256(encoded).hexdigest(),
                        "rows": [
                            {
                                "node_id": row.node_id,
                                "target": row.target_value,
                                "is_exact": row.is_exact,
                                "is_terminal": row.is_terminal,
                                "metadata": row.metadata,
                            }
                            for row in data.rows
                        ],
                    })
    return results


def audit(root: Path) -> dict[str, Any]:
    """Measure ordinary, genuinely solved, and explicitly synthetic cases."""
    cases = {}
    late, _ = endgame.__wrapped__()
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("ordinary", "proved", "synthetic"):
            for prediction in (None, 42.0):
                for alpha in (1.0, 0.25):
                    case = f"{name}-prediction-{prediction}-alpha-{alpha}"
                    directory = root / case
                    directory.mkdir()
                    runner = make_runner(
                        monkeypatch,
                        state=late if name == "proved" else None,
                        alpha=alpha,
                    )
                    if name == "synthetic":
                        node, child = tuple(runner.iter_profile_nodes())
                        node.tree_evaluation.direct_value = (
                            canonical_value.make_forced_value(score=9)
                        )
                        node.tree_evaluation.tree_value = (
                            canonical_value.make_estimate_value(score=3)
                        )
                        child.tree_evaluation.direct_value = (
                            canonical_value.make_estimate_value(score=2)
                        )
                        child.tree_evaluation.tree_value = (
                            canonical_value.make_estimate_value(score=2)
                        )
                    before = values(runner)
                    patch_data = None
                    if prediction is not None:
                        patch = model_patch(runner, monkeypatch, prediction, directory)
                        if name != "proved":
                            patch = replace(patch, rows=(patch.rows[0],))
                        applied = runner.apply_reevaluation_patch(patch)
                        patch_data = {
                            "applied": applied,
                            "rows": [
                                {
                                    "node_id": row.node_id,
                                    "direct_value": row.direct_value,
                                    "is_exact": row.is_exact,
                                    "is_terminal": row.is_terminal,
                                    "source": row.metadata["source"],
                                }
                                for row in patch.rows
                            ],
                        }
                    cases[case] = {
                        "before": before,
                        "after": values(runner),
                        "patch": patch_data,
                        "dataset": dataset_matrix(runner, directory),
                    }
    return {
        "cases": cases,
        "matrix_outputs": sum(len(case["dataset"]) for case in cases.values()),
    }


def main() -> None:
    """Write a new evidence artifact, refusing to overwrite any retained result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="morpion-certainty-audit-") as scratch:
        result = audit(Path(scratch))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            checkpoint_payload_to_jsonable(result), stream, indent=2, sort_keys=True
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
