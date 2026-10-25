"""Audit before temporarily moving an unpinned tree in a disposable E2E copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    file_digest,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    reevaluation_patch_from_dict,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.e2e_bootstrap.checks import growth_oracle, science_hashes
from docs.research.e2e_bootstrap.worker import arguments
from docs.research.receipt_model_evidence.measure import acknowledge
from scripts.report_morpion_rolling_recovery import analyze
from tests.environments.test_morpion_activation_completion import cold_state
from tests.environments.test_morpion_receipt_model_evidence import grow


def main() -> None:
    """No deletion; selected model bytes, descriptors and original workspace survive."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("copy", type=Path)
    parser.add_argument("original_experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, original = args.copy.resolve(), args.original_experiment.resolve()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    report = analyze(paths, retain=(1,))
    assert report["validated"] and report["conditional_candidates"] == [4], report
    assert report["current_contract_reclaimable_bytes"] == 0
    state = load_bootstrap_run_state(paths.run_state_path)
    old = publication_records(state.metadata)[3]
    preserved = root.parent / "preserved"
    preserved.mkdir()
    original_hashes = {
        str(p.relative_to(original)): file_digest(p)
        for p in original.rglob("*")
        if p.is_file()
    }
    authority, payload = paths.run_state_path.read_bytes(), cold_state(paths)
    active_model = committed_evaluator(root, state)[0]
    frozen = {
        e["physical_bundle"]: artifact_digest(root / e["physical_bundle"])
        for e in report["receipt_model_evidence"]
    }
    growth_oracle(root, root.parent / "oracle-before")
    moved = []
    for name, relative in (
        ("checkpoint", old["checkpoint_path"]),
        ("science", old["tree_snapshot_path"]),
    ):
        source = root / relative
        if name == "science":
            source = source.parent
        files = {
            str(p.relative_to(source)): file_digest(p)
            for p in source.rglob("*")
            if p.is_file()
        }
        source.rename(preserved / name)
        assert files == {
            str(p.relative_to(preserved / name)): file_digest(p)
            for p in (preserved / name).rglob("*")
            if p.is_file()
        }
        moved.append({
            "original": str(source),
            "preserved": str(preserved / name),
            "files": files,
        })
    # Pending new M3 work is preserved separately while exact older ACKs use the
    # single create-only slot. Its admission is covered by the real focused tests.
    pending = paths.pipeline_reevaluation_patch_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.rename(preserved / "pending-M3.json")
    for label in ("06-check", "22-check"):
        raw = (original / "evidence" / label / "pending.json").read_bytes()
        proposal = reevaluation_patch_from_dict(json.loads(raw))
        for _ in range(3):
            assert publish_reevaluation_patch_if_absent(
                proposal, paths.pipeline_reevaluation_patch_path
            )
            assert paths.pipeline_reevaluation_patch_path.read_bytes() == raw
            assert acknowledge(paths)
            assert paths.run_state_path.read_bytes() == authority
            assert cold_state(paths) == payload
    activate_model(paths, json.loads((original / "M2-request.json").read_bytes()))
    assert paths.run_state_path.read_bytes() == authority
    assert committed_evaluator(root, state)[0] == active_model
    assert frozen == {p: artifact_digest(root / p) for p in frozen}
    growth_oracle(root, root.parent / "oracle-after")
    oracle = json.loads((root.parent / "oracle-before/expected.json").read_bytes())
    assert (root.parent / "oracle-before/expected.json").read_bytes() == (
        root.parent / "oracle-after/expected.json"
    ).read_bytes()
    after = grow(arguments(root), paths)
    assert after.generation == 9 and cold_state(paths) == oracle["payload"]
    assert science_hashes(root / after.latest_tree_snapshot_path) == oracle["science"]
    (preserved / "pending-M3.json").rename(paths.pipeline_reevaluation_patch_path)
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == pending
    incomplete = analyze(paths, retain=(1,))
    assert not incomplete["validated"]
    assert original_hashes == {
        str(p.relative_to(original)): file_digest(p)
        for p in original.rglob("*")
        if p.is_file()
    }
    args.output.write_text(
        json.dumps(
            {
                "action": "temporary unavailability in disposable copy only; no deletion",
                "complete_graph_before": report,
                "preserved_payloads": moved,
                "historical_model_hashes_before_and_after": frozen,
                "exact_M1_and_M2_ack_count": 6,
                "authority_receipts_cursor_model_epoch_unchanged_by_ack": True,
                "full_checkpoint_and_science_equal": True,
                "selector_search_and_rollout_rng_equal": True,
                "historical_activation_ack_equal": True,
                "real_G9_continuation_equals_unmodified_oracle": True,
                "pending_M3_preserved_and_restored": True,
                "incomplete_graph_refuses_retirement": incomplete,
                "original_experiment_file_hashes_unchanged": original_hashes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
