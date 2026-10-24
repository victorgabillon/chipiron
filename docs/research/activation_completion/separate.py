"""Demonstrate descriptor/payload lifetime separation only in a disposable E2E copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    object_digest,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.e2e_bootstrap.checks import growth_oracle
from scripts.report_morpion_rolling_recovery import analyze
from tests.environments.test_morpion_activation_completion import cold_state


def main() -> None:
    """Require a verified unpinned old payload; move it aside without deleting bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("copy", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    root = options.copy.resolve()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    report = analyze(paths, retain=(1,))
    assert report["validated"], report["blockers"]
    # Only a generated fixture with an independently unpinned activation is admissible.
    assert report["current_contract_reclaimable_bytes"] == 0
    state = load_bootstrap_run_state(paths.run_state_path)
    request = json.loads(options.request.read_text())
    records = publication_records(state.metadata)
    matches = [
        r
        for r in records
        if r.get("activation_request_sha256") == object_digest(request)
    ]
    assert len(matches) == 1
    old = matches[0]
    assert state.generation > old["generation"]
    assert report["conditional_candidates"] == [old["generation"]], report
    assert old["descriptor_path"] in report["retained_commitment_files"]
    authority = paths.run_state_path.read_bytes()
    pending = (
        paths.pipeline_reevaluation_patch_path.read_bytes()
        if paths.pipeline_reevaluation_patch_path.exists()
        else None
    )
    payload = cold_state(paths)
    model = committed_evaluator(root, state)[0]
    assert model != request["candidate"]
    growth_oracle(root, root.parent / "oracle-before")
    unavailable = root.parent / "unavailable"
    unavailable.mkdir(exist_ok=False)
    moved = []
    for name, relative in (
        ("checkpoint", old["checkpoint_path"]),
        ("science", old["tree_snapshot_path"]),
    ):
        source = root / relative
        if name == "science":
            source = source.parent
        source.rename(unavailable / name)
        moved.append(str(source.relative_to(root)))
    assert activate_model(paths, request) == state
    assert paths.run_state_path.read_bytes() == authority
    assert (
        paths.pipeline_reevaluation_patch_path.read_bytes()
        if paths.pipeline_reevaluation_patch_path.exists()
        else None
    ) == pending
    assert cold_state(paths) == payload
    assert committed_evaluator(root, state)[0] == model
    growth_oracle(root, root.parent / "oracle-after")
    assert (root.parent / "oracle-before/expected.json").read_bytes() == (
        root.parent / "oracle-after/expected.json"
    ).read_bytes()
    incomplete = analyze(paths, retain=(1,))
    assert not incomplete["validated"]
    options.output.write_text(
        json.dumps(
            {
                "action": "disposable-copy-only; payload moved aside, not deleted",
                "before_graph": report,
                "unavailable_payloads": moved,
                "acknowledged_original_m2_request": True,
                "m3_model": model,
                "authority_receipts_cursor_and_pending_bytes_unchanged": True,
                "full_cold_checkpoint_payload_equal": True,
                "next_growth_full_checkpoint_and_scientific_files_equal": True,
                "incomplete_graph_still_refuses_deletion": incomplete,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
