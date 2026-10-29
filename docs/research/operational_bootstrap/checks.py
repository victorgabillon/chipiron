"""Configure the existing full scientific oracle for the exact operational inputs."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
from docs.research.e2e_bootstrap import checks
from docs.research.e2e_bootstrap.worker import arguments


def verify(root: Path, *, full: bool = True) -> None:
    """Keep authoritative recovery checks; select the existing independent full audit."""
    workspace = root / "workspace"
    config = load_bootstrap_config(workspace / "bootstrap_config.json")
    with patch.object(
        checks, "arguments", lambda p: _scientific_args(arguments(p), config)
    ):
        result = checks.verify(
            workspace, root / f"verified-{time.time_ns()}", full=full
        )
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapPaths,
        dataset_publication,
        training_publication,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        publication_records,
    )
    from chipiron.environments.morpion.bootstrap.run_state import (
        load_bootstrap_run_state,
    )

    paths = MorpionBootstrapPaths.from_work_dir(workspace)
    before = paths.run_state_path.read_bytes()
    state = load_bootstrap_run_state(paths.run_state_path)
    accepted = []
    for record in publication_records(state.metadata):
        item = {"generation": record["generation"]}
        if "accepted_dataset" in record:
            item["dataset"] = dataset_publication.accepted_dataset_identity(
                paths, state, record["generation"]
            )
        if "accepted_training" in record:
            item["training"] = training_publication.accepted_selected_model(
                paths, state, record["generation"]
            )
        if len(item) > 1:
            accepted.append(item)
    assert paths.run_state_path.read_bytes() == before
    result["validation_level"] = "independent-audit" if full else "routine-recovery"
    result["verified_accepted_identities"] = accepted
    from atomheart.games.morpion.dynamics import MorpionDynamics

    from chipiron.environments.morpion.learning import decode_morpion_state_ref_payload

    # Reuse the existing leaderboard and legal replay decoder; no new record authority.
    leaderboard = workspace / "leaderboard.jsonl"
    verified_records = []
    if leaderboard.exists():
        for line in leaderboard.read_text().splitlines():
            entry = json.loads(line)
            state = decode_morpion_state_ref_payload(entry["state_ref_payload"])
            assert state.variant.value == entry["variant"] == "5T"
            assert MorpionDynamics().is_terminal_state(state)
            assert int(state.moves) == entry["moves_since_start"]
            assert 36 + int(state.moves) == entry["total_points"]
            verified_records.append({
                "fingerprint": entry["state_fingerprint"],
                "moves": int(state.moves),
                "total_points": 36 + int(state.moves),
                "variant": entry["variant"],
                "legal_replay_and_terminal": True,
            })
    if result.get("certificates", {}).get("terminal"):
        assert verified_records
    result["verified_leaderboard"] = verified_records
    encoded = json.dumps(result, indent=2)
    (root / f"verification-{time.time_ns()}.json").write_text(encoded)
    (root / "latest-verification.json").write_text(encoded)
