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


def verify(root: Path) -> None:
    """Compare current checkpoint, all accepted rows, values and actual patch rows."""
    workspace = root / "workspace"
    config = load_bootstrap_config(workspace / "bootstrap_config.json")
    with patch.object(
        checks, "arguments", lambda p: _scientific_args(arguments(p), config)
    ):
        result = checks.verify(
            workspace, root / f"verified-{time.time_ns()}", full=True
        )
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
    if result["certificates"]["terminal"]:
        assert verified_records
    result["verified_leaderboard"] = verified_records
    (root / "latest-verification.json").write_text(json.dumps(result, indent=2))
