"""Propose bounded preparation intent alongside the existing Dataset acceptance save."""

import math
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import prepared_dataset_storage as storage
from .generation_publication_io import read_object, require
from .persistent_worker_state import monitored_identity
from .prepared_dataset_recipe import dataset_recipe

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineStageClaim
    from .run_state import MorpionBootstrapRunState


def _deadline(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    claim: MorpionPipelineStageClaim,
    owner: dict[str, Any],
) -> tuple[float, float]:
    """Capture original claim/job bounds, never a renewed preparation timeout."""
    assert args.continual_learning is not None
    now = time.monotonic()
    # A persistent worker resets this clock at original dispatch, not at acceptance.
    status = paths.pipeline_dir / "worker_processes" / "dataset_worker.json"
    # /proc start ticks use boot time (including suspend), not CLOCK_MONOTONIC.
    age = time.clock_gettime(time.CLOCK_BOOTTIME) - int(
        owner["start_ticks"]
    ) / os.sysconf("SC_CLK_TCK")
    require(age >= 0, "Invalid Dataset process clock")
    started = now - age
    if status.exists():
        job = read_object(status)
        require(
            job["identity"] == owner and job["phase"] == "running",
            "Dataset job identity differs",
        )
        started = job["job_started_monotonic_s"]
    require(
        isinstance(started, (float, int))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= started <= now,
        "Invalid Dataset job clock",
    )
    wall = time.time()
    claim_end = datetime.fromisoformat(claim.expires_at_utc).timestamp()
    end = min(
        now + claim_end - wall, started + args.continual_learning.worker_max_seconds
    )
    require(end > now, "Original Dataset job allowance exhausted")
    return end, wall + end - now


def intent_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    args: MorpionBootstrapArgs,
    proposal: dict[str, Any],
    accepted: dict[str, Any],
    claim: MorpionPipelineStageClaim,
) -> MorpionBootstrapRunState:
    """Join eligible preparation intent with scientific acceptance in ONE state save."""
    recipe = dataset_recipe(args, paths.work_dir / proposal["manifest"]["rows_path"])
    if recipe is None:
        return state
    # Unmonitored legacy invocations retain their exact private Training path.
    if not os.environ.get("MORPION_RESOURCE_MONITOR_PID"):
        return state
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _Budget,
    )

    owner = monitored_identity()
    end, wall_end = _deadline(paths, args, claim, owner)
    key = accepted["descriptor_sha256"]
    data = storage.entries(state)
    if (storage.directory(paths, key) / "retired.json").exists():
        return state  # Retirement is final for this operational root.
    if key in data:
        return state  # Exact original allowance, including accepted retries.
    limits = _Budget.configured()
    assert args.continual_learning is not None
    aggregate_cap = min(
        limits.scratch_bytes, args.continual_learning.max_scratch_mb * 2**20
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation_reuse import (
        _limit,
        requested,
    )

    # Charge the shared prior store once at its configured upper bound. Its own
    # builder additionally checks actual prior/new coexistence within this budget.
    prior_reservation = _limit() if requested() else 0
    remaining = (
        aggregate_cap
        - prior_reservation
        - sum(r["reserved_bytes"] for r in data.values())
    )
    # No eligible reservation means ordinary private fallback, not partial reuse.
    if len(data) == 2 or remaining <= proposal["rows_bytes"] + 2 * limits.chunk_bytes:
        return state
    assert args.continual_learning is not None
    record = {
        "schema": 1,
        "accepted": accepted,
        "source": proposal["admission"]["source"],
        "context": storage.context(paths, state),
        "recipe": recipe.identity(),
        "rows": {
            "path": proposal["manifest"]["rows_path"],
            "sha256": proposal["rows_sha256"],
            "bytes": proposal["rows_bytes"],
        },
        "generation": proposal["admission"]["generation"],
        "phase": "PREPARING",
        "dataset_claim_id": claim.claim_id,
        "token": uuid.uuid4().hex,
        "owner": owner,
        "deadline": end,
        "utc_deadline": wall_end,
        "reserved_bytes": remaining,
        "ready": None,
    }
    return storage.updated(state, {**data, key: record})
