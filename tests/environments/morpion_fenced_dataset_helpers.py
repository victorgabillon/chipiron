"""Explicit fresh fenced genesis and tiny synthetic source helpers."""

from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_closed_helpers import setup_closed_baseline
from tests.environments.morpion_durable_v2_helpers import scientific_producer


def setup_fenced(root: Path) -> Any:
    """Create a fresh two-node workspace, never adopt an existing authority."""
    args, paths, state = setup_closed_baseline(root, initialize=False)
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=scientific_producer(args),
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        fenced_dataset=True,
    )
    return args, paths, state


def local_leaderboard(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Keep all optional leaderboard writes inside this new fixture."""
    persist = stages.persist_certified_leaderboard_candidates
    monkeypatch.setattr(
        stages,
        "persist_certified_leaderboard_candidates",
        lambda **kwargs: persist(**kwargs, leaderboard_path=root / "leaderboard.jsonl"),
    )


def run_dataset(args: Any, **kwargs: Any) -> Any:
    """Use the actual production bounded reader and fenced Dataset entrypoint."""
    return stages.run_pipeline_dataset_stage(
        args,
        generation=kwargs.pop("generation", 1),
        use_streaming_sharded_reader=True,
        allow_scientific_v2=True,
        **kwargs,
    )
