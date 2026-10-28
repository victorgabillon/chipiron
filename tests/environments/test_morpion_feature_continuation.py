"""Exact feature witnesses captured from the independently installed foundation."""

import hashlib
import json
from pathlib import Path

import pytest
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.state import Variant

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    extract_morpion_features,
    morpion_feature_names,
)
from chipiron.environments.morpion.types import MorpionDynamics

_BASELINE = json.loads(
    (Path(__file__).parents[1] / "data/morpion_features_baseline.json").read_text()
)


@pytest.mark.parametrize(
    "walk", _BASELINE["walks"], ids=lambda walk: f"{walk['variant']}-{walk['seed']}"
)
def test_entire_feature_vector_matches_frozen_legal_walk(walk: dict) -> None:
    """Preserve ordered float bits across both variants, depth and true terminals."""
    assert _BASELINE["baseline_commit"] == "f290fd7ea8746f3fedef70853d716b1ce06a0e36"
    assert list(morpion_feature_names()) == _BASELINE["feature_names"]
    dynamics = MorpionDynamics()
    state = dynamics.wrap_atomheart_state(
        initial_state(variant=Variant(walk["variant"]))
    )
    witnesses = {row["depth"]: row["sha256"] for row in walk["witnesses"]}
    seen = set()
    for depth in range(len(walk["actions"]) + 1):
        if depth in witnesses:
            features = extract_morpion_features(state, dynamics)
            rows = [(k, v.hex()) for k, v in features.items()]
            actual = hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()
            ).hexdigest()
            assert actual == witnesses[depth], (walk["variant"], walk["seed"], depth)
            seen.add(depth)
        if depth < len(walk["actions"]):
            state = dynamics.step(state, tuple(walk["actions"][depth])).next_state
    assert seen == witnesses.keys()
    assert state.is_game_over()
