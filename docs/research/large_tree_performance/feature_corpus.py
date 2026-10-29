"""Capture exact feature witnesses using the independently frozen baseline wheel.

Run with that environment, PYTHONPATH=.; destination must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from atomheart.games.morpion import initial_state
from atomheart.games.morpion.state import Variant

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    extract_morpion_features,
    morpion_feature_names,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState


def feature_hash(state: MorpionState, dynamics: MorpionDynamics) -> str:
    """Hash names and exact float hex representations in their canonical order."""
    rows = [(k, v.hex()) for k, v in extract_morpion_features(state, dynamics).items()]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    """Record entire legal trajectories, including their genuine terminal state."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("destination", type=Path)
    options = parser.parse_args()
    dynamics = MorpionDynamics()
    walks = []
    for variant in Variant:
        for seed in (4, 17, 29, 103):
            rng = Random(seed)
            state = dynamics.wrap_atomheart_state(initial_state(variant=variant))
            actions = []
            witnesses = []
            for depth in range(121):
                legal = dynamics.all_legal_actions(state)
                if depth in (0, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100) or not legal:
                    witnesses.append({
                        "depth": depth,
                        "sha256": feature_hash(state, dynamics),
                    })
                if not legal:
                    break
                action = rng.choice(legal)
                actions.append(action)
                state = dynamics.step(state, action).next_state
            assert state.is_game_over(), (
                "Bounded witness walk must reach a true terminal."
            )
            walks.append({
                "variant": variant.value,
                "seed": seed,
                "actions": actions,
                "witnesses": witnesses,
            })
    output = {
        "baseline_commit": "f290fd7ea8746f3fedef70853d716b1ce06a0e36",
        "feature_names": morpion_feature_names(),
        "walks": walks,
    }
    with options.destination.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(len(walks), sum(len(w["witnesses"]) for w in walks))


if __name__ == "__main__":
    main()
