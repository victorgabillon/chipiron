"""Create private, reproducible tiny linear inputs for the bounded operator experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    save_bootstrap_config,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)
from docs.research.e2e_bootstrap.worker import arguments


def main() -> None:
    """Write config and an explicitly initialized M1; never present M1 as trained."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-initial-steps", type=int, default=1000)
    options = parser.parse_args()
    root = options.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args = replace(
        arguments(root),
        max_growth_steps_per_cycle=options.max_initial_steps,
        max_rows=128,
    )
    save_bootstrap_config(
        bootstrap_config_from_args(args), root / "bootstrap_config.json"
    )
    torch.manual_seed(4)
    model_args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(model_args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        list(model.parameters())[-1].fill_(42)
    save_morpion_model_bundle(model, root / "seed", model_args=model_args)


if __name__ == "__main__":
    main()
