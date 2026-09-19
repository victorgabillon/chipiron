"""Smoke-test published dependency contracts using the installed Chipiron wheel."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import sysconfig
from pathlib import Path


def require_installed_package(name: str, distribution: str) -> None:
    """Reject editable/source overrides and print the actual imported distribution."""
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    paths = sysconfig.get_paths()
    installed_roots = [Path(paths[key]).resolve() for key in ("purelib", "platlib")]
    if origin is None or not any(
        Path(origin).resolve().is_relative_to(root) for root in installed_roots
    ):
        message = (
            f"{name} must come from this environment's installed wheel; got {origin}."
        )
        raise RuntimeError(message)
    for entry in sys.path:
        source_candidate = Path(entry or ".") / name / "__init__.py"
        if source_candidate.is_file() and not any(
            source_candidate.resolve().is_relative_to(root) for root in installed_roots
        ):
            message = f"Source override on sys.path: {source_candidate}."
            raise RuntimeError(message)
    print(f"{name} {importlib.metadata.version(distribution)}: {origin}")


def main() -> None:
    """Exercise the real evaluator builder and the extracted checkpoint API."""
    for name, distribution in (
        ("chipiron", "chipiron"),
        ("coral", "algorhino-coral"),
        ("anemone", "algorhino-anemone"),
        ("atomheart", "atomheart"),
    ):
        require_installed_package(name, distribution)

    from anemone.checkpoints import build_atoms, deserialize_checkpoint_atom

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        build_morpion_regressor,
        morpion_evaluator_v1_model_args,
    )
    from chipiron.utils.small_tools import resolve_package_path

    resolve_package_path(
        "package://scripts/one_match/inputs/gui_launcher/exp_options.yaml"
    )

    branch = ("morpion", 3, -4)
    assert (
        deserialize_checkpoint_atom(build_atoms.serialize_checkpoint_atom(branch))
        == branch
    )
    args = morpion_evaluator_v1_model_args()
    model = build_morpion_regressor(args)
    assert args.input_dim == 25
    assert args.relation_bias_scale == 0.25
    assert sum(parameter.numel() for parameter in model.parameters()) == 106_049
    print("Installed evaluator-v1 and checkpoint contracts passed.")


if __name__ == "__main__":
    main()
