"""Compatibility tests for legacy chess trainer checkpoint helper paths."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def test_legacy_checkpoint_helper_import_paths_reexport_canonical_helpers() -> None:
    """Legacy checkpoint helper paths should re-export canonical helpers."""
    pytest.importorskip("coral")
    canonical = importlib.import_module(
        "chipiron.scripts.learn_nn_supervised.checkpoint_helpers"
    )
    legacy = _import_legacy_checkpoint_helpers()
    factory = importlib.import_module("chipiron.learningprocesses.nn_trainer.factory")

    assert legacy.safe_nn_architecture_save is canonical.safe_nn_architecture_save
    assert legacy.safe_nn_param_save is canonical.safe_nn_param_save
    assert legacy.safe_nn_trainer_save is canonical.safe_nn_trainer_save
    assert legacy.get_optimizer_file_path_from is canonical.get_optimizer_file_path_from
    assert legacy.get_scheduler_file_path_from is canonical.get_scheduler_file_path_from
    assert (
        legacy.get_folder_training_copies_path_from
        is canonical.get_folder_training_copies_path_from
    )

    assert factory.safe_nn_architecture_save is canonical.safe_nn_architecture_save
    assert factory.safe_nn_param_save is canonical.safe_nn_param_save
    assert factory.safe_nn_trainer_save is canonical.safe_nn_trainer_save


def _import_legacy_checkpoint_helpers() -> Any:
    """Import the dependency-light legacy checkpoint helper module."""
    return importlib.import_module(
        "chipiron.learningprocesses.nn_trainer.checkpoint_helpers"
    )
