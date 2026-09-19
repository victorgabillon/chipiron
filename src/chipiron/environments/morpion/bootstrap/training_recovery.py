"""CLI wrapper for Morpion stale training recovery."""

from __future__ import annotations

from .pipeline.training_recovery import (
    DEFAULT_STALE_GRACE_SECONDS,
    StaleTrainingState,
    inspect_training_state_for_recovery,
    inspect_training_states_for_work_dir,
    main,
    recover_stale_training_state,
)

__all__ = [
    "DEFAULT_STALE_GRACE_SECONDS",
    "StaleTrainingState",
    "inspect_training_state_for_recovery",
    "inspect_training_states_for_work_dir",
    "main",
    "recover_stale_training_state",
]


if __name__ == "__main__":
    raise SystemExit(main())
