"""Opt-in continuous experiment settings stored in the existing configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState

CONTINUAL_KEY = "continual_learning"


@dataclass(frozen=True, slots=True)
class MorpionContinualLearningPolicy:
    """Small explicit policy; accepted results remain the only candidate authority."""

    training_policy: str = "continue_from_previous_accepted_weights"
    seed: int = 0
    optimizer_state_policy: str = "continue_adam"
    automatic_activation: bool = False
    dataset_every_generations: int = 1
    training_every_generations: int = 1
    selection_every_rounds: int = 1
    activation_every_rounds: int = 1
    max_tree_nodes: int = 200_000
    max_elapsed_seconds: int = 43_000
    worker_max_seconds: int = 7_200
    max_group_rss_mb: int = 8_192
    min_available_ram_mb: int = 8_192
    min_free_disk_mb: int = 20_480
    max_workspace_mb: int = 32_768
    max_scratch_mb: int = 4_096
    shutdown_grace_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate declared limits; budgets are never extended automatically."""
        from .generation_publication_io import require

        require(
            self.training_policy
            in {"continue_from_previous_accepted_weights", "fresh_from_scratch"},
            "Unknown continual training policy.",
        )
        require(
            self.optimizer_state_policy in {"continue_adam", "reset_each_round"},
            "Unknown continual optimizer policy.",
        )
        require(
            isinstance(self.automatic_activation, bool),
            "automatic_activation must be boolean.",
        )
        for key, value in asdict(self).items():
            if key not in {
                "training_policy",
                "optimizer_state_policy",
                "automatic_activation",
                "seed",
            }:
                require(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0,
                    f"{key} must be a positive integer.",
                )
        require(
            isinstance(self.seed, int)
            and not isinstance(self.seed, bool)
            and 0 <= self.seed < 2**63,
            "seed must be an integer in [0, 2**63).",
        )


def policy_from_metadata(
    metadata: dict[str, Any],
) -> MorpionContinualLearningPolicy | None:
    """Load only an explicitly configured policy; old experiments retain their mode."""
    raw = metadata.get("continual_learning")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        from .generation_publication_io import require

        require(False, "continual_learning must be an explicit policy object.")
    assert isinstance(raw, dict)
    return MorpionContinualLearningPolicy(**raw)


def validate_continual_context(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    policy: MorpionContinualLearningPolicy | None,
) -> None:
    """Reject policy downgrade, historical adoption and changed declared budgets."""
    from .config import bootstrap_config_sha256, load_bootstrap_config
    from .generation_publication_io import require

    marker = state.metadata.get(CONTINUAL_KEY)
    if marker is None:
        require(policy is None, "Continual learning requires explicit fresh genesis.")
        return
    require(
        marker == {"schema_version": 1}
        and isinstance(marker["schema_version"], int)
        and not isinstance(marker["schema_version"], bool),
        "Unsupported continual-learning genesis marker.",
    )
    config = load_bootstrap_config(paths.bootstrap_config_path)
    expected = policy_from_metadata(config.metadata)
    require(
        expected is not None and policy == expected,
        "Continual learning policy changed or was silently disabled.",
    )
    require(
        bootstrap_config_sha256(config) == state.metadata["bootstrap_config_hash"],
        "Continual configuration or resource budgets changed.",
    )
