"""Opt-in single-source learning windows; all transitions require Growth ownership.

This is a Morpion scientific policy, not generic attempt coordination. Accepted
references and the ordinary publication commit remain the scientific authority.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .generation_publication_io import (
    object_digest,
    publication_records,
    read_committed_descriptor,
    require,
)

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState

CONFIG_KEY = "learning_schedule"
WINDOW_KEY = "learning_window_v1"


@dataclass(frozen=True, slots=True)
class BoundedGrowthWindow:
    """A fixed ordinary-cycle gate, never a completion-time-driven cutover."""

    growth_lookahead_cycles: int

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded cycle allowances."""
        require(
            isinstance(self.growth_lookahead_cycles, int)
            and not isinstance(self.growth_lookahead_cycles, bool)
            and self.growth_lookahead_cycles > 0,
            "Growth lookahead must be a positive integer.",
        )

    def validate_supported(self, args: MorpionBootstrapArgs) -> None:
        """Expose configuration validation without a reverse runtime import."""
        validate_supported(args)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize only an explicit opt-in; old config bytes stay unchanged."""
        return {
            "schema_version": 1,
            "kind": "bounded_growth_window_v1",
            "growth_lookahead_cycles": self.growth_lookahead_cycles,
            "queue_capacity": 1,
            "cutover": "window_end",
            "refresh": "owned_full",
        }


def schedule_from_metadata(metadata: dict[str, Any]) -> BoundedGrowthWindow | None:
    """Reject unknown versions/options rather than interpreting a partial policy."""
    if CONFIG_KEY not in metadata:
        return None
    raw = metadata[CONFIG_KEY]
    require(isinstance(raw, dict), "Invalid learning schedule.")
    assert isinstance(raw, dict)
    lookahead = raw.get("growth_lookahead_cycles")
    require(isinstance(lookahead, int), "Growth lookahead must be a positive integer.")
    assert isinstance(lookahead, int)
    schedule = BoundedGrowthWindow(lookahead)
    require(
        raw == schedule.to_metadata()
        and isinstance(raw.get("schema_version"), int)
        and not isinstance(raw.get("schema_version"), bool)
        and isinstance(raw.get("queue_capacity"), int)
        and not isinstance(raw.get("queue_capacity"), bool),
        "Unsupported learning schedule.",
    )
    return schedule


def context_marker(schedule: BoundedGrowthWindow | None) -> dict[str, Any]:
    """Old binaries reject the new context instead of silently changing policy."""
    return (
        {"schema_version": 1}
        if schedule is None
        else {
            "schema_version": 2,
            "learning_schedule_sha256": object_digest(schedule.to_metadata()),
        }
    )


def initial_window(schedule: BoundedGrowthWindow | None) -> dict[str, Any]:
    """Fresh genesis only; no adoption or inferred state for existing runs."""
    return (
        {}
        if schedule is None
        else {
            WINDOW_KEY: {
                "policy": schedule.to_metadata(),
                "active": None,
                "completed": [],
                "skipped": [],
            }
        }
    )


def validate_supported(args: MorpionBootstrapArgs) -> None:
    """Keep v1 narrow without altering any historical recipe or default."""
    if args.learning_schedule is None:
        return
    policy = args.continual_learning
    require(
        policy is not None
        and policy.automatic_activation
        and policy.dataset_every_generations == policy.training_every_generations
        and args.pipeline_mode == "artifact_pipeline"
        and args.runtime_checkpoint_format == "sharded"
        and args.training_export_mode == "sharded"
        and args.save_after_seconds == 0
        and not args.growth_skip_training_export
        and not args.growth_save_and_exit
        and not args.diagnostic_stop_after_growth
        and args.max_growth_steps_per_cycle > 0,
        "Bounded Growth requires continual automatic activation, equal cadences "
        "and a closed sharded publication each ordinary cycle.",
    )


def window(state: MorpionBootstrapRunState) -> dict[str, Any] | None:
    """Validate the durable policy binding before consuming its scheduling state."""
    raw = state.metadata.get(WINDOW_KEY)
    marker = state.metadata.get("continual_learning")
    if raw is None:
        require(
            marker is None or marker == {"schema_version": 1},
            "Missing durable learning window.",
        )
        return None
    require(isinstance(raw, dict), "Invalid durable learning window.")
    assert isinstance(raw, dict)
    require(
        set(raw) == {"policy", "active", "completed", "skipped"},
        "Invalid window fields.",
    )
    schedule = schedule_from_metadata({CONFIG_KEY: raw["policy"]})
    require(marker == context_marker(schedule), "Learning schedule identity changed.")
    require(
        isinstance(raw["completed"], list) and isinstance(raw["skipped"], list),
        "Invalid window history.",
    )
    slot = raw["active"]
    if slot is not None:
        require(isinstance(slot, dict), "Invalid active learning slot.")
        records = publication_records(state.metadata)
        source = next((r for r in records if r["id"] == slot["source"]["id"]), None)
        require(source is not None, "Learning source is not committed.")
        assert source is not None and schedule is not None
        require(
            slot["source"] == source_identity(source)
            and "retired_payload_v1" not in source
            and isinstance(slot["source_cycle"], int)
            and not isinstance(slot["source_cycle"], bool)
            and isinstance(slot["target_cycle"], int)
            and not isinstance(slot["target_cycle"], bool)
            and slot["target_cycle"]
            == slot["source_cycle"] + schedule.growth_lookahead_cycles
            and slot["source_cycle"] <= state.cycle_index <= slot["target_cycle"],
            "Learning source or cycle bound changed.",
        )
    return raw


def source_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Keep publication, tree and producer identities together, never model aliases."""
    return {
        k: record[k]
        for k in ("id", "generation", "descriptor_sha256", "scientific_identity")
    }


def selected_source(state: MorpionBootstrapRunState) -> int | None:
    """Select the reserved source only; no newest-source substitution."""
    data = window(state)
    return (
        None
        if data is None or data["active"] is None
        else int(data["active"]["source"]["generation"])
    )


def source_record(
    state: MorpionBootstrapRunState, slot: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the immutable source through authoritative publication roots."""
    return next(
        r
        for r in publication_records(state.metadata)
        if r["id"] == slot["source"]["id"]
    )


def wait_reason(state: MorpionBootstrapRunState) -> str | None:
    """Return a passive wait reason; None permits ordinary Growth."""
    data = window(state)
    if data is None or data["active"] is None:
        return None
    slot = data["active"]
    if state.cycle_index < slot["target_cycle"]:
        return None
    record = source_record(state, slot)
    return (
        "await_dataset"
        if "accepted_dataset" not in record
        else "await_training"
        if "accepted_training" not in record
        else "await_learning_resolution"
    )


def validate_source(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, generation: int
) -> None:
    """Extra admission/acceptance fence, without weakening existing validation."""
    data = window(state)
    if data is None:
        return
    from .config import bootstrap_config_sha256, load_bootstrap_config

    config = load_bootstrap_config(paths.bootstrap_config_path)
    require(
        config.metadata.get(CONFIG_KEY) == data["policy"]
        and bootstrap_config_sha256(config) == state.metadata["bootstrap_config_hash"],
        "Persisted learning configuration changed.",
    )
    require(
        state.metadata.get("experiment_termination") is None,
        "Experiment is terminated.",
    )
    require(
        selected_source(state) == generation,
        "Generation is not the reserved learning source.",
    )
    head = publication_records(state.metadata)[-1]
    descriptor = read_committed_descriptor(paths.work_dir, head)
    # Scheduling compares authenticated identities. Existing scientific consumers
    # still validate all capsule/model bytes before reading or mutating the tree.
    require(
        (head["generation"], head["checkpoint_path"], head["tree_snapshot_path"])
        == (
            state.generation,
            state.latest_runtime_checkpoint_path,
            state.latest_tree_snapshot_path,
        )
        and descriptor["bootstrap_config_hash"]
        == state.metadata["bootstrap_config_hash"]
        and descriptor.get("evaluator_contract") == 1
        and descriptor.get("active_evaluator") == data["active"]["incumbent"],
        "Learning incumbent or committed head changed.",
    )


def validate_resolution(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, generation: int
) -> None:
    """All new activation entry points must meet the same owned fixed gate."""
    if window(state) is None:
        return
    validate_source(paths, state, generation)
    require(
        wait_reason(state) == "await_learning_resolution",
        "Learning cutover gate is not ready.",
    )


def close_window(
    state: MorpionBootstrapRunState, *, outcome: dict[str, Any]
) -> MorpionBootstrapRunState:
    """Return a proposed state; caller commits it under existing ownership."""
    data = window(state)
    assert data is not None and data["active"] is not None
    slot = data["active"]
    record = source_record(state, slot)
    require("accepted_training" in record, "Cannot resolve unaccepted learning.")
    closed = {
        **slot,
        "accepted_training": record["accepted_training"],
        "resolution_head": publication_records(state.metadata)[-1]["id"],
        **outcome,
    }
    return replace(
        state,
        metadata={
            **state.metadata,
            WINDOW_KEY: {
                **data,
                "active": None,
                "completed": [*data["completed"], closed],
            },
        },
    )


def on_publication(
    paths: MorpionBootstrapPaths,
    previous: MorpionBootstrapRunState | None,
    state: MorpionBootstrapRunState,
    incumbent: dict[str, Any] | None,
    activation: dict[str, Any] | None,
) -> MorpionBootstrapRunState:
    """Join reservation/skip/closure to the ordinary atomic publication commit."""
    # Validate the old cycle bound before admitting a proposed next cycle.
    data = window(previous if previous is not None else state)
    if data is None or previous is None:
        return state
    from .config import load_bootstrap_config

    # Context validation already binds the complete policy. Publication needs
    # only its cadence; avoid coupling the policy parser back to transitions.
    policy = load_bootstrap_config(paths.bootstrap_config_path).metadata[
        "continual_learning"
    ]
    require(isinstance(policy, dict), "Missing continual learning policy.")
    assert isinstance(policy, dict)
    cadence = policy["dataset_every_generations"]
    require(
        isinstance(cadence, int) and not isinstance(cadence, bool) and cadence > 0,
        "Invalid learning cadence.",
    )
    head = publication_records(state.metadata)[-1]
    require(
        state.cycle_index in (previous.cycle_index, previous.cycle_index + 1),
        "Invalid publication cycle transition.",
    )
    ordinary = state.cycle_index > previous.cycle_index
    slot = data["active"]
    if ordinary:
        require(
            state.cycle_index == previous.cycle_index + 1,
            "Non-sequential Growth cycle.",
        )
        require(wait_reason(previous) is None, "Growth would exceed the learning gate.")
    eligible = state.generation > 1 and state.generation % cadence == 0
    if activation is not None:
        if eligible:
            skipped = {
                "source": source_identity(head),
                "reason": "non_growth_publication",
                "window_source": None if slot is None else slot["source"]["id"],
            }
            state = replace(
                state,
                metadata={
                    **state.metadata,
                    WINDOW_KEY: {**data, "skipped": [*data["skipped"], skipped]},
                },
            )
        require(
            slot is not None and not ordinary,
            "Activation has no pending learning window.",
        )
        assert slot is not None
        require(
            activation["request"]["training_generation"] == slot["source"]["generation"]
            and activation["request"]["previous_model"] == slot["incumbent"],
            "Activation retargeted the learning slot.",
        )
        return close_window(
            state,
            outcome={
                "outcome": "activated",
                "request_sha256": activation["request_sha256"],
            },
        )
    if slot is not None:
        require(
            incumbent == slot["incumbent"], "Growth changed the learning incumbent."
        )
    if not eligible:
        return state
    if not ordinary or slot is not None:
        skipped = {
            "source": source_identity(head),
            "reason": "skipped_by_active_window"
            if ordinary
            else "non_growth_publication",
            "window_source": None if slot is None else slot["source"]["id"],
        }
        return replace(
            state,
            metadata={
                **state.metadata,
                WINDOW_KEY: {**data, "skipped": [*data["skipped"], skipped]},
            },
        )
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    assert schedule is not None and incumbent is not None
    slot = {
        "source": source_identity(head),
        "source_cycle": state.cycle_index,
        "target_cycle": state.cycle_index + schedule.growth_lookahead_cycles,
        "incumbent": incumbent,
    }
    return replace(
        state, metadata={**state.metadata, WINDOW_KEY: {**data, "active": slot}}
    )
