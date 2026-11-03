"""Opt-in single-source learning windows; all transitions require Growth ownership.

This is a Morpion scientific policy, not generic attempt coordination. Accepted
references and the ordinary publication commit remain the scientific authority.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

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
READY_WINDOW_KEY = "learning_window_v2"


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


@dataclass(frozen=True, slots=True)
class ReadyBoundedGrowthWindow:
    """Resolve accepted learning at the next boundary, with a hard cycle cap."""

    max_growth_cycles: int

    def __post_init__(self) -> None:
        """Require an explicit finite, unambiguous cycle allowance."""
        require(
            isinstance(self.max_growth_cycles, int)
            and not isinstance(self.max_growth_cycles, bool)
            and self.max_growth_cycles > 0,
            "Maximum Growth cycles must be a positive integer.",
        )

    def validate_supported(self, args: MorpionBootstrapArgs) -> None:
        """Retain the fixed policy's supported scientific domain."""
        validate_supported(args)

    def to_metadata(self) -> dict[str, Any]:
        """Give completion-driven scheduling its own scientific identity."""
        return {
            "schema_version": 1,
            "kind": "ready_bounded_growth_window_v1",
            "max_growth_cycles": self.max_growth_cycles,
            "queue_capacity": 1,
            "cutover": "next_owned_boundary",
            "refresh": "owned_full",
        }


type LearningSchedule = BoundedGrowthWindow | ReadyBoundedGrowthWindow


def window_key(schedule: LearningSchedule | None) -> str:
    """Select the explicit format without rewriting historical state."""
    return (
        READY_WINDOW_KEY
        if isinstance(schedule, ReadyBoundedGrowthWindow)
        else WINDOW_KEY
    )


def cycle_limit(schedule: LearningSchedule) -> int:
    """Return ordinary cycles, never publication generations or elapsed time."""
    return (
        schedule.max_growth_cycles
        if isinstance(schedule, ReadyBoundedGrowthWindow)
        else schedule.growth_lookahead_cycles
    )


def limit_key(schedule: LearningSchedule) -> str:
    """Keep the historical target field unchanged."""
    return (
        "limit_cycle"
        if isinstance(schedule, ReadyBoundedGrowthWindow)
        else "target_cycle"
    )


def schedule_from_metadata(metadata: dict[str, Any]) -> LearningSchedule | None:
    """Reject unknown versions/options rather than interpreting a partial policy."""
    if CONFIG_KEY not in metadata:
        return None
    raw = metadata[CONFIG_KEY]
    require(isinstance(raw, dict), "Invalid learning schedule.")
    assert isinstance(raw, dict)
    schedule: LearningSchedule
    if raw.get("kind") == "ready_bounded_growth_window_v1":
        maximum = raw.get("max_growth_cycles")
        require(
            isinstance(maximum, int),
            "Maximum Growth cycles must be a positive integer.",
        )
        assert isinstance(maximum, int)
        schedule = ReadyBoundedGrowthWindow(maximum)
    else:
        lookahead = raw.get("growth_lookahead_cycles")
        require(
            isinstance(lookahead, int), "Growth lookahead must be a positive integer."
        )
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


def context_marker(schedule: LearningSchedule | None) -> dict[str, Any]:
    """Old binaries reject the new context instead of silently changing policy."""
    return (
        {"schema_version": 1}
        if schedule is None
        else {
            "schema_version": 3
            if isinstance(schedule, ReadyBoundedGrowthWindow)
            else 2,
            "learning_schedule_sha256": object_digest(schedule.to_metadata()),
        }
    )


def initial_window(schedule: LearningSchedule | None) -> dict[str, Any]:
    """Fresh genesis only; no adoption or inferred state for existing runs."""
    return (
        {}
        if schedule is None
        else {
            window_key(schedule): {
                "policy": schedule.to_metadata(),
                "active": None,
                "completed": [],
                "skipped": [],
                **(
                    {"decisions": []}
                    if isinstance(schedule, ReadyBoundedGrowthWindow)
                    else {}
                ),
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
    require(
        not (WINDOW_KEY in state.metadata and READY_WINDOW_KEY in state.metadata),
        "Mixed learning window versions.",
    )
    key = READY_WINDOW_KEY if READY_WINDOW_KEY in state.metadata else WINDOW_KEY
    raw = state.metadata.get(key)
    marker = state.metadata.get("continual_learning")
    if raw is None:
        require(
            key != READY_WINDOW_KEY
            and (marker is None or marker == {"schema_version": 1}),
            "Missing durable learning window.",
        )
        return None
    require(isinstance(raw, dict), "Invalid durable learning window.")
    assert isinstance(raw, dict)
    require(
        set(raw)
        == {"policy", "active", "completed", "skipped"}
        | ({"decisions"} if key == READY_WINDOW_KEY else set()),
        "Invalid window fields.",
    )
    schedule = schedule_from_metadata({CONFIG_KEY: raw["policy"]})
    require(
        marker == context_marker(schedule) and key == window_key(schedule),
        "Learning schedule identity changed.",
    )
    require(
        isinstance(raw["completed"], list) and isinstance(raw["skipped"], list),
        "Invalid window history.",
    )
    if key == READY_WINDOW_KEY:
        require(
            isinstance(marker, dict)
            and isinstance(marker.get("schema_version"), int)
            and not isinstance(marker.get("schema_version"), bool),
            "Invalid ready context version.",
        )
        require(isinstance(raw["decisions"], list), "Invalid boundary decisions.")
        _validate_decisions(raw, state)
    slot = raw["active"]
    if slot is not None:
        require(isinstance(slot, dict), "Invalid active learning slot.")
        records = publication_records(state.metadata)
        source = next((r for r in records if r["id"] == slot["source"]["id"]), None)
        require(source is not None, "Learning source is not committed.")
        assert source is not None and schedule is not None
        bound = limit_key(schedule)
        if isinstance(schedule, ReadyBoundedGrowthWindow):
            require(
                set(slot)
                == {"source", "source_cycle", "limit_cycle", "incumbent", "sequence"}
                and isinstance(slot["sequence"], int)
                and not isinstance(slot["sequence"], bool)
                and slot["sequence"] == len(raw["completed"]) + 1,
                "Invalid ready learning slot.",
            )
        require(
            slot["source"] == source_identity(source)
            and "retired_payload_v1" not in source
            and isinstance(slot["source_cycle"], int)
            and not isinstance(slot["source_cycle"], bool)
            and isinstance(slot[bound], int)
            and not isinstance(slot[bound], bool)
            and slot[bound] == slot["source_cycle"] + cycle_limit(schedule)
            and slot["source_cycle"] <= state.cycle_index <= slot[bound],
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


def wait_reason(
    state: MorpionBootstrapRunState,
) -> Literal["await_dataset", "await_training", "await_learning_resolution"] | None:
    """Return a passive wait reason; None permits ordinary Growth."""
    data = window(state)
    if data is None or data["active"] is None:
        return None
    slot = data["active"]
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    assert schedule is not None
    record = source_record(state, slot)
    if isinstance(schedule, ReadyBoundedGrowthWindow) and "accepted_training" in record:
        return "await_learning_resolution"
    if state.cycle_index < slot[limit_key(schedule)]:
        return None
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
    """All activation entry points must meet their explicit owned policy gate."""
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
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    if isinstance(schedule, ReadyBoundedGrowthWindow):
        head = publication_records(state.metadata)[-1]
        data = _record_decision(
            data,
            state,
            parent=head["parent_id"]
            if outcome["outcome"] == "activated"
            else head["id"],
            before=state.cycle_index,
            action=outcome["outcome"],
            accepted=record["accepted_training"],
        )
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
            window_key(schedule): {
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
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    assert schedule is not None
    key = window_key(schedule)
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
    if ordinary and isinstance(schedule, ReadyBoundedGrowthWindow):
        data = _record_decision(
            data,
            state,
            parent=publication_records(previous.metadata)[-1]["id"],
            before=previous.cycle_index,
            action="grow",
            accepted=None,
        )
        state = replace(state, metadata={**state.metadata, key: data})
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
                    key: {**data, "skipped": [*data["skipped"], skipped]},
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
                key: {**data, "skipped": [*data["skipped"], skipped]},
            },
        )
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    assert schedule is not None and incumbent is not None
    slot = {
        "source": source_identity(head),
        "source_cycle": state.cycle_index,
        limit_key(schedule): state.cycle_index + cycle_limit(schedule),
        **(
            {"sequence": len(data["completed"]) + 1}
            if isinstance(schedule, ReadyBoundedGrowthWindow)
            else {}
        ),
        "incumbent": incumbent,
    }
    return replace(state, metadata={**state.metadata, key: {**data, "active": slot}})


def on_no_publication(
    previous: MorpionBootstrapRunState, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """Count a persisted no-save Growth attempt without inventing a publication."""
    data = window(previous)
    if data is None:
        return state
    schedule = schedule_from_metadata({CONFIG_KEY: data["policy"]})
    if not isinstance(schedule, ReadyBoundedGrowthWindow):
        return state
    before = publication_records(previous.metadata)
    after = publication_records(state.metadata)
    require(
        state.generation == previous.generation
        and state.cycle_index == previous.cycle_index + 1
        and before == after
        and wait_reason(previous) is None,
        "Invalid no-save Growth boundary.",
    )
    head = after[-1]["id"]
    data = _record_decision(
        data,
        state,
        parent=head,
        before=previous.cycle_index,
        action="no_save",
        accepted=None,
    )
    return replace(state, metadata={**state.metadata, READY_WINDOW_KEY: data})


def _record_decision(
    data: dict[str, Any],
    state: MorpionBootstrapRunState,
    *,
    parent: str,
    before: int,
    action: str,
    accepted: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind one decision to an existing commit, never to a poll or timing file."""
    slot = data["active"]
    return {
        **data,
        "decisions": [
            *data["decisions"],
            {
                "sequence": len(data["completed"]) + 1,
                "source": None
                if slot is None
                else {
                    k: slot["source"][k]
                    for k in ("id", "generation", "descriptor_sha256")
                },
                "parent_publication": parent,
                "publication": publication_records(state.metadata)[-1]["id"],
                "before_cycle": before,
                "after_cycle": state.cycle_index,
                "accepted_training": accepted,
                "action": action,
            },
        ],
    }


def _validate_decisions(data: dict[str, Any], state: MorpionBootstrapRunState) -> None:
    """Validate the bounded commit history without consulting observational clocks."""
    records = {r["id"]: r for r in publication_records(state.metadata)}
    decisions = data["decisions"]
    require(
        len(decisions) <= len(records) + len(data["completed"]) + state.cycle_index + 1,
        "Boundary decisions exceed committed work.",
    )
    previous_cycle = -1
    sequence = 1
    for decision in decisions:
        require(
            isinstance(decision, dict)
            and set(decision)
            == {
                "sequence",
                "source",
                "parent_publication",
                "publication",
                "before_cycle",
                "after_cycle",
                "accepted_training",
                "action",
            },
            "Invalid boundary decision fields.",
        )
        require(
            all(
                isinstance(decision[k], int) and not isinstance(decision[k], bool)
                for k in ("sequence", "before_cycle", "after_cycle")
            )
            and decision["sequence"] == sequence
            and -1
            <= decision["before_cycle"]
            <= decision["after_cycle"]
            <= state.cycle_index
            and previous_cycle == decision["before_cycle"]
            and decision["publication"] in records
            and decision["parent_publication"] in records,
            "Invalid boundary decision identity or cycle.",
        )
        head = records[decision["publication"]]
        source = decision["source"]
        if source is not None:
            require(
                isinstance(source, dict)
                and source.get("id") in records
                and source
                == {
                    k: records[source["id"]][k]
                    for k in ("id", "generation", "descriptor_sha256")
                },
                "Invalid boundary source.",
            )
        if decision["action"] == "no_save":
            require(
                decision["accepted_training"] is None
                and decision["after_cycle"] == decision["before_cycle"] + 1
                and decision["publication"] == decision["parent_publication"],
                "Invalid no-save boundary decision.",
            )
        elif decision["action"] == "grow":
            require(
                decision["accepted_training"] is None
                and decision["after_cycle"] == decision["before_cycle"] + 1
                and head["parent_id"] == decision["parent_publication"],
                "Invalid Growth boundary decision.",
            )
        else:
            require(
                decision["action"] in {"retained", "activated"}
                and source is not None
                and decision["accepted_training"] is not None
                and decision["accepted_training"]
                == records[source["id"]].get("accepted_training")
                and decision["after_cycle"] == decision["before_cycle"]
                and decision["parent_publication"]
                == (
                    head["parent_id"]
                    if decision["action"] == "activated"
                    else head["id"]
                ),
                "Invalid learning resolution decision.",
            )
            require(
                sequence <= len(data["completed"])
                and data["completed"][sequence - 1]["outcome"] == decision["action"]
                and data["completed"][sequence - 1]["accepted_training"]
                == decision["accepted_training"]
                and data["completed"][sequence - 1]["resolution_head"]
                == decision["publication"],
                "Resolution history changed.",
            )
            sequence += 1
        previous_cycle = decision["after_cycle"]
    require(
        previous_cycle == state.cycle_index and sequence == len(data["completed"]) + 1,
        "Incomplete boundary decisions.",
    )
