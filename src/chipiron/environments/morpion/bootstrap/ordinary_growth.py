"""The ordinary Growth write set, separate from its captured computation source.

Callers hold Growth ownership. This is not a metadata merge: the result may write
only tree/cycle observations and the already admitted computation config hash.
Accepted worker authority comes from the latest durable state, with all other
changes treated as a computation conflict.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .generation_publication_io import (
    PUBLICATION_KEY,
    publication_records,
    require,
    require_growth_source,
)
from .run_state import MorpionBootstrapRunState
from .worker_result_io import (
    DATASET_NAMESPACE,
    TRAINING_NAMESPACE,
    read_proposal,
    result_protocol_enabled,
    workspace_epoch,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bootstrap_paths import MorpionBootstrapPaths


# These are the outputs of cycle_runtime and pipeline.stages, not worker outputs.
_CYCLE_METADATA = frozenset({
    "bootstrap_config_hash",
    "runtime_checkpoint_path",
    "bootstrap_applied_control",
    "bootstrap_applied_runtime_control",
    "bootstrap_effective_runtime",
    "bootstrap_effective_runtime_hash",
    "training_skipped_reason",
    "growth_status",
    "checkpoint_skipped",
    "checkpoint_skipped_reason",
    "branch_count",
    "tree_branch_limit",
    "experiment_termination",
})
_SAVED_METADATA = frozenset({
    "tree",
    "memory",
    "state_eviction",
    "checkpoint",
    "training_export",
    "training_export_profile",
})
_WORKER_METADATA = frozenset({
    PUBLICATION_KEY,
    "continuous_pipeline_v1",
    "prepared_dataset_v1",
})
_PREREQUISITE_METADATA = frozenset({
    "bootstrap_config_hash",
    "managed_sessions_v1",
    "experiment_termination",
})
_ACCEPTED = frozenset({"accepted_dataset", "accepted_training"})
_WORKER_PROOFS = frozenset({"continuous_ready", "continuous_retirement"})
_WORKER_HISTORY = frozenset({
    "continuous_admissions",
    "continuous_dispositions",
    "continuous_consumed",
})


def _without(
    metadata: Mapping[str, object], fields: frozenset[str]
) -> dict[str, object]:
    """Compare the complete complement of an explicitly owned set of fields."""
    return {key: value for key, value in metadata.items() if key not in fields}


def _worker_history(
    paths: MorpionBootstrapPaths,
    source: MorpionBootstrapRunState,
    latest: MorpionBootstrapRunState,
) -> None:
    """Only append accepted references to existing, otherwise identical records."""
    if PUBLICATION_KEY not in source.metadata:
        require(
            PUBLICATION_KEY not in latest.metadata,
            "Growth publication protocol changed during computation.",
        )
        return
    before = source.metadata[PUBLICATION_KEY]
    after = latest.metadata[PUBLICATION_KEY]
    require(
        _without(before, frozenset({"generations"}))
        == _without(after, frozenset({"generations"})),
        "Growth publication head or protocol changed during computation.",
    )
    old = publication_records(source.metadata)
    new = publication_records(latest.metadata)
    require(len(old) == len(new), "Growth computation checkpoint changed.")
    for captured, current in zip(old, new, strict=True):
        require(
            _without(captured, _ACCEPTED | _WORKER_PROOFS | _WORKER_HISTORY)
            == _without(current, _ACCEPTED | _WORKER_PROOFS | _WORKER_HISTORY),
            "Growth scientific source record changed during computation.",
        )
        for key in _ACCEPTED | _WORKER_PROOFS:
            require(
                key not in captured or current.get(key) == captured[key],
                "Accepted worker result was removed or replaced.",
            )
            if key in current and key not in captured and key in _ACCEPTED:
                dataset = key == "accepted_dataset"
                require(
                    result_protocol_enabled(
                        latest, "fenced_dataset" if dataset else "durable_training", key
                    ),
                    "Worker acceptance lacks its scientific protocol.",
                )
                proposal = read_proposal(
                    paths,
                    current[key],
                    namespace=DATASET_NAMESPACE if dataset else TRAINING_NAMESPACE,
                )
                admission = proposal["admission"]
                admitted_source = (
                    admission["source"] if dataset else admission["dataset"]["source"]
                )
                require(
                    admission["identity"] == workspace_epoch(latest)
                    and admission["generation"] == current["generation"]
                    and admitted_source
                    == {
                        k: current[k]
                        for k in ("id", "descriptor_sha256", "scientific_identity")
                    },
                    "Accepted worker proof belongs to another scientific source.",
                )
                require(
                    dataset
                    or admission["dataset"]["reference"]
                    == current.get("accepted_dataset"),
                    "Accepted Training lost its exact Dataset input.",
                )
            if key in _WORKER_PROOFS and key in current:
                require(
                    "continuous_pipeline_v1" in source.metadata,
                    "Continuous proof lacks its scientific protocol.",
                )
        for key in _WORKER_HISTORY:
            earlier, later = captured.get(key, []), current.get(key, [])
            require(
                isinstance(earlier, list)
                and isinstance(later, list)
                and later[: len(earlier)] == earlier
                and (later == earlier or "continuous_pipeline_v1" in source.metadata),
                "Continuous worker history was removed or replaced.",
            )


@dataclass(frozen=True)
class OrdinaryGrowthUpdate:
    """Capture the durable baseline and the config already admitted for computation.

    Relaunch validation precedes capture. Neither a later authority change nor a
    later config-file change is admitted by this result.
    """

    source: MorpionBootstrapRunState
    result: MorpionBootstrapRunState
    config_hash: str

    def apply(
        self, paths: MorpionBootstrapPaths, latest: MorpionBootstrapRunState
    ) -> MorpionBootstrapRunState:
        """Fence computation prerequisites and replace only ordinary Growth fields.

        The receipt/publication owner subsequently authenticates dependencies and
        evaluates the current session and scheduling rules before its sole save.
        """
        source, result = self.source, self.result
        require_growth_source(
            replace(latest, metadata=source.metadata) == source,
            "Growth computation checkpoint, evaluator or cycle changed.",
        )
        require(
            _without(latest.metadata, _WORKER_METADATA | _PREREQUISITE_METADATA)
            == _without(source.metadata, _WORKER_METADATA | _PREREQUISITE_METADATA),
            "Growth prerequisite or unowned metadata changed outside the publication contract.",
        )
        require_growth_source(
            _without(latest.metadata, _WORKER_METADATA)
            == _without(source.metadata, _WORKER_METADATA),
            "Growth prerequisite or unowned metadata changed.",
        )
        _worker_history(paths, source, latest)
        require(
            result.metadata.get("bootstrap_config_hash") == self.config_hash,
            "Growth result differs from its captured configuration.",
        )
        require(
            result.generation in (source.generation, source.generation + 1)
            and result.cycle_index in (source.cycle_index, source.cycle_index + 1),
            "Nonsequential ordinary Growth result.",
        )
        saved = result.generation != source.generation
        require(
            not saved or result.cycle_index == source.cycle_index + 1,
            "Ordinary Growth cannot publish an activation or pre-switch drain.",
        )
        fields = _CYCLE_METADATA | (_SAVED_METADATA if saved else frozenset())
        require(
            _without(result.metadata, fields) == _without(source.metadata, fields),
            "Ordinary Growth result changed worker or scientific authority.",
        )
        # Explicit field ownership also catches newly introduced run-state fields.
        expected = replace(
            source,
            generation=result.generation,
            cycle_index=result.cycle_index,
            latest_runtime_checkpoint_path=result.latest_runtime_checkpoint_path,
            latest_tree_snapshot_path=result.latest_tree_snapshot_path,
            tree_size_at_last_save=result.tree_size_at_last_save,
            last_save_unix_s=result.last_save_unix_s,
            latest_frontier_status=result.latest_frontier_status,
            active_evaluator_name=result.active_evaluator_name,
            metadata=result.metadata,
        )
        require(expected == result, "Ordinary Growth wrote an unowned state field.")
        if not saved:
            require(
                replace(
                    result,
                    cycle_index=source.cycle_index,
                    active_evaluator_name=source.active_evaluator_name,
                    metadata=source.metadata,
                )
                == source,
                "Unsaved Growth changed checkpoint observations.",
            )
        metadata = dict(latest.metadata)
        for key in fields:
            if key in result.metadata:
                metadata[key] = result.metadata[key]
            else:
                metadata.pop(key, None)
        return replace(
            latest,
            generation=result.generation,
            cycle_index=result.cycle_index,
            latest_runtime_checkpoint_path=result.latest_runtime_checkpoint_path,
            latest_tree_snapshot_path=result.latest_tree_snapshot_path,
            tree_size_at_last_save=result.tree_size_at_last_save,
            last_save_unix_s=result.last_save_unix_s,
            latest_frontier_status=result.latest_frontier_status,
            active_evaluator_name=result.active_evaluator_name,
            metadata=metadata,
        )
