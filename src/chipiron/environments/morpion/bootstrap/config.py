"""Persisted bootstrap configuration helpers for Morpion runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_MODEL_KIND,
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
)

from .bootstrap_errors import InvalidReevaluationBlendAlphaError
from .record_status import (
    MORPION_BOOTSTRAP_GAME,
    MORPION_BOOTSTRAP_INITIAL_PATTERN,
    MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    MORPION_BOOTSTRAP_VARIANT,
)

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
    from .pipeline_config import (
        MorpionEvaluatorUpdatePolicy,
        MorpionPipelineMode,
        MorpionPipelineStage,
        MorpionTrainingExportMode,
    )
    from .pv_family_targets import PvFamilyTargetPolicy

from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    DEFAULT_MORPION_TRAINING_EXPORT_MODE,
)

BOOTSTRAP_CONFIG_HASH_METADATA_KEY = "bootstrap_config_hash"
DEFAULT_MORPION_TREE_BRANCH_LIMIT = 128


def _empty_metadata() -> dict[str, object]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRuntimeConfig:
    """Runtime controls for one persistent Morpion bootstrap run."""

    save_after_tree_growth_factor: float
    save_after_seconds: float
    max_growth_steps_per_cycle: int
    tree_branch_limit: int
    reevaluation_blend_alpha: float = 1.0
    min_available_ram_mb: int | None = None
    candidate_checkpoint_load_headroom_factor: float = 60.0
    candidate_checkpoint_load_min_headroom_mb: int = 512

    def __post_init__(self) -> None:
        """Validate runtime scalar controls."""
        if isinstance(self.reevaluation_blend_alpha, bool) or not (
            0.0 <= self.reevaluation_blend_alpha <= 1.0
        ):
            raise InvalidReevaluationBlendAlphaError
        if self.min_available_ram_mb is not None and (
            isinstance(self.min_available_ram_mb, bool) or self.min_available_ram_mb < 0
        ):
            raise MalformedMorpionBootstrapConfigError.invalid_int(
                "runtime.min_available_ram_mb"
            )
        if (
            isinstance(self.candidate_checkpoint_load_headroom_factor, bool)
            or self.candidate_checkpoint_load_headroom_factor < 0
        ):
            raise MalformedMorpionBootstrapConfigError.invalid_float(
                "runtime.candidate_checkpoint_load_headroom_factor"
            )
        if (
            isinstance(self.candidate_checkpoint_load_min_headroom_mb, bool)
            or self.candidate_checkpoint_load_min_headroom_mb < 0
        ):
            raise MalformedMorpionBootstrapConfigError.invalid_int(
                "runtime.candidate_checkpoint_load_min_headroom_mb"
            )


@dataclass(frozen=True, slots=True)
class MorpionBootstrapDatasetConfig:
    """Dataset extraction controls for one persistent Morpion bootstrap run."""

    require_exact_or_terminal: bool
    min_depth: int | None
    min_visit_count: int | None
    max_rows: int | None
    use_backed_up_value: bool
    family_target_policy: PvFamilyTargetPolicy = "none"
    family_prediction_blend: float = 0.25


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRolloutConfig:
    """Search expansion rollout settings for Morpion bootstrap.

    Disabled by default to preserve old Python launcher behavior.
    ``max_extra_steps=None`` means Anemone rollout continues until a normal stop
    condition.
    """

    enabled: bool = False
    max_extra_steps: int | None = None
    action_selector_kind: str = "random_legal_prefer_openable"
    random_seed: int | None = 0
    stop_on_existing_node: bool = False

    def __post_init__(self) -> None:
        """Validate rollout scalar controls."""
        if isinstance(self.max_extra_steps, bool):
            raise MalformedMorpionBootstrapConfigError.invalid_int(
                "search.rollout.max_extra_steps"
            )
        if self.max_extra_steps is not None and self.max_extra_steps < 0:
            raise MalformedMorpionBootstrapConfigError.invalid_int(
                "search.rollout.max_extra_steps"
            )
        if isinstance(self.random_seed, bool):
            raise MalformedMorpionBootstrapConfigError.invalid_int(
                "search.rollout.random_seed"
            )


@dataclass(frozen=True, slots=True)
class MorpionBootstrapSearchConfig:
    """Search behavior controls for one persistent Morpion bootstrap run."""

    rollout: MorpionBootstrapRolloutConfig = field(
        default_factory=MorpionBootstrapRolloutConfig
    )


@dataclass(frozen=True, slots=True)
class MorpionBootstrapExperimentIdentityConfig:
    """Semantic identity fields that define one Morpion bootstrap run."""

    game: str
    variant: str
    initial_pattern: str
    initial_point_count: int


@dataclass(frozen=True, slots=True)
class MorpionBootstrapConfig:
    """Canonical persisted configuration for one Morpion bootstrap run."""

    experiment: MorpionBootstrapExperimentIdentityConfig
    runtime: MorpionBootstrapRuntimeConfig
    dataset: MorpionBootstrapDatasetConfig
    evaluators: MorpionEvaluatorsConfig
    search: MorpionBootstrapSearchConfig = field(
        default_factory=MorpionBootstrapSearchConfig
    )
    validation_fraction: float = 0.2
    validation_seed: int = 0
    evaluator_update_policy: MorpionEvaluatorUpdatePolicy = (
        DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY
    )
    pipeline_mode: MorpionPipelineMode = DEFAULT_MORPION_PIPELINE_MODE
    training_export_mode: MorpionTrainingExportMode = (
        DEFAULT_MORPION_TRAINING_EXPORT_MODE
    )
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


class MalformedMorpionBootstrapConfigError(TypeError):
    """Raised when one persisted bootstrap config payload is malformed."""

    @classmethod
    def invalid_json(cls, path: str | Path) -> MalformedMorpionBootstrapConfigError:
        """Return the invalid persisted JSON config error."""
        return cls(f"Morpion bootstrap config at {path!s} is not valid JSON.")

    @classmethod
    def invalid_top_level_mapping(cls) -> MalformedMorpionBootstrapConfigError:
        """Return the malformed top-level payload error."""
        return cls("Morpion bootstrap config must be a mapping with string keys.")

    @classmethod
    def invalid_section(cls, section_name: str) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed section error."""
        return cls(
            f"Morpion bootstrap config field `{section_name}` must be a mapping."
        )

    @classmethod
    def invalid_required_str(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed required-string field error."""
        return cls(f"Morpion bootstrap config field `{field_name}` must be a string.")

    @classmethod
    def invalid_bool(cls, field_name: str) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed bool field error."""
        return cls(f"Morpion bootstrap config field `{field_name}` must be a bool.")

    @classmethod
    def invalid_int(cls, field_name: str) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed integer-like field error."""
        return cls(
            f"Morpion bootstrap config field `{field_name}` must be integer-like."
        )

    @classmethod
    def invalid_float(cls, field_name: str) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed float-like field error."""
        return cls(f"Morpion bootstrap config field `{field_name}` must be float-like.")

    @classmethod
    def invalid_metadata(cls) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed metadata field error."""
        return cls("Morpion bootstrap config field `metadata` must be a mapping.")

    @classmethod
    def invalid_evaluators(
        cls,
    ) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed evaluators field error."""
        return cls(
            "Morpion bootstrap config field `evaluators` must contain a valid "
            "Morpion evaluator mapping."
        )

    @classmethod
    def invalid_feature_names(
        cls,
        field_name: str,
    ) -> MalformedMorpionBootstrapConfigError:
        """Return one malformed feature-names field error."""
        return cls(
            f"Morpion bootstrap config field `{field_name}` must be a list or tuple of strings."
        )


class UnsafeMorpionBootstrapConfigChangeError(ValueError):
    """Raised when one relaunch changes unsafe bootstrap config fields."""


class IncompatibleStageBootstrapConfigError(ValueError):
    """Raised when one worker drifts from the persisted bootstrap config."""

    @classmethod
    def for_field(
        cls,
        *,
        stage: str,
        field_name: str,
        persisted_value: object,
        requested_value: object,
        stage_uses_field: bool,
    ) -> IncompatibleStageBootstrapConfigError:
        """Build one deterministic stage/config compatibility error."""
        stage_message = (
            f"field {field_name!r} is used by {stage}"
            if stage_uses_field
            else f"field {field_name!r} differs from the persisted bootstrap config"
        )
        return cls(
            "Incompatible Morpion bootstrap config for "
            f"pipeline stage {stage!r}: {stage_message}, but bootstrap config is "
            "already persisted. Change bootstrap_config.json intentionally or start "
            "a new work_dir "
            f"(persisted={persisted_value!r}, requested={requested_value!r})."
        )


# These fields control growth worker batching/checkpoint cadence and are
# intentionally mutable between relaunches. They affect how much search work a
# process does and when it checkpoints, but not the persisted experiment
# protocol used by dataset/training/reevaluation.
GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS = frozenset(
    {
        "max_growth_steps_per_cycle",
        "tree_branch_limit",
        "reevaluation_blend_alpha",
        "min_available_ram_mb",
        "candidate_checkpoint_load_headroom_factor",
        "candidate_checkpoint_load_min_headroom_mb",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    }
)

GROWTH_SEARCH_BOOTSTRAP_STAGE_VALUE_FIELDS = frozenset(
    {
        "rollout_after_opening",
        "rollout_max_extra_steps",
        "rollout_action_selector_kind",
        "rollout_random_seed",
        "rollout_stop_on_existing_node",
    }
)

GROWTH_SEARCH_BOOTSTRAP_CONFIG_DIFF_FIELDS = frozenset(
    {
        "search.rollout.enabled",
        "search.rollout.max_extra_steps",
        "search.rollout.action_selector_kind",
        "search.rollout.random_seed",
        "search.rollout.stop_on_existing_node",
    }
)

STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS: dict[str, frozenset[str]] = {
    "dataset": GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    "dataset_worker": GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    "training": GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    "training_worker": GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    "reevaluation": GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
}

RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS = (
    GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS
)


def bootstrap_config_from_args(args: MorpionBootstrapArgs) -> MorpionBootstrapConfig:
    """Build the canonical persisted bootstrap config from current args."""
    return MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=MORPION_BOOTSTRAP_GAME,
            variant=MORPION_BOOTSTRAP_VARIANT,
            initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
            initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
        ),
        runtime=MorpionBootstrapRuntimeConfig(
            save_after_tree_growth_factor=args.save_after_tree_growth_factor,
            save_after_seconds=args.save_after_seconds,
            max_growth_steps_per_cycle=args.max_growth_steps_per_cycle,
            tree_branch_limit=args.tree_branch_limit,
            reevaluation_blend_alpha=args.reevaluation_blend_alpha,
            min_available_ram_mb=args.min_available_ram_mb,
            candidate_checkpoint_load_headroom_factor=(
                args.candidate_checkpoint_load_headroom_factor
            ),
            candidate_checkpoint_load_min_headroom_mb=(
                args.candidate_checkpoint_load_min_headroom_mb
            ),
        ),
        dataset=MorpionBootstrapDatasetConfig(
            require_exact_or_terminal=args.require_exact_or_terminal,
            min_depth=args.min_depth,
            min_visit_count=args.min_visit_count,
            max_rows=args.max_rows,
            use_backed_up_value=args.use_backed_up_value,
            family_target_policy=args.dataset_family_target_policy,
            family_prediction_blend=args.dataset_family_prediction_blend,
        ),
        evaluators=args.resolved_evaluators_config(),
        search=args.search,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        evaluator_update_policy=args.evaluator_update_policy,
        pipeline_mode=args.pipeline_mode,
        training_export_mode=args.training_export_mode,
    )


def bootstrap_config_to_dict(config: MorpionBootstrapConfig) -> dict[str, object]:
    """Serialize one bootstrap config into JSON-friendly data."""
    return {
        "experiment": {
            "game": config.experiment.game,
            "variant": config.experiment.variant,
            "initial_pattern": config.experiment.initial_pattern,
            "initial_point_count": config.experiment.initial_point_count,
        },
        "runtime": {
            "save_after_tree_growth_factor": config.runtime.save_after_tree_growth_factor,
            "save_after_seconds": config.runtime.save_after_seconds,
            "max_growth_steps_per_cycle": config.runtime.max_growth_steps_per_cycle,
            "tree_branch_limit": config.runtime.tree_branch_limit,
            "reevaluation_blend_alpha": config.runtime.reevaluation_blend_alpha,
            "min_available_ram_mb": config.runtime.min_available_ram_mb,
            "candidate_checkpoint_load_headroom_factor": (
                config.runtime.candidate_checkpoint_load_headroom_factor
            ),
            "candidate_checkpoint_load_min_headroom_mb": (
                config.runtime.candidate_checkpoint_load_min_headroom_mb
            ),
        },
        "dataset": {
            "require_exact_or_terminal": config.dataset.require_exact_or_terminal,
            "min_depth": config.dataset.min_depth,
            "min_visit_count": config.dataset.min_visit_count,
            "max_rows": config.dataset.max_rows,
            "use_backed_up_value": config.dataset.use_backed_up_value,
            "family_target_policy": config.dataset.family_target_policy,
            "family_prediction_blend": config.dataset.family_prediction_blend,
        },
        "search": {
            "rollout": {
                "enabled": config.search.rollout.enabled,
                "max_extra_steps": config.search.rollout.max_extra_steps,
                "action_selector_kind": config.search.rollout.action_selector_kind,
                "random_seed": config.search.rollout.random_seed,
                "stop_on_existing_node": config.search.rollout.stop_on_existing_node,
            }
        },
        "evaluators": _evaluators_config_to_dict(config.evaluators),
        "validation_fraction": config.validation_fraction,
        "validation_seed": config.validation_seed,
        "evaluator_update_policy": config.evaluator_update_policy,
        "pipeline_mode": config.pipeline_mode,
        "training_export_mode": config.training_export_mode,
        "metadata": dict(config.metadata),
    }


def bootstrap_config_from_dict(data: object) -> MorpionBootstrapConfig:
    """Deserialize one bootstrap config from JSON-friendly data."""
    from .evaluator_config import MorpionEvaluatorsConfig

    if not _is_str_key_mapping(data):
        raise MalformedMorpionBootstrapConfigError.invalid_top_level_mapping()

    payload = cast("Mapping[str, object]", data)
    experiment = _require_section_mapping(
        payload.get("experiment"), section_name="experiment"
    )
    runtime = _require_section_mapping(payload.get("runtime"), section_name="runtime")
    dataset = _require_section_mapping(payload.get("dataset"), section_name="dataset")
    evaluators_data = _require_section_mapping(
        payload.get("evaluators"),
        section_name="evaluators",
    )
    evaluator_entries = _require_section_mapping(
        evaluators_data.get("evaluators"),
        section_name="evaluators.evaluators",
    )

    try:
        evaluators = MorpionEvaluatorsConfig(
            evaluators={
                evaluator_name: _evaluator_spec_from_config_payload(
                    evaluator_name=evaluator_name,
                    spec_payload=spec_payload,
                )
                for evaluator_name, spec_payload in evaluator_entries.items()
            }
        )
    except (TypeError, ValueError) as exc:
        raise MalformedMorpionBootstrapConfigError.invalid_evaluators() from exc

    return MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=_required_str(experiment.get("game"), field_name="experiment.game"),
            variant=_required_str(
                experiment.get("variant"),
                field_name="experiment.variant",
            ),
            initial_pattern=_required_str(
                experiment.get("initial_pattern"),
                field_name="experiment.initial_pattern",
            ),
            initial_point_count=_coerce_int(
                experiment.get("initial_point_count"),
                field_name="experiment.initial_point_count",
            ),
        ),
        runtime=MorpionBootstrapRuntimeConfig(
            save_after_tree_growth_factor=_coerce_float(
                runtime.get("save_after_tree_growth_factor"),
                field_name="runtime.save_after_tree_growth_factor",
            ),
            save_after_seconds=_coerce_float(
                runtime.get("save_after_seconds"),
                field_name="runtime.save_after_seconds",
            ),
            max_growth_steps_per_cycle=_coerce_int(
                runtime.get("max_growth_steps_per_cycle"),
                field_name="runtime.max_growth_steps_per_cycle",
            ),
            tree_branch_limit=_coerce_int(
                runtime.get(
                    "tree_branch_limit",
                    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
                ),
                field_name="runtime.tree_branch_limit",
            ),
            reevaluation_blend_alpha=_coerce_float(
                runtime.get("reevaluation_blend_alpha", 1.0),
                field_name="runtime.reevaluation_blend_alpha",
            ),
            min_available_ram_mb=_optional_int(
                runtime.get("min_available_ram_mb"),
                field_name="runtime.min_available_ram_mb",
            ),
            candidate_checkpoint_load_headroom_factor=_coerce_float(
                runtime.get("candidate_checkpoint_load_headroom_factor", 60.0),
                field_name="runtime.candidate_checkpoint_load_headroom_factor",
            ),
            candidate_checkpoint_load_min_headroom_mb=_coerce_int(
                runtime.get("candidate_checkpoint_load_min_headroom_mb", 512),
                field_name="runtime.candidate_checkpoint_load_min_headroom_mb",
            ),
        ),
        dataset=MorpionBootstrapDatasetConfig(
            require_exact_or_terminal=_required_bool(
                dataset.get("require_exact_or_terminal"),
                field_name="dataset.require_exact_or_terminal",
            ),
            min_depth=_optional_int(
                dataset.get("min_depth"),
                field_name="dataset.min_depth",
            ),
            min_visit_count=_optional_int(
                dataset.get("min_visit_count"),
                field_name="dataset.min_visit_count",
            ),
            max_rows=_optional_int(
                dataset.get("max_rows"),
                field_name="dataset.max_rows",
            ),
            use_backed_up_value=_required_bool(
                dataset.get("use_backed_up_value"),
                field_name="dataset.use_backed_up_value",
            ),
            family_target_policy=cast(
                "PvFamilyTargetPolicy",
                _required_str(
                    dataset.get("family_target_policy", "none"),
                    field_name="dataset.family_target_policy",
                ),
            ),
            family_prediction_blend=_coerce_float(
                dataset.get("family_prediction_blend", 0.25),
                field_name="dataset.family_prediction_blend",
            ),
        ),
        evaluators=evaluators,
        search=_search_config_from_payload(payload.get("search")),
        validation_fraction=_coerce_float(
            payload.get("validation_fraction", 0.2),
            field_name="validation_fraction",
        ),
        validation_seed=_coerce_int(
            payload.get("validation_seed", 0),
            field_name="validation_seed",
        ),
        evaluator_update_policy=cast(
            "MorpionEvaluatorUpdatePolicy",
            _required_str(
                payload.get(
                    "evaluator_update_policy",
                    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
                ),
                field_name="evaluator_update_policy",
            ),
        ),
        pipeline_mode=cast(
            "MorpionPipelineMode",
            _required_str(
                payload.get("pipeline_mode", DEFAULT_MORPION_PIPELINE_MODE),
                field_name="pipeline_mode",
            ),
        ),
        training_export_mode=cast(
            "MorpionTrainingExportMode",
            _required_str(
                payload.get(
                    "training_export_mode",
                    DEFAULT_MORPION_TRAINING_EXPORT_MODE,
                ),
                field_name="training_export_mode",
            ),
        ),
        metadata=_metadata_dict(payload.get("metadata")),
    )


def load_bootstrap_config(path: str | Path) -> MorpionBootstrapConfig:
    """Load one persisted bootstrap config from JSON."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MalformedMorpionBootstrapConfigError.invalid_json(path) from exc
    return bootstrap_config_from_dict(loaded)


def save_bootstrap_config(config: MorpionBootstrapConfig, path: str | Path) -> None:
    """Persist one bootstrap config as canonical UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        bootstrap_config_to_canonical_json(config) + "\n",
        encoding="utf-8",
    )


def bootstrap_config_to_canonical_json(config: MorpionBootstrapConfig) -> str:
    """Return one stable canonical JSON string for hashing or persistence."""
    return json.dumps(
        bootstrap_config_to_dict(config),
        indent=2,
        sort_keys=True,
    )


def bootstrap_config_sha256(config: MorpionBootstrapConfig) -> str:
    """Return a stable hash for one canonical bootstrap config."""
    return hashlib.sha256(
        bootstrap_config_to_canonical_json(config).encode("utf-8")
    ).hexdigest()


def diff_bootstrap_configs(
    previous: MorpionBootstrapConfig,
    current: MorpionBootstrapConfig,
) -> tuple[str, ...]:
    """Return a stable list of changed config field paths."""
    differences: list[str] = []
    differences.extend(
        _diff_dataclass_section(
            previous.experiment, current.experiment, prefix="experiment"
        )
    )
    differences.extend(
        _diff_dataclass_section(previous.runtime, current.runtime, prefix="runtime")
    )
    differences.extend(
        _diff_dataclass_section(previous.dataset, current.dataset, prefix="dataset")
    )
    if previous.evaluators != current.evaluators:
        differences.append("evaluators")
    differences.extend(
        _diff_dataclass_section(
            previous.search.rollout,
            current.search.rollout,
            prefix="search.rollout",
        )
    )
    if previous.validation_fraction != current.validation_fraction:
        differences.append("validation_fraction")
    if previous.validation_seed != current.validation_seed:
        differences.append("validation_seed")
    if previous.evaluator_update_policy != current.evaluator_update_policy:
        differences.append("evaluator_update_policy")
    if previous.pipeline_mode != current.pipeline_mode:
        differences.append("pipeline_mode")
    if previous.training_export_mode != current.training_export_mode:
        differences.append("training_export_mode")
    if previous.metadata != current.metadata:
        differences.append("metadata")
    return tuple(differences)


def validate_bootstrap_config_change(
    previous: MorpionBootstrapConfig,
    current: MorpionBootstrapConfig,
) -> None:
    """Validate whether a relaunch config change can continue one run safely."""
    unsafe_changes = [
        field_name
        for field_name in _diff_dataclass_section(
            previous.experiment,
            current.experiment,
            prefix="experiment",
        )
    ]
    if not unsafe_changes:
        return

    rendered_changes = "; ".join(
        f"{field_name}: {_resolve_diff_value(previous, field_name)!r} -> {_resolve_diff_value(current, field_name)!r}"
        for field_name in unsafe_changes
    )
    raise UnsafeMorpionBootstrapConfigChangeError(
        "Unsafe Morpion bootstrap config change(s): " + rendered_changes
    )


def dataset_stage_owned_bootstrap_fields() -> tuple[str, ...]:
    """Return bootstrap-args fields owned by dataset extraction workers."""
    return (
        "require_exact_or_terminal",
        "use_backed_up_value",
        "dataset_family_target_policy",
        "dataset_family_prediction_blend",
        "min_depth",
        "min_visit_count",
        "max_rows",
        "training_export_mode",
    )


def training_stage_owned_bootstrap_fields() -> tuple[str, ...]:
    """Return bootstrap-args fields owned by training workers."""
    return (
        "batch_size",
        "num_epochs",
        "learning_rate",
        "shuffle",
        "model_kind",
        "hidden_dim",
        "validation_fraction",
        "validation_seed",
        "evaluators_config",
        "evaluator_family_preset",
        "training_evaluator_names",
        "training_max_rows",
        "training_row_chunk_size",
        "skip_evaluator_diagnostics",
        "evaluator_diagnostics_max_rows",
    )


def growth_stage_owned_bootstrap_fields() -> tuple[str, ...]:
    """Return bootstrap-args fields owned by growth/runtime workers."""
    return (
        "max_growth_steps_per_cycle",
        "save_after_tree_growth_factor",
        "save_after_seconds",
        "tree_branch_limit",
        "reevaluation_blend_alpha",
        "min_available_ram_mb",
        "candidate_checkpoint_load_headroom_factor",
        "candidate_checkpoint_load_min_headroom_mb",
        "growth_memory_profile",
        "growth_memory_profile_top_n",
        "growth_memory_profile_sample_nodes",
        "growth_memory_profile_recursive",
        "growth_memory_profile_recursive_max_objects",
        "growth_memory_profile_recursive_context_node_cap",
        "growth_memory_profile_recursive_events",
        "growth_memory_profile_recursive_complete_map",
        "rollout_after_opening",
        "rollout_max_extra_steps",
        "rollout_action_selector_kind",
        "rollout_random_seed",
        "rollout_stop_on_existing_node",
        "evaluator_update_policy",
        "training_export_mode",
    )


def reevaluation_stage_owned_bootstrap_fields() -> tuple[str, ...]:
    """Return bootstrap-args fields owned by reevaluation workers."""
    return ()


def bootstrap_fields_owned_by_stage(stage: MorpionPipelineStage) -> tuple[str, ...]:
    """Return bootstrap-args fields owned by one pipeline stage."""
    if stage in {"dataset", "dataset_worker"}:
        return dataset_stage_owned_bootstrap_fields()
    if stage in {"training", "training_worker"}:
        return training_stage_owned_bootstrap_fields()
    if stage == "growth":
        return growth_stage_owned_bootstrap_fields()
    if stage == "reevaluation":
        return reevaluation_stage_owned_bootstrap_fields()
    if stage == "loop":
        return (
            *growth_stage_owned_bootstrap_fields(),
            *dataset_stage_owned_bootstrap_fields(),
            *training_stage_owned_bootstrap_fields(),
        )
    return ()


def validate_stage_bootstrap_config_compatibility(
    *,
    stage: MorpionPipelineStage,
    persisted_config: MorpionBootstrapConfig,
    requested_config: MorpionBootstrapConfig,
) -> None:
    """Validate that one stage matches the persisted bootstrap config."""
    owned_fields = set(bootstrap_fields_owned_by_stage(stage))
    irrelevant_fields = STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS.get(stage, frozenset())
    persisted_values = _stage_bootstrap_config_field_values(persisted_config)
    requested_values = _stage_bootstrap_config_field_values(requested_config)
    for field_name in sorted(persisted_values):
        persisted_value = persisted_values[field_name]
        requested_value = requested_values[field_name]
        if field_name in GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS:
            continue
        if (
            stage in {"growth", "loop"}
            and field_name in GROWTH_SEARCH_BOOTSTRAP_STAGE_VALUE_FIELDS
        ):
            continue
        if field_name in irrelevant_fields:
            continue
        if persisted_value != requested_value:
            raise IncompatibleStageBootstrapConfigError.for_field(
                stage=stage,
                field_name=field_name,
                persisted_value=persisted_value,
                requested_value=requested_value,
                stage_uses_field=_config_field_is_owned_by_stage(
                    field_name,
                    owned_fields,
                ),
            )


def _diff_dataclass_section(
    previous: (
        MorpionBootstrapExperimentIdentityConfig
        | MorpionBootstrapRuntimeConfig
        | MorpionBootstrapDatasetConfig
        | MorpionBootstrapRolloutConfig
    ),
    current: (
        MorpionBootstrapExperimentIdentityConfig
        | MorpionBootstrapRuntimeConfig
        | MorpionBootstrapDatasetConfig
        | MorpionBootstrapRolloutConfig
    ),
    *,
    prefix: str,
) -> list[str]:
    """Return changed field paths for one flat dataclass section."""
    return [
        f"{prefix}.{field_info.name}"
        for field_info in fields(previous)
        if getattr(previous, field_info.name) != getattr(current, field_info.name)
    ]


def _resolve_diff_value(
    config: MorpionBootstrapConfig, dotted_field_name: str
) -> object:
    """Resolve one dotted config field path against one config object."""
    if "." not in dotted_field_name:
        return getattr(config, dotted_field_name)
    section_name, field_name = dotted_field_name.split(".", maxsplit=1)
    return getattr(getattr(config, section_name), field_name)


def _stage_bootstrap_config_field_values(
    config: MorpionBootstrapConfig,
) -> dict[str, object]:
    """Return config values keyed by their closest bootstrap-args field name."""
    return {
        "experiment.game": config.experiment.game,
        "experiment.variant": config.experiment.variant,
        "experiment.initial_pattern": config.experiment.initial_pattern,
        "experiment.initial_point_count": config.experiment.initial_point_count,
        "max_growth_steps_per_cycle": config.runtime.max_growth_steps_per_cycle,
        "save_after_tree_growth_factor": config.runtime.save_after_tree_growth_factor,
        "save_after_seconds": config.runtime.save_after_seconds,
        "tree_branch_limit": config.runtime.tree_branch_limit,
        "reevaluation_blend_alpha": config.runtime.reevaluation_blend_alpha,
        "min_available_ram_mb": config.runtime.min_available_ram_mb,
        "candidate_checkpoint_load_headroom_factor": (
            config.runtime.candidate_checkpoint_load_headroom_factor
        ),
        "candidate_checkpoint_load_min_headroom_mb": (
            config.runtime.candidate_checkpoint_load_min_headroom_mb
        ),
        "rollout_after_opening": config.search.rollout.enabled,
        "rollout_max_extra_steps": config.search.rollout.max_extra_steps,
        "rollout_action_selector_kind": config.search.rollout.action_selector_kind,
        "rollout_random_seed": config.search.rollout.random_seed,
        "rollout_stop_on_existing_node": config.search.rollout.stop_on_existing_node,
        "require_exact_or_terminal": config.dataset.require_exact_or_terminal,
        "min_depth": config.dataset.min_depth,
        "min_visit_count": config.dataset.min_visit_count,
        "max_rows": config.dataset.max_rows,
        "use_backed_up_value": config.dataset.use_backed_up_value,
        "dataset_family_target_policy": config.dataset.family_target_policy,
        "dataset_family_prediction_blend": config.dataset.family_prediction_blend,
        "evaluators": config.evaluators,
        "validation_fraction": config.validation_fraction,
        "validation_seed": config.validation_seed,
        "evaluator_update_policy": config.evaluator_update_policy,
        "pipeline_mode": config.pipeline_mode,
        "training_export_mode": config.training_export_mode,
    }


def _config_field_is_owned_by_stage(field_name: str, owned_fields: set[str]) -> bool:
    """Return whether one persisted config field is represented by owned args."""
    if field_name in owned_fields:
        return True
    if field_name == "evaluators":
        return bool(
            {
                "batch_size",
                "num_epochs",
                "learning_rate",
                "model_kind",
                "hidden_dim",
                "validation_fraction",
                "validation_seed",
                "evaluators_config",
                "evaluator_family_preset",
            }
            & owned_fields
        )
    return False


def _required_optional_int(
    mapping: Mapping[str, object],
    key: str,
    *,
    field_name: str,
) -> int | None:
    """Return one required nullable integer-like field or raise."""
    if key not in mapping:
        raise MalformedMorpionBootstrapConfigError.invalid_int(field_name)
    return _optional_int(mapping.get(key), field_name=field_name)


def _search_config_from_payload(value: object) -> MorpionBootstrapSearchConfig:
    """Deserialize the required search config section."""
    search = _require_section_mapping(value, section_name="search")
    rollout = _require_section_mapping(
        search.get("rollout"), section_name="search.rollout"
    )
    return MorpionBootstrapSearchConfig(
        rollout=MorpionBootstrapRolloutConfig(
            enabled=_required_bool(
                rollout.get("enabled"),
                field_name="search.rollout.enabled",
            ),
            max_extra_steps=_required_optional_int(
                rollout,
                "max_extra_steps",
                field_name="search.rollout.max_extra_steps",
            ),
            action_selector_kind=_required_str(
                rollout.get("action_selector_kind"),
                field_name="search.rollout.action_selector_kind",
            ),
            random_seed=_required_optional_int(
                rollout,
                "random_seed",
                field_name="search.rollout.random_seed",
            ),
            stop_on_existing_node=_required_bool(
                rollout.get("stop_on_existing_node"),
                field_name="search.rollout.stop_on_existing_node",
            ),
        )
    )


def _evaluator_spec_from_config_payload(
    *,
    evaluator_name: str,
    spec_payload: object,
) -> MorpionEvaluatorSpec:
    """Deserialize one evaluator spec from JSON-friendly data."""
    from .evaluator_config import MorpionEvaluatorSpec

    section_name = f"evaluators.evaluators.{evaluator_name}"
    spec_mapping = _require_section_mapping(spec_payload, section_name=section_name)

    return MorpionEvaluatorSpec(
        name=_required_str(
            spec_mapping.get("name"),
            field_name=f"{section_name}.name",
        ),
        model_type=_required_str(
            spec_mapping.get("model_type"),
            field_name=f"{section_name}.model_type",
        ),
        hidden_sizes=_optional_int_tuple(
            spec_mapping.get("hidden_sizes"),
            field_name=f"{section_name}.hidden_sizes",
        ),
        num_epochs=_coerce_int(
            spec_mapping.get("num_epochs"),
            field_name=f"{section_name}.num_epochs",
        ),
        batch_size=_coerce_int(
            spec_mapping.get("batch_size"),
            field_name=f"{section_name}.batch_size",
        ),
        learning_rate=_coerce_float(
            spec_mapping.get("learning_rate"),
            field_name=f"{section_name}.learning_rate",
        ),
        feature_subset_name=_required_str(
            spec_mapping.get(
                "feature_subset_name",
                DEFAULT_MORPION_FEATURE_SUBSET_NAME,
            ),
            field_name=f"{section_name}.feature_subset_name",
        ),
        feature_names=_optional_str_tuple(
            spec_mapping.get("feature_names"),
            field_name=f"{section_name}.feature_names",
        ),
        graph_max_tokens=_coerce_int(
            spec_mapping.get("graph_max_tokens", 1536),
            field_name=f"{section_name}.graph_max_tokens",
        ),
        graph_input_feature_dim=_coerce_int(
            spec_mapping.get(
                "graph_input_feature_dim",
                MORPION_GRAPH_TOKEN_FEATURE_DIM,
            ),
            field_name=f"{section_name}.graph_input_feature_dim",
        ),
        graph_d_model=_coerce_int(
            spec_mapping.get("graph_d_model", 64),
            field_name=f"{section_name}.graph_d_model",
        ),
        graph_n_head=_coerce_int(
            spec_mapping.get("graph_n_head", 4),
            field_name=f"{section_name}.graph_n_head",
        ),
        graph_n_layer=_coerce_int(
            spec_mapping.get("graph_n_layer", 2),
            field_name=f"{section_name}.graph_n_layer",
        ),
        graph_dim_feedforward=_coerce_int(
            spec_mapping.get("graph_dim_feedforward", 256),
            field_name=f"{section_name}.graph_dim_feedforward",
        ),
        graph_dropout_ratio=_coerce_float(
            spec_mapping.get("graph_dropout_ratio", 0.0),
            field_name=f"{section_name}.graph_dropout_ratio",
        ),
        graph_pooling=_required_str(
            spec_mapping.get("graph_pooling", "value_token"),
            field_name=f"{section_name}.graph_pooling",
        ),
        graph_output_tanh=_required_bool(
            spec_mapping.get("graph_output_tanh", True),
            field_name=f"{section_name}.graph_output_tanh",
        ),
    )


def _evaluators_config_to_dict(config: MorpionEvaluatorsConfig) -> dict[str, object]:
    """Serialize one evaluator config into JSON-friendly data."""
    return {
        "evaluators": {
            name: _evaluator_spec_to_dict(config.evaluators[name])
            for name in sorted(config.evaluators)
        }
    }


def _evaluator_spec_to_dict(spec: MorpionEvaluatorSpec) -> dict[str, object]:
    """Serialize one evaluator spec into JSON-friendly data."""
    payload: dict[str, object] = {
        "name": spec.name,
        "model_type": spec.model_type,
        "hidden_sizes": None if spec.hidden_sizes is None else list(spec.hidden_sizes),
        "num_epochs": spec.num_epochs,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "feature_subset_name": spec.feature_subset_name,
        "feature_names": list(spec.feature_names),
    }
    if (
        spec.model_type == MORPION_GRAPH_MODEL_KIND
        or _has_non_default_graph_evaluator_settings(spec)
    ):
        payload.update(_graph_evaluator_settings_to_dict(spec))
    return payload


def _has_non_default_graph_evaluator_settings(spec: MorpionEvaluatorSpec) -> bool:
    """Return whether graph settings differ from dataclass defaults."""
    return (
        spec.graph_max_tokens != 1536
        or spec.graph_input_feature_dim != MORPION_GRAPH_TOKEN_FEATURE_DIM
        or spec.graph_d_model != 64
        or spec.graph_n_head != 4
        or spec.graph_n_layer != 2
        or spec.graph_dim_feedforward != 256
        or spec.graph_dropout_ratio != 0.0
        or spec.graph_pooling != "value_token"
        or spec.graph_output_tanh is not True
    )


def _graph_evaluator_settings_to_dict(
    spec: MorpionEvaluatorSpec,
) -> dict[str, object]:
    """Serialize graph-specific evaluator settings."""
    return {
        "graph_max_tokens": spec.graph_max_tokens,
        "graph_input_feature_dim": spec.graph_input_feature_dim,
        "graph_d_model": spec.graph_d_model,
        "graph_n_head": spec.graph_n_head,
        "graph_n_layer": spec.graph_n_layer,
        "graph_dim_feedforward": spec.graph_dim_feedforward,
        "graph_dropout_ratio": spec.graph_dropout_ratio,
        "graph_pooling": spec.graph_pooling,
        "graph_output_tanh": spec.graph_output_tanh,
    }


def _is_str_key_mapping(value: object) -> bool:
    """Return whether ``value`` is a mapping with string keys."""
    if not isinstance(value, Mapping):
        return False
    mapping = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in mapping)


def _require_section_mapping(value: object, *, section_name: str) -> dict[str, object]:
    """Return one config section mapping or raise clearly."""
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapConfigError.invalid_section(section_name)
    return dict(cast("Mapping[str, object]", value))


def _required_str(value: object, *, field_name: str) -> str:
    """Return one required string field or raise."""
    if isinstance(value, str):
        return value
    raise MalformedMorpionBootstrapConfigError.invalid_required_str(field_name)


def _required_bool(value: object, *, field_name: str) -> bool:
    """Return one required bool field or raise."""
    if isinstance(value, bool):
        return value
    raise MalformedMorpionBootstrapConfigError.invalid_bool(field_name)


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return one integer-like field or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapConfigError.invalid_int(field_name)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise MalformedMorpionBootstrapConfigError.invalid_int(field_name) from exc
    raise MalformedMorpionBootstrapConfigError.invalid_int(field_name)


def _optional_int(value: object, *, field_name: str) -> int | None:
    """Return one optional integer-like field or raise."""
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _coerce_float(value: object, *, field_name: str) -> float:
    """Return one float-like field or raise."""
    if isinstance(value, bool):
        raise MalformedMorpionBootstrapConfigError.invalid_float(field_name)
    try:
        if isinstance(value, int | float | str):
            return float(value)
    except ValueError as exc:
        raise MalformedMorpionBootstrapConfigError.invalid_float(field_name) from exc
    raise MalformedMorpionBootstrapConfigError.invalid_float(field_name)


def _optional_int_tuple(value: object, *, field_name: str) -> tuple[int, ...] | None:
    """Return one optional integer tuple field or raise."""
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise MalformedMorpionBootstrapConfigError.invalid_int(field_name)
    items = cast("list[object] | tuple[object, ...]", value)
    return tuple(_coerce_int(item, field_name=field_name) for item in items)


def _optional_str_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Return one optional string tuple field or raise."""
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise MalformedMorpionBootstrapConfigError.invalid_feature_names(field_name)
    items = cast("list[object] | tuple[object, ...]", value)
    if not all(isinstance(item, str) for item in items):
        raise MalformedMorpionBootstrapConfigError.invalid_feature_names(field_name)
    return tuple(cast("str", item) for item in items)


def _metadata_dict(value: object) -> dict[str, object]:
    """Return one metadata mapping or raise."""
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise MalformedMorpionBootstrapConfigError.invalid_metadata()
    return dict(cast("Mapping[str, object]", value))


__all__ = [
    "BOOTSTRAP_CONFIG_HASH_METADATA_KEY",
    "DEFAULT_MORPION_TREE_BRANCH_LIMIT",
    "GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS",
    "GROWTH_SEARCH_BOOTSTRAP_CONFIG_DIFF_FIELDS",
    "GROWTH_SEARCH_BOOTSTRAP_STAGE_VALUE_FIELDS",
    "RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS",
    "STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS",
    "IncompatibleStageBootstrapConfigError",
    "MalformedMorpionBootstrapConfigError",
    "MorpionBootstrapConfig",
    "MorpionBootstrapDatasetConfig",
    "MorpionBootstrapExperimentIdentityConfig",
    "MorpionBootstrapRolloutConfig",
    "MorpionBootstrapRuntimeConfig",
    "MorpionBootstrapSearchConfig",
    "UnsafeMorpionBootstrapConfigChangeError",
    "bootstrap_config_from_args",
    "bootstrap_config_from_dict",
    "bootstrap_config_sha256",
    "bootstrap_config_to_canonical_json",
    "bootstrap_config_to_dict",
    "bootstrap_config_to_dict",
    "bootstrap_fields_owned_by_stage",
    "dataset_stage_owned_bootstrap_fields",
    "diff_bootstrap_configs",
    "growth_stage_owned_bootstrap_fields",
    "load_bootstrap_config",
    "reevaluation_stage_owned_bootstrap_fields",
    "save_bootstrap_config",
    "training_stage_owned_bootstrap_fields",
    "validate_bootstrap_config_change",
    "validate_stage_bootstrap_config_compatibility",
]
