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
    )
    from .pv_family_targets import PvFamilyTargetPolicy

from .pipeline_config import (
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
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
    validation_fraction: float = 0.2
    validation_seed: int = 0
    evaluator_update_policy: MorpionEvaluatorUpdatePolicy = (
        DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY
    )
    pipeline_mode: MorpionPipelineMode = DEFAULT_MORPION_PIPELINE_MODE
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
        "save_after_seconds",
        "save_after_tree_growth_factor",
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
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        evaluator_update_policy=args.evaluator_update_policy,
        pipeline_mode=args.pipeline_mode,
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
        "evaluators": _evaluators_config_to_dict(config.evaluators),
        "validation_fraction": config.validation_fraction,
        "validation_seed": config.validation_seed,
        "evaluator_update_policy": config.evaluator_update_policy,
        "pipeline_mode": config.pipeline_mode,
        "metadata": dict(config.metadata),
    }


def bootstrap_config_from_dict(data: object) -> MorpionBootstrapConfig:
    """Deserialize one bootstrap config from JSON-friendly data."""
    from .bootstrap_loop import MorpionEvaluatorsConfig, MorpionEvaluatorSpec

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
                evaluator_name: MorpionEvaluatorSpec(
                    name=_required_str(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("name"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.name",
                    ),
                    model_type=_required_str(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("model_type"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.model_type",
                    ),
                    hidden_sizes=_optional_int_tuple(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("hidden_sizes"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.hidden_sizes",
                    ),
                    num_epochs=_coerce_int(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("num_epochs"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.num_epochs",
                    ),
                    batch_size=_coerce_int(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("batch_size"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.batch_size",
                    ),
                    learning_rate=_coerce_float(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("learning_rate"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.learning_rate",
                    ),
                    feature_subset_name=_required_str(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get(
                            "feature_subset_name",
                            DEFAULT_MORPION_FEATURE_SUBSET_NAME,
                        ),
                        field_name=f"evaluators.evaluators.{evaluator_name}.feature_subset_name",
                    ),
                    feature_names=_optional_str_tuple(
                        _require_section_mapping(
                            spec_payload,
                            section_name=f"evaluators.evaluators.{evaluator_name}",
                        ).get("feature_names"),
                        field_name=f"evaluators.evaluators.{evaluator_name}.feature_names",
                    ),
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
    if previous.validation_fraction != current.validation_fraction:
        differences.append("validation_fraction")
    if previous.validation_seed != current.validation_seed:
        differences.append("validation_seed")
    if previous.evaluator_update_policy != current.evaluator_update_policy:
        differences.append("evaluator_update_policy")
    if previous.pipeline_mode != current.pipeline_mode:
        differences.append("pipeline_mode")
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
    )


def growth_stage_owned_bootstrap_fields() -> tuple[str, ...]:
    """Return bootstrap-args fields owned by growth/runtime workers."""
    return (
        "max_growth_steps_per_cycle",
        "save_after_tree_growth_factor",
        "save_after_seconds",
        "tree_branch_limit",
        "evaluator_update_policy",
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
    ),
    current: (
        MorpionBootstrapExperimentIdentityConfig
        | MorpionBootstrapRuntimeConfig
        | MorpionBootstrapDatasetConfig
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
    }


def _config_field_is_owned_by_stage(
    field_name: str, owned_fields: set[str]
) -> bool:
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
    return {
        "name": spec.name,
        "model_type": spec.model_type,
        "hidden_sizes": None if spec.hidden_sizes is None else list(spec.hidden_sizes),
        "num_epochs": spec.num_epochs,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "feature_subset_name": spec.feature_subset_name,
        "feature_names": list(spec.feature_names),
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
    return tuple(_coerce_int(item, field_name=field_name) for item in value)


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
    "IncompatibleStageBootstrapConfigError",
    "MalformedMorpionBootstrapConfigError",
    "MorpionBootstrapConfig",
    "MorpionBootstrapDatasetConfig",
    "MorpionBootstrapExperimentIdentityConfig",
    "MorpionBootstrapRuntimeConfig",
    "RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS",
    "STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS",
    "UnsafeMorpionBootstrapConfigChangeError",
    "bootstrap_config_from_args",
    "bootstrap_config_from_dict",
    "bootstrap_config_sha256",
    "bootstrap_config_to_canonical_json",
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
