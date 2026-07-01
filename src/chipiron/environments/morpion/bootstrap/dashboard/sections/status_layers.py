"""Dashboard control and effective-state status layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from chipiron.environments.morpion.bootstrap.config import (
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapConfig,
    load_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapRuntimeControl,
    bootstrap_control_from_metadata,
    bootstrap_runtime_control_from_metadata,
    effective_runtime_config_from_metadata,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    canonical_morpion_evaluator_names,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
        MorpionBootstrapPaths,
    )

__all__ = [
    "applied_runtime_control",
    "baseline_tree_branch_limit",
    "build_next_control",
    "configured_evaluator_names",
    "dataset_status_summary",
    "effective_runtime_config",
    "effective_runtime_hash",
    "effective_state_summary",
    "evaluator_control_status_summary",
    "evaluator_set_summary",
    "force_evaluator_option_index",
    "force_evaluator_options",
    "format_force_evaluator_option",
    "format_force_evaluator_state",
    "has_pending_control_changes",
    "is_stale_forced_evaluator",
    "load_applied_control",
    "load_bootstrap_config_or_none",
    "load_run_state",
    "pending_control_fields",
    "pending_control_sections",
    "render_dataset_control_section",
    "render_effective_state_section",
    "render_evaluator_control_section",
    "render_pending_changes_section",
    "render_runtime_control_section",
    "render_scheduling_control_section",
    "render_status_layers",
    "runtime_status_summary",
    "scheduling_status_summary",
    "tree_branch_limit_input_value",
]


def load_run_state(paths: MorpionBootstrapPaths) -> Any:
    """Return the latest run state when present, else one initialized default state."""
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path)
    return initialize_bootstrap_run_state()


def load_applied_control(paths: MorpionBootstrapPaths) -> MorpionBootstrapControl:
    """Return the last control known to have been applied at a cycle boundary."""
    if not paths.run_state_path.is_file():
        return MorpionBootstrapControl()
    run_state = load_bootstrap_run_state(paths.run_state_path)
    return bootstrap_control_from_metadata(
        run_state.metadata.get(BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY)
    )


def load_bootstrap_config_or_none(
    paths: MorpionBootstrapPaths,
) -> MorpionBootstrapConfig | None:
    """Return the persisted bootstrap config when present."""
    if not paths.bootstrap_config_path.is_file():
        return None
    return load_bootstrap_config(paths.bootstrap_config_path)


def baseline_tree_branch_limit(config: MorpionBootstrapConfig | None) -> int:
    """Return the configured baseline tree branch limit or the stable default."""
    if config is None:
        return DEFAULT_MORPION_TREE_BRANCH_LIMIT
    return config.runtime.tree_branch_limit


def applied_runtime_control(run_state: Any) -> MorpionBootstrapRuntimeControl:
    """Return the last applied runtime-control subsection from run-state metadata."""
    return bootstrap_runtime_control_from_metadata(
        getattr(run_state, "metadata", {}).get(
            BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY
        )
    )


def effective_runtime_config(
    run_state: Any,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Return the effective runtime config from run-state metadata when present."""
    return effective_runtime_config_from_metadata(
        getattr(run_state, "metadata", {}).get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    )


def effective_runtime_hash(run_state: Any) -> str | None:
    """Return the effective-runtime hash from run-state metadata when present."""
    value = getattr(run_state, "metadata", {}).get(
        BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY
    )
    return value if isinstance(value, str) else None


def tree_branch_limit_input_value(
    *,
    runtime_control: MorpionBootstrapRuntimeControl,
    baseline_tree_branch_limit: int | None,
) -> int:
    """Return the displayed tree-branch-limit value for the dashboard input."""
    if runtime_control.tree_branch_limit is not None:
        return runtime_control.tree_branch_limit
    if baseline_tree_branch_limit is not None:
        return baseline_tree_branch_limit
    return DEFAULT_MORPION_TREE_BRANCH_LIMIT


def _field_status_summary(
    *,
    baseline: object | None,
    current_override: object | None,
    applied_override: object | None,
    effective: object | None,
) -> dict[str, object | None]:
    """Return one stable four-layer control-state summary."""
    return {
        "baseline": baseline,
        "current_override": current_override,
        "applied_override": applied_override,
        "effective": effective,
    }


def _effective_control_value(
    *,
    baseline: object | None,
    override: object | None,
) -> object | None:
    """Return the control value currently in force for one baseline/override pair."""
    return baseline if override is None else override


def dataset_status_summary(
    config: MorpionBootstrapConfig | None,
    current_control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> dict[str, dict[str, object | None]]:
    """Return one stable dataset-control status summary."""
    baseline_max_rows = None if config is None else config.dataset.max_rows
    baseline_use_backed_up_value = (
        None if config is None else config.dataset.use_backed_up_value
    )
    return {
        "max_rows": _field_status_summary(
            baseline=baseline_max_rows,
            current_override=current_control.max_rows,
            applied_override=applied_control.max_rows,
            effective=_effective_control_value(
                baseline=baseline_max_rows,
                override=applied_control.max_rows,
            ),
        ),
        "use_backed_up_value": _field_status_summary(
            baseline=baseline_use_backed_up_value,
            current_override=current_control.use_backed_up_value,
            applied_override=applied_control.use_backed_up_value,
            effective=_effective_control_value(
                baseline=baseline_use_backed_up_value,
                override=applied_control.use_backed_up_value,
            ),
        ),
    }


def scheduling_status_summary(
    config: MorpionBootstrapConfig | None,
    current_control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> dict[str, dict[str, object | None]]:
    """Return one stable scheduling-control status summary."""
    baseline_growth_steps = (
        None if config is None else config.runtime.max_growth_steps_per_cycle
    )
    baseline_save_after_seconds = (
        None if config is None else config.runtime.save_after_seconds
    )
    baseline_growth_factor = (
        None if config is None else config.runtime.save_after_tree_growth_factor
    )
    return {
        "max_growth_steps_per_cycle": _field_status_summary(
            baseline=baseline_growth_steps,
            current_override=current_control.max_growth_steps_per_cycle,
            applied_override=applied_control.max_growth_steps_per_cycle,
            effective=_effective_control_value(
                baseline=baseline_growth_steps,
                override=applied_control.max_growth_steps_per_cycle,
            ),
        ),
        "save_after_seconds": _field_status_summary(
            baseline=baseline_save_after_seconds,
            current_override=current_control.save_after_seconds,
            applied_override=applied_control.save_after_seconds,
            effective=_effective_control_value(
                baseline=baseline_save_after_seconds,
                override=applied_control.save_after_seconds,
            ),
        ),
        "save_after_tree_growth_factor": _field_status_summary(
            baseline=baseline_growth_factor,
            current_override=current_control.save_after_tree_growth_factor,
            applied_override=applied_control.save_after_tree_growth_factor,
            effective=_effective_control_value(
                baseline=baseline_growth_factor,
                override=applied_control.save_after_tree_growth_factor,
            ),
        ),
    }


def is_stale_forced_evaluator(
    forced_evaluator: str | None,
    configured_evaluator_names: tuple[str, ...],
) -> bool:
    """Return whether one forced evaluator is absent from current config."""
    return (
        forced_evaluator is not None
        and forced_evaluator not in configured_evaluator_names
    )


def evaluator_control_status_summary(
    *,
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object | dict[str, object | None]]:
    """Return one stable evaluator-control status summary."""
    current_mode = "auto" if control.force_evaluator is None else "forced"
    applied_mode = "auto" if applied_control.force_evaluator is None else "forced"
    return {
        "selection_mode": _field_status_summary(
            baseline="auto",
            current_override=current_mode,
            applied_override=applied_mode,
            effective=applied_mode,
        ),
        "forced_evaluator": _field_status_summary(
            baseline=None,
            current_override=control.force_evaluator,
            applied_override=applied_control.force_evaluator,
            effective=applied_control.force_evaluator,
        ),
        "current_force_evaluator_is_stale": is_stale_forced_evaluator(
            control.force_evaluator,
            configured_evaluator_names,
        ),
        "applied_force_evaluator_is_stale": is_stale_forced_evaluator(
            applied_control.force_evaluator,
            configured_evaluator_names,
        ),
    }


def runtime_status_summary(
    *,
    baseline_tree_branch_limit: int,
    current_runtime_control: MorpionBootstrapRuntimeControl,
    applied_runtime_control: MorpionBootstrapRuntimeControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    effective_runtime_hash: str | None,
) -> dict[str, object | dict[str, object | None]]:
    """Return one stable runtime-control status summary."""
    return {
        "tree_branch_limit": _field_status_summary(
            baseline=baseline_tree_branch_limit,
            current_override=current_runtime_control.tree_branch_limit,
            applied_override=applied_runtime_control.tree_branch_limit,
            effective=None
            if effective_runtime_config is None
            else effective_runtime_config.tree_branch_limit,
        ),
        "effective_runtime_hash": effective_runtime_hash,
    }


def pending_control_fields(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> tuple[str, ...]:
    """Return stable field names that still differ from the last applied control."""
    pending_fields: list[str] = []
    if control.max_rows != applied_control.max_rows:
        pending_fields.append("max_rows")
    if control.use_backed_up_value != applied_control.use_backed_up_value:
        pending_fields.append("use_backed_up_value")
    if control.max_growth_steps_per_cycle != applied_control.max_growth_steps_per_cycle:
        pending_fields.append("max_growth_steps_per_cycle")
    if control.save_after_seconds != applied_control.save_after_seconds:
        pending_fields.append("save_after_seconds")
    if (
        control.save_after_tree_growth_factor
        != applied_control.save_after_tree_growth_factor
    ):
        pending_fields.append("save_after_tree_growth_factor")
    if control.force_evaluator != applied_control.force_evaluator:
        pending_fields.append("force_evaluator")
    if control.runtime.tree_branch_limit != applied_control.runtime.tree_branch_limit:
        pending_fields.append("runtime.tree_branch_limit")
    return tuple(pending_fields)


def pending_control_sections(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> tuple[str, ...]:
    """Return stable control-section names containing unapplied changes."""
    fields = set(pending_control_fields(control, applied_control))
    sections: list[str] = []
    if {"max_rows", "use_backed_up_value"} & fields:
        sections.append("dataset")
    if {
        "max_growth_steps_per_cycle",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    } & fields:
        sections.append("scheduling")
    if "force_evaluator" in fields:
        sections.append("evaluator selection")
    if "runtime.tree_branch_limit" in fields:
        sections.append("runtime")
    return tuple(sections)


def effective_state_summary(
    *,
    run_summary: Any,
    run_state: Any,
    current_control: MorpionBootstrapControl,
    baseline_tree_branch_limit: int,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    latest_dataset_rows: object | None,
    pending_changes: bool,
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object | None]:
    """Return one compact operator-facing effective-state summary."""
    active_evaluator = getattr(run_summary, "latest_active_evaluator_name", None)
    if active_evaluator is None:
        active_evaluator = getattr(run_state, "active_evaluator_name", None)
    evaluator_set = evaluator_set_summary(configured_evaluator_names)
    return {
        "active_evaluator": active_evaluator,
        "forced_evaluator_request": current_control.force_evaluator,
        "forced_evaluator_request_label": format_force_evaluator_state(
            current_control.force_evaluator,
            configured_evaluator_names=configured_evaluator_names,
        ),
        "baseline_tree_branch_limit": baseline_tree_branch_limit,
        "effective_tree_branch_limit": None
        if effective_runtime_config is None
        else effective_runtime_config.tree_branch_limit,
        "runtime_override_status": "set"
        if current_control.runtime.tree_branch_limit is not None
        else "unset",
        "evaluator_set_label": evaluator_set["label"],
        "configured_evaluator_count": evaluator_set["count"],
        "configured_evaluator_names": evaluator_set["configured_evaluator_names"],
        "is_canonical_evaluator_family": evaluator_set["is_canonical_family"],
        "latest_dataset_rows": latest_dataset_rows,
        "control_pending_application": pending_changes,
    }


def _summary_layer(
    summary: dict[str, dict[str, object | None]],
    layer: str,
) -> dict[str, object | None]:
    """Return one stable status layer extracted from a field summary."""
    return {
        field_name: field_summary[layer]
        for field_name, field_summary in summary.items()
    }


def _status_layer(value: object) -> dict[str, object | None]:
    """Return one validated dashboard status-layer mapping."""
    if not isinstance(value, dict):
        return {}
    return cast("dict[str, object | None]", value)


def render_pending_changes_section(
    *,
    st: Any,
    pending_changes: bool,
    pending_sections: tuple[str, ...],
    pending_fields: tuple[str, ...],
) -> None:
    """Render one compact summary of unapplied control changes."""
    st.subheader("Pending Changes")
    if pending_changes:
        st.warning("Pending changes will apply at the next cycle boundary.")
        st.write("Pending sections:", pending_sections)
        st.write("Pending fields:", pending_fields)
    else:
        st.success("Control file matches the last applied cycle boundary state.")
        st.write("Pending sections:", ())
        st.write("Pending fields:", ())


def render_effective_state_section(
    *,
    st: Any,
    summary: dict[str, object | None],
) -> None:
    """Render one compact operator-facing effective-state panel."""
    st.subheader("Effective State")
    columns = st.columns(4)
    columns[0].metric("Active evaluator", _format_value(summary["active_evaluator"]))
    columns[1].metric(
        "Forced evaluator request",
        _format_value(summary["forced_evaluator_request_label"]),
    )
    columns[2].metric(
        "Effective tree branch limit",
        _format_value(summary["effective_tree_branch_limit"]),
    )
    columns[3].metric(
        "Pending control file",
        "yes" if bool(summary["control_pending_application"]) else "no",
    )
    st.write("Evaluator set:", _format_value(summary["evaluator_set_label"]))
    st.write(
        "Configured evaluators:",
        _format_value(summary["configured_evaluator_names"]),
    )
    st.write("Summary:", summary)


def evaluator_set_summary(
    configured_evaluator_names: tuple[str, ...],
) -> dict[str, object]:
    """Return one compact evaluator-set summary for dashboard/operator views."""
    sorted_names = tuple(sorted(configured_evaluator_names))
    canonical_names = tuple(sorted(canonical_morpion_evaluator_names()))
    is_canonical_family = sorted_names == canonical_names
    if not sorted_names:
        label = "no configured evaluators"
    elif is_canonical_family:
        label = "canonical 8-model family"
    else:
        label = f"custom ({len(sorted_names)} evaluators)"
    return {
        "label": label,
        "count": len(sorted_names),
        "configured_evaluator_names": sorted_names,
        "is_canonical_family": is_canonical_family,
    }


def render_status_layers(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render baseline/current/applied/effective layers for one control section."""
    columns = st.columns(4)
    columns[0].write("Baseline")
    columns[0].write(_summary_layer(summary, "baseline"))
    columns[1].write("Current override")
    columns[1].write(_summary_layer(summary, "current_override"))
    columns[2].write("Last applied")
    columns[2].write(_summary_layer(summary, "applied_override"))
    columns[3].write("Effective")
    columns[3].write(_summary_layer(summary, "effective"))


def render_dataset_control_section(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render the dataset-control summary block."""
    st.subheader("Dataset / Training Data Extraction")
    st.caption("Unchecked inputs inherit the dataset baseline from persisted config.")
    render_status_layers(st=st, summary=summary)


def render_scheduling_control_section(
    *,
    st: Any,
    summary: dict[str, dict[str, object | None]],
) -> None:
    """Render the cycle-scheduling control summary block."""
    st.subheader("Cycle Scheduling")
    st.caption(
        "Scheduling overrides are persisted in the control file and apply at cycle boundaries."
    )
    render_status_layers(st=st, summary=summary)


def render_evaluator_control_section(
    *,
    st: Any,
    summary: dict[str, object | dict[str, object | None]],
) -> None:
    """Render the evaluator-selection control summary block."""
    st.subheader("Evaluator Selection")
    st.caption(
        "Auto keeps normal evaluator selection. Forced persists one explicit evaluator name."
    )
    render_status_layers(
        st=st,
        summary={
            "selection_mode": _status_layer(summary["selection_mode"]),
            "forced_evaluator": _status_layer(summary["forced_evaluator"]),
        },
    )
    st.write(
        "Stale forced evaluator flags:",
        {
            "current": summary["current_force_evaluator_is_stale"],
            "applied": summary["applied_force_evaluator_is_stale"],
        },
    )


def render_runtime_control_section(
    *,
    st: Any,
    summary: dict[str, object | dict[str, object | None]],
) -> None:
    """Render the runtime-control summary block."""
    st.subheader("Runtime Control")
    st.caption(
        "Baseline comes from persisted config, override comes from the control file, "
        "and effective runtime reflects what the loop has actually applied. On an "
        "existing persisted tree, only non-increasing tree branch limit changes are supported."
    )
    render_status_layers(
        st=st,
        summary={"tree_branch_limit": _status_layer(summary["tree_branch_limit"])},
    )
    st.write(
        "Effective runtime hash:", _format_value(summary["effective_runtime_hash"])
    )


def configured_evaluator_names(
    config: MorpionBootstrapConfig | None,
) -> tuple[str, ...]:
    """Return configured evaluator names from persisted bootstrap config."""
    if config is None:
        return ()
    return tuple(config.evaluators.evaluators)


def force_evaluator_options(
    *,
    configured_evaluator_names: tuple[str, ...],
    current_force_evaluator: str | None,
) -> tuple[str, ...]:
    """Return selectable forced evaluators from config, preserving any stale current value."""
    options = list(configured_evaluator_names)
    if current_force_evaluator is not None and current_force_evaluator not in options:
        options.append(current_force_evaluator)
    return tuple(options)


def has_pending_control_changes(
    control: MorpionBootstrapControl,
    applied_control: MorpionBootstrapControl,
) -> bool:
    """Return whether the control file differs from the last applied control."""
    return control != applied_control


def build_next_control(
    *,
    override_max_growth_steps_per_cycle: bool,
    max_growth_steps_per_cycle: int,
    override_max_rows: bool,
    max_rows: int,
    override_use_backed_up_value: bool,
    use_backed_up_value: bool,
    override_save_after_seconds: bool,
    save_after_seconds: float,
    override_save_after_tree_growth_factor: bool,
    save_after_tree_growth_factor: float,
    override_tree_branch_limit: bool,
    tree_branch_limit: int,
    force_evaluator_mode: str,
    force_evaluator: str,
) -> MorpionBootstrapControl:
    """Build one persisted control payload from tri-state dashboard inputs."""
    return MorpionBootstrapControl(
        max_growth_steps_per_cycle=max_growth_steps_per_cycle
        if override_max_growth_steps_per_cycle
        else None,
        max_rows=max_rows if override_max_rows else None,
        use_backed_up_value=use_backed_up_value
        if override_use_backed_up_value
        else None,
        save_after_seconds=save_after_seconds if override_save_after_seconds else None,
        save_after_tree_growth_factor=save_after_tree_growth_factor
        if override_save_after_tree_growth_factor
        else None,
        force_evaluator=resolved_force_evaluator(
            force_evaluator_mode=force_evaluator_mode,
            force_evaluator=force_evaluator,
        ),
        runtime=MorpionBootstrapRuntimeControl(
            tree_branch_limit=tree_branch_limit if override_tree_branch_limit else None
        ),
    )


def force_evaluator_option_index(
    options: tuple[str, ...],
    current_force_evaluator: str | None,
) -> int:
    """Return the current forced-evaluator index for one select widget."""
    if not options:
        return 0
    if current_force_evaluator is None:
        return 0
    try:
        return options.index(current_force_evaluator)
    except ValueError:
        return 0


def format_force_evaluator_state(
    value: str | None,
    *,
    configured_evaluator_names: tuple[str, ...],
) -> str:
    """Render one requested forced evaluator state for operator display."""
    if value is None:
        return "auto"
    if is_stale_forced_evaluator(value, configured_evaluator_names):
        return f"{value} (stale / not configured)"
    return value


def format_force_evaluator_option(
    value: str,
    *,
    configured_evaluator_names: tuple[str, ...] = (),
) -> str:
    """Render one forced-evaluator option for the Streamlit select widget."""
    if not value:
        return "No configured evaluators"
    if is_stale_forced_evaluator(
        value,
        configured_evaluator_names,
    ):
        return f"{value} (stale / not configured)"
    return value


def resolved_force_evaluator(
    *,
    force_evaluator_mode: str,
    force_evaluator: str,
) -> str | None:
    """Resolve the tri-state dashboard force-evaluator controls into persisted data."""
    if force_evaluator_mode != "forced" or not force_evaluator:
        return None
    return force_evaluator
