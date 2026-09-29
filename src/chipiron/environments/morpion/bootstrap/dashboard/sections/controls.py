"""Existing persistent controls, with an explicit preview before cycle-boundary apply."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.control import (
    bootstrap_control_to_dict,
    load_bootstrap_control,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    control_float_value as _control_float_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    control_number_value as _control_number_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    applied_runtime_control as _applied_runtime_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    baseline_tree_branch_limit as _baseline_tree_branch_limit,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    build_next_control as _build_next_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    configured_evaluator_names as _configured_evaluator_names,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    dataset_status_summary as _dataset_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_runtime_config as _effective_runtime_config,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_runtime_hash as _effective_runtime_hash,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    effective_state_summary as _effective_state_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    evaluator_control_status_summary as _evaluator_control_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    force_evaluator_option_index as _force_evaluator_option_index,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    force_evaluator_options as _force_evaluator_options,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    format_force_evaluator_option as _format_force_evaluator_option,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    has_pending_control_changes as _has_pending_control_changes,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_applied_control as _load_applied_control,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_bootstrap_config_or_none as _load_bootstrap_config_or_none,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    load_run_state as _load_run_state,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_fields as _pending_control_fields,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    pending_control_sections as _pending_control_sections,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_dataset_control_section as _render_dataset_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_effective_state_section as _render_effective_state_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_evaluator_control_section as _render_evaluator_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_pending_changes_section as _render_pending_changes_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_runtime_control_section as _render_runtime_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    render_scheduling_control_section as _render_scheduling_control_section,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    runtime_status_summary as _runtime_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    scheduling_status_summary as _scheduling_status_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.status_layers import (
    tree_branch_limit_input_value as _tree_branch_limit_input_value,
)

if TYPE_CHECKING:
    from pathlib import Path


def render_controls(st: Any, work_dir: Path, snapshot: Any) -> None:
    """Render baseline/requested/applied/effective layers and preserve save semantics."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    config = _load_bootstrap_config_or_none(paths)
    control = load_bootstrap_control(paths.control_path)
    applied_control = _load_applied_control(paths)
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        load_morpion_bootstrap_run_view,
        summarize_bootstrap_run,
    )

    summary = summarize_bootstrap_run(load_morpion_bootstrap_run_view(work_dir))
    latest_dataset_rows = snapshot.dataset_rows
    run_state = _load_run_state(paths)
    pending_changes = _has_pending_control_changes(control, applied_control)
    configured_evaluator_names = _configured_evaluator_names(config)
    force_evaluator_options = _force_evaluator_options(
        configured_names=configured_evaluator_names,
        current_force_evaluator=control.force_evaluator,
    )
    baseline_tree_branch_limit = _baseline_tree_branch_limit(config)
    applied_runtime_control = _applied_runtime_control(run_state)
    effective_runtime_config = _effective_runtime_config(run_state)
    effective_runtime_hash = _effective_runtime_hash(run_state)
    tree_branch_limit_input_value = _tree_branch_limit_input_value(
        runtime_control=control.runtime,
        baseline_limit=baseline_tree_branch_limit,
    )
    pending_fields = _pending_control_fields(control, applied_control)
    pending_sections = _pending_control_sections(control, applied_control)
    dataset_summary = _dataset_status_summary(config, control, applied_control)
    scheduling_summary = _scheduling_status_summary(config, control, applied_control)
    evaluator_summary = _evaluator_control_status_summary(
        control=control,
        applied_control=applied_control,
        configured_names=configured_evaluator_names,
    )
    runtime_summary = _runtime_status_summary(
        baseline_limit=baseline_tree_branch_limit,
        current_runtime_control=control.runtime,
        applied_runtime=applied_runtime_control,
        effective_runtime=effective_runtime_config,
        runtime_hash=effective_runtime_hash,
    )
    _render_pending_changes_section(
        st=st,
        pending_changes=pending_changes,
        pending_sections=pending_sections,
        pending_fields=pending_fields,
    )
    _render_effective_state_section(
        st=st,
        summary=_effective_state_summary(
            run_summary=summary,
            run_state=run_state,
            current_control=control,
            baseline_limit=baseline_tree_branch_limit,
            effective_runtime=effective_runtime_config,
            latest_dataset_rows=latest_dataset_rows,
            pending_changes=pending_changes,
            configured_names=configured_evaluator_names,
        ),
    )
    st.caption(
        "Unchecked controls inherit the persisted bootstrap config baseline. "
        "Checked controls persist explicit overrides to the control file."
    )

    with st.container():
        _render_dataset_control_section(st=st, summary=dataset_summary)
        override_max_rows = st.checkbox(
            "Persist explicit override for max rows",
            value=control.max_rows is not None,
        )
        max_rows = st.number_input(
            "Max rows",
            min_value=0,
            value=_control_number_value(control.max_rows),
            disabled=not override_max_rows,
        )
        override_use_backed_up_value = st.checkbox(
            "Persist explicit override for use backed-up value",
            value=control.use_backed_up_value is not None,
        )
        use_backed_up_value = st.checkbox(
            "Use backed-up value",
            value=False
            if control.use_backed_up_value is None
            else control.use_backed_up_value,
            disabled=not override_use_backed_up_value,
        )

        _render_scheduling_control_section(st=st, summary=scheduling_summary)
        override_max_growth_steps_per_cycle = st.checkbox(
            "Persist explicit override for max growth steps per cycle",
            value=control.max_growth_steps_per_cycle is not None,
        )
        max_growth_steps_per_cycle = st.number_input(
            "Max growth steps per cycle",
            min_value=0,
            value=_control_number_value(control.max_growth_steps_per_cycle),
            disabled=not override_max_growth_steps_per_cycle,
        )
        override_save_after_seconds = st.checkbox(
            "Persist explicit override for save after seconds",
            value=control.save_after_seconds is not None,
        )
        save_after_seconds = st.number_input(
            "Save after seconds",
            min_value=0.0,
            value=_control_float_value(control.save_after_seconds),
            step=1.0,
            disabled=not override_save_after_seconds,
        )
        override_save_after_tree_growth_factor = st.checkbox(
            "Persist explicit override for save after tree growth factor",
            value=control.save_after_tree_growth_factor is not None,
        )
        save_after_tree_growth_factor = st.number_input(
            "Save after tree growth factor",
            min_value=0.0,
            value=_control_float_value(
                control.save_after_tree_growth_factor, default=2.0
            ),
            step=0.1,
            disabled=not override_save_after_tree_growth_factor,
        )

        _render_evaluator_control_section(st=st, summary=evaluator_summary)
        force_evaluator_mode = st.radio(
            "Evaluator selection mode",
            options=("auto", "forced"),
            index=0 if control.force_evaluator is None else 1,
            horizontal=True,
            help="Auto inherits normal evaluator selection. Forced persists an explicit evaluator override.",
        )
        selectable_force_evaluator_options = (
            force_evaluator_options if force_evaluator_options else ("",)
        )

        def _format_option(value: str) -> str:
            return _format_force_evaluator_option(
                value,
                configured_names=configured_evaluator_names,
            )

        force_evaluator = st.selectbox(
            "Forced evaluator",
            options=selectable_force_evaluator_options,
            index=_force_evaluator_option_index(
                selectable_force_evaluator_options,
                control.force_evaluator,
            ),
            disabled=(force_evaluator_mode != "forced" or not force_evaluator_options),
            format_func=_format_option,
        )

        _render_runtime_control_section(st=st, summary=runtime_summary)
        override_tree_branch_limit = st.checkbox(
            "Persist explicit override for tree branch limit",
            value=control.runtime.tree_branch_limit is not None,
        )
        tree_branch_limit = st.number_input(
            "Tree branch limit",
            min_value=1,
            value=tree_branch_limit_input_value,
            disabled=not override_tree_branch_limit,
        )

        next_control = _build_next_control(
            override_max_growth_steps_per_cycle=override_max_growth_steps_per_cycle,
            max_growth_steps_per_cycle=max_growth_steps_per_cycle,
            override_max_rows=override_max_rows,
            max_rows=max_rows,
            override_use_backed_up_value=override_use_backed_up_value,
            use_backed_up_value=use_backed_up_value,
            override_save_after_seconds=override_save_after_seconds,
            save_after_seconds=save_after_seconds,
            override_save_after_tree_growth_factor=override_save_after_tree_growth_factor,
            save_after_tree_growth_factor=save_after_tree_growth_factor,
            override_tree_branch_limit=override_tree_branch_limit,
            tree_branch_limit=tree_branch_limit,
            force_evaluator_mode=force_evaluator_mode,
            force_evaluator=force_evaluator,
        )
        requested = bootstrap_control_to_dict(control)
        proposed = bootstrap_control_to_dict(next_control)
        diff = [
            {
                "Field": key,
                "Current request": repr(requested.get(key)),
                "Proposed": repr(value),
            }
            for key, value in proposed.items()
            if requested.get(key) != value
        ]
        st.caption("Pending edit preview · applies at the existing cycle boundary")
        if diff:
            st.dataframe(diff, hide_index=True, width="stretch")
        else:
            st.caption("No edits to apply.")
        if st.button("Apply changes", disabled=not diff):
            save_bootstrap_control(next_control, paths.control_path)
            st.success(
                "Saved control changes. They will apply at the next cycle boundary."
            )
