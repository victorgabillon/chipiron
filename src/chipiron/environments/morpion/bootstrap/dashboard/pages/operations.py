"""Progressively disclosed operational state and existing experiment controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import (
    page_header,
    worker_strip,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Keep operational density here, outside the first-screen run summary."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.controls import (
        render_controls,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.observability import (
        observability_summary_from_metadata,
        render_observability_section,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.run_control import (
        render_run_control_section,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import mapping

    paths = MorpionBootstrapPaths.from_work_dir(snapshot.work_dir)
    page_header(
        st,
        snapshot,
        "Operations",
        "Persistence, pipeline evidence and deliberate experiment controls.",
    )
    st.subheader("Pipeline")
    worker_strip(st, snapshot)
    st.caption(
        "Worker liveness remains unknown unless process evidence establishes it. A saved stage status is historical evidence."
    )
    with st.expander("Lifecycle & launcher"):
        render_run_control_section(st=st, paths=paths)
    with st.expander("Storage · detailed disk usage"):
        st.caption(
            "Scans artifact sizes only when requested; large runs can take a moment."
        )
        if st.button("Measure storage"):
            from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
                build_disk_usage_summary,
            )
            from chipiron.environments.morpion.bootstrap.dashboard.sections.disk_usage import (
                render_disk_usage_section,
            )

            with st.spinner("Measuring stored artifacts…"):
                render_disk_usage_section(
                    st=st, summary=build_disk_usage_summary(snapshot.work_dir)
                )
    with st.expander("Memory & persistence · checkpoint/export diagnostics"):
        render_observability_section(
            st=st,
            summary=observability_summary_from_metadata(
                mapping(snapshot.run_state.get("metadata"))
            ),
        )
        st.write(
            "Latest checkpoint:",
            snapshot.run_state.get("latest_runtime_checkpoint_path", "unknown"),
        )
        st.json(snapshot.run_state)
    with st.expander("Full history · datasets, evaluator changes and all loss curves"):
        st.caption(
            "Overview keeps recent observations bounded. Full history is loaded only on request."
        )
        if st.checkbox("Load full historical analysis"):
            render_full_history(st, paths)
    with st.expander("Modify running experiment"):
        st.caption(
            "Changes are persistent requests. The existing runtime applies them at cycle boundaries."
        )
        try:
            render_controls(st, snapshot.work_dir, snapshot)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(
                f"Controls unavailable until the artifact error is resolved: {exc}"
            )
    with st.expander("Debug · effective configuration and raw artifact paths"):
        st.code(str(snapshot.work_dir))
        st.write("Configured baseline")
        st.json(snapshot.config)
        st.write("Prepared manifest")
        st.json(snapshot.manifest)
        st.write("Pipeline dataset / training status")
        st.json(snapshot.dataset)
        st.json(snapshot.training)
        st.write("Run metadata (effective runtime and applied controls)")
        st.json(mapping(snapshot.run_state.get("metadata")))
        from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
            read_artifact,
        )

        errors: list[str] = []
        st.write("Current requested control")
        st.json(read_artifact(snapshot.work_dir / "control.json", errors))
        for error in errors:
            st.warning(error)


def render_full_history(st: Any, paths: Any) -> None:
    """Retain the original reusable scientific history plots behind a lazy action."""
    from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
        cached_build_morpion_bootstrap_dashboard_data,
        cached_dashboard_data_freshness_tokens,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
        downsample_loss_series_by_name,
        downsample_series,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.plot import (
        plot_active_evaluator,
        plot_certified_record_score,
        plot_dataset_size,
        plot_evaluator_losses,
        plot_tree_size,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
        render_plot,
    )

    try:
        with st.spinner("Loading historical artifacts…"):
            data = cached_build_morpion_bootstrap_dashboard_data(
                str(paths.work_dir), cached_dashboard_data_freshness_tokens(paths)
            )
        for plot, series in (
            (plot_tree_size, data.tree_num_nodes),
            (plot_dataset_size, data.dataset_num_rows),
            (plot_active_evaluator, data.active_evaluator),
            (plot_certified_record_score, data.certified_record_score),
        ):
            render_plot(
                st, lambda plot=plot, series=series: plot(downsample_series(series))
            )
        logarithmic = st.checkbox("Log scale for loss")
        render_plot(
            st,
            lambda: plot_evaluator_losses(
                downsample_loss_series_by_name(data.evaluator_loss_by_name),
                log_scale=logarithmic,
            ),
        )
        st.caption(
            "All recorded generations; plots retain the existing display downsampling."
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Historical artifacts unavailable: {exc}")
