"""Active-model provenance and generation-specific learning diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import (
    PALETTE,
    metric_row,
    page_header,
)
from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
    active_model_summary,
    display,
    mapping,
)

from .overview import loss_rows, trend

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Separate the active bundle's provenance from planned or subsequent training."""
    from chipiron.environments.morpion.bootstrap.dashboard.sections.evaluator_diagnostics import (
        render_evaluator_training_diagnostics_section,
    )

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
        cached_evaluator_loss_freshness_tokens,
        cached_load_evaluator_loss_series_for_dashboard,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
        downsample_loss_series_by_name,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.plot import (
        plot_evaluator_losses,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
        render_plot,
    )

    page_header(
        st,
        snapshot,
        "Evaluator",
        "Model identity, learning evidence and the data behind it.",
    )
    summary = active_model_summary(snapshot)
    with st.container(border=True):
        st.caption("ACTIVE EVALUATOR")
        st.subheader(display(snapshot.evaluator_name))
        metric_row(
            st,
            [
                ("Model generation", summary["Generation"]),
                ("Source", summary["Model source"]),
                ("Parameters", summary["Parameter count"]),
            ],
        )
        st.caption("Architecture: " + display(summary["Architecture"]))
        st.caption(
            "Representation: "
            + display(summary["Representation"])
            + " · relation_bias_scale: "
            + display(summary["relation_bias_scale"])
        )
        with st.expander("Active model provenance"):
            st.dataframe(
                [
                    {"Field": key, "Value": display(value)}
                    for key, value in summary.items()
                ],
                hide_index=True,
                width="stretch",
            )
            st.json(snapshot.active_model)
            st.json(snapshot.model_manifest)
            st.json(snapshot.model_args)
    configured = mapping(mapping(snapshot.config.get("evaluators")).get("evaluators"))
    results = mapping(snapshot.training.get("evaluator_results"))
    names = sorted(
        set(configured)
        | set(results)
        | ({str(snapshot.evaluator_name)} if snapshot.evaluator_name else set())
    )
    if not names:
        st.info("No configured or trained evaluator metadata is available yet.")
        return

    st.subheader("Loss comparison")
    st.caption(
        "All recorded evaluator validation-loss curves on the same axes. "
        "Use the selector below for one model's detailed diagnostics."
    )
    paths = MorpionBootstrapPaths.from_work_dir(snapshot.work_dir)
    try:
        loss_by_name = cached_load_evaluator_loss_series_for_dashboard(
            str(snapshot.work_dir),
            cached_evaluator_loss_freshness_tokens(paths),
        )
        logarithmic = st.checkbox(
            "Log scale for comparison",
            key="evaluator-all-losses-log-scale",
        )
        render_plot(
            st,
            lambda: plot_evaluator_losses(
                downsample_loss_series_by_name(loss_by_name),
                log_scale=logarithmic,
            ),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"Evaluator loss history temporarily unavailable: {exc}")

    selected = st.selectbox(
        "Inspect evaluator",
        names,
        index=names.index(str(snapshot.evaluator_name))
        if str(snapshot.evaluator_name) in names
        else 0,
    )
    result = mapping(results.get(selected))
    metrics = result
    if not metrics:
        metrics = mapping(mapping(snapshot.event.get("evaluators")).get(selected))
    st.caption(
        "Latest saved training · generation "
        + display(snapshot.training.get("generation", snapshot.event.get("generation")))
    )
    metric_row(
        st,
        [
            ("Train loss", metrics.get("train_loss", result.get("final_loss"))),
            ("Validation loss", metrics.get("validation_loss")),
            ("Train MAE", metrics.get("train_mae")),
            ("Validation MAE", metrics.get("validation_mae")),
        ],
    )
    losses = loss_rows(snapshot, selected)
    trend(
        st,
        title="Learning history · " + selected,
        rows=losses,
        color=PALETTE["green"],
        key="evaluator-loss",
    )
    metric_row(
        st,
        [
            ("Training rows", metrics.get("num_train_samples")),
            ("Validation rows", metrics.get("num_validation_samples")),
            ("Epochs", metrics.get("num_epochs")),
            ("Learning rate", metrics.get("learning_rate")),
        ],
    )
    st.caption(
        "Loss and MAE describe the recorded dataset; lower supervised loss does not establish stronger gameplay."
    )
    render_evaluator_training_diagnostics_section(
        st=st, work_dir=snapshot.work_dir, selected_evaluator_name=selected
    )
    with st.expander("Training artifact details"):
        st.write("Configured next training (not active-bundle provenance)")
        st.json(mapping(configured.get(selected)))
        st.write("Latest saved training result")
        st.json(result)
        st.write("Training status and dataset provenance")
        st.json(snapshot.training)
        st.json(snapshot.dataset)
