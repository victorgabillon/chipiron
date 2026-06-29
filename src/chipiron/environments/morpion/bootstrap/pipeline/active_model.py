"""Active-model resolution helpers for Morpion artifact-pipeline growth."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    ResolvedActiveMorpionModelBundle,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_active_model,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

LOGGER = logging.getLogger(__name__)


def _resolve_pipeline_active_model_for_growth(
    *,
    paths: MorpionBootstrapPaths,
    force_evaluator: str | None,
) -> ResolvedActiveMorpionModelBundle:
    """Resolve the active model for artifact-pipeline growth from the pipeline artifact."""
    if not paths.pipeline_active_model_path.is_file():
        LOGGER.info(
            "[growth] active_model_status source=none evaluator=none model_bundle=none"
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )

    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    if force_evaluator is not None and active_model.evaluator_name != force_evaluator:
        LOGGER.warning(
            "[growth] active_model_force_evaluator_mismatch requested=%s active=%s artifact=%s",
            force_evaluator,
            active_model.evaluator_name,
            paths.pipeline_active_model_path,
        )
    model_bundle_path = paths.resolve_work_dir_path(active_model.model_bundle_path)
    if model_bundle_path is None or not model_bundle_path.exists():
        LOGGER.warning(
            "[growth] active_model_missing_bundle source=pipeline_active_model generation=%s evaluator=%s model_bundle=%s artifact=%s",
            active_model.generation,
            active_model.evaluator_name,
            model_bundle_path,
            paths.pipeline_active_model_path,
        )
        LOGGER.info(
            "[growth] active_model_status source=none evaluator=none model_bundle=none"
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )
    LOGGER.info(
        "[growth] active_model_status source=pipeline_active_model generation=%s evaluator=%s model_bundle=%s",
        active_model.generation,
        active_model.evaluator_name,
        model_bundle_path,
    )
    return ResolvedActiveMorpionModelBundle(
        active_evaluator_name=active_model.evaluator_name,
        model_bundle_path=model_bundle_path,
    )
