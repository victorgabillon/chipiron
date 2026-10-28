"""New scientific sources and tiny cold scalar checkpoints, never saved experiments."""

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from anemone.node_evaluation.common.canonical_value import make_estimate_value
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.dynamics import MorpionDynamics

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    run_pipeline_growth_stage,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    save_scientific_training_generation,
)
from tests.environments.morpion_generation_publication_helpers import PublicationRunner
from tests.environments.morpion_patch_receipt_helpers import (
    BASELINE,
    MODEL,
    receipt_args,
)
from tests.environments.morpion_scientific_reader_helpers import ScientificFixtureNode
from tests.environments.test_morpion_sharded_training_v2 import Evaluation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapArgs,
        MorpionBootstrapRunState,
    )


def scientific_args(root: Path) -> MorpionBootstrapArgs:
    """Keep the existing receipt fixture thresholds and activate only explicit API calls."""
    return replace(
        receipt_args(root),
        training_export_mode="sharded",
        growth_skip_training_export=False,
    )


def scientific_producer(args: MorpionBootstrapArgs) -> dict[str, str]:
    """Keep a fixed config/evaluator/source identity across generated generations."""
    return {
        "revision": "durable-v2-synthetic-fixture",
        "evaluator": "declared-estimates",
        "config_sha256": bootstrap_config_sha256(bootstrap_config_from_args(args)),
    }


def scientific_nodes(values: dict[str, float]) -> list[ScientificFixtureNode]:
    """Gain an incoming parent when a third node joins; no proof claims are invented."""
    dynamics = MorpionDynamics()
    boards = [initial_state()]
    for _ in range(2):
        boards.append(
            dynamics.step(
                boards[-1], dynamics.all_legal_actions(boards[-1])[0]
            ).next_state
        )
    names = ["root", *sorted(k for k in values if k != "root")]
    result = []
    for name in names:
        parents = () if name == "root" else ("root",)
        children = tuple(names[1:]) if name == "root" else ()
        if "new" in values:
            if name == "leaf":
                parents += ("new",)
            if name == "new":
                children = ("leaf",)
        result.append(
            ScientificFixtureNode(
                name,
                0 if name == "root" else (2 if name == "leaf" else 1),
                boards[0 if name == "root" else (2 if name == "leaf" else 1)],
                Evaluation(make_estimate_value(score=values[name]), None),
                parents,
                children,
                3,
                {"fixture": "durable-scientific-v2", "node": name},
            )
        )
    return result


class ScientificPublicationRunner(PublicationRunner):
    """Use the authoritative scientific writer around the existing receipt fixture."""

    def export_scientific_training_snapshot(
        self, output_dir: str | Path, *, generation: int, producer: Mapping[str, str]
    ) -> Path:
        """Export all Values, links and references, never v1 fallback payloads."""
        assert self.values is not None
        path, _stats = save_scientific_training_generation(
            nodes=scientific_nodes(self.values),
            root_node_id="root",
            output_dir=output_dir,
            generation=generation,
            producer=producer,
        )
        return path


def setup_scientific_baseline(
    root: Path, *, initialize: bool = True
) -> tuple[MorpionBootstrapArgs, MorpionBootstrapPaths, MorpionBootstrapRunState]:
    """Prepare a new v2/checkpoint pair without creating or adopting historical state."""
    args = scientific_args(root)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    checkpoint = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint.write_text(json.dumps(BASELINE, sort_keys=True) + "\n")
    model = root / MODEL
    model.mkdir(parents=True)
    (model / "param.pt").write_bytes(b"synthetic-model-identity\x00")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="default",
            model_bundle_path=MODEL,
            updated_at_utc="2026-10-21T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    runner = ScientificPublicationRunner()
    runner.values = dict(BASELINE)
    source = runner.export_scientific_training_snapshot(
        root / "genesis-scientific-v2", generation=1, producer=scientific_producer(args)
    )
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        cycle_index=0,
        tree_size_at_last_save=2,
        last_save_unix_s=1e9,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        metadata={"bootstrap_config_hash": bootstrap_config_sha256(config)},
    )
    if initialize:
        state = initialize_scientific_v2_publication(
            paths, state, producer=scientific_producer(args)
        )
    return args, paths, state


def run_scientific_publication(
    args: MorpionBootstrapArgs,
    runner: ScientificPublicationRunner,
    *,
    max_cycles: int = 1,
) -> MorpionBootstrapRunState:
    """Run one explicit experimental serial Growth invocation."""
    return run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=max_cycles,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=scientific_producer(args),
    )


def publish_scientific_patch(paths: MorpionBootstrapPaths) -> None:
    """Use the existing patch publisher, binding the committed scientific source."""
    from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        committed_scientific_source,
    )
    from tests.environments.morpion_patch_receipt_helpers import (
        fixture_patch,
        publish_fixture_patch,
    )

    state = load_bootstrap_run_state(paths.run_state_path)
    patch = fixture_patch()
    patch = replace(
        patch,
        metadata={
            **patch.metadata,
            "scientific_publication": committed_scientific_source(
                paths.work_dir / state.latest_tree_snapshot_path
            ),
        },
    )
    publish_fixture_patch(paths, patch)
