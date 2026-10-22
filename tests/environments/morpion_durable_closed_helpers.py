"""Closed-layout fixtures over the existing serial Growth/receipt test harness."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap import run_pipeline_growth_stage
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    save_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    scientific_nodes,
    scientific_producer,
    setup_scientific_baseline,
)


class ClosedPublicationRunner(ScientificPublicationRunner):
    """Closedpublicationrunner."""

    def export_closed_scientific_snapshot(
        self,
        destination: Any,
        *,
        generation: int,
        producer: Any,
        committed_source: Any = None,
        _validation_result: Any = None,
    ) -> Any:
        """Export closed scientific snapshot."""
        path, self.closed_stats = save_closed_scientific_snapshot(
            nodes=scientific_nodes(self.values),
            root_node_id="root",
            destination=Path(destination),
            generation=generation,
            producer=producer,
            committed_source=committed_source,
            _validation_result=_validation_result,
        )
        return path


def setup_closed_baseline(root: Path, *, initialize: Any = True) -> Any:
    """Prepare a new closed snapshot and matching scalar fixture checkpoint."""
    args, paths, state = setup_scientific_baseline(root, initialize=False)
    runner = ClosedPublicationRunner()
    from tests.environments.morpion_patch_receipt_helpers import BASELINE

    runner.values = dict(BASELINE)
    source = runner.export_closed_scientific_snapshot(
        root / "genesis-closed", generation=1, producer=scientific_producer(args)
    )
    state = replace(state, latest_tree_snapshot_path=paths.relative_to_work_dir(source))
    if initialize:
        state = initialize_scientific_v2_publication(
            paths,
            state,
            producer=scientific_producer(args),
            closed_scientific_snapshot=True,
        )
    return args, paths, state


def run_closed_publication(args: Any, runner: Any, *, max_cycles: int = 1) -> Any:
    """Run closed publication."""
    return run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=max_cycles,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=scientific_producer(args),
        closed_scientific_snapshot=True,
    )


def publish_closed_patch(paths: Any, patch_id: str) -> Any:
    """Publish closed patch."""
    from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        committed_scientific_source,
    )
    from tests.environments.morpion_patch_receipt_helpers import (
        fixture_patch,
        publish_fixture_patch,
    )

    state = load_bootstrap_run_state(paths.run_state_path)
    patch = fixture_patch(patch_id)
    publish_fixture_patch(
        paths,
        replace(
            patch,
            tree_generation=state.generation,
            metadata={
                **patch.metadata,
                "scientific_publication": committed_scientific_source(
                    paths.work_dir / state.latest_tree_snapshot_path
                ),
            },
        ),
    )
