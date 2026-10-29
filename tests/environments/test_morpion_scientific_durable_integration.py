"""Cross-stack seams; scientific v2 publication remains explicitly unsupported."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, get_args

import pytest

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    artifact_digest,
    capsule_path,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    save_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import publish_fixture_patch
from tests.environments.test_morpion_sharded_training_v2 import PRODUCER, nodes_fixture

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("directory", ["tree_exports_sharded", "custom-v2"])
def test_v1_capsule_refuses_scientific_v2_before_sealing(
    tmp_path: Path, directory: str
) -> None:
    """A future writer adapter cannot sneak v2 through the generic capsule seal."""
    _args, paths = setup_publication_baseline(tmp_path)
    original_state = paths.run_state_path.read_bytes()
    with growth_receipt_ownership(paths):
        publication = DurableGenerationPublication(paths)
        state = publication.prepare(load_bootstrap_run_state(paths.run_state_path))
        publication.begin_cycle(None)
        assert publication.staging is not None
        checkpoint = publication.checkpoint_path(2, "json-zst")
        runner = PublicationRunner()
        runner.values = {"root": 2.0, "leaf": 4.0}
        runner.save_checkpoint(checkpoint)
        export, _ = save_scientific_training_generation(
            nodes=nodes_fixture(),
            root_node_id="0",
            output_dir=publication.staging / directory,
            generation=1,
            producer=PRODUCER,
        )
        candidate = replace(
            state,
            generation=2,
            latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
            latest_tree_snapshot_path=paths.relative_to_work_dir(export),
        )
        objects = set((publication.root / "objects").iterdir())
        with pytest.raises(GenerationPublicationError, match="scientific v1"):
            publication.publish(candidate)
        assert set((publication.root / "objects").iterdir()) == objects
        assert not (publication.staging / ".generation_publication_v1.json").exists()
        assert paths.run_state_path.read_bytes() == original_state


def test_scientific_writer_cannot_mutate_a_sealed_capsule(tmp_path: Path) -> None:
    """All cooperative writers must honor the immutable-inode boundary."""
    _args, paths = setup_publication_baseline(tmp_path)
    state = load_bootstrap_run_state(paths.run_state_path)
    capsule = capsule_path(tmp_path, publication_records(state.metadata)[-1]["id"])
    before = artifact_digest(capsule)
    with pytest.raises(GenerationPublicationError, match="immutable publication"):
        save_scientific_training_generation(
            nodes=nodes_fixture(),
            root_node_id="0",
            output_dir=capsule / "new-scientific-v2",
            generation=1,
            producer=PRODUCER,
        )
    assert artifact_digest(capsule) == before


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_committed_v1_dataset_parity_with_all_legacy_policies(
    tmp_path: Path, policy: str
) -> None:
    """The merged reader retains capsule pins and exact legacy Dataset bytes."""
    args, paths = setup_publication_baseline(tmp_path)
    publish_fixture_patch(paths)
    state = run_publication(args, PublicationRunner())
    assert state.latest_tree_snapshot_path is not None
    path = tmp_path / state.latest_tree_snapshot_path
    reference = load_morpion_sharded_training_tree_snapshot(path)
    args = replace(args, dataset_family_target_policy=policy)
    expected = extract_rows_from_training_snapshot(
        args=args, snapshot=reference, generation=state.generation
    )
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.format_version == 1
        assert tuple(view.iter_nodes_in_export_order()) == reference.nodes
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            stream = context.stream(state.generation)
            assert tuple(stream.rows) == expected.rows
            assert stream.metadata == expected.metadata
            save_morpion_supervised_rows_streaming(
                rows=context.rows(),
                metadata=stream.metadata,
                path=tmp_path / "indexed.jsonl",
            )
        view.validate_source_identity()
    save_morpion_supervised_rows_streaming(
        rows=expected.rows,
        metadata=expected.metadata,
        path=tmp_path / "reference.jsonl",
    )
    assert (tmp_path / "indexed.jsonl").read_bytes() == (
        tmp_path / "reference.jsonl"
    ).read_bytes()
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as reused:
        assert reused.index_reused
