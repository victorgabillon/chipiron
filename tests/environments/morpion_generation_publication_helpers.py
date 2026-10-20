"""Generated v1 exports and tiny scalar checkpoints; no search or workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anemone.training_export import save_training_tree_snapshot
from anemone.training_export.builders import build_training_tree_snapshot
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    run_pipeline_growth_stage,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    save_morpion_sharded_training_tree_from_live_nodes,
)
from tests.environments.morpion_patch_receipt_helpers import (
    BASELINE,
    ColdScalarRunner,
    setup_receipt_baseline,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapArgs,
        MorpionBootstrapPaths,
        MorpionBootstrapRunState,
    )


@dataclass
class PublicationNode:
    """Expose precisely the public fields read by the released v1 builder."""

    id: str
    depth: int
    state: object
    direct_value: float
    backed_up_value: float | None
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    visit_count: int = 3
    is_terminal: bool = False
    is_exact: bool = False
    metadata: dict[str, object] | None = None


def synthetic_nodes(values: dict[str, float]) -> tuple[PublicationNode, ...]:
    """Keep real serializable Morpion refs but perform no search or state restoration."""
    state_ref = MorpionStateCheckpointCodec().dump_state_ref(initial_state())
    names = ["root", *sorted(k for k in values if k != "root")]
    return tuple(
        PublicationNode(
            id=name,
            depth=int(name != "root"),
            state=state_ref,
            direct_value=values[name],
            backed_up_value=max(values.values()) if name == "root" else None,
            parent_ids=() if name == "root" else ("root",),
            child_ids=tuple(names[1:]) if name == "root" else (),
            metadata={"fixture": "publication", "node": name},
        )
        for name in names
    )


def _scalar(value: object | None) -> float | None:
    """Match the scalar extractor used by both fixture export paths."""
    return None if value is None else float(cast("float", value))


def write_v1_export(values: dict[str, float], root: Path, generation: int) -> Path:
    """Call the unchanged additive writer with complete deterministic nodes."""
    path, _stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=synthetic_nodes(values),
        root_node_id="root",
        output_dir=root,
        generation=generation,
        state_ref_dumper=lambda state: state,
        direct_value_extractor=_scalar,
        backed_up_value_extractor=_scalar,
    )
    return path


class PublicationRunner(ColdScalarRunner):
    """Use actual scientific v1 exports around PR #79's non-idempotent scalar fixture."""

    def __init__(self) -> None:
        """Record model load provenance alongside the inherited exact apply counter."""
        super().__init__()
        self.model_paths: list[Path] = []

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Keep checkpoint semantics while observing the frozen evaluator path."""
        super().load_or_create(
            tree_snapshot_path,
            model_bundle_path,
            effective_runtime_config,
            reevaluate_tree=reevaluate_tree,
        )
        if model_bundle_path is not None:
            self.model_paths.append(Path(model_bundle_path))

    def export_sharded_training_tree_snapshot(
        self, output_dir: str | Path, *, generation: int
    ) -> Path:
        """Exercise the real legacy writer inside the new private staging directory."""
        assert self.values is not None
        return write_v1_export(self.values, Path(output_dir), generation)

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Exercise the real flat serializer for 'both' mode parity."""
        assert self.values is not None
        snapshot = build_training_tree_snapshot(
            synthetic_nodes(self.values),
            root_node_id="root",
            state_ref_dumper=lambda state: state,
            direct_value_extractor=_scalar,
            backed_up_value_extractor=_scalar,
        )
        save_training_tree_snapshot(snapshot, output_path)


def setup_publication_baseline(
    work_dir: Path,
    *,
    values: dict[str, float] | None = None,
    adopt_publication: bool = True,
) -> tuple[MorpionBootstrapArgs, MorpionBootstrapPaths]:
    """Adopt a generated sharded v1 baseline, leaving every legacy payload untouched."""
    args, paths = setup_receipt_baseline(work_dir, adopt=False)
    values = dict(BASELINE) if values is None else values
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.latest_runtime_checkpoint_path is not None
    import json

    (work_dir / state.latest_runtime_checkpoint_path).write_text(
        json.dumps(values, sort_keys=True) + "\n"
    )
    snapshot = write_v1_export(values, paths.sharded_tree_snapshot_dir, 1)
    state = replace(
        load_bootstrap_run_state(paths.run_state_path),
        latest_tree_snapshot_path=paths.relative_to_work_dir(snapshot),
        tree_size_at_last_save=len(values),
        last_save_unix_s=1e9,
    )
    save_bootstrap_run_state(state, paths.run_state_path)
    args = replace(args, training_export_mode="sharded")
    run_pipeline_growth_stage(
        args, PublicationRunner(), max_cycles=0, checkpoint_bound_receipts=True
    )
    args = replace(args, growth_skip_training_export=False)
    if adopt_publication:
        run_publication(args, PublicationRunner(), max_cycles=0)
    return args, paths


def run_publication(
    args: MorpionBootstrapArgs, runner: PublicationRunner, *, max_cycles: int = 1
) -> MorpionBootstrapRunState:
    """Run the experimental API solely with generated fixtures."""
    return run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=max_cycles,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
    )
