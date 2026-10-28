"""Controlled synthetic graph fixtures for resource tests, never saved experiments."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anemone.node_evaluation.common import canonical_value
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from atomheart.games.morpion.dynamics import MorpionDynamics

from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    save_scientific_training_generation,
    value_score,
)
from tests.environments.test_morpion_sharded_training_v2 import Evaluation

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ScientificFixtureNode:
    """Declared evaluator values; geometry is real, scores are controlled inputs."""

    id: str
    depth: int
    state: Any
    tree_evaluation: Evaluation
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    visit_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def generated_scientific_source(
    root: Path,
    count: int,
    *,
    padding: int = 0,
    id_padding: int = 0,
    generations: int = 1,
    legacy_root: Path | None = None,
) -> Path:
    """Generate exact-size bounded-degree DAGs with no production experiment input.

    Boards follow a deterministic legal move at each depth. Duplicate boards
    under distinct IDs make a controlled resource fixture, not a claim about
    search transposition uniqueness or certified game optima.
    """
    assert count >= 4 and generations in (1, 2)
    codec = MorpionStateCheckpointCodec()
    dynamics = MorpionDynamics()
    boards = [codec.load_state_ref((0, ()))]
    for _ in range((count + 1).bit_length()):
        state = boards[-1]
        boards.append(
            dynamics.step(state, dynamics.all_legal_actions(state)[0]).next_state
        )
    ids = ["x" * id_padding + f"node-{i:06d}-é" for i in range(count)]
    nodes = []
    estimate = canonical_value.make_estimate_value
    forced = canonical_value.make_forced_value
    for i, node_id in enumerate(ids):
        depth = (i + 1).bit_length() - 1
        direct = (forced if i % 13 == 0 else estimate)(score=float(i % 17) + 0.25)
        tree = None if i % 5 == 0 else estimate(score=float(i % 11) - 0.5)
        children = tuple(ids[j] for j in (2 * i + 1, 2 * i + 2) if j < count)
        nodes.append(
            ScientificFixtureNode(
                node_id,
                depth,
                boards[depth],
                Evaluation(direct, tree),
                () if i == 0 else (ids[(i - 1) // 2],),
                children,
                i % 19,
                {"synthetic_padding": "p" * padding, "fraction": 0.1},
            )
        )
    producer = {
        "revision": "controlled-scientific-reader-fixture-v1",
        "evaluator": "declared-values-not-game-certification",
        "config_sha256": "1" * 64,
    }
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id=ids[0],
        output_dir=root,
        generation=1,
        producer=producer,
    )
    if legacy_root is not None:
        save_morpion_sharded_training_tree_from_live_nodes(
            nodes=nodes,
            root_node_id=ids[0],
            output_dir=legacy_root,
            generation=1,
            state_ref_dumper=codec.dump_state_ref,
            direct_value_extractor=value_score,
            backed_up_value_extractor=value_score,
        )
    if generations == 2:
        new_parent = (count - 2) // 2 - 1
        nodes[-1].parent_ids += (ids[new_parent],)
        nodes[new_parent].child_ids += (ids[-1],)
        nodes[1].tree_evaluation.direct_value = estimate(score=99.0)
        path, _ = save_scientific_training_generation(
            nodes=nodes,
            root_node_id=ids[0],
            output_dir=root,
            generation=2,
            producer=producer,
        )
        if legacy_root is not None:
            save_morpion_sharded_training_tree_from_live_nodes(
                nodes=nodes,
                root_node_id=ids[0],
                output_dir=legacy_root,
                generation=2,
                state_ref_dumper=codec.dump_state_ref,
                direct_value_extractor=value_score,
                backed_up_value_extractor=value_score,
            )
    return path
