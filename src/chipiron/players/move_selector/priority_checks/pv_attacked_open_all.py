"""Priority check opening all branches when the principal variation is tactically attacked."""


import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone.hooks.search_hooks import FeatureExtractor
from anemone.node_selector.opening_instructions import (
    OpeningInstructor,
    OpeningInstructions,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone.trees import Tree


def _deepest_existing_node_on_pv(root: AlgorithmNode[Any]) -> AlgorithmNode[Any]:
    """Return the deepest already-expanded node along the current PV sequence."""
    node: AlgorithmNode[Any] = root
    best_branch_sequence = getattr(root.tree_evaluation, "best_branch_sequence", ())

    for branch in best_branch_sequence:
        child = node.branches_children.get(branch)
        if child is None:
            break
        node = child

    return node


@dataclass(frozen=True)
class PvAttackedOpenAllPriorityCheck:
    """Open all branches at the current PV node when it is tactically threatened."""

    opening_instructor: OpeningInstructor
    feature_extractor: FeatureExtractor | None
    random_generator: random.Random
    probability: float = 0.5
    feature_key: str = "tactical_threat"

    def maybe_choose_opening(
        self,
        tree: "Tree[AlgorithmNode[Any]]",
        latest_tree_expansions: Any,
    ) -> OpeningInstructions[AlgorithmNode[Any]] | None:
        """Optionally return opening instructions that override the base selector."""
        _ = latest_tree_expansions


        print(f"DEBUG PV-attacked opening all branchnches ")

        if self.feature_extractor is None:
            print(f"DEBUG PV-attacked opening all branches: no feature extractor, skipping")

            return None

        target = _deepest_existing_node_on_pv(tree.root_node)



        if target.is_over():
            print(f"DEBUG PV-attacked opening all branches: PV node is terminal, skipping") 
            return None

        if self.probability < 1.0 and self.random_generator.random() > self.probability:
            print(f"DEBUG PV-attacked opening all branches: random check failed, skipping")
            return None



        features = self.feature_extractor.features(target.state)
        print("PV target depth:", target.tree_depth, "id:", target.id)
        print("pv len:", len(getattr(tree.root_node.tree_evaluation, "best_branch_sequence", ())))
        print("feature:", features[self.feature_key])
        print("non_opened_branches:", len(target.non_opened_branches))
        print("all_branches_generated:", target.all_branches_generated)

        if self.feature_key not in features:
            raise RuntimeError(
                f"Priority check '{self.__class__.__name__}' "
                f"requires feature '{self.feature_key}', "
                f"but extractor returned keys {list(features.keys())}"
            )

        if not features[self.feature_key]:
            print(f"DEBUGI PV-attacked opening all branches: feature '{self.feature_key}' not present, skipping")
            return None


        if target.all_branches_generated:
            print(f"DEBUG PV-attacked opening all branches: no non-opened branches at PV node, skipping")
            return None

        branches_to_open = self.opening_instructor.all_branches_to_open(target)
        print(f"DEBUGAAA PV-attacked opening all branches: opening {len(branches_to_open)} branches at nodewith state {target.state}")
        return create_instructions_to_open_all_branches(
            branches_to_play=branches_to_open,
            node_to_open=target,
        )
