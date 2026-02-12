"""Priority check opening all branches when the principal variation is tactically attacked."""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone.hooks.search_hooks import FeatureExtractor
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
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


class FeatureKeyMissingError(RuntimeError):
    """Raised when a required feature key is missing from the extractor output."""

    def __init__(
        self, priority_check_name: str, feature_key: str, available_keys: list[str]
    ) -> None:
        """Construct an error message indicating the missing feature key and available keys."""
        message = f"Priority check '{priority_check_name}' requires feature '{feature_key}', but extractor returned keys {available_keys}"
        super().__init__(message)


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

        if self.feature_extractor is None:
            return None

        target = _deepest_existing_node_on_pv(tree.root_node)

        if target.is_over():
            return None

        if self.probability < 1.0 and self.random_generator.random() > self.probability:
            return None

        features = self.feature_extractor.features(target.state)

        if self.feature_key not in features:
            raise FeatureKeyMissingError(
                priority_check_name=self.__class__.__name__,
                feature_key=self.feature_key,
                available_keys=list(features.keys()),
            )

        if not features[self.feature_key]:
            return None

        if target.all_branches_generated:
            return None

        branches_to_open = self.opening_instructor.all_branches_to_open(target)

        return create_instructions_to_open_all_branches(
            branches_to_play=branches_to_open,
            node_to_open=target,
        )
