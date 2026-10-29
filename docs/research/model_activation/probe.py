"""Research-only refresh comparisons; no publication authority or production route.

The oracle deliberately does not call Anemone backup, propagation, value selection
or branch ordering. The bounded probe reuses those operations on ONE live graph.
"""

from __future__ import annotations

from itertools import batched
from typing import Any

from anemone.node_evaluation.direct.node_direct_evaluator import EvaluationQueries
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from docs.research.rolling_snapshot.measure import runner

_PARTIAL_PROOF = "Unreconstructible partial tree certificate"


def nodes(instance: Any, *, reverse: bool = False) -> Any:
    """Traverse the already resident depth registry without another node list."""
    descendants = instance._require_runtime().tree.descendants
    depths = descendants.range()
    for depth in reversed(depths) if reverse else depths:
        yield from descendants[depth].values()


def graph(count: int = 76) -> Any:
    """Generate a fresh legal Morpion DAG using the existing Growth runner."""
    instance = runner()
    instance.load_or_create(
        None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    while instance.current_tree_size() < count:
        instance.grow(1)
        assert instance.last_growth_steps > 0
    return instance


def exact(value: Any) -> bool:
    """Inspect the value itself, never the node effective-value flag."""
    return value is not None and value.certainty in (
        Certainty.FORCED,
        Certainty.TERMINAL,
    )


def reference(instance: Any, evaluator: Any) -> dict[int, dict[str, Any]]:
    """Small-fixture independent default max oracle, including PV ties and certainty.

    Its O(N) result is ONLY a test oracle. Default single-agent semantics only;
    externally certified partial tree values have no proof reconstruction contract.
    Growth's existing depth factor is deliberately retained exactly, not rounded.
    """
    result = {}
    for node in nodes(instance, reverse=True):
        old = node.tree_evaluation
        if exact(old.tree_value) and not node.tree_node.all_branches_generated:
            raise ValueError(_PARTIAL_PROOF)
        if node.state.is_game_over() or exact(old.direct_value):
            direct = old.direct_value
        else:
            raw = evaluator.evaluate_batch_items([node])[0]
            assert raw.certainty == Certainty.ESTIMATE
            direct = Value(
                score=(1 / 0.99999999) ** node.tree_depth * raw.score,
                certainty=Certainty.ESTIMATE,
            )
        children = [(b, c) for b, c in node.iter_child_links() if c is not None]
        ordered = sorted(
            children,
            key=lambda bc: (
                -result[bc[1].id]["effective"].score,
                -int(exact(result[bc[1].id]["effective"])),
                len(result[bc[1].id]["pv"]),
                bc[1].id,
                str(bc[0]),
            ),
        )
        tree = None
        pv = ()
        if ordered:
            branch, child = ordered[0]
            chosen = result[child.id]["effective"]
            proven = (
                node.tree_node.all_branches_generated
                and len(children) == node.tree_node.child_link_count()
                and all(exact(result[c.id]["effective"]) for _, c in children)
            )
            tree = Value(
                score=chosen.score,
                certainty=Certainty.FORCED if proven else Certainty.ESTIMATE,
                over_event=chosen.over_event if proven else None,
            )
            pv = (branch, *result[child.id]["pv"])
        if tree is None:
            effective, source = direct, "direct_self"
        elif (
            direct is None
            or node.tree_node.all_branches_generated
            or ((direct.score, exact(direct)) <= (tree.score, exact(tree)))
        ):
            effective, source = tree, "tree_child"
        else:
            effective, source = direct, "direct_self"
        result[node.id] = dict(
            direct=direct, tree=tree, effective=effective, source=source, pv=pv
        )
    return result


def state(instance: Any) -> dict[int, dict[str, Any]]:
    """Capture all small-oracle value and principal-variation fields."""
    return {
        n.id: dict(
            direct=n.tree_evaluation.direct_value,
            tree=n.tree_evaluation.tree_value,
            effective=n.tree_evaluation.effective_value,
            source=n.tree_evaluation.effective_value_source.value,
            pv=tuple(n.tree_evaluation.best_branch_sequence),
        )
        for n in nodes(instance)
    }


def differences(instance: Any, expected: dict) -> list[dict]:
    """Report every unequal scientific field without normalization."""
    actual = state(instance)
    return [
        dict(
            node_id=key,
            fields=[
                field for field in expected[key] if expected[key][field] != value[field]
            ],
        )
        for key, value in actual.items()
        if expected[key] != value
    ]


def direct_phase(instance: Any, evaluator: Any, batch_size: int = 64) -> dict:
    """Experimental two-pass kernel: preserve direct certificates, fail on unknowns."""
    runtime = instance._require_runtime()
    for node in nodes(instance):
        if (
            exact(node.tree_evaluation.tree_value)
            and not node.tree_node.all_branches_generated
        ):
            raise ValueError(_PARTIAL_PROOF)
    runtime.set_evaluator(evaluator)
    count, batches, largest = 0, 0, 0
    eligible = (
        n
        for n in nodes(instance)
        if not n.state.is_game_over() and not exact(n.tree_evaluation.direct_value)
    )
    for batch in batched(eligible, batch_size, strict=False):
        outcome = runtime.tree_manager.node_evaluator.evaluate_nodes(
            batch,
            evaluation_queries=EvaluationQueries(),
            clear_existing_direct_values=True,
            skip_terminal_nodes=True,
        )
        count += outcome.reevaluated_count
        batches += 1
        largest = max(largest, len(batch))
    return dict(evaluations=count, windows=batches, max_window=largest)


def backup_phase(instance: Any) -> int:
    """One unconditional deepest-first pass; never incremental dirty inference."""
    count = 0
    for node in nodes(instance, reverse=True):
        branches = {b for b, child in node.iter_child_links() if child is not None}
        if branches:
            node.tree_evaluation.backup_from_children(branches, branches)
            count += 1
    return count


def selector_phase(instance: Any) -> None:
    """Rebuild derived caches while retaining RNG and selection step count."""
    runtime = instance._require_runtime()
    runtime.tree_manager.refresh_exploration_indices(tree=runtime.tree)
    runtime._invalidate_selector_cache_if_supported()
    selector = runtime.node_selector
    selector.refresh_state_for_checkpoint(
        tree=runtime.tree,
        objective=runtime.tree.root_node.tree_evaluation.required_objective,
        latest_tree_expansions=runtime.latest_tree_expansions,
    )


def bounded_probe(instance: Any, evaluator: Any, batch_size: int = 64) -> dict:
    """Run the research kernel; never publish or activate a model."""
    result = direct_phase(instance, evaluator, batch_size)
    result["backed_up_nodes"] = backup_phase(instance)
    selector_phase(instance)
    return result
