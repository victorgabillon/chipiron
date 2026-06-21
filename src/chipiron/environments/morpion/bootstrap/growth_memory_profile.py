"""Opt-in shallow memory attribution for Morpion growth runtimes."""

from __future__ import annotations

import gc
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sized
from itertools import islice

from .pipeline_memory import current_rss_mb, format_metric

LOGGER = logging.getLogger(__name__)


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    module = value_type.__module__
    qualname = value_type.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def _format_pairs(pairs: Iterable[tuple[str, object]]) -> str:
    return "[" + ", ".join(f"({name!r}, {value!r})" for name, value in pairs) + "]"


def _safe_len(value: object) -> int | None:
    try:
        return len(value) if isinstance(value, Sized) else None
    except (TypeError, RuntimeError):
        return None


def _runner_attribute_sizes(runner: object, *, top_n: int) -> list[tuple[str, int]]:
    try:
        runner_vars = vars(runner)
    except TypeError:
        return []
    pairs: list[tuple[str, int]] = []
    for name, value in runner_vars.items():
        try:
            shallow_size = sys.getsizeof(value)
        except TypeError:
            continue
        pairs.append((name, shallow_size))
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]


def _top_gc_type_counts(*, top_n: int) -> list[tuple[str, int]]:
    counts = Counter(_qualified_type_name(obj) for obj in gc.get_objects())
    return counts.most_common(top_n)


def _iter_from_candidate(candidate: object) -> Iterator[object] | None:
    if isinstance(candidate, Mapping):
        return iter(candidate.values())
    if isinstance(candidate, (str, bytes, bytearray)):
        return None
    try:
        return iter(candidate) if isinstance(candidate, Iterable) else None
    except TypeError:
        return None


def _node_iterator_from_runner(runner: object) -> tuple[Iterator[object] | None, str]:
    for method_name in ("iter_nodes", "live_nodes", "nodes"):
        method = getattr(runner, method_name, None)
        if not callable(method):
            continue
        try:
            iterator = _iter_from_candidate(method())
        except Exception:
            continue
        if iterator is not None:
            return iterator, method_name

    for attr_path in (
        ("node_store",),
        ("nodes",),
        ("tree", "nodes"),
        ("search_tree", "nodes"),
        ("runtime", "nodes"),
        ("runtime", "tree", "nodes"),
        ("runtime", "search_tree", "nodes"),
    ):
        value: object = runner
        found = True
        for attr_name in attr_path:
            if not hasattr(value, attr_name):
                found = False
                break
            value = getattr(value, attr_name)
        if not found:
            continue
        iterator = _iter_from_candidate(value)
        if iterator is not None:
            return iterator, ".".join(attr_path)
    return None, "no_node_iterator"


def _shallow_size(value: object | None) -> int:
    if value is None:
        return 0
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def _get_field(node: object, field_name: str) -> object | None:
    if isinstance(node, Mapping):
        return node.get(field_name)
    return getattr(node, field_name, None)


def _truthy_fraction(count: int, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    return count / sample_size


def _avg(total: int | float, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    return total / sample_size


def _node_sample_summary(
    runner: object,
    *,
    sample_nodes: int,
) -> tuple[dict[str, object] | None, str]:
    if sample_nodes <= 0:
        return {
            "sample_size": 0,
            "avg_node_shallow_bytes": None,
            "avg_node_dict_shallow_bytes": None,
            "avg_metadata_shallow_bytes": None,
            "avg_state_payload_shallow_bytes": None,
            "avg_children_shallow_bytes": None,
            "avg_children_len": None,
            "state_payload_fraction": None,
        }, "disabled"

    iterator, source = _node_iterator_from_runner(runner)
    if iterator is None:
        return None, source
    sample = list(islice(iterator, sample_nodes))
    sample_size = len(sample)
    totals = {
        "node": 0,
        "node_dict": 0,
        "metadata": 0,
        "state_payload": 0,
        "children": 0,
        "children_len": 0,
        "parents": 0,
        "legal_actions": 0,
        "value": 0,
    }
    state_payload_count = 0
    terminal_count = 0
    exact_count = 0
    metadata_len_total = 0
    metadata_len_seen = 0

    for node in sample:
        totals["node"] += _shallow_size(node)
        node_dict = getattr(node, "__dict__", None)
        totals["node_dict"] += _shallow_size(node_dict)

        metadata = _get_field(node, "metadata")
        totals["metadata"] += _shallow_size(metadata)
        metadata_len = _safe_len(metadata)
        if metadata_len is not None:
            metadata_len_seen += 1
            metadata_len_total += metadata_len

        state_payload = _get_field(node, "state_ref_payload")
        totals["state_payload"] += _shallow_size(state_payload)
        if state_payload is not None:
            state_payload_count += 1

        children = _get_field(node, "children")
        totals["children"] += _shallow_size(children)
        children_len = _safe_len(children)
        if children_len is not None:
            totals["children_len"] += children_len

        totals["parents"] += _shallow_size(_get_field(node, "parents"))
        totals["legal_actions"] += _shallow_size(_get_field(node, "legal_actions"))
        value = _get_field(node, "value")
        if value is None:
            value = _get_field(node, "evaluation")
        totals["value"] += _shallow_size(value)

        if bool(_get_field(node, "terminal")) or bool(_get_field(node, "is_terminal")):
            terminal_count += 1
        if bool(_get_field(node, "exact")) or bool(_get_field(node, "is_exact")):
            exact_count += 1

    return {
        "source": source,
        "sample_size": sample_size,
        "avg_node_shallow_bytes": format_metric(_avg(totals["node"], sample_size)),
        "avg_node_dict_shallow_bytes": format_metric(
            _avg(totals["node_dict"], sample_size)
        ),
        "avg_metadata_shallow_bytes": format_metric(
            _avg(totals["metadata"], sample_size)
        ),
        "avg_metadata_len": format_metric(
            _avg(metadata_len_total, metadata_len_seen)
            if metadata_len_seen > 0
            else None
        ),
        "avg_state_payload_shallow_bytes": format_metric(
            _avg(totals["state_payload"], sample_size)
        ),
        "avg_children_shallow_bytes": format_metric(
            _avg(totals["children"], sample_size)
        ),
        "avg_children_len": format_metric(_avg(totals["children_len"], sample_size)),
        "avg_parents_shallow_bytes": format_metric(_avg(totals["parents"], sample_size)),
        "avg_legal_actions_shallow_bytes": format_metric(
            _avg(totals["legal_actions"], sample_size)
        ),
        "avg_value_shallow_bytes": format_metric(_avg(totals["value"], sample_size)),
        "state_payload_fraction": format_metric(
            _truthy_fraction(state_payload_count, sample_size)
        ),
        "terminal_fraction": format_metric(_truthy_fraction(terminal_count, sample_size)),
        "exact_fraction": format_metric(_truthy_fraction(exact_count, sample_size)),
    }, source


def _density_metrics(
    *,
    node_count: int | None,
    branch_count: int | None,
) -> dict[str, object]:
    rss_mb = current_rss_mb()
    rss_per_100k_nodes = None
    if rss_mb is not None and node_count is not None and node_count > 0:
        rss_per_100k_nodes = rss_mb / node_count * 100_000
    branch_count_per_node = None
    if branch_count is not None and node_count is not None and node_count > 0:
        branch_count_per_node = branch_count / node_count
    return {
        "rss_mb": format_metric(rss_mb),
        "node_count": node_count,
        "branch_count": branch_count,
        "rss_mb_per_100k_nodes": format_metric(rss_per_100k_nodes),
        "branch_count_per_node": format_metric(branch_count_per_node),
    }


def log_growth_runtime_memory_profile(
    *,
    runner: object,
    generation: int,
    event: str,
    node_count: int | None,
    branch_count: int | None,
    sample_nodes: int,
    top_n: int,
) -> None:
    """Log bounded shallow memory attribution for a live growth runtime.

    All object-size fields are shallow ``sys.getsizeof`` measurements. The helper
    avoids recursive object graph traversal and limits node inspection to
    ``sample_nodes`` so profile mode remains usable on large trees.
    """
    density = _density_metrics(node_count=node_count, branch_count=branch_count)
    LOGGER.info(
        "[growth-profile] event=%s generation=%s rss_mb=%s node_count=%s "
        "branch_count=%s rss_mb_per_100k_nodes=%s branch_count_per_node=%s",
        event,
        generation,
        density["rss_mb"],
        density["node_count"],
        density["branch_count"],
        density["rss_mb_per_100k_nodes"],
        density["branch_count_per_node"],
    )

    LOGGER.info(
        "[growth-profile] event=%s top_gc_types=%s",
        event,
        _format_pairs(_top_gc_type_counts(top_n=top_n)),
    )

    runner_attrs = _runner_attribute_sizes(runner, top_n=top_n)
    if runner_attrs:
        LOGGER.info(
            "[growth-profile] event=%s runner_attrs=%s",
            event,
            _format_pairs(runner_attrs),
        )
    else:
        LOGGER.info("[growth-profile] event=%s runner_attrs unavailable", event)

    sample_summary, reason = _node_sample_summary(runner, sample_nodes=sample_nodes)
    if sample_summary is None:
        LOGGER.info(
            "[growth-profile] event=%s node_sample unavailable reason=%s",
            event,
            reason,
        )
        return
    sample_text = " ".join(
        f"{name}={value}" for name, value in sample_summary.items()
    )
    LOGGER.info("[growth-profile] event=%s node_sample %s", event, sample_text)


__all__ = ["log_growth_runtime_memory_profile"]
