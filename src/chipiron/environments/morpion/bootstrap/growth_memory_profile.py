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

_PROJECT_TYPE_FILTERS = (
    "anemone.",
    "chipiron.",
    "valanga.",
    "atomheart.",
    "coral.",
)

_NODE_SAMPLE_FIELDS = (
    "metadata",
    "state_ref_payload",
    "state_handle",
    "children",
    "parents",
    "successors",
    "branches",
    "evaluation",
    "node_evaluation",
    "max_evaluation",
    "selector_state",
    "linoo_state",
    "pv_state",
    "decision_ordering_state",
    "branch_frontier_state",
)


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


def _top_gc_type_counts(
    *,
    top_n: int,
    project_only: bool = False,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for obj in gc.get_objects():
        type_name = _qualified_type_name(obj)
        if project_only and not any(
            token in type_name.lower() for token in _PROJECT_TYPE_FILTERS
        ):
            continue
        counts[type_name] += 1
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
    for method_name in (
        "profile_iter_nodes",
        "iter_profile_nodes",
        "_profile_iter_nodes",
        "iter_nodes",
        "live_nodes",
        "nodes",
    ):
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
        ("_runtime", "nodes"),
        ("_runtime", "tree", "nodes"),
        ("_runtime", "search_tree", "nodes"),
        ("_runtime", "graph", "nodes"),
        ("_runtime", "node_store"),
        ("_runtime", "node_store", "nodes"),
        ("_runtime", "_nodes"),
        ("_runtime", "_tree", "nodes"),
        ("_runtime", "_search_tree", "nodes"),
        ("_runtime", "_node_store"),
        ("_runtime", "_node_store", "nodes"),
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


def _branch_iterator_from_runner(runner: object) -> tuple[Iterator[object] | None, str]:
    for method_name in (
        "profile_iter_branches",
        "iter_profile_branches",
        "_profile_iter_branches",
        "iter_branches",
        "live_branches",
        "branches",
    ):
        method = getattr(runner, method_name, None)
        if not callable(method):
            continue
        try:
            iterator = _iter_from_candidate(method())
        except Exception:
            continue
        if iterator is not None:
            return iterator, method_name
    return None, "no_branch_iterator"


def _shallow_size(value: object | None) -> int:
    if value is None:
        return 0
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def _safe_getattr(
    value: object,
    attr_name: str,
    default: object | None = None,
) -> object | None:
    try:
        return getattr(value, attr_name)
    except Exception:
        return default


def _safe_vars(value: object) -> Mapping[object, object] | None:
    raw_dict = _safe_getattr(value, "__dict__")
    if isinstance(raw_dict, Mapping):
        return raw_dict
    return None


def _get_field(node: object, field_name: str) -> object | None:
    if isinstance(node, Mapping):
        return node.get(field_name)
    return _safe_getattr(node, field_name)


def _truthy_fraction(count: int, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    return count / sample_size


def _avg(total: int | float, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    return total / sample_size


def _sample_nodes_from_runner(
    runner: object,
    *,
    sample_nodes: int,
) -> tuple[list[object] | None, str]:
    if sample_nodes <= 0:
        return [], "disabled"
    iterator, source = _node_iterator_from_runner(runner)
    if iterator is None:
        return None, source
    try:
        return list(islice(iterator, sample_nodes)), source
    except Exception:
        return None, source


def _top_node_attr_names(sample: list[object]) -> list[str]:
    attr_name_counts: Counter[str] = Counter()
    for node in sample:
        node_dict = _safe_vars(node)
        if node_dict is None:
            continue
        attr_name_counts.update(
            str(attr_name) for attr_name in node_dict.keys() if attr_name is not None
        )
    return [attr_name for attr_name, _count in attr_name_counts.most_common(10)]


def _node_sample_summary_from_sample(
    sample: list[object],
    *,
    source: str,
) -> dict[str, object]:
    sample_size = len(sample)
    totals = {
        "node": 0,
        "node_dict": 0,
        "dict_len": 0,
    }
    state_payload_count = 0
    terminal_count = 0
    exact_count = 0
    attr_name_counts: Counter[str] = Counter()
    field_totals: dict[str, int] = {field_name: 0 for field_name in _NODE_SAMPLE_FIELDS}

    for node in sample:
        totals["node"] += _shallow_size(node)
        node_dict = _safe_vars(node)
        totals["node_dict"] += _shallow_size(node_dict)
        node_dict_len = _safe_len(node_dict)
        if node_dict_len is not None:
            totals["dict_len"] += node_dict_len
        if node_dict is not None:
            attr_name_counts.update(
                str(attr_name) for attr_name in node_dict.keys() if attr_name is not None
            )

        state_payload = None
        for field_name in _NODE_SAMPLE_FIELDS:
            value = _get_field(node, field_name)
            field_totals[field_name] += _shallow_size(value)
            if field_name == "state_ref_payload":
                state_payload = value
        if state_payload is not None:
            state_payload_count += 1

        if bool(_get_field(node, "terminal")) or bool(_get_field(node, "is_terminal")):
            terminal_count += 1
        if bool(_get_field(node, "exact")) or bool(_get_field(node, "is_exact")):
            exact_count += 1

    summary: dict[str, object] = {
        "source": source,
        "sample_size": sample_size,
        "avg_node_shallow_bytes": format_metric(_avg(totals["node"], sample_size)),
        "avg_node_dict_shallow_bytes": format_metric(
            _avg(totals["node_dict"], sample_size)
        ),
        "avg_dict_len": format_metric(_avg(totals["dict_len"], sample_size)),
        "state_payload_fraction": format_metric(
            _truthy_fraction(state_payload_count, sample_size)
        ),
        "terminal_fraction": format_metric(_truthy_fraction(terminal_count, sample_size)),
        "exact_fraction": format_metric(_truthy_fraction(exact_count, sample_size)),
        "top_node_attrs": _format_pairs(attr_name_counts.most_common(10)),
    }
    for field_name in _NODE_SAMPLE_FIELDS:
        avg_key = f"avg_{field_name}_shallow_bytes"
        summary[avg_key] = format_metric(_avg(field_totals[field_name], sample_size))
    summary["avg_state_payload_shallow_bytes"] = summary[
        "avg_state_ref_payload_shallow_bytes"
    ]
    return summary


def _node_attr_sample_summaries(
    sample: list[object],
    *,
    top_attr_names: list[str],
) -> list[dict[str, object]]:
    sample_size = len(sample)
    summaries: list[dict[str, object]] = []
    for attr_name in top_attr_names[:10]:
        try:
            shallow_total = 0
            attr_dict_total = 0
            attr_dict_len_total = 0
            attr_dict_seen = 0
            type_counts: Counter[str] = Counter()
            child_attr_counts: Counter[str] = Counter()

            for node in sample:
                node_dict = _safe_vars(node)
                attr_value = None if node_dict is None else node_dict.get(attr_name)
                shallow_total += _shallow_size(attr_value)
                if attr_value is None:
                    continue

                type_counts[_qualified_type_name(attr_value)] += 1
                attr_dict = _safe_vars(attr_value)
                if attr_dict is None:
                    continue
                attr_dict_total += _shallow_size(attr_dict)
                attr_dict_len = _safe_len(attr_dict)
                if attr_dict_len is not None:
                    attr_dict_len_total += attr_dict_len
                    attr_dict_seen += 1
                child_attr_counts.update(
                    str(child_attr_name)
                    for child_attr_name in attr_dict.keys()
                    if child_attr_name is not None
                )

            summaries.append(
                {
                    "attr": attr_name,
                    "sample_size": sample_size,
                    "top_types": _format_pairs(type_counts.most_common(5)),
                    "avg_shallow_bytes": format_metric(
                        _avg(shallow_total, sample_size)
                    ),
                    "avg_dict_shallow_bytes": format_metric(
                        _avg(attr_dict_total, attr_dict_seen)
                        if attr_dict_seen > 0
                        else None
                    ),
                    "avg_dict_len": format_metric(
                        _avg(attr_dict_len_total, attr_dict_seen)
                        if attr_dict_seen > 0
                        else None
                    ),
                    "top_child_attrs": _format_pairs(child_attr_counts.most_common(10)),
                }
            )
        except Exception as exc:
            summaries.append(
                {
                    "attr": attr_name,
                    "sample_size": sample_size,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return summaries


def _node_sample_summary(
    runner: object,
    *,
    sample_nodes: int,
) -> tuple[dict[str, object] | None, str]:
    if sample_nodes <= 0:
        return {
            "source": "disabled",
            "sample_size": 0,
            "avg_node_shallow_bytes": None,
            "avg_node_dict_shallow_bytes": None,
            "avg_dict_len": None,
            "top_node_attrs": [],
        }, "disabled"

    sample, source = _sample_nodes_from_runner(runner, sample_nodes=sample_nodes)
    if sample is None:
        return None, source
    return _node_sample_summary_from_sample(sample, source=source), source


def _branch_sample_summary(
    runner: object,
    *,
    sample_branches: int,
) -> tuple[dict[str, object] | None, str]:
    if sample_branches <= 0:
        return None, "disabled"
    iterator, source = _branch_iterator_from_runner(runner)
    if iterator is None:
        return None, source
    sample = list(islice(iterator, sample_branches))
    sample_size = len(sample)
    if sample_size == 0:
        return None, source
    branch_total = 0
    branch_dict_total = 0
    for branch in sample:
        branch_total += _shallow_size(branch)
        branch_dict_total += _shallow_size(getattr(branch, "__dict__", None))
    return {
        "source": source,
        "sample_size": sample_size,
        "avg_branch_key_shallow_bytes": format_metric(
            _avg(branch_total, sample_size)
        ),
        "avg_branch_dict_shallow_bytes": format_metric(
            _avg(branch_dict_total, sample_size)
        ),
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
    LOGGER.info(
        "[growth-profile] event=%s top_project_gc_types=%s",
        event,
        _format_pairs(_top_gc_type_counts(top_n=top_n, project_only=True)),
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

    try:
        sample, source = _sample_nodes_from_runner(runner, sample_nodes=sample_nodes)
        if sample is not None:
            for attr_summary in _node_attr_sample_summaries(
                sample,
                top_attr_names=_top_node_attr_names(sample),
            ):
                attr_text = " ".join(
                    f"{name}={value}" for name, value in attr_summary.items()
                )
                LOGGER.info(
                    "[growth-profile] event=%s node_attr_sample source=%s %s",
                    event,
                    source,
                    attr_text,
                )
    except Exception as exc:
        LOGGER.info(
            "[growth-profile] event=%s node_attr_sample unavailable reason=%s: %s",
            event,
            type(exc).__name__,
            exc,
        )

    try:
        branch_summary, _reason = _branch_sample_summary(
            runner,
            sample_branches=sample_nodes,
        )
        if branch_summary is not None:
            branch_text = " ".join(
                f"{name}={value}" for name, value in branch_summary.items()
            )
            LOGGER.info("[growth-profile] event=%s branch_sample %s", event, branch_text)
    except Exception as exc:
        LOGGER.info(
            "[growth-profile] event=%s branch_sample unavailable reason=%s: %s",
            event,
            type(exc).__name__,
            exc,
        )


__all__ = ["log_growth_runtime_memory_profile"]
