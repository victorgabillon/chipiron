"""Bounded Dataset passes over PR #75's indexed generation view.

The existing source index is the only reader. Disposable TEMP tables on its
connection hold scalar family context and row deltas; no states are copied into
these tables. Python retains one source node, one family summary and one row.
"""
# ruff: noqa: TRY003

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.training_export.model import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
)

from chipiron.environments.morpion.learning.target_confidence import (
    DatasetConfidenceContext,
    DatasetConfidenceError,
    finalize_confidence_metadata,
    validate_confidence_policy,
)
from chipiron.environments.morpion.learning.tree_to_dataset import (
    iter_morpion_supervised_rows_from_training_snapshot,
    morpion_supervised_rows_metadata_from_training_snapshot,
)

from .cycle_dataset import MorpionSupervisedRowsStream
from .dataset_family_targets import (
    _dataset_row_with_family_target_metadata,
    _node_raw_target,
    _node_raw_target_is_certified,
)
from .pv_family_targets import PvFamilyTargetPolicy, PvFamilyTargets, family_node_target

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator, Iterator

    from anemone.training_export import TrainingNodeSnapshot

    from chipiron.environments.morpion.learning import MorpionSupervisedRow

    from .bootstrap_args import MorpionBootstrapArgs
    from .sharded_training_reader import ShardedTrainingGenerationView


class UnsupportedStreamingDatasetError(ValueError):
    """Signal a source requiring the legacy policy implementation."""


@dataclass(frozen=True)
class IndexedDatasetNodes:
    """Expose repeatable export order and count using the existing view only."""

    view: ShardedTrainingGenerationView

    def __iter__(self) -> Iterator[TrainingNodeSnapshot]:
        """Reconstruct one node at a time."""
        return self.view.iter_nodes_in_export_order()

    def __len__(self) -> int:
        """Return the indexed count without reconstruction."""
        return self.view.node_count()


@dataclass(frozen=True)
class IndexedDatasetSource:
    """Small extraction source; never constructs a TrainingTreeSnapshot."""

    view: ShardedTrainingGenerationView

    @property
    def root_node_id(self) -> str:
        """Return the generation root."""
        return self.view.root_node_id

    @property
    def nodes(self) -> IndexedDatasetNodes:
        """Return a repeatable bounded traversal."""
        return IndexedDatasetNodes(self.view)

    @property
    def metadata(self) -> dict[str, Any]:
        """Match the legacy sharded loader's extraction format identity."""
        if self.view.format_version == 2:
            return dict(self.view.snapshot_metadata)
        return {
            "format_kind": TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
            "format_version": TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
        }


class DiskDatasetFamilyContext:
    """Compute exact family policies using disk-backed scalar keyed data."""

    def __init__(
        self,
        source: IndexedDatasetSource,
        connection: sqlite3.Connection,
        args: MorpionBootstrapArgs,
    ) -> None:
        """Populate TEMP tables; errors are cleaned up by the owning context."""
        if not 0.0 <= args.dataset_family_prediction_blend <= 1.0:
            raise ValueError("dataset_family_prediction_blend must be between 0 and 1.")
        if args.dataset_family_target_policy not in get_args(PvFamilyTargetPolicy):
            raise UnsupportedStreamingDatasetError("unsupported_family_target_policy")
        self.source = source
        self.connection = connection
        self.args = args
        validate_confidence_policy(
            args.dataset_confidence_policy, args.dataset_family_target_policy
        )
        self.confidence = None
        if args.dataset_confidence_policy == "selected_value_v2":
            if source.view.format_version != 2:
                raise DatasetConfidenceError("Dataset v2 requires scientific v2 input")
            self.confidence = DatasetConfidenceContext(
                lambda node_id: source.view.load_value_contracts((node_id,))[0],
                source.view.provenance,
            )
        self._active_scientific_rows: Generator[MorpionSupervisedRow] | None = None
        connection.executescript("""
            CREATE TEMP TABLE dataset_nodes (
                id TEXT COLLATE BINARY PRIMARY KEY,
                raw TEXT NOT NULL, prediction TEXT NOT NULL, exact INTEGER NOT NULL,
                selected TEXT, representative TEXT, visiting INTEGER DEFAULT 0,
                previous TEXT
            );
            CREATE TEMP TABLE dataset_families (
                representative TEXT COLLATE BINARY PRIMARY KEY, first_id TEXT COLLATE BINARY NOT NULL, payload TEXT NOT NULL
            );
            CREATE TEMP TABLE dataset_deltas (
                sequence INTEGER PRIMARY KEY, exact INTEGER NOT NULL, delta TEXT NOT NULL
            );
        """)
        for index, node in enumerate(source.nodes):
            raw = _node_raw_target(node, use_backed_up_value=args.use_backed_up_value)
            prediction = node.direct_value_scalar
            if prediction is None:
                prediction = raw
            if (
                source.view.format_version != 2 and (raw is None or prediction is None)
            ) or not all(math.isfinite(v) for v in (raw, prediction) if v is not None):
                raise UnsupportedStreamingDatasetError(
                    "missing_or_nonfinite_family_scalar"
                )
            # JSON scalars preserve signed zero; SQLite REAL storage does not.
            connection.execute(
                "INSERT INTO dataset_nodes(id, raw, prediction, exact) VALUES (?, ?, ?, ?)",
                (
                    node.node_id,
                    json.dumps(raw),
                    json.dumps(prediction),
                    int(node.is_exact or node.is_terminal)
                    if self.confidence is None
                    else int(
                        _node_raw_target_is_certified(
                            node,
                            use_backed_up_value=args.use_backed_up_value,
                            confidence=self.confidence,
                        )
                    ),
                ),
            )
            if index % 256 == 255:
                connection.commit()
        connection.commit()
        for index, node in enumerate(source.nodes):
            selected: str | None = None
            best: float | None = None
            for child_id in sorted(node.child_ids, key=str):
                row = connection.execute(
                    "SELECT raw FROM dataset_nodes WHERE id=?", (child_id,)
                ).fetchone()
                if row is None:
                    continue
                decoded = json.loads(row[0])
                if decoded is None:
                    continue
                value = float(decoded)
                if best is None or value > best:
                    best, selected = value, child_id
            connection.execute(
                "UPDATE dataset_nodes SET selected=? WHERE id=?",
                (selected, node.node_id),
            )
            if index % 256 == 255:
                connection.commit()
        connection.commit()
        self._resolve_representatives()
        connection.execute(
            "CREATE INDEX temp.dataset_family_order ON dataset_nodes(representative, id)"
        )
        for (representative,) in connection.execute(
            "SELECT DISTINCT representative FROM dataset_nodes ORDER BY representative"
        ):
            self._build_family(str(representative))
        connection.commit()

    def _resolve_representatives(self) -> None:
        """Memoize paths on disk, including arbitrarily long chains and cycle checks."""
        connection = self.connection
        for (node_id,) in connection.execute(
            "SELECT id FROM dataset_nodes ORDER BY id"
        ):
            current = str(node_id)
            previous: str | None = None
            while True:
                selected, representative, visiting = connection.execute(
                    "SELECT selected, representative, visiting FROM dataset_nodes WHERE id=?",
                    (current,),
                ).fetchone()
                if representative is not None:
                    break
                if visiting:
                    raise ValueError(
                        "Principal-variation family traversal found a cycle."
                    )
                connection.execute(
                    "UPDATE dataset_nodes SET visiting=1, previous=? WHERE id=?",
                    (previous, current),
                )
                previous = current
                if selected is None:
                    representative = current
                    break
                current = str(selected)
            while previous is not None:
                node = previous
                previous = connection.execute(
                    "SELECT previous FROM dataset_nodes WHERE id=?", (node,)
                ).fetchone()[0]
                connection.execute(
                    "UPDATE dataset_nodes SET representative=?, visiting=0, previous=NULL WHERE id=?",
                    (representative, node),
                )
            connection.commit()

    def _values(
        self, representative: str, column: str, *, exact: bool = False
    ) -> Iterator[float]:
        """Yield float scalars in the oracle's sorted node-ID order."""
        if column not in {"raw", "prediction"}:
            raise ValueError(column)
        query = f"SELECT {column} FROM dataset_nodes WHERE representative=?"
        if exact:
            query += " AND exact=1"
        for (value,) in self.connection.execute(
            query + " ORDER BY id", (representative,)
        ):
            decoded = json.loads(value)
            if decoded is not None:
                yield float(decoded)

    def _build_family(self, representative: str) -> None:
        """Use Python sum/min/max in legacy order, including compensated sum."""
        size, exact_count, first_id = self.connection.execute(
            "SELECT COUNT(*), SUM(exact), MIN(id) FROM dataset_nodes WHERE representative=?",
            (representative,),
        ).fetchone()
        missing_exact = self.connection.execute(
            "SELECT id FROM dataset_nodes WHERE representative=? AND exact=1 AND raw='null' ORDER BY id LIMIT 1",
            (representative,),
        ).fetchone()
        if missing_exact is not None:
            # The reference accesses raw_targets[id] for every exact family
            # member. Missing certainty/values are never silently synthesized.
            raise KeyError(missing_exact[0])
        prediction_count = self.connection.execute(
            "SELECT COUNT(*) FROM dataset_nodes WHERE representative=? AND prediction!='null'",
            (representative,),
        ).fetchone()[0]
        exact_target = max(
            self._values(representative, "raw", exact=True), default=None
        )
        exact_min = min(self._values(representative, "raw", exact=True), default=None)
        payload: dict[str, Any] = {
            "size": size,
            "exact_count": exact_count,
            "mean": sum(self._values(representative, "prediction")) / prediction_count
            if prediction_count
            else 0.0,
            "min": min(self._values(representative, "prediction"), default=0.0),
            "exact_target": exact_target,
            "exact_rule": "pv_exact_family_multi_max"
            if exact_min != exact_target
            else "pv_exact_family",
        }
        for node_id, exact in self.connection.execute(
            "SELECT id,exact FROM dataset_nodes WHERE representative=? AND raw='null' ORDER BY id",
            (representative,),
        ):
            # Some policies legitimately need no raw scalar; others raise the
            # same KeyError as the full reference, even for filtered nodes.
            family_node_target(
                raw_targets={},
                node_id=node_id,
                is_exact=bool(exact),
                family_mean=payload["mean"],
                family_min=payload["min"],
                exact_family_target=exact_target,
                exact_family_rule=payload["exact_rule"],
                family_target_policy=self.args.dataset_family_target_policy,
                family_prediction_blend=self.args.dataset_family_prediction_blend,
            )
        self.connection.execute(
            "INSERT INTO dataset_families VALUES (?, ?, ?)",
            (representative, first_id, json.dumps(payload)),
        )

    def adjust_row(self, row: MorpionSupervisedRow) -> MorpionSupervisedRow:
        """Reuse the existing row assembly and policy arithmetic with singleton maps."""
        raw_json, selected, representative, family_json = self.connection.execute(
            "SELECT n.raw, n.selected, n.representative, f.payload FROM dataset_nodes n JOIN dataset_families f ON f.representative=n.representative WHERE n.id=?",
            (row.node_id,),
        ).fetchone()
        raw = float(json.loads(raw_json))
        family = json.loads(family_json)
        effective, rule = family_node_target(
            raw_targets={row.node_id: raw},
            node_id=row.node_id,
            is_exact=(row.is_exact or row.is_terminal)
            if self.confidence is None
            else row.is_exact,
            family_mean=family["mean"],
            family_min=family["min"],
            exact_family_target=family["exact_target"],
            exact_family_rule=family["exact_rule"],
            family_target_policy=self.args.dataset_family_target_policy,
            family_prediction_blend=self.args.dataset_family_prediction_blend,
        )
        targets = PvFamilyTargets(
            effective_targets={row.node_id: effective},
            representative_by_node={row.node_id: representative},
            family_size_by_node={row.node_id: family["size"]},
            family_has_exact_by_node={row.node_id: bool(family["exact_count"])},
            family_exact_target_by_node={row.node_id: family["exact_target"]},
            family_target_rule_by_node={row.node_id: rule},
            family_num_exact_by_node={row.node_id: family["exact_count"]},
            num_families=0,
            mean_family_size=None,
            max_family_size=None,
        )
        return _dataset_row_with_family_target_metadata(
            row,
            raw_target=raw,
            selected_child_id=selected,
            family_targets=targets,
            scientific_confidence=self.confidence is not None,
        )

    def close_rows(self) -> None:
        """Release a partial v2 pass before dropping its TEMP tables."""
        if self._active_scientific_rows is not None:
            self._active_scientific_rows.close()
            self._active_scientific_rows = None

    def rows(self) -> Iterator[MorpionSupervisedRow]:
        """Start one sequential scientific pass; keep v1 iterator behavior intact."""
        if self.source.view.format_version == 2:
            self.close_rows()
            self._active_scientific_rows = self._iter_rows()
            return self._active_scientific_rows
        return self._iter_rows()

    def _iter_rows(self) -> Generator[MorpionSupervisedRow]:
        """Repeat exactly the oracle's filter, validation and max_rows order."""
        args = self.args
        for row in iter_morpion_supervised_rows_from_training_snapshot(
            self.source,
            require_exact_or_terminal=args.require_exact_or_terminal,
            min_depth=args.min_depth,
            min_visit_count=args.min_visit_count,
            max_rows=args.max_rows,
            use_backed_up_value=args.use_backed_up_value,
            confidence=self.confidence,
            # Only v2's opt-in index proves canonicality for every immutable
            # record and verifies the source plus cached payload digest. Keep
            # the identical JSON normalization without replaying each state on
            # both summary and output passes. The end guard still gates rename.
            _validated_state_ref_normalizer=checkpoint_payload_to_jsonable
            if self.source.view.format_version == 2
            else None,
        ):
            yield self.adjust_row(row)
        # The writer has not renamed its temporary output yet at this boundary.
        self.source.view.validate_source_identity()

    def summary(self) -> dict[str, object]:
        """Stage deltas on disk, then sum in oracle order with bounded Python RAM."""
        connection = self.connection
        connection.execute("DELETE FROM dataset_deltas")
        rows_in_exact = 0
        row_count = 0
        for row_count, row in enumerate(self.rows(), 1):
            exact = bool(row.metadata["family_has_exact_or_terminal"])
            rows_in_exact += int(exact)
            delta = abs(row.metadata["effective_target"] - row.metadata["raw_target"])
            connection.execute(
                "INSERT INTO dataset_deltas VALUES (?, ?, ?)",
                (row_count, int(exact), json.dumps(delta)),
            )
            if row_count % 256 == 0:
                connection.commit()
        connection.commit()

        def deltas(exact: int | None = None) -> Iterator[float]:
            query = "SELECT delta FROM dataset_deltas"
            parameters = () if exact is None else (exact,)
            if exact is not None:
                query += " WHERE exact=?"
            for (value,) in connection.execute(
                query + " ORDER BY sequence", parameters
            ):
                yield float(json.loads(value))

        def mean(exact: int | None = None) -> float | None:
            count = (
                row_count
                if exact is None
                else (rows_in_exact if exact else row_count - rows_in_exact)
            )
            return sum(deltas(exact)) / count if count else None

        num_families = connection.execute(
            "SELECT COUNT(*) FROM dataset_families"
        ).fetchone()[0]
        exact_families = 0
        max_size: int | None = None
        for (payload,) in connection.execute("SELECT payload FROM dataset_families"):
            family = json.loads(payload)
            exact_families += int(bool(family["exact_count"]))
            max_size = max(max_size or 0, family["size"])
        # Match the oracle's family insertion order (smallest member ID).
        size_sum = sum(
            float(json.loads(payload)["size"])
            for (payload,) in connection.execute(
                "SELECT payload FROM dataset_families ORDER BY first_id"
            )
        )
        return {
            "dataset_family_target_policy": self.args.dataset_family_target_policy,
            "dataset_family_prediction_blend": self.args.dataset_family_prediction_blend,
            "fraction_rows_in_exact_family": rows_in_exact / row_count
            if row_count
            else None,
            "num_exact_families": exact_families,
            "mean_abs_effective_minus_raw_on_exact_families": mean(1),
            "mean_abs_effective_minus_raw_on_non_exact_families": mean(0),
            "effective_minus_raw_mean_abs": mean(),
            "effective_minus_raw_max_abs": max(deltas(), default=None),
            "num_pv_families": num_families,
            "mean_pv_family_size": size_sum / num_families if num_families else None,
            "max_pv_family_size": max_size,
        }

    def scratch_bytes(self) -> int:
        """Return TEMP database logical bytes before cleanup, not a disk high water mark."""
        pages = self.connection.execute("PRAGMA temp.page_count").fetchone()[0]
        page_size = self.connection.execute("PRAGMA temp.page_size").fetchone()[0]
        return int(pages * page_size)

    def stream(self, generation: int) -> MorpionSupervisedRowsStream:
        """Compute the complete header before the repeatable write pass."""
        args = self.args
        if self.confidence is not None:
            self.confidence.require_generation(generation)
        metadata = morpion_supervised_rows_metadata_from_training_snapshot(
            self.source,
            require_exact_or_terminal=args.require_exact_or_terminal,
            min_depth=args.min_depth,
            min_visit_count=args.min_visit_count,
            max_rows=args.max_rows,
            use_backed_up_value=args.use_backed_up_value,
            metadata={"bootstrap_generation": generation},
            confidence=self.confidence,
        )
        metadata.update(self.summary())
        finalize_confidence_metadata(metadata)
        return MorpionSupervisedRowsStream(rows=self.rows(), metadata=metadata)


@contextmanager
def disk_dataset_context(
    source: IndexedDatasetSource,
    args: MorpionBootstrapArgs,
) -> Iterator[DiskDatasetFamilyContext]:
    """Scope all TEMP storage to one Dataset attempt, including failed attempts."""
    with source.view.temporary_workspace() as connection:
        context = None
        try:
            context = DiskDatasetFamilyContext(source, connection, args)
            yield context
        finally:
            if context is not None:
                context.close_rows()
            for table in ("dataset_deltas", "dataset_families", "dataset_nodes"):
                connection.execute(f"DROP TABLE IF EXISTS temp.{table}")
            connection.commit()
