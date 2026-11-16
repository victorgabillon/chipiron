"""Current direct origins observed at Growth's existing native write boundary.

This is volatile transaction state, never an acceptance ledger. Only the existing
capsule and run-state commit authenticate its sealed form. No per-node history.
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from valanga.evaluations import Certainty

from .generation_publication_io import artifact_digest, object_digest, require
from .pipeline_artifacts import MorpionReevaluationPatch
from .prediction_lineage import PatchDisposition, PredictionOrigin
from .reevaluation_transaction import GROWTH_NUMERIC_POLICY

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GrowthPredictionOrigins:
    """One private runner's last writes and inherited, authenticated current map."""

    generation: int = 1
    parent_checkpoint: str | None = None
    creation: dict[str, int] = field(default_factory=dict)
    current: dict[str, PredictionOrigin] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    patch_bytes: dict[str, bytes] = field(default_factory=dict)
    writes: dict[str, tuple[str, str]] = field(default_factory=dict)
    expected_values: dict[str, object] = field(default_factory=dict)
    model_sha256: str | None = None
    operation: str = "growth_birth"
    disposition: PatchDisposition | None = None
    patch: MorpionReevaluationPatch | None = None
    revision: int = 0
    saved: tuple[str, int] | None = None
    valid: bool = True
    allowed_evaluator: object | None = None

    def bind_model(self, model: Path) -> None:
        """Bind the actual loaded bytes; a requested evaluator is never an origin."""
        self.model_sha256 = artifact_digest(model)

    def observe(self, node: Any, value: Any) -> None:
        """Record a successful native direct store, including an equal score."""
        require(self.valid, "Prediction-origin transaction was invalidated.")
        key = str(node.id)
        self.revision += 1
        self.expected_values[key] = value
        if value is not None and value.certainty == Certainty.ESTIMATE:
            require(self.model_sha256 is not None, "Neural write has no loaded model.")
            assert self.model_sha256 is not None
            require(
                self.operation == "full_refresh" or key not in self.creation,
                "Uninstrumented rewrite of an existing direct estimate during Growth.",
            )
            self.writes[key] = (self.operation, self.model_sha256)
        else:
            self.current.pop(key, None)
            self.writes.pop(key, None)

    def attach(
        self, runtime: Any, *, fresh: bool = False, rollback: bool = False
    ) -> None:
        """Observe the existing native store; never replace its scoring or propagation.

        Fresh construction evaluates exactly its newly created root before returning.
        That one birth is qualified by the constructor, not an imported local stamp.
        Subsequent writes are observed at the native store itself.
        """
        nodes = runtime._nodes_by_public_id()  # pylint: disable=protected-access
        if fresh:
            require(len(nodes) == 1 and not self.creation, "Not a fresh root birth.")
            root = runtime.tree.root_node
            require(
                root.tree_evaluation.direct_evaluation_version is not None,
                "Fresh native root was not evaluated.",
            )
            self.operation = "genesis"
            self.observe(root, root.tree_evaluation.direct_value)
        elif rollback:
            require(
                set(nodes) == set(self.expected_values)
                and all(
                    node.tree_evaluation.direct_value == self.expected_values[key]
                    for key, node in nodes.items()
                ),
                "Native rollback differs from saved prediction observations.",
            )
            # Deserialization creates new immutable Value objects. This is the
            # existing checkpoint rollback, not a new prediction or origin.
            self.expected_values = {
                key: node.tree_evaluation.direct_value for key, node in nodes.items()
            }
        else:
            require(
                set(nodes) == set(self.creation), "Restored origin inventory differs."
            )
            estimates = {
                key
                for key, node in nodes.items()
                if node.tree_evaluation.direct_value is not None
                and node.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
            }
            require(
                estimates == set(self.current),
                "Restored direct origin coverage differs.",
            )
            self.expected_values = {
                key: node.tree_evaluation.direct_value for key, node in nodes.items()
            }
        evaluator = runtime.tree_manager.node_evaluator
        self.allowed_evaluator = evaluator.master_state_value_evaluator
        original = evaluator._store_direct_value  # pylint: disable=protected-access

        def store(*, node: Any, value: Any) -> None:
            require(
                evaluator.master_state_value_evaluator is self.allowed_evaluator,
                "Evaluator changed outside the owned model binding.",
            )
            try:
                original(node=node, value=value)
                self.observe(node, value)
            except BaseException:
                self.valid = False
                raise

        evaluator._store_direct_value = store  # pylint: disable=protected-access

    def apply_patch(
        self,
        runtime: Any,
        patch: MorpionReevaluationPatch,
        protected: tuple[str, ...],
        result: Any,
    ) -> None:
        """Partition native applied/missing results, preserving certificate semantics."""
        require(
            self.patch is None, "Only one pending patch belongs to a Growth commit."
        )
        require(
            not result.skipped_node_ids, "Unqualified native skipped prediction IDs."
        )
        missing = set(result.missing_node_ids)
        skipped = set(protected)
        nodes = runtime._nodes_by_public_id()  # pylint: disable=protected-access
        applied: list[str] = []
        reused: list[str] = []
        for row in patch.rows:
            if row.node_id in missing or row.node_id in skipped:
                continue
            node = nodes[row.node_id]
            value = node.tree_evaluation.direct_value
            if row.metadata.get("source") == "active_model_reevaluation":
                require(
                    value is not None and value.certainty == Certainty.ESTIMATE,
                    "Native prediction did not produce a direct estimate.",
                )
                applied.append(row.node_id)
                # External model generations are not Anemone's local counter.
                node.tree_evaluation.direct_evaluation_version = None
                # The accepted receipt supplies the patch model at sealing, never
                # the evaluator currently loaded by Growth.
                self.writes[row.node_id] = ("accepted_patch", "")
            else:
                require(
                    value is not None and value.certainty == Certainty.TERMINAL,
                    "Certified reuse did not preserve a terminal value.",
                )
                reused.append(row.node_id)
                self.current.pop(row.node_id, None)
                self.writes.pop(row.node_id, None)
            self.expected_values[row.node_id] = value
            self.revision += 1
        disposition = PatchDisposition(
            tuple(applied), protected, tuple(result.missing_node_ids), tuple(reused)
        )
        disposition.validate(tuple(row.node_id for row in patch.rows))
        require(
            result.applied_count == len(applied) + len(reused),
            "Native applied count differs from observed IDs.",
        )
        self.disposition, self.patch = disposition, patch

    def preflight_patch(
        self, runtime: Any, patch: MorpionReevaluationPatch, blend: float
    ) -> None:
        """Reject unsupported writes before any native mutation."""
        require(
            self.patch is None and self.valid,
            "Invalid or already applied prediction transaction.",
        )
        require(blend == 1.0, "Prediction origins do not qualify blended writes.")
        require(
            len({row.node_id for row in patch.rows}) == len(patch.rows),
            "Duplicate prediction patch IDs.",
        )
        nodes = runtime._nodes_by_public_id()  # pylint: disable=protected-access
        for row in patch.rows:
            if (
                row.metadata.get("source") == "terminal_existing_value"
                and row.node_id in nodes
            ):
                node = nodes[row.node_id]
                value = node.tree_evaluation.direct_value
                require(
                    node.state.is_game_over()
                    and value is not None
                    and value.certainty == Certainty.TERMINAL
                    and value.score == row.direct_value,
                    "Fabricated certified-value reuse.",
                )
            model = row.metadata.get("source") == "active_model_reevaluation"
            terminal = row.metadata.get("source") == "terminal_existing_value"
            require(
                row.backed_up_value is None
                and (
                    (model and row.is_exact is False and row.is_terminal is False)
                    or (terminal and row.is_exact is True and row.is_terminal is True)
                ),
                "Unqualified direct prediction source.",
            )
            if model:
                require(
                    row.metadata.get("numeric_policy") == GROWTH_NUMERIC_POLICY,
                    "Prediction patch requires the existing Growth numeric recipe.",
                )

    def checkpoint(self, runtime: Any, path: Path) -> None:
        """Bind the observation revision to the exact successfully written checkpoint."""
        self.audit(runtime)
        self.saved = (artifact_digest(path), self.revision)

    def audit(self, runtime: Any) -> None:
        """Refuse unobserved mutations rather than authenticating a proposed map."""
        require(self.valid, "Prediction-origin transaction was invalidated.")
        nodes = runtime._nodes_by_public_id()  # pylint: disable=protected-access
        require(
            set(self.expected_values) == set(nodes), "Unobserved node birth or removal."
        )
        for key, node in nodes.items():
            require(
                node.tree_evaluation.direct_value is self.expected_values[key],
                "Direct value changed outside observed Growth writes.",
            )
            if node.tree_evaluation.direct_value is not None and (
                node.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
            ):
                require(
                    key in self.current or key in self.writes,
                    "Direct estimate has no actual write origin.",
                )

    def native_recipe(self) -> str:
        """Hash the existing numerical rule independently of evaluator identity."""
        return object_digest({"numeric_policy": GROWTH_NUMERIC_POLICY})

    def fork(self) -> GrowthPredictionOrigins:
        """Snapshot volatile maps for the existing native node-budget rollback."""
        return replace(
            self,
            creation=dict(self.creation),
            current=dict(self.current),
            evidence=dict(self.evidence),
            patch_bytes=dict(self.patch_bytes),
            writes=dict(self.writes),
            expected_values=dict(self.expected_values),
        )
