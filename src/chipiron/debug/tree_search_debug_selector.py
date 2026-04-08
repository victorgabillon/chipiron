"""Debug wrapper for one-shot tree-search branch selectors."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from anemone._valanga_types import AnyTurnState
from anemone.debug import build_live_debug_environment

if TYPE_CHECKING:
    from anemone.tree_and_value_branch_selector import TreeAndValueBranchSelector
    from valanga.game import Seed
    from valanga.policy import NotifyProgressCallable, Recommendation


def _is_non_string_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Return whether a value is a non-string sequence for JSON serialization."""
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


class DebugTreeSearchSelector[StateT: AnyTurnState]:
    """Record one Anemone live-debug session for each tree-search recommendation."""

    def __init__(
        self,
        *,
        base: TreeAndValueBranchSelector[StateT],
        session_root: Path | str,
        state_to_debug_string: Callable[[StateT], str],
        snapshot_format: str = "svg",
    ) -> None:
        """Store the wrapped selector and the output location for move sessions."""
        self.base = base
        self.session_root = Path(session_root)
        self.state_to_debug_string = state_to_debug_string
        self.snapshot_format = snapshot_format
        self._move_index = 0

        self.session_root.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-debug attributes to the wrapped selector."""
        return getattr(self.base, name)

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Run the real exploration inside Anemone's live debug environment."""
        tree_exploration = self.base.create_tree_exploration(
            state=state,
            notify_progress=notify_progress,
        )
        session_directory = self._make_session_dir(state=state, seed=seed)
        environment = build_live_debug_environment(
            tree_exploration=tree_exploration,
            session_directory=session_directory,
            snapshot_format=self.snapshot_format,
        )

        self.base.random_generator.seed(seed)
        try:
            result = environment.controlled_exploration.explore(
                random_generator=self.base.random_generator
            )
            recommendation = cast("Recommendation", result.branch_recommendation)
            self._write_summary(
                session_directory=environment.session_directory,
                state=state,
                seed=seed,
                recommendation=recommendation,
            )
            self._move_index += 1
            return recommendation
        finally:
            environment.finalize()

    def print_info(self) -> None:
        """Delegate informational printing to the wrapped selector."""
        self.base.print_info()

    def _make_session_dir(self, *, state: StateT, seed: Seed) -> Path:
        """Create a readable per-move session directory."""
        state_label = self._sanitize(self.state_to_debug_string(state))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_label = self._sanitize(str(seed))
        return self.session_root / (
            f"move_{self._move_index:03d}_state_{state_label}_"
            f"seed_{seed_label}_{timestamp}"
        )

    def _write_summary(
        self,
        *,
        session_directory: Path,
        state: StateT,
        seed: Seed,
        recommendation: Recommendation,
    ) -> None:
        """Persist a small move-level summary beside the Anemone session files."""
        summary: dict[str, Any] = {
            "move_index": self._move_index,
            "state_debug": self.state_to_debug_string(state),
            "state_repr": str(state),
            "seed": seed,
            "recommended_move": str(recommendation.recommended_name),
            "recommended_move_name": str(recommendation.recommended_name),
        }
        if recommendation.evaluation is not None:
            summary["evaluation"] = self._serialize(recommendation.evaluation)
        if recommendation.policy is not None:
            summary["policy"] = self._serialize(recommendation.policy)
        if recommendation.branch_evals is not None:
            summary["branch_evaluations"] = self._serialize(recommendation.branch_evals)

        (session_directory / "move_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def _sanitize(self, value: str) -> str:
        """Return a filesystem-friendly fragment for session directory names."""
        cleaned = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in value.strip()
        )
        cleaned = cleaned.strip("_")
        if not cleaned:
            return "state"
        return cleaned[:48]

    def _serialize(self, value: Any) -> Any:
        """Convert nested recommendation data into JSON-safe builtins."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            return self._serialize(asdict(cast("Any", value)))
        if isinstance(value, Mapping):
            return {
                str(key): self._serialize(item_value)
                for key, item_value in cast("Mapping[object, Any]", value).items()
            }
        if _is_non_string_sequence(value):
            return self._serialize_sequence(value)
        return str(value)

    def _serialize_sequence(self, value: Sequence[object]) -> list[Any]:
        """Serialize one generic sequence into JSON-safe builtins."""
        return [self._serialize(item) for item in value]
