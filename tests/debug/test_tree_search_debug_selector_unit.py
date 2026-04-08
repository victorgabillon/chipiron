"""Direct-file unit tests for the tree-search debug wrapper."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from random import Random
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/chipiron/debug/tree_search_debug_selector.py"
)


@dataclass(frozen=True, slots=True)
class _FakeRecommendation:
    recommended_name: str
    evaluation: object | None = None
    policy: object | None = None
    branch_evals: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _FakeState:
    value: int

    def __str__(self) -> str:
        return f"n={self.value}"


class _FakeBaseSelector:
    def __init__(self, recommendation: _FakeRecommendation) -> None:
        self.random_generator = Random()
        self.recommendation = recommendation
        self.create_calls: list[tuple[object, object]] = []

    def create_tree_exploration(
        self,
        *,
        state: object,
        notify_progress: object | None = None,
    ) -> object:
        self.create_calls.append((state, notify_progress))
        return SimpleNamespace(marker="tree-exploration")

    def print_info(self) -> None:
        """Compatibility stub."""


def _load_debug_selector_module(
    tmp_path: Path,
) -> tuple[types.ModuleType, dict[str, object]]:
    """Load the wrapper module with lightweight dependency stubs."""
    captured: dict[str, object] = {}

    anemone_module = types.ModuleType("anemone")
    anemone_debug_module = types.ModuleType("anemone.debug")
    valanga_module = types.ModuleType("valanga")
    valanga_game_module = types.ModuleType("valanga.game")
    valanga_policy_module = types.ModuleType("valanga.policy")

    def build_live_debug_environment(
        *,
        tree_exploration: object,
        session_directory: str | Path,
        snapshot_format: str = "svg",
    ) -> object:
        session_path = Path(session_directory)
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "index.html").write_text("debug", encoding="utf-8")
        (session_path / "session.json").write_text(
            json.dumps({"entry_count": 1, "is_live": True}),
            encoding="utf-8",
        )
        (session_path / "snapshots").mkdir(exist_ok=True)

        captured["tree_exploration"] = tree_exploration
        captured["session_directory"] = session_path
        captured["snapshot_format"] = snapshot_format
        captured["finalized"] = False

        recommendation = _FakeRecommendation(
            recommended_name="half",
            evaluation={"score": -7.0},
            policy={"half": 1.0},
            branch_evals={"half": {"score": -7.0}, "dec1": {"score": -14.0}},
        )

        class _FakeControlledExploration:
            def explore(self, *, random_generator: Random) -> object:
                captured["explore_random"] = random_generator
                return SimpleNamespace(branch_recommendation=recommendation)

        class _FakeEnvironment:
            controlled_exploration = _FakeControlledExploration()
            session_directory = session_path

            def finalize(self) -> None:
                captured["finalized"] = True

        return _FakeEnvironment()

    valanga_game_module.Seed = int
    valanga_policy_module.NotifyProgressCallable = object
    valanga_policy_module.Recommendation = _FakeRecommendation
    anemone_debug_module.build_live_debug_environment = build_live_debug_environment

    stub_modules = {
        "anemone": anemone_module,
        "anemone.debug": anemone_debug_module,
        "valanga": valanga_module,
        "valanga.game": valanga_game_module,
        "valanga.policy": valanga_policy_module,
    }
    original_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    try:
        spec = importlib.util.spec_from_file_location(
            "chipiron.debug.tree_search_debug_selector",
            MODULE_PATH,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    captured["tmp_path"] = tmp_path
    return module, captured


def test_debug_tree_search_selector_records_per_move_summary(tmp_path: Path) -> None:
    """The wrapper should create a per-move session directory and summary file."""
    module, captured = _load_debug_selector_module(tmp_path)
    base_selector = _FakeBaseSelector(
        recommendation=_FakeRecommendation(recommended_name="unused")
    )
    selector = module.DebugTreeSearchSelector(
        base=base_selector,
        session_root=tmp_path / "runs" / "debug" / "integer_reduction" / "match-1",
        state_to_debug_string=lambda state: str(state.value),
        snapshot_format="dot",
    )

    recommendation = selector.recommend(
        state=_FakeState(15),
        seed=37,
        notify_progress="progress-callback",
    )

    move_directory = captured["session_directory"]
    assert isinstance(move_directory, Path)
    summary = json.loads(
        (move_directory / "move_summary.json").read_text(encoding="utf-8")
    )

    assert base_selector.create_calls == [(_FakeState(15), "progress-callback")]
    assert captured["snapshot_format"] == "dot"
    assert captured["finalized"] is True
    assert move_directory.name.startswith("move_000_state_15_seed_37_")
    assert summary["move_index"] == 0
    assert summary["state_debug"] == "15"
    assert summary["state_repr"] == "n=15"
    assert summary["recommended_move_name"] == "half"
    assert summary["branch_evaluations"]["dec1"]["score"] == -14.0
    assert recommendation.recommended_name == "half"
