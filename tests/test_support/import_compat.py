"""Test-only import bootstrap for lightweight characterization suites.

The current installed dependencies do not fully match the production source tree
that these tests need to exercise.  PR1 is intentionally test-only, so this
helper keeps the compatibility shims local to the new tests instead of changing
runtime code.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

_BOOTSTRAPPED = False

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CHIPIRON_SRC = SRC_ROOT / "chipiron"
SIBLING_ROOT = REPO_ROOT.parent
ATOMHEART_SRC = SIBLING_ROOT / "atomheart" / "src"
ANEMONE_SRC = SIBLING_ROOT / "anemone" / "src"
TEST_OUTPUT_DIR = Path("/tmp/chipiron-test-output")


def bootstrap_test_imports() -> None:
    """Make local source imports deterministic for the PR1 characterization tests."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    _configure_test_environment()
    _ensure_source_paths()
    _seed_namespace_packages()
    _patch_valanga()
    _install_stub_modules()

    _BOOTSTRAPPED = True


def _configure_test_environment() -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CHIPIRON_OUTPUT_DIR", str(TEST_OUTPUT_DIR))
    os.environ.setdefault(
        "ML_FLOW_URI_PATH",
        f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns.db').as_posix()}",
    )
    os.environ.setdefault(
        "ML_FLOW_URI_PATH_TEST",
        f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns_test.db').as_posix()}",
    )


def _ensure_source_paths() -> None:
    for path in (SRC_ROOT, ATOMHEART_SRC, ANEMONE_SRC):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _seed_namespace_packages() -> None:
    _seed_package("chipiron", CHIPIRON_SRC)
    _seed_package("chipiron.games", CHIPIRON_SRC / "games")
    _seed_package("chipiron.displays", CHIPIRON_SRC / "displays")
    _seed_package("chipiron.players", CHIPIRON_SRC / "players")
    utils_pkg = _seed_package("chipiron.utils", CHIPIRON_SRC / "utils")
    _seed_package("chipiron.scripts", CHIPIRON_SRC / "scripts")
    _seed_package("chipiron.games.domain.match", CHIPIRON_SRC / "games" / "domain" / "match")
    _seed_package("chipiron.players.move_selector", CHIPIRON_SRC / "players" / "move_selector")

    if ATOMHEART_SRC.exists():
        atomheart_pkg = ATOMHEART_SRC / "atomheart"
        _seed_package("atomheart", atomheart_pkg)
        _seed_package("atomheart.games", atomheart_pkg / "games")
        _seed_package("atomheart.games.chess", atomheart_pkg / "games" / "chess")
        _seed_package("atomheart.games.chess.move", atomheart_pkg / "games" / "chess" / "move")
        _seed_package("atomheart.games.chess.board", atomheart_pkg / "games" / "chess" / "board")

    _seed_package("anemone", ANEMONE_SRC / "anemone" if ANEMONE_SRC.exists() else TEST_OUTPUT_DIR)
    _seed_package(
        "anemone.progress_monitor",
        ANEMONE_SRC / "anemone" / "progress_monitor"
        if ANEMONE_SRC.exists()
        else TEST_OUTPUT_DIR,
    )

    # `game_manager.py` imports `MyPath` from the package root.
    utils_pkg.MyPath = str


def _seed_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    _attach_to_parent(name, module)
    return module


def _attach_to_parent(name: str, module: ModuleType) -> None:
    if "." not in name:
        return
    parent_name, child_name = name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, module)


def _patch_valanga() -> None:
    import valanga
    import valanga.evaluations as evaluations

    if not hasattr(valanga, "Role"):
        class Role(StrEnum):
            """Minimal role enum used only to import newer sibling sources in tests."""

            SOLO = "solo"
            WHITE = "white"
            BLACK = "black"

        valanga.Role = Role

    if not hasattr(valanga, "SoloRole"):
        class SoloRole(StrEnum):
            """Minimal solo-role enum used only in test imports."""

            SOLO = "solo"

        valanga.SoloRole = SoloRole

    if not hasattr(valanga, "Outcome"):
        class Outcome(StrEnum):
            """Minimal outcome enum used only in test imports."""

            WIN = "win"
            DRAW = "draw"
            LOSS = "loss"
            ABORTED = "aborted"
            UNKNOWN = "unknown"

        valanga.Outcome = Outcome

    if not hasattr(evaluations, "Value"):
        evaluations.Value = float
    if not hasattr(evaluations, "Certainty"):
        evaluations.Certainty = float

    @dataclass(frozen=True, slots=True)
    class OverEvent:
        """Small stand-in for the newer generic-role over-event API."""

        outcome: Any
        termination: Any | None = None
        winner: Any | None = None

        def is_draw(self) -> bool:
            return self.outcome == valanga.Outcome.DRAW

        def is_win_for(self, role: Any) -> bool:
            return self.outcome == valanga.Outcome.WIN and self.winner == role

        @property
        def is_win(self) -> bool:
            return self.outcome == valanga.Outcome.WIN

        def __class_getitem__(cls, item: object) -> type["OverEvent"]:
            _ = item
            return cls

    # The installed class has an older constructor. Overriding it locally keeps
    # the tests focused on chipiron behavior rather than packaging drift.
    valanga.OverEvent = OverEvent


def _install_stub_modules() -> None:
    _install_chipiron_player_stubs()
    _install_anemone_stubs()
    _install_atomheart_stubs()


def _install_chipiron_player_stubs() -> None:
    player_factory_module = ModuleType("chipiron.players.factory_higher_level")
    player_factory_module.MoveFunction = object
    player_factory_module.create_player_observer_factory = lambda *args, **kwargs: None
    sys.modules[player_factory_module.__name__] = player_factory_module
    _attach_to_parent(player_factory_module.__name__, player_factory_module)

    @dataclass
    class CompatPlayerArgs:
        """Tiny replacement for player args used in characterization tests."""

        name: str
        main_move_selector: object
        oracle_play: bool = False

        def is_human(self) -> bool:
            maybe_callable = getattr(self.main_move_selector, "is_human", None)
            if callable(maybe_callable):
                return bool(maybe_callable())
            return bool(getattr(self.main_move_selector, "human", False))

    @dataclass
    class CompatPlayerFactoryArgs:
        """Tiny replacement for player factory args used in characterization tests."""

        player_args: CompatPlayerArgs
        seed: int

    class CompatPlayerHandle:
        """Minimal closeable player handle used by the runtime fixtures."""

        def close(self) -> None:
            return None

    players_pkg = sys.modules["chipiron.players"]
    players_pkg.PlayerArgs = CompatPlayerArgs
    players_pkg.PlayerFactoryArgs = CompatPlayerFactoryArgs
    players_pkg.PlayerHandle = CompatPlayerHandle
    players_pkg.InProcessPlayerHandle = CompatPlayerHandle
    players_pkg.PlayerProcess = CompatPlayerHandle

    move_selector_module = ModuleType("chipiron.players.move_selector.tree_and_value_args")

    @dataclass(frozen=True)
    class TreeAndValueAppArgs:
        """Small stand-in exposing only the field used by player_ui_info."""

        anemone_args: object

    move_selector_module.TreeAndValueAppArgs = TreeAndValueAppArgs
    sys.modules[move_selector_module.__name__] = move_selector_module
    _attach_to_parent(move_selector_module.__name__, move_selector_module)


def _install_anemone_stubs() -> None:
    progress_module = ModuleType("anemone.progress_monitor.progress_monitor")

    @dataclass(frozen=True)
    class TreeBranchLimitArgs:
        """Small stand-in carrying only the tree branch limit."""

        tree_branch_limit: int

    @dataclass(frozen=True)
    class AllStoppingCriterionArgs:
        """Small stand-in used by a few player-facing modules."""

        tree_branch_limit: int | None = None

    progress_module.TreeBranchLimitArgs = TreeBranchLimitArgs
    progress_module.AllStoppingCriterionArgs = AllStoppingCriterionArgs
    sys.modules[progress_module.__name__] = progress_module
    _attach_to_parent(progress_module.__name__, progress_module)


def _install_atomheart_stubs() -> None:
    chess_move_module = sys.modules.get("atomheart.games.chess.move")
    if chess_move_module is None:
        chess_move_module = ModuleType("atomheart.games.chess.move")
        sys.modules[chess_move_module.__name__] = chess_move_module
        _attach_to_parent(chess_move_module.__name__, chess_move_module)
    chess_move_module.MoveUci = str

    move_factory_module = ModuleType("atomheart.games.chess.move.move_factory")

    class MoveFactory:
        """Placeholder move factory for GameManager imports in tests."""

    move_factory_module.MoveFactory = MoveFactory
    sys.modules[move_factory_module.__name__] = move_factory_module
    _attach_to_parent(move_factory_module.__name__, move_factory_module)

    board_utils_module = ModuleType("atomheart.games.chess.board.utils")

    @dataclass(frozen=True)
    class FenPlusHistory:
        """Placeholder chess snapshot type used by GUI/player protocols."""

        current_fen: str = ""

    board_utils_module.FenPlusHistory = FenPlusHistory
    sys.modules[board_utils_module.__name__] = board_utils_module
    _attach_to_parent(board_utils_module.__name__, board_utils_module)
