"""Direct-file unit tests for integer-reduction tree evaluator logic."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/chipiron/environments/integer_reduction/players/evaluators/integer_reduction_state_evaluator.py"
)
WIRING_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/chipiron/environments/integer_reduction/players/evaluators/wiring.py"
)


def _load_integer_reduction_evaluator_modules() -> (
    tuple[types.ModuleType, types.ModuleType, type[object]]
):
    """Load the evaluator modules with lightweight stubs for external dependencies."""

    class Outcome:
        WIN = "win"

    class State:
        pass

    class Certainty:
        TERMINAL = "terminal"
        ESTIMATE = "estimate"

    @dataclass(frozen=True, slots=True)
    class Value:
        score: float
        certainty: str
        over_event: object | None = None

    @dataclass(frozen=True, slots=True)
    class OverEvent:
        outcome: object
        termination: object | None
        winner: object | None

        def __class_getitem__(cls, _item: object) -> type[OverEvent]:
            return cls

    class EvaluationScale(StrEnum):
        SYMMETRIC_UNIT_INTERVAL = "symmetric_unit_interval"
        ENTIRE_REAL_AXIS = "entire_real_axis"
        STOCKFISH_BASED = "stockfish_based"

    @dataclass(frozen=True, slots=True)
    class IntegerReductionState:
        value: int
        steps: int = 0

        def is_game_over(self) -> bool:
            return self.value == 1

    class MasterStateValueEvaluator:
        pass

    class OverEventDetector:
        pass

    class StateEvaluator:
        def __class_getitem__(cls, _item: object) -> type[StateEvaluator]:
            return cls

    stub_modules = {
        "anemone": types.ModuleType("anemone"),
        "anemone.node_evaluation": types.ModuleType("anemone.node_evaluation"),
        "anemone.node_evaluation.direct": types.ModuleType(
            "anemone.node_evaluation.direct"
        ),
        "anemone.node_evaluation.direct.protocols": types.ModuleType(
            "anemone.node_evaluation.direct.protocols"
        ),
        "valanga": types.ModuleType("valanga"),
        "valanga.evaluations": types.ModuleType("valanga.evaluations"),
        "valanga.over_event": types.ModuleType("valanga.over_event"),
        "chipiron": types.ModuleType("chipiron"),
        "chipiron.core": types.ModuleType("chipiron.core"),
        "chipiron.core.evaluation_scale": types.ModuleType(
            "chipiron.core.evaluation_scale"
        ),
        "chipiron.environments": types.ModuleType("chipiron.environments"),
        "chipiron.environments.integer_reduction": types.ModuleType(
            "chipiron.environments.integer_reduction"
        ),
        "chipiron.environments.integer_reduction.players": types.ModuleType(
            "chipiron.environments.integer_reduction.players"
        ),
        "chipiron.environments.integer_reduction.players.evaluators": types.ModuleType(
            "chipiron.environments.integer_reduction.players.evaluators"
        ),
        "chipiron.environments.integer_reduction.types": types.ModuleType(
            "chipiron.environments.integer_reduction.types"
        ),
        "chipiron.players": types.ModuleType("chipiron.players"),
        "chipiron.players.boardevaluators": types.ModuleType(
            "chipiron.players.boardevaluators"
        ),
        "chipiron.players.boardevaluators.board_evaluator": types.ModuleType(
            "chipiron.players.boardevaluators.board_evaluator"
        ),
    }

    stub_modules["anemone.node_evaluation.direct.protocols"].MasterStateValueEvaluator = (
        MasterStateValueEvaluator
    )
    stub_modules["anemone.node_evaluation.direct.protocols"].OverEventDetector = (
        OverEventDetector
    )
    stub_modules["valanga"].Outcome = Outcome
    stub_modules["valanga"].State = State
    stub_modules["valanga.evaluations"].Certainty = Certainty
    stub_modules["valanga.evaluations"].Value = Value
    stub_modules["valanga.over_event"].OverEvent = OverEvent
    stub_modules["chipiron.core.evaluation_scale"].EvaluationScale = EvaluationScale
    stub_modules["chipiron.environments.integer_reduction.types"].IntegerReductionState = (
        IntegerReductionState
    )
    stub_modules["chipiron.players.boardevaluators.board_evaluator"].StateEvaluator = (
        StateEvaluator
    )

    for name, module in stub_modules.items():
        sys.modules[name] = module

    evaluator_spec = importlib.util.spec_from_file_location(
        "chipiron.environments.integer_reduction.players.evaluators.integer_reduction_state_evaluator",
        EVALUATOR_PATH,
    )
    assert evaluator_spec is not None
    assert evaluator_spec.loader is not None
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    sys.modules[evaluator_spec.name] = evaluator_module
    evaluator_spec.loader.exec_module(evaluator_module)

    wiring_spec = importlib.util.spec_from_file_location(
        "chipiron.environments.integer_reduction.players.evaluators.wiring",
        WIRING_PATH,
    )
    assert wiring_spec is not None
    assert wiring_spec.loader is not None
    wiring_module = importlib.util.module_from_spec(wiring_spec)
    sys.modules[wiring_spec.name] = wiring_module
    wiring_spec.loader.exec_module(wiring_module)

    return evaluator_module, wiring_module, IntegerReductionState


def test_integer_reduction_state_evaluator_orders_states_for_maximizing_search() -> (
    None
):
    """Fewer steps should score higher, with terminal states marked as certain."""
    evaluator_module, _, state_cls = _load_integer_reduction_evaluator_modules()

    evaluator = evaluator_module.IntegerReductionStateEvaluator()

    quick_state = evaluator.evaluate(state_cls(2, steps=2))
    slow_state = evaluator.evaluate(state_cls(2, steps=5))
    terminal_state = evaluator.evaluate(state_cls(1, steps=3))

    assert quick_state.score > slow_state.score
    assert quick_state.score == -2.0
    assert slow_state.score == -5.0
    assert terminal_state.score == -3.0
    assert terminal_state.certainty == evaluator_module.Certainty.TERMINAL


def test_integer_reduction_master_evaluator_and_wiring_expose_terminal_reward() -> (
    None
):
    """The tree evaluator builder and GUI wiring should share the same heuristic."""
    evaluator_module, wiring_module, state_cls = (
        _load_integer_reduction_evaluator_modules()
    )

    master = evaluator_module.build_integer_reduction_master_evaluator(
        evaluation_scale=object()
    )
    terminal_value = master.evaluate(state_cls(1, steps=4))
    non_terminal_value = master.evaluate(state_cls(8, steps=2))

    assert terminal_value.score == -4.0
    assert terminal_value.certainty == evaluator_module.Certainty.TERMINAL
    assert terminal_value.over_event is not None
    assert non_terminal_value.score == -2.0

    wiring = wiring_module.IntegerReductionEvalWiring()
    chi = wiring.build_chi()
    assert chi.evaluate(state_cls(2, steps=1)).score > chi.evaluate(
        state_cls(2, steps=3)
    ).score
    assert wiring.build_oracle() is None
