"""Focused tests for the PR3 role-driven environment assembly path."""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from valanga import Transition

import chipiron.games.domain.game.game_manager_factory as game_manager_factory_module
from chipiron.core.request_context import RequestContext
from chipiron.displays.gui_protocol import (
    UpdGameStatus,
    UpdNeedHumanAction,
    UpdStateGeneric,
)
from chipiron.environments.base import Environment
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_manager_factory import (
    GameManagerFactory,
    MissingParticipantAssignmentForRoleError,
)
from chipiron.games.domain.game.game_rules import GameOutcome, OutcomeKind
from chipiron.players import PlayerArgs, PlayerFactoryArgs
from chipiron.players.communications.player_message import (
    EvMove,
    PlayerRequest,
    TurnStatePlusHistory,
)
from chipiron.players.move_selector.human import GuiHumanPlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.random_args import RandomSelectorArgs


@dataclass(frozen=True, slots=True)
class ActionSet:
    """Tiny iterable action container compatible with runtime dynamics calls."""

    actions: tuple[str, ...]

    def __iter__(self) -> Any:  # pragma: no cover - exercised indirectly
        """Iterate over legal actions."""
        return iter(self.actions)

    def get_all(self) -> tuple[str, ...]:
        """Return all legal actions."""
        return self.actions


@dataclass(frozen=True, slots=True)
class RoleCycleState:
    """Minimal state carrying the environment role cycle and current actor."""

    roles: tuple[str, ...]
    turn: str
    remaining_moves: int
    tag: int = 0

    def is_game_over(self) -> bool:
        """Return whether the deterministic move budget is exhausted."""
        return self.remaining_moves <= 0


@dataclass(frozen=True, slots=True)
class RoleCycleDynamics:
    """Deterministic dynamics cycling through environment-declared roles."""

    roles: tuple[str, ...]
    valid_action: str = "advance"

    def legal_actions(self, state: RoleCycleState) -> ActionSet:
        """Return the single legal action until terminal state."""
        if state.is_game_over():
            return ActionSet(())
        return ActionSet((self.valid_action,))

    def action_name(self, state: RoleCycleState, action: str) -> str:
        """Return the stable transport name for an action."""
        _ = state
        return self.action_from_name(state, action)

    def action_from_name(self, state: RoleCycleState, name: str) -> str:
        """Parse an action name into the concrete branch key."""
        _ = state
        if name != self.valid_action:
            raise UnknownRoleCycleActionError(name)
        return name

    def step(self, state: RoleCycleState, action: str) -> Transition[RoleCycleState]:
        """Advance to the next role in the declared environment order."""
        self.action_from_name(state, action)
        next_remaining = state.remaining_moves - 1
        next_turn = state.turn
        if next_remaining > 0:
            current_index = self.roles.index(state.turn)
            next_turn = self.roles[(current_index + 1) % len(self.roles)]
        return Transition(
            RoleCycleState(
                roles=self.roles,
                turn=next_turn,
                remaining_moves=next_remaining,
                tag=state.tag + 1,
            ),
            modifications=None,
            is_over=next_remaining <= 0,
            over_event=None,
            info={"action": action},
        )


class UnknownRoleCycleActionError(ValueError):
    """Raised when a role-cycle test action is not supported."""

    def __init__(self, action_name: str) -> None:
        """Initialize the error with the invalid action name."""
        super().__init__(f"Unknown role-cycle action: {action_name!r}")


class RoleCycleRules:
    """Minimal rules adapter returning a draw at terminal states."""

    def outcome(self, state: RoleCycleState) -> GameOutcome | None:
        """Return a terminal outcome once the move budget is exhausted."""
        if not state.is_game_over():
            return None
        return GameOutcome(kind=OutcomeKind.DRAW)

    def pretty_result(self, state: RoleCycleState, outcome: GameOutcome) -> str:
        """Return a stable human-readable result."""
        _ = state
        _ = outcome
        return "draw"

    def assessment(self, state: RoleCycleState) -> None:
        """Return no non-terminal assessment for the toy environment."""
        _ = state
        return

    def pretty_assessment(self, state: RoleCycleState, assessment: object) -> str:
        """Return a placeholder assessment string."""
        _ = state
        _ = assessment
        return "no assessment"


@dataclass(frozen=True, slots=True)
class RoleCycleGuiEncoder:
    """Small GUI encoder used by the role-driven factory tests."""

    game_kind: GameKind = GameKind.CHECKERS

    def make_state_payload(
        self,
        *,
        state: RoleCycleState,
        seed: int | None,
    ) -> UpdStateGeneric:
        """Encode the role-cycle state into a generic GUI payload."""
        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=(f"turn:{state.turn}",),
            adapter_payload={"turn": state.turn, "remaining_moves": state.remaining_moves},
            seed=seed,
        )

    def make_status_payload(self, *, status: Any) -> UpdGameStatus:
        """Encode the current playing status."""
        return UpdGameStatus(status=status)


@dataclass(frozen=True, slots=True)
class RoleCyclePlayerRequestEncoder:
    """Small request encoder producing role-aware test snapshots."""

    game_kind: GameKind = GameKind.CHECKERS

    def make_move_request(
        self,
        *,
        state: RoleCycleState,
        seed: int,
        scope: Any,
    ) -> PlayerRequest[str]:
        """Encode a move request for the toy role-cycle environment."""
        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                role_to_play=state.turn,
                snapshot=f"remaining={state.remaining_moves}",
                historical_actions=None,
            ),
        )


class NoOpEvaluator:
    """Minimal evaluator satisfying the factory dependency surface."""

    def evaluate(self, state: RoleCycleState) -> tuple[None, float]:
        """Return a neutral evaluation."""
        _ = state
        return None, 0.0

    def add_evaluation(self, player_color: Any, evaluation: float) -> None:
        """Ignore external evaluations for the toy role-cycle environment."""
        _ = player_color
        _ = evaluation


@dataclass
class CloseableHandle:
    """Tiny closeable handle returned by the fake player observer factory."""

    role: str
    closed: bool = False

    def close(self) -> None:
        """Mark the handle as closed."""
        self.closed = True


@dataclass
class FakeObserverRecorder:
    """Capture which environment roles are wired to engine participants."""

    created_roles: list[str] = field(default_factory=list)
    requests_by_role: dict[str, list[PlayerRequest[str]]] = field(default_factory=dict)

    def build_factory(
        self,
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: object,
        universal_behavior: bool,
    ) -> Any:
        """Return a fake observer factory that records role-based creation."""
        _ = each_player_has_its_own_thread
        _ = implementation_args
        _ = universal_behavior

        def factory(
            player_factory_args: PlayerFactoryArgs,
            player_role: str,
            main_thread_mailbox: queue.Queue[Any],
        ) -> tuple[CloseableHandle, Any]:
            _ = player_factory_args
            _ = main_thread_mailbox
            self.created_roles.append(player_role)
            self.requests_by_role.setdefault(player_role, [])
            handle = CloseableHandle(role=player_role)

            def record_request(request: PlayerRequest[str]) -> None:
                self.requests_by_role[player_role].append(request)

            return handle, record_request

        return factory


def make_player_factory_args(*, name: str, human: bool) -> PlayerFactoryArgs:
    """Build small player factory args for role-driven assembly tests."""
    return PlayerFactoryArgs(
        player_args=PlayerArgs(
            name=name,
            main_move_selector=(
                GuiHumanPlayerArgs(type=MoveSelectorTypes.GUI_HUMAN)
                if human
                else RandomSelectorArgs()
            ),
            oracle_play=False,
        ),
        seed=11,
    )


def make_role_environment(
    *,
    roles: tuple[str, ...],
    remaining_moves: int,
    recorder: FakeObserverRecorder,
) -> Environment[RoleCycleState, str, str]:
    """Create a generic role-driven environment for the PR3 assembly tests."""

    def normalize_start_tag(tag: Any) -> str:
        return str(tag)

    def make_initial_state(tag: str) -> RoleCycleState:
        return RoleCycleState(roles=roles, turn=tag, remaining_moves=remaining_moves)

    return Environment(
        game_kind=GameKind.CHECKERS,
        roles=roles,
        rules=RoleCycleRules(),
        dynamics=RoleCycleDynamics(roles=roles),
        gui_encoder=RoleCycleGuiEncoder(),
        player_encoder=RoleCyclePlayerRequestEncoder(),
        make_player_observer_factory=recorder.build_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )


def make_game_manager_factory(*, display_queue: queue.Queue[Any]) -> GameManagerFactory:
    """Build a small factory with a subscribed GUI publisher queue."""
    factory = GameManagerFactory(
        env_deps=SimpleNamespace(),
        output_folder_path=None,
        main_thread_mailbox=queue.Queue(),
        game_manager_state_evaluator=NoOpEvaluator(),
        move_factory=SimpleNamespace(),
        implementation_args=SimpleNamespace(),
        universal_behavior=False,
    )
    factory.session_id = "session-pr3"
    factory.match_id = "match-pr3"
    factory.subscribe(display_queue)
    return factory


def make_game_args(*, start_role: str) -> Any:
    """Build minimal game args for the role-driven session tests."""
    return SimpleNamespace(
        game_kind=GameKind.CHECKERS,
        starting_position=SimpleNamespace(get_start_tag=lambda: start_role),
        each_player_has_its_own_thread=False,
        max_half_moves=None,
    )


def make_move_event(
    request: PlayerRequest[str],
    *,
    player_name: str,
    player_role: str,
) -> EvMove:
    """Build a player move event from an issued role-based request."""
    return EvMove(
        branch_name="advance",
        corresponding_state_tag=request.state.current_state_tag,
        ctx=RequestContext(
            request_id=request.ctx.request_id if request.ctx is not None else 0,
            role_to_play=player_role,
        ),
        player_name=player_name,
        player_role=player_role,
        evaluation=None,
    )


def test_factory_assembles_participants_from_environment_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GameManagerFactory should bind participants from the environment role set."""
    recorder = FakeObserverRecorder()
    environment = make_role_environment(
        roles=("alpha", "beta", "gamma"),
        remaining_moves=3,
        recorder=recorder,
    )
    monkeypatch.setattr(game_manager_factory_module, "make_environment", lambda **_: environment)

    display_queue: queue.Queue[Any] = queue.Queue()
    factory = make_game_manager_factory(display_queue=display_queue)
    session = factory.create(
        args_game_manager=make_game_args(start_role="gamma"),
        participant_factory_args_by_role={
            "alpha": make_player_factory_args(name="alpha-engine", human=False),
            "beta": make_player_factory_args(name="beta-human", human=True),
            "gamma": make_player_factory_args(name="gamma-engine", human=False),
        },
        game_seed=7,
    )

    assert recorder.created_roles == ["alpha", "gamma"]
    assert session.controller.human_roles == {"beta"}
    assert set(session.controller.engine_request_by_role) == {"alpha", "gamma"}
    assert session.manager.participant_id_by_role == {
        "alpha": "alpha-engine",
        "beta": "beta-human",
        "gamma": "gamma-engine",
    }
    assert display_queue.empty()


def test_factory_rejects_missing_participant_assignment_for_environment_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment-declared roles should all require participant assignments."""
    recorder = FakeObserverRecorder()
    environment = make_role_environment(
        roles=("alpha", "beta", "gamma"),
        remaining_moves=2,
        recorder=recorder,
    )
    monkeypatch.setattr(game_manager_factory_module, "make_environment", lambda **_: environment)

    factory = make_game_manager_factory(display_queue=queue.Queue())

    with pytest.raises(MissingParticipantAssignmentForRoleError):
        factory.create(
            args_game_manager=make_game_args(start_role="alpha"),
            participant_factory_args_by_role={
                "alpha": make_player_factory_args(name="alpha-engine", human=False),
                "beta": make_player_factory_args(name="beta-engine", human=False),
            },
            game_seed=9,
        )


def test_controller_start_uses_state_turn_not_environment_role_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial dispatch should target `state.turn`, not the first declared role."""
    recorder = FakeObserverRecorder()
    environment = make_role_environment(
        roles=("alpha", "beta", "gamma"),
        remaining_moves=3,
        recorder=recorder,
    )
    monkeypatch.setattr(game_manager_factory_module, "make_environment", lambda **_: environment)

    session = make_game_manager_factory(display_queue=queue.Queue()).create(
        args_game_manager=make_game_args(start_role="gamma"),
        participant_factory_args_by_role={
            "alpha": make_player_factory_args(name="alpha-engine", human=False),
            "beta": make_player_factory_args(name="beta-engine", human=False),
            "gamma": make_player_factory_args(name="gamma-engine", human=False),
        },
        game_seed=13,
    )

    session.controller.start()

    assert recorder.requests_by_role["gamma"][0].ctx is not None
    assert recorder.requests_by_role["gamma"][0].ctx.role_to_play == "gamma"
    assert recorder.requests_by_role["alpha"] == []
    assert recorder.requests_by_role["beta"] == []


def test_single_role_environment_can_publish_human_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solo environment should request human input without white/black placeholders."""
    recorder = FakeObserverRecorder()
    environment = make_role_environment(
        roles=("solo",),
        remaining_moves=1,
        recorder=recorder,
    )
    monkeypatch.setattr(game_manager_factory_module, "make_environment", lambda **_: environment)

    display_queue: queue.Queue[Any] = queue.Queue()
    session = make_game_manager_factory(display_queue=display_queue).create(
        args_game_manager=make_game_args(start_role="solo"),
        participant_factory_args_by_role={
            "solo": make_player_factory_args(name="solo-human", human=True),
        },
        game_seed=21,
    )

    session.controller.start()

    payload = display_queue.get_nowait().payload
    assert isinstance(payload, UpdNeedHumanAction)
    assert payload.ctx.role_to_play == "solo"
    assert session.controller.pending_role == "solo"
    assert recorder.requests_by_role == {}


def test_controller_rejects_wrong_role_response_and_advances_by_state_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role-driven dispatch should still reject wrong-role responses."""
    recorder = FakeObserverRecorder()
    environment = make_role_environment(
        roles=("alpha", "beta", "gamma"),
        remaining_moves=3,
        recorder=recorder,
    )
    monkeypatch.setattr(game_manager_factory_module, "make_environment", lambda **_: environment)

    session = make_game_manager_factory(display_queue=queue.Queue()).create(
        args_game_manager=make_game_args(start_role="alpha"),
        participant_factory_args_by_role={
            "alpha": make_player_factory_args(name="alpha-engine", human=False),
            "beta": make_player_factory_args(name="beta-engine", human=False),
            "gamma": make_player_factory_args(name="gamma-engine", human=False),
        },
        game_seed=34,
    )

    session.controller.start()
    alpha_request = recorder.requests_by_role["alpha"][0]
    session.controller.handle_player_action(
        make_move_event(
            alpha_request,
            player_name="beta-engine",
            player_role="beta",
        )
    )

    assert session.manager.game.state.tag == 0
    assert recorder.requests_by_role["beta"] == []

    session.controller.handle_player_action(
        make_move_event(
            alpha_request,
            player_name="alpha-engine",
            player_role="alpha",
        )
    )

    assert session.manager.game.state.turn == "beta"
    assert session.manager.game.state.tag == 1
    assert recorder.requests_by_role["beta"][0].ctx is not None
    assert recorder.requests_by_role["beta"][0].ctx.role_to_play == "beta"
