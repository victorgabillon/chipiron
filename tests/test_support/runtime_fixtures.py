"""Shared deterministic runtime fixtures for PR1 characterization tests."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, TypeVar

from test_support.import_compat import bootstrap_test_imports

bootstrap_test_imports()

from valanga import Color, Outcome, OverEvent, Transition

from chipiron.displays.gui_protocol import (
    CmdBackOneMove,
    CmdSetStatus,
    GuiCommand,
    HumanActionChosen,
    Scope,
    UpdGameStatus,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
    UpdStateGeneric,
)
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.final_game_result import GameReport
from chipiron.games.domain.game.game import Game, ObservableGame
from chipiron.games.domain.game.game_manager import GameManager
from chipiron.games.domain.game.game_playing_status import GamePlayingStatus, PlayingStatus
from chipiron.games.domain.game.game_rules import GameOutcome, OutcomeKind
from chipiron.games.domain.game.progress_collector import PlayerProgressCollectorObservable
from chipiron.games.runtime.orchestrator.match_controller import MatchController
from chipiron.games.runtime.orchestrator.match_orchestrator import MatchOrchestrator
from chipiron.players.communications.player_message import EvMove, PlayerEvent, PlayerRequest, TurnStatePlusHistory

PayloadT = TypeVar("PayloadT")


def other_color(color: Color) -> Color:
    """Return the opposing color for the tiny deterministic runtime game."""
    return Color.BLACK if color is Color.WHITE else Color.WHITE


@dataclass(frozen=True, slots=True)
class ActionSet:
    """Tiny branch-key generator compatible with the runtime code paths."""

    actions: tuple[str, ...]

    def __iter__(self):
        return iter(self.actions)

    def get_all(self) -> tuple[str, ...]:
        """Return all legal actions in deterministic order."""
        return self.actions


@dataclass(frozen=True, slots=True)
class CounterState:
    """Tiny two-color state used to characterize current white/black runtime logic."""

    turn: Color
    remaining_moves: int
    tag: int = 0
    last_actor: Color | None = None

    def is_game_over(self) -> bool:
        """Return whether the deterministic move budget has been exhausted."""
        return self.remaining_moves <= 0


class CounterDynamics:
    """Minimal deterministic dynamics with a single legal action."""

    VALID_ACTION = "advance"

    def legal_actions(self, state: CounterState) -> ActionSet:
        if state.is_game_over():
            return ActionSet(())
        return ActionSet((self.VALID_ACTION,))

    def step(self, state: CounterState, action: str) -> Transition[CounterState]:
        normalized_action = self.action_from_name(state, action)
        next_state = CounterState(
            turn=other_color(state.turn),
            remaining_moves=state.remaining_moves - 1,
            tag=state.tag + 1,
            last_actor=state.turn,
        )
        over_event = None
        if next_state.is_game_over():
            over_event = OverEvent(
                outcome=Outcome.WIN,
                termination="counter_exhausted",
                winner=state.turn,
            )
        return Transition(
            next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=over_event,
            info={"action": normalized_action},
        )

    def action_name(self, state: CounterState, action: str) -> str:
        _ = state
        return self.action_from_name(state, action)

    def action_from_name(self, state: CounterState, name: str) -> str:
        _ = state
        if name != self.VALID_ACTION:
            raise ValueError(f"Unknown counter action: {name!r}")
        return name


class CounterRules:
    """Minimal rules adapter exposing a terminal winner for the tiny game."""

    def outcome(self, state: CounterState) -> GameOutcome | None:
        if not state.is_game_over():
            return None
        return GameOutcome(kind=OutcomeKind.WIN, winner=state.last_actor)

    def pretty_result(self, state: CounterState, outcome: GameOutcome) -> str:
        _ = state
        return f"{outcome.winner} wins"

    def assessment(self, state: CounterState) -> None:
        _ = state
        return None

    def pretty_assessment(self, state: CounterState, assessment: object) -> str:
        _ = state
        _ = assessment
        return "no assessment"


class NoOpStateEvaluator:
    """Minimal evaluator satisfying the GameManager dependency surface."""

    def __init__(self) -> None:
        self.external_evaluations: list[tuple[Color, float]] = []

    def evaluate(self, state: CounterState) -> tuple[None, float]:
        _ = state
        return None, 0.0

    def add_evaluation(self, player_color: Color, evaluation: float) -> None:
        self.external_evaluations.append((player_color, evaluation))


@dataclass(frozen=True, slots=True)
class SimpleGuiEncoder:
    """Small GUI encoder publishing generic state/status payloads."""

    game_kind: GameKind = GameKind.CHECKERS

    def make_state_payload(
        self,
        *,
        state: CounterState,
        seed: int | None,
    ) -> UpdStateGeneric:
        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=(f"remaining:{state.remaining_moves}",),
            adapter_payload={
                "turn": state.turn.value,
                "remaining_moves": state.remaining_moves,
            },
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdGameStatus:
        return UpdGameStatus(status=status)


@dataclass(frozen=True, slots=True)
class SimplePlayerRequestEncoder:
    """Small request encoder producing picklable test snapshots."""

    game_kind: GameKind = GameKind.CHECKERS

    def make_move_request(
        self,
        *,
        state: CounterState,
        seed: int,
        scope: Scope,
    ) -> PlayerRequest[str]:
        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                turn=state.turn,
                snapshot=f"remaining={state.remaining_moves}",
                historical_actions=None,
            ),
        )


@dataclass
class CloseableHandle:
    """Closeable player-handle stand-in used by the orchestrator tests."""

    closed_count: int = 0

    def close(self) -> None:
        self.closed_count += 1


@dataclass(slots=True)
class RuntimeHarness:
    """Bundle of tiny runtime objects used by the characterization tests."""

    scope: Scope
    mailbox: queue.Queue[Any]
    display_queue: queue.Queue[Any]
    publisher: GuiPublisher
    game_manager: GameManager[CounterState]
    controller: MatchController
    orchestrator: MatchOrchestrator
    engine_requests: dict[Color, list[PlayerRequest[str]]]
    closeables: list[CloseableHandle]


@dataclass(slots=True)
class OrchestratorRun:
    """Background orchestrator execution state for mailbox-loop tests."""

    thread: threading.Thread
    results: list[GameReport] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)


def build_runtime_harness(
    *,
    start_turn: Color = Color.WHITE,
    remaining_moves: int = 2,
    human_colors: set[Color] | None = None,
    max_half_moves: int | None = None,
    seed: int = 7,
) -> RuntimeHarness:
    """Create a tiny fully wired runtime harness around the current code paths."""
    if human_colors is None:
        human_colors = {Color.WHITE, Color.BLACK}

    scope = Scope(session_id="session-1", match_id="match-1", game_id="game-1")
    mailbox: queue.Queue[Any] = queue.Queue()
    display_queue: queue.Queue[Any] = queue.Queue()
    publisher = GuiPublisher(
        out=display_queue,
        schema_version=1,
        game_kind=GameKind.CHECKERS,
        scope=scope,
    )

    game = Game(
        state=CounterState(turn=start_turn, remaining_moves=remaining_moves),
        dynamics=CounterDynamics(),
        playing_status=GamePlayingStatus(),
        seed_=seed,
    )
    observable_game = ObservableGame(
        game=game,
        gui_encoder=SimpleGuiEncoder(),
        scope=scope,
        player_encoder=SimplePlayerRequestEncoder(),
    )
    observable_game.register_display(publisher)

    engine_requests: dict[Color, list[PlayerRequest[str]]] = {
        Color.WHITE: [],
        Color.BLACK: [],
    }
    closeables = [CloseableHandle(), CloseableHandle()]
    progress_collector = PlayerProgressCollectorObservable(publishers=[publisher])

    game_manager = GameManager(
        game=observable_game,
        display_state_evaluator=NoOpStateEvaluator(),
        output_folder_path=None,
        args=SimpleNamespace(max_half_moves=max_half_moves),
        player_color_to_id={
            Color.WHITE: "white-player",
            Color.BLACK: "black-player",
        },
        main_thread_mailbox=mailbox,
        players=closeables,
        move_factory=object(),
        progress_collector=progress_collector,
        rules=CounterRules(),
    )
    controller = MatchController(
        scope=scope,
        game_manager=game_manager,
        engine_request_by_color={
            Color.WHITE: engine_requests[Color.WHITE].append,
            Color.BLACK: engine_requests[Color.BLACK].append,
        },
        human_colors=set(human_colors),
    )

    return RuntimeHarness(
        scope=scope,
        mailbox=mailbox,
        display_queue=display_queue,
        publisher=publisher,
        game_manager=game_manager,
        controller=controller,
        orchestrator=MatchOrchestrator(mailbox),
        engine_requests=engine_requests,
        closeables=closeables,
    )


def drain_gui_payloads(display_queue: queue.Queue[Any]) -> list[Any]:
    """Drain and return GUI payloads currently waiting in the display queue."""
    payloads: list[Any] = []
    while True:
        try:
            payloads.append(display_queue.get_nowait().payload)
        except queue.Empty:
            return payloads


def wait_for_gui_payload(
    display_queue: queue.Queue[Any],
    payload_type: type[PayloadT],
    *,
    timeout: float = 1.0,
) -> PayloadT:
    """Wait until a specific GUI payload type appears on the display queue."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            update = display_queue.get(timeout=0.01)
        except queue.Empty:
            continue
        payload = update.payload
        if isinstance(payload, payload_type):
            return payload
    raise AssertionError(f"Timed out waiting for GUI payload {payload_type.__name__}")


def start_orchestrator_thread(harness: RuntimeHarness) -> OrchestratorRun:
    """Run the mailbox loop in a background thread and capture its result."""
    run = OrchestratorRun(thread=threading.Thread(target=lambda: None, daemon=True))

    def runner() -> None:
        try:
            report = harness.orchestrator.play_one_game(
                game_manager=harness.game_manager,
                controller=harness.controller,
            )
        except BaseException as exc:  # pragma: no cover - surfaced back to the test
            run.errors.append(exc)
        else:
            run.results.append(report)

    run.thread = threading.Thread(target=runner, daemon=True)
    run.thread.start()
    return run


def wait_for_thread_result(run: OrchestratorRun, *, timeout: float = 1.0) -> GameReport:
    """Wait for a background orchestrator run to finish and return its report."""
    run.thread.join(timeout)
    if run.thread.is_alive():
        raise AssertionError("Timed out waiting for the orchestrator thread to finish.")
    if run.errors:
        raise run.errors[0]
    if not run.results:
        raise AssertionError("The orchestrator thread finished without producing a report.")
    return run.results[0]


def make_human_action_command(
    *,
    scope: Scope,
    action_name: str,
    ctx: Any | None,
    corresponding_state_tag: int | None,
) -> GuiCommand:
    """Create a GUI command carrying a human action."""
    return GuiCommand(
        schema_version=1,
        scope=scope,
        payload=HumanActionChosen(
            action_name=action_name,
            ctx=ctx,
            corresponding_state_tag=corresponding_state_tag,
        ),
    )


def make_status_command(*, scope: Scope, status: PlayingStatus) -> GuiCommand:
    """Create a GUI command changing the playing status."""
    return GuiCommand(
        schema_version=1,
        scope=scope,
        payload=CmdSetStatus(status=status),
    )


def make_back_command(*, scope: Scope) -> GuiCommand:
    """Create a GUI command rewinding one move."""
    return GuiCommand(
        schema_version=1,
        scope=scope,
        payload=CmdBackOneMove(),
    )


def make_player_move(
    request: PlayerRequest[str],
    *,
    branch_name: str = "advance",
    player_name: str = "engine-player",
) -> EvMove:
    """Build a player move payload from a previously issued player request."""
    return EvMove(
        branch_name=branch_name,
        corresponding_state_tag=request.state.current_state_tag,
        ctx=request.ctx,
        player_name=player_name,
        color_to_play=request.state.turn,
        evaluation=None,
    )


def make_player_event(
    request: PlayerRequest[str],
    *,
    branch_name: str = "advance",
    player_name: str = "engine-player",
) -> PlayerEvent:
    """Wrap a player move payload into the mailbox envelope used by the loop."""
    return PlayerEvent(
        schema_version=1,
        scope=request.scope,
        payload=make_player_move(
            request=request,
            branch_name=branch_name,
            player_name=player_name,
        ),
    )
