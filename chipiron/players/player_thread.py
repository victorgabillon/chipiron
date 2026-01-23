"""player_thread.py

Generic, game-agnostic player worker process.

This process:
- reads `PlayerRequest[SnapT]` from an input queue
- calls the generic runtime handler
- emits dataclass events to an output queue
"""

from __future__ import annotations

import multiprocessing
import queue
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from chipiron.players.communications.player_runtime import handle_player_request
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from chipiron.players.communications.player_message import PlayerRequest
    from chipiron.players.game_player import GamePlayer
    from chipiron.utils.dataclass import IsDataclass
    from chipiron.utils.queue_protocols import PutGetQueue, PutQueue

SnapT = TypeVar("SnapT")
RuntimeT = TypeVar("RuntimeT")
BuildArgsT = TypeVar("BuildArgsT")


class StopEvent(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class BuildGamePlayer(Protocol[SnapT, RuntimeT]):
    def __call__(
        self, args: object, queue_out: PutQueue[IsDataclass]
    ) -> GamePlayer[SnapT, RuntimeT]: ...


class PlayerProcess(multiprocessing.Process, Generic[SnapT, RuntimeT, BuildArgsT]):
    """Run a `GamePlayer` in a separate process.

    This is intentionally game-agnostic: game construction is injected via `build_game_player`.
    """

    queue_in: PutGetQueue[PlayerRequest[SnapT] | None]
    queue_out: PutQueue[IsDataclass]
    _build_game_player: BuildGamePlayer[SnapT, RuntimeT]
    _build_args: BuildArgsT
    _stop_event: StopEvent

    def __init__(
        self,
        *,
        build_game_player: BuildGamePlayer[SnapT, RuntimeT],
        build_args: BuildArgsT,
        queue_in: PutGetQueue[PlayerRequest[SnapT] | None],
        queue_out: PutQueue[IsDataclass],
        daemon: bool = False,
    ) -> None:
        super().__init__(daemon=daemon)
        self._stop_event = multiprocessing.Event()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self._build_game_player = build_game_player
        self._build_args = build_args

    def stop(self) -> None:
        """Request a clean stop.

        Note: calling `stop()` from the parent process only works if the `PlayerProcess`
        is still alive and shares the IPC primitives. As an additional spawn-safe option,
        we also try to send a poison-pill (`None`) into the input queue.
        """
        self._stop_event.set()
        try:
            self.queue_in.put(None)
        except Exception:
            # Best-effort: if the queue is already closed or not writable, termination
            # is still handled by external process control.
            pass

    def close(self) -> None:
        """Shutdown hook used by the game manager.

        Tries a graceful stop first (poison-pill) and then falls back to
        `terminate()` if the process doesn't exit quickly.
        """
        self.stop()

        try:
            self.join(timeout=1.0)
        except Exception:
            # If join isn't possible for some reason, fall back to terminate.
            pass

        if self.is_alive():
            try:
                self.terminate()
                self.join(timeout=1.0)
            except Exception:
                pass

    def run(self) -> None:
        """Main loop.

        Receives `PlayerRequest[SnapT]` objects from `queue_in` and dispatches them
        through `handle_player_request`. If a `None` is received, the process exits.
        """
        game_player = self._build_game_player(self._build_args, self.queue_out)
        chipiron_logger.info("Started player process: %s", game_player)

        while not self._stop_event.is_set():
            try:
                request = self.queue_in.get(timeout=0.1)
            except queue.Empty:
                continue

            if request is None:
                break

            handle_player_request(
                request=request,
                game_player=game_player,
                out_queue=self.queue_out,
            )
