"""Module that contains the SyzygyProcess class."""

import copy
import multiprocessing
import queue
from typing import Any

from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import AnySyzygyTable


class SyzygyProcess(multiprocessing.Process):
    """A class that extends the Process class from the multiprocessing module.

    This class represents a separate process that runs in parallel with the main program.

    Attributes:
        syzygy_table (AnySyzygyTable): The SyzygyTable object used for tablebase lookups.
        queue_board (queue.Queue): The queue used for receiving board messages from the main program.

    """

    def __init__(
        self, syzygy_table: AnySyzygyTable, queue_board: queue.Queue[Any]
    ) -> None:
        """Initialize a new instance of the SyzygyProcess class.

        Args:
            syzygy_table (AnySyzygyTable): The SyzygyTable object used for tablebase lookups.
            queue_board (queue.Queue): The queue used for receiving board messages from the main program.

        """
        multiprocessing.Process.__init__(self, daemon=False)
        self._stop_event = multiprocessing.Event()
        self.syzygy_table = syzygy_table
        self.queue_board = queue_board

    def run(self) -> None:
        """Override the Process run method.

        This method is called when the process is started.

        It continuously listens for board messages from the main program,
        performs tablebase lookups using the SyzygyTable object,
        and sends back the corresponding move messages.
        """
        print("Started Syzygy thread : ", self.syzygy_table)

        while not self.stopped():
            try:
                message = self.queue_board.get(False)
            except queue.Empty:
                pass
            else:
                if message["type"] == "board":
                    board = message["board"]
                    queue_reply = message["queue_reply"]
                    print("syzygy thread got ", board)
                    move = self.syzygy_table.best_move(board)
                    message = {
                        "type": "move",
                        "move": move,
                        "corresponding_board": board.fen,
                    }
                    deep_copy_message = copy.deepcopy(message)
                    print("sending ", message)
                    queue_reply.put(deep_copy_message)

    def stop(self) -> None:
        """Stop the SyzygyProcess by setting the stop event."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """Check if the SyzygyProcess is stopped.

        Returns:
            bool: True if the process is stopped, False otherwise.

        """
        return self._stop_event.is_set()
