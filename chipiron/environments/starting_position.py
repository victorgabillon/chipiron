from typing import Protocol

from valanga import StateTag


class StartingPositionArgs(Protocol):
    def get_start_tag(self) -> StateTag: ...
