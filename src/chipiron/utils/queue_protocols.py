"""Module for queue protocols."""

from typing import Protocol


class PutQueue[T](Protocol):
    """Protocol for queues that only support put operations."""

    def put(
        self,
        item: T,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an item into the queue."""
        ...


class PutGetQueue[T](Protocol):
    """Protocol for queues that support put and get operations."""

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Put an item into the queue."""
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Get an item from the queue."""
        ...
