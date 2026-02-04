"""Module for queue protocols."""
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")


class PutQueue(Protocol[T_contra]):
    """Protocol for queues that only support put operations."""

    def put(
        self,
        item: T_contra,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an item into the queue."""
        ...


class PutGetQueue(Protocol[T]):
    """Protocol for queues that support put and get operations."""

    def put(
        self, item: T, block: bool = True, timeout: float | None = None
    ) -> None:
        """Put an item into the queue."""
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Get an item from the queue."""
        ...
