from __future__ import annotations

from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")


class PutQueue(Protocol[T_contra]):
    def put(
        self,
        item: T_contra,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...

class PutGetQueue(Protocol[T]):
    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None: ...
    def get(self, block: bool = True, timeout: float | None = None) -> T: ...
