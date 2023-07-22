from typing import Protocol,TypeVar

T= TypeVar("T")

class Comparable(Protocol[T]):
    def __eq__(self : T, other:T):
        ...

    def __lt__(self: T, other: T):
        ...
