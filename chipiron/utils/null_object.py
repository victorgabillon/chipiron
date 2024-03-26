from typing import Self, Any


class NullObject:
    """ Null objects always and reliably "do nothing." """

    def __init__(self, *args: Any, **kwargs: Any) -> None: pass

    def __call__(self, *args: Any, **kwargs: Any) -> Self: return self

    def __repr__(self) -> str: return "Null(  )"

    def __nonzero__(self) -> int: return 0

    def __getattr__(self, name: Any) -> Self: return self

    def __setattr__(self, name: Any, value: Any) -> None: pass

    def __delattr__(self, name: Any) -> None: pass
