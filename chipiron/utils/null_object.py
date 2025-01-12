"""
Module that contains the NullObject class.


"""

from typing import Any, Self


class NullObject:
    """
    The NullObject class is a null object implementation in Python. Null objects are objects that always
     and reliably "do nothing." They are often used as placeholders or default values when an actual object
      is not available or needed.

    The NullObject class provides the following methods:
    - __init__: Initializes the NullObject instance.
    - __call__: Allows the NullObject instance to be called as a function.
    - __repr__: Returns a string representation of the NullObject instance.
    - __nonzero__: Returns 0 to indicate that the NullObject instance is considered False.
    - __getattr__: Handles attribute access on the NullObject instance.
    - __setattr__: Handles attribute assignment on the NullObject instance.
    - __delattr__: Handles attribute deletion on the NullObject instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Self:
        return self

    def __repr__(self) -> str:
        return "Null(  )"

    def __nonzero__(self) -> int:
        return 0

    def __getattr__(self, name: Any) -> Self:
        return self

    def __setattr__(self, name: Any, value: Any) -> None:
        pass

    def __delattr__(self, name: Any) -> None:
        pass
