"""
Module to check if an object is a dataclass
"""
from typing import ClassVar, Protocol, Any


class DataClass(Protocol):
    """
    Protocol to represent a dataclass.

    This protocol is used to check if an object is a dataclass by checking
    for the presence of the `__dataclass_fields__` attribute.
    """
    __dataclass_fields__: dict[str, Any]


class IsDataclass(Protocol):
    """
    Protocol to represent a dataclass.

    This protocol is used to check if an object is a dataclass by checking
    for the presence of the `__dataclass_fields__` attribute.
    """
    __dataclass_fields__: ClassVar[dict[Any, Any]]
