"""
Module to check if an object is a dataclass
"""
from collections.abc import Iterable
from enum import Enum
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


def custom_asdict_factory(data: Iterable[Any]) -> dict[Any, Any]:
    def convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data)
