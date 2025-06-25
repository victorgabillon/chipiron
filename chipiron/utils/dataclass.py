"""
Module to check if an object is a dataclass
"""

from collections.abc import Iterable
from enum import Enum
from typing import Any, ClassVar, Protocol


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


# used as factory function in one of the option of as_dict in order to make the enum be replaced by str (useful for yamlization)
def custom_asdict_factory(data: Iterable[Any]) -> dict[Any, Any]:
    def convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, list):
            return [convert_value(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        else:
            return obj

    return dict((k, convert_value(v)) for k, v in data)
