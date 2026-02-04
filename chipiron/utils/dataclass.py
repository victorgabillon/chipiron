"""Module to check if an object is a dataclass."""

from collections.abc import Iterable
from enum import Enum
from typing import Any, ClassVar, Protocol, no_type_check


class DataClass(Protocol):
    """Protocol to represent a dataclass.

    This protocol is used to check if an object is a dataclass by checking
    for the presence of the `__dataclass_fields__` attribute.
    """

    __dataclass_fields__: dict[str, Any]


class IsDataclass(Protocol):
    """Protocol to represent a dataclass.

    This protocol is used to check if an object is a dataclass by checking
    for the presence of the `__dataclass_fields__` attribute.
    """

    __dataclass_fields__: ClassVar[dict[Any, Any]]


# used as factory function in one of the option of as_dict in order to make the enum be replaced by str (useful for yamlization)
# pyright: ignore
def custom_asdict_factory(data: Iterable[tuple[Any, Any]]) -> dict[Any, Any]:
    """Create a custom asdict factory function.

    Args:
        data (Iterable[tuple[Any, Any]]): The input data to be converted.

    Returns:
        dict[Any, Any]: The converted dictionary.

    """
    @no_type_check
    def convert_value(obj: Any) -> Any:
        """Convert a value to a serializable format.

        Args:
            obj (Any): The input value to be converted.

        Returns:
            Any: The converted value.

        """
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, list):
            return [convert_value(item) for item in obj]
        elif isinstance(obj, dict):
            result: dict[Any, Any] = {k: convert_value(v) for k, v in obj.items()}
            return result
        else:
            return obj

    return dict((k, convert_value(v)) for k, v in data)
