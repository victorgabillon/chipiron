from typing import ClassVar, Dict, Protocol, Any, Type


def my_dataclass[DC](klass: Type[DC]) -> Type[DC]:
    ...


class DataClass(Protocol):
    __dataclass_fields__: dict[str, Any]


class IsDataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[dict]
