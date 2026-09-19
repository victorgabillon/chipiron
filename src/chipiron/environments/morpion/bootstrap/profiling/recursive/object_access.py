"""Low-level object access helpers for recursive profiling."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sized
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    FunctionType,
    MethodType,
    ModuleType,
)
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

ATOMIC_TYPES = (str, bytes, bytearray, int, float, bool, type(None))
SKIP_DEEP_TYPES = (
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    CodeType,
    type,
)
CONTAINER_TYPES = (list, tuple, set, frozenset)
CONTAINER_VALUE_TYPES = (Mapping, list, tuple, set, frozenset)

__all__ = [
    "all_attr_paths",
    "first_attr_path",
    "iter_direct_field_entries",
    "iter_object_attribute_values",
    "len_or_none",
    "qualified_type_name",
    "raw_attr_path",
    "raw_getattr",
    "raw_getattr_present",
    "safe_call_no_args",
    "safe_object_dict",
    "should_skip_deep",
    "slot_names",
    "small_len_bucket",
]


def qualified_type_name(value: object) -> str:
    """Return a stable module-qualified type name for ``value``."""
    value_type = type(value)
    module = value_type.__module__
    qualname = value_type.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def len_or_none(value: object) -> int | None:
    """Return ``len(value)`` when cheap and supported."""
    try:
        return len(value) if isinstance(value, Sized) else None
    except (TypeError, RuntimeError):
        return None


def small_len_bucket(length: int) -> str:
    """Return a compact bucket label for small container lengths."""
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 9:
        return "5-9"
    if length <= 24:
        return "10-24"
    if length <= 99:
        return "25-99"
    return "100+"


def should_skip_deep(value: object) -> bool:
    """Return whether generic deep traversal should treat ``value`` as shallow."""
    return isinstance(value, SKIP_DEEP_TYPES)


def raw_getattr(value: object, attr_name: str) -> object | None:
    """Read a concrete attribute/slot without using ``dir`` or properties."""
    try:
        result: object = object.__getattribute__(value, attr_name)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return result


def raw_getattr_present(value: object, attr_name: str) -> tuple[bool, object | None]:
    """Return whether a concrete attribute exists plus its raw value."""
    try:
        result: object = object.__getattribute__(value, attr_name)
    except Exception:  # pylint: disable=broad-exception-caught
        return False, None
    return True, result


def safe_call_no_args(value: object) -> object | None:
    """Call a no-argument callable, returning None when it raises."""
    if not callable(value):
        return value
    try:
        return cast("Callable[[], object]", value)()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def raw_attr_path(root: object, path: tuple[str, ...]) -> object | None:
    """Follow one raw attribute path without consulting dynamic attribute lists."""
    value: object | None = root
    for index, attr_name in enumerate(path):
        if value is None:
            return None
        value = raw_getattr(value, attr_name)
        if (
            index == len(path) - 1
            and attr_name.startswith("profile_")
            and callable(value)
        ):
            value = safe_call_no_args(value)
    return value


def first_attr_path(root: object, paths: tuple[tuple[str, ...], ...]) -> object | None:
    """Return the first non-None value found through raw attribute paths."""
    for path in paths:
        value = raw_attr_path(root, path)
        if value is not None:
            return value
    return None


def all_attr_paths(
    root: object, paths: tuple[tuple[str, ...], ...]
) -> tuple[object, ...]:
    """Return unique non-None values found through raw attribute paths."""
    values: list[object] = []
    seen_ids: set[int] = set()
    for path in paths:
        value = raw_attr_path(root, path)
        if value is None:
            continue
        value_id = id(value)
        if value_id in seen_ids:
            continue
        seen_ids.add(value_id)
        values.append(value)
    return tuple(values)


def safe_object_dict(value: object) -> Mapping[object, object] | None:
    """Return the concrete ``__dict__`` mapping when it is safely readable."""
    raw_dict = raw_getattr(value, "__dict__")
    if isinstance(raw_dict, Mapping):
        return raw_dict
    return None


def iter_direct_field_entries(value: object) -> Iterator[tuple[str, object]]:
    """Yield raw direct fields from __dict__ entries and declared slots."""
    seen_names: set[str] = set()
    raw_dict = safe_object_dict(value)
    if raw_dict is not None:
        for field_name, field_value in raw_dict.items():
            if not isinstance(field_name, str) or field_name in seen_names:
                continue
            seen_names.add(field_name)
            yield field_name, field_value
    for slot_name in slot_names(value):
        if slot_name in seen_names:
            continue
        present, slot_value = raw_getattr_present(value, slot_name)
        if not present:
            continue
        seen_names.add(slot_name)
        yield slot_name, slot_value


def slot_names(type_or_obj: object) -> tuple[str, ...]:
    """Return declared slot names across the MRO without consulting ``dir``."""
    value_type = type_or_obj if isinstance(type_or_obj, type) else type(type_or_obj)
    seen: set[str] = set()
    ordered_names: list[str] = []
    for base_type in value_type.__mro__:
        raw_slots = getattr(base_type, "__slots__", ())
        candidate_names: tuple[object, ...]
        if isinstance(raw_slots, str):
            candidate_names = (raw_slots,)
        else:
            try:
                candidate_names = tuple(raw_slots)
            except TypeError:
                candidate_names = ()
        for slot_name in candidate_names:
            if not isinstance(slot_name, str):
                continue
            if slot_name in {"__weakref__", "__dict__"} or slot_name in seen:
                continue
            seen.add(slot_name)
            ordered_names.append(slot_name)
    return tuple(ordered_names)


def iter_object_attribute_values(value: object) -> Iterator[object]:
    """Yield direct attribute containers and slot values for traversal."""
    raw_dict = safe_object_dict(value)
    if raw_dict is not None:
        yield raw_dict
    for slot_name in slot_names(value):
        slot_value = raw_getattr(value, slot_name)
        if slot_value is not None:
            yield slot_value
