"""Live control-file helpers for one Morpion bootstrap run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bootstrap_loop import MorpionBootstrapArgs

BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY = "bootstrap_applied_control"


@dataclass(frozen=True, slots=True)
class MorpionBootstrapControl:
    """Optional per-cycle overrides for one running Morpion bootstrap loop."""

    max_growth_steps_per_cycle: int | None = None
    max_rows: int | None = None
    use_backed_up_value: bool | None = None
    save_after_seconds: float | None = None
    save_after_tree_growth_factor: float | None = None
    force_evaluator: str | None = None


def load_bootstrap_control(path: Path) -> MorpionBootstrapControl:
    """Load one tolerant bootstrap control file or return the empty control.

    Missing, malformed, or wrong-shaped payloads are treated as an empty control so
    a broken live control file never stops the bootstrap loop.
    """
    if not path.is_file():
        return MorpionBootstrapControl()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return MorpionBootstrapControl()
    if not isinstance(loaded, dict):
        return MorpionBootstrapControl()

    return MorpionBootstrapControl(
        max_growth_steps_per_cycle=_optional_int(
            loaded.get("max_growth_steps_per_cycle")
        ),
        max_rows=_optional_int(loaded.get("max_rows")),
        use_backed_up_value=_optional_bool(loaded.get("use_backed_up_value")),
        save_after_seconds=_optional_float(loaded.get("save_after_seconds")),
        save_after_tree_growth_factor=_optional_float(
            loaded.get("save_after_tree_growth_factor")
        ),
        force_evaluator=_optional_force_evaluator(loaded.get("force_evaluator")),
    )


def save_bootstrap_control(control: MorpionBootstrapControl, path: Path) -> None:
    """Persist one bootstrap control file as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bootstrap_control_to_dict(control), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def apply_control_to_args(
    args: MorpionBootstrapArgs,
    control: MorpionBootstrapControl,
) -> MorpionBootstrapArgs:
    """Return args overridden by the non-None live control fields."""
    replacements: dict[str, object] = {}
    if control.max_growth_steps_per_cycle is not None:
        replacements["max_growth_steps_per_cycle"] = control.max_growth_steps_per_cycle
    if control.max_rows is not None:
        replacements["max_rows"] = control.max_rows
    if control.use_backed_up_value is not None:
        replacements["use_backed_up_value"] = control.use_backed_up_value
    if control.save_after_seconds is not None:
        replacements["save_after_seconds"] = control.save_after_seconds
    if control.save_after_tree_growth_factor is not None:
        replacements["save_after_tree_growth_factor"] = (
            control.save_after_tree_growth_factor
        )
    if not replacements:
        return args
    return replace(args, **replacements)


def bootstrap_control_to_dict(control: MorpionBootstrapControl) -> dict[str, object]:
    """Serialize one bootstrap control into JSON-friendly data."""
    return dict(asdict(control))


def bootstrap_control_from_metadata(value: object) -> MorpionBootstrapControl:
    """Deserialize one applied-control metadata payload tolerantly."""
    if not isinstance(value, dict):
        return MorpionBootstrapControl()
    return MorpionBootstrapControl(
        max_growth_steps_per_cycle=_optional_int(value.get("max_growth_steps_per_cycle")),
        max_rows=_optional_int(value.get("max_rows")),
        use_backed_up_value=_optional_bool(value.get("use_backed_up_value")),
        save_after_seconds=_optional_float(value.get("save_after_seconds")),
        save_after_tree_growth_factor=_optional_float(
            value.get("save_after_tree_growth_factor")
        ),
        force_evaluator=_optional_force_evaluator(value.get("force_evaluator")),
    )


def _optional_int(value: object) -> int | None:
    """Return one optional integer-like control field."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _optional_float(value: object) -> float | None:
    """Return one optional float-like control field."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _optional_bool(value: object) -> bool | None:
    """Return one optional bool control field."""
    return value if isinstance(value, bool) else None


def _optional_force_evaluator(value: object) -> str | None:
    """Return one optional forced evaluator name, normalizing auto-like values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "auto":
        return None
    return stripped


__all__ = [
    "BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY",
    "MorpionBootstrapControl",
    "apply_control_to_args",
    "bootstrap_control_from_metadata",
    "bootstrap_control_to_dict",
    "load_bootstrap_control",
    "save_bootstrap_control",
]