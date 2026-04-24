"""Live control-file helpers for one Morpion bootstrap run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_loop import MorpionBootstrapArgs
    from .config import MorpionBootstrapConfig

BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY = "bootstrap_applied_control"
BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY = "bootstrap_applied_runtime_control"
BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY = "bootstrap_effective_runtime"
BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY = "bootstrap_effective_runtime_hash"


@dataclass(frozen=True, slots=True)
class MorpionBootstrapRuntimeControl:
    """Optional cycle-boundary overrides for supported Anemone runtime settings."""

    tree_branch_limit: int | None = None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapEffectiveRuntimeConfig:
    """Normalized runtime/search configuration applied for one bootstrap cycle."""

    tree_branch_limit: int


@dataclass(frozen=True, slots=True)
class MorpionBootstrapControl:
    """Optional per-cycle overrides for one running Morpion bootstrap loop."""

    max_growth_steps_per_cycle: int | None = None
    max_rows: int | None = None
    use_backed_up_value: bool | None = None
    save_after_seconds: float | None = None
    save_after_tree_growth_factor: float | None = None
    force_evaluator: str | None = None
    runtime: MorpionBootstrapRuntimeControl = field(
        default_factory=MorpionBootstrapRuntimeControl
    )


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
        runtime=_optional_runtime_control(loaded.get("runtime")),
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


def bootstrap_runtime_control_to_dict(
    control: MorpionBootstrapRuntimeControl,
) -> dict[str, object]:
    """Serialize one runtime-control subsection into JSON-friendly data."""
    return dict(asdict(control))


def bootstrap_control_from_metadata(value: object) -> MorpionBootstrapControl:
    """Deserialize one applied-control metadata payload tolerantly."""
    if not isinstance(value, dict):
        return MorpionBootstrapControl()
    return MorpionBootstrapControl(
        max_growth_steps_per_cycle=_optional_int(
            value.get("max_growth_steps_per_cycle")
        ),
        max_rows=_optional_int(value.get("max_rows")),
        use_backed_up_value=_optional_bool(value.get("use_backed_up_value")),
        save_after_seconds=_optional_float(value.get("save_after_seconds")),
        save_after_tree_growth_factor=_optional_float(
            value.get("save_after_tree_growth_factor")
        ),
        force_evaluator=_optional_force_evaluator(value.get("force_evaluator")),
        runtime=_optional_runtime_control(value.get("runtime")),
    )


def effective_runtime_config_from_config_and_control(
    config: MorpionBootstrapConfig,
    control: MorpionBootstrapControl,
) -> MorpionBootstrapEffectiveRuntimeConfig:
    """Derive the explicit runtime config to apply on the next cycle boundary."""
    tree_branch_limit = control.runtime.tree_branch_limit
    if tree_branch_limit is None:
        tree_branch_limit = config.runtime.tree_branch_limit
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=tree_branch_limit,
    )


def effective_runtime_config_to_dict(
    config: MorpionBootstrapEffectiveRuntimeConfig,
) -> dict[str, object]:
    """Serialize one effective runtime config into JSON-friendly data."""
    return dict(asdict(config))


def effective_runtime_config_sha256(
    config: MorpionBootstrapEffectiveRuntimeConfig,
) -> str:
    """Return a stable hash for one effective runtime config payload."""
    return hashlib.sha256(
        json.dumps(
            effective_runtime_config_to_dict(config),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def effective_runtime_config_from_metadata(
    value: object,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Deserialize one effective-runtime metadata payload tolerantly."""
    if not isinstance(value, dict):
        return None
    tree_branch_limit = _optional_int(value.get("tree_branch_limit"))
    if tree_branch_limit is None:
        return None
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=tree_branch_limit,
    )


def bootstrap_runtime_control_from_metadata(
    value: object,
) -> MorpionBootstrapRuntimeControl:
    """Deserialize one applied runtime-control metadata payload tolerantly."""
    return _optional_runtime_control(value)


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


def _optional_runtime_control(value: object) -> MorpionBootstrapRuntimeControl:
    """Return one optional runtime-control subsection from JSON-friendly data."""
    if not isinstance(value, dict):
        return MorpionBootstrapRuntimeControl()
    return MorpionBootstrapRuntimeControl(
        tree_branch_limit=_optional_int(value.get("tree_branch_limit")),
    )


__all__ = [
    "BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY",
    "BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY",
    "BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY",
    "BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY",
    "MorpionBootstrapControl",
    "MorpionBootstrapEffectiveRuntimeConfig",
    "MorpionBootstrapRuntimeControl",
    "apply_control_to_args",
    "bootstrap_control_from_metadata",
    "bootstrap_control_to_dict",
    "bootstrap_runtime_control_from_metadata",
    "bootstrap_runtime_control_to_dict",
    "effective_runtime_config_from_config_and_control",
    "effective_runtime_config_from_metadata",
    "effective_runtime_config_sha256",
    "effective_runtime_config_to_dict",
    "load_bootstrap_control",
    "save_bootstrap_control",
]
