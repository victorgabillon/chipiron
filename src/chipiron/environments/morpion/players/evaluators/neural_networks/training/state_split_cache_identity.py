"""Conservative, path-independent semantics for the optional split artifact.

This deliberately fences CPU and GPU Torch builds. It does not inspect a device
or change Torch settings. Installed code is immutable for a process lifetime;
restart workers after changing an installation, just as for model execution.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path

FORMAT = "chipiron-state-split-v1"
ENVIRONMENT = "CHIPIRON_STATE_SPLIT_CACHE_DIR"


def digest(path: Path) -> str:
    """Read content with bounded hashing memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value: object) -> bytes:
    """Use one finite, canonical JSON representation for identity and payloads."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@lru_cache(maxsize=1)
def semantic_fingerprint() -> str:
    """Bind actual transitive codec/row-reader code, runtime and serialization.

    Hash installed bytes, not just version labels or unverified RECORD entries.
    The broad foundation fence deliberately misses on unrelated foundation edits.
    Torch does not compute membership; its build label/version module additionally
    prevents unqualified cross-backend reuse without initializing its GPU runtime.
    """
    members: dict[str, str] = {}
    package = Path(__file__).parents[6]
    for path in sorted(package.rglob("*.py")):
        members[f"chipiron/{path.relative_to(package)}"] = digest(path)
    versions: dict[str, str] = {}
    for name in (
        "atomheart",
        "valanga",
        "algorhino-anemone",
        "algorhino-coral",
        "parsley-coco",
        "numpy",
        "orjson",
        "ijson",
        "zstandard",
        "pyarrow",
    ):
        distribution = importlib.metadata.distribution(name)
        import_name = {
            "algorhino-anemone": "anemone",
            "algorhino-coral": "coral",
            "parsley-coco": "parsley",
        }.get(name, name)
        spec = importlib.util.find_spec(import_name)
        expected = Path(str(distribution.locate_file(f"{import_name}/__init__.py")))
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).resolve() != expected.resolve()
        ):
            message = f"Split cache requires unshadowed installed dependency: {name}"
            raise ValueError(message)
        versions[name] = distribution.version
        files = distribution.files
        if files is None:
            message = f"Split cache requires installed dependency evidence: {name}"
            raise ValueError(message)
        for item in sorted(files):
            if str(item).endswith((".py", ".pyd", ".dll")) or ".so" in item.name:
                members[f"{name}/{item}"] = digest(
                    Path(str(distribution.locate_file(item)))
                )
    torch = importlib.metadata.distribution("torch")
    versions["torch"] = torch.version
    members["torch/version.py"] = digest(
        Path(str(torch.locate_file("torch/version.py")))
    )
    # Python's JSON module and accelerator govern float/target serialization.

    members["python"] = digest(Path(sys.executable))
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    if library.is_file():
        members["libpython"] = digest(library)
    accelerator = importlib.util.find_spec("_json")
    if accelerator is None or accelerator.origin is None:
        message = "Split cache requires JSON accelerator identity"
        raise ValueError(message)
    members["_json"] = (
        accelerator.origin
        if accelerator.origin in {"built-in", "frozen"}
        else digest(Path(accelerator.origin))
    )
    for path in sorted(Path(json.__file__).parent.glob("*.py")):
        members[f"json/{path.name}"] = digest(path)
    identity = {
        "format": FORMAT,
        "python": sys.version,
        "implementation": sys.implementation.name,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "versions": versions,
        "members": members,
    }
    return hashlib.sha256(encoded(identity)).hexdigest()
