"""Immutable foreign seeds bound by fresh genesis, never local acceptances."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .derived.provenance import inside
from .generation_publication_io import artifact_digest, file_digest, require

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import MorpionBootstrapConfig
    from .run_state import MorpionBootstrapRunState

KEY = "fork_origin_v1"


def validate(
    paths: MorpionBootstrapPaths,
    config: MorpionBootstrapConfig,
    state: MorpionBootstrapRunState,
    *,
    verify: bool = False,
) -> dict[str, Any] | None:
    """Adopt only a bounded genesis reference, with optional scientific byte checks."""
    ref = config.metadata.get(KEY)
    require(
        ref == state.metadata.get(KEY), "Fork origin binding is missing or changed."
    )
    if ref is None:
        return None
    from .lifecycle import Metadata
    from .managed_sessions import KEY as MANAGED

    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256"}
        and ref["path"] == "genesis/fork/origin.json",
        "Malformed fork origin reference.",
    )
    reader = Metadata()
    path = inside(paths.work_dir, ref["path"])
    require(reader.digest(path) == ref["sha256"], "Fork origin descriptor changed.")
    origin = reader.read(path)
    require(
        origin.get("schema_version") == 1
        and type(origin["schema_version"]) is int
        and origin.get("work_dir") == str(paths.work_dir.resolve())
        and origin.get("enrollment_id") == config.metadata[MANAGED]["enrollment_id"]
        and isinstance(origin.get("candidate_seeds"), dict)
        and set(origin["candidate_seeds"]) == set(config.evaluators.evaluators)
        and origin.get("dataset_policy") == "regenerate_local",
        "Unsupported or foreign fork genesis.",
    )
    require(len(origin["candidate_seeds"]) <= 128, "Fork candidate bound exceeded.")
    for name, seed in origin["candidate_seeds"].items():
        if seed is None:
            continue
        require(
            set(seed) == {"kind", "model", "weights_sha256", "origin"}
            and seed["kind"] == "fork_seed"
            and set(seed["model"]) == {"path", "sha256"}
            and seed["model"]["path"] == f"genesis/fork/seeds/{name}",
            "Malformed imported candidate seed.",
        )
        bundle = inside(paths.work_dir, seed["model"]["path"])
        for filename in (
            "param.pt",
            "optimizer.pt",
            "morpion_manifest.json",
            "morpion_regressor_args.json",
        ):
            require(
                inside(bundle, filename).is_file(), "Incomplete fork weights/Adam seed."
            )
        if verify:
            require(
                artifact_digest(bundle) == seed["model"]["sha256"]
                and file_digest(bundle / "param.pt") == seed["weights_sha256"],
                "Imported candidate bytes changed.",
            )
    return origin


def seeds(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, *, verify: bool
) -> dict[str, Any]:
    """Resolve only the local immutable import descriptor; never follow ancestor paths."""
    from .config import load_bootstrap_config

    if KEY not in state.metadata:
        return {}
    origin = validate(
        paths, load_bootstrap_config(paths.bootstrap_config_path), state, verify=verify
    )
    return {} if origin is None else origin["candidate_seeds"]


def preserve(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    persisted: MorpionBootstrapConfig,
    requested: MorpionBootstrapConfig,
) -> MorpionBootstrapConfig:
    """Preserve this one validated immutable field before ordinary exact-hash checks."""
    if validate(paths, persisted, state) is None:
        return requested
    return replace(
        requested, metadata={**requested.metadata, KEY: persisted.metadata[KEY]}
    )
