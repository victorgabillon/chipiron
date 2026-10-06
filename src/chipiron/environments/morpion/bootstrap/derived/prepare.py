"""Prepare an immutable-source derivation without constructing a live search runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import zstandard

from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_dict,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.legacy_graph.config import (
    LegacyGraphConfig,
    legacy_graph_config_from_dict,
)

from .provenance import (
    PROVENANCE_NAME,
    SCHEMA,
    DerivationError,
    atomic_json,
    inside,
    read_json,
    sha256,
)


def migrate_config(
    raw: dict[str, Any], *, growth_steps: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Wrap legacy fields, preserving values/schema; apply only the growth override."""
    result = copy.deepcopy(raw)
    mapping = []
    expected = set(LegacyGraphConfig().to_dict()) - {"representation"}
    for name, spec in result["evaluators"]["evaluators"].items():
        legacy_keys = {key for key in spec if key.startswith("graph_")}
        if not legacy_keys:
            continue
        if (
            legacy_keys != expected
            or spec.get("model_type") != "entity_token_transformer_value_net"
            or "legacy_graph_tokens" in spec
            or any(key.startswith("entity_") for key in spec)
        ):
            message = f"Ambiguous legacy evaluator fields: {name}"
            raise DerivationError(message)
        block = {"representation": "graph_tokens_v1"}
        for key in sorted(expected):
            value = spec.pop(key)
            block[key] = value
            mapping.append({
                "evaluator": name,
                "legacy_field": key,
                "legacy_value": value,
                "current_field": f"legacy_graph_tokens.{key}",
                "current_value": value,
                "equivalence": "unchanged historical graph_tokens_v1 field; no entity conversion",
            })
        legacy_graph_config_from_dict(block)
        spec["legacy_graph_tokens"] = block
    result["runtime"]["max_growth_steps_per_cycle"] = growth_steps
    bootstrap_config_from_dict(result)
    return result, mapping


def _inventory(root: Path) -> dict[str, list[int]]:
    return {
        str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in root.rglob("*")
        if path.is_file()
    }


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    with (
        (path / "metadata.json.zst").open("rb") as handle,
        zstandard.ZstdDecompressor().stream_reader(handle) as reader,
    ):
        return cast("dict[str, Any]", json.loads(reader.read()))


def inspect_source(
    source: Path,
    target: Path,
    *,
    generation: int,
    expected_nodes: int,
    growth_steps: int,
    max_generations: int,
    search_seed: int,
    rollout_seed: int,
    training_seed: int,
    code_sha: str,
) -> dict[str, Any]:
    """Read source artifacts/config and emit a complete deterministic experiment plan."""
    source = source.resolve()
    target = target.resolve()
    if target.exists():
        message = f"Refusing existing target: {target}"
        raise DerivationError(message)
    if (
        source == target
        or target.is_relative_to(source)
        or source.is_relative_to(target)
    ):
        message = "Source and target must be disjoint."
        raise DerivationError(message)
    if growth_steps < 1 or max_generations < 1 or expected_nodes < 1:
        message = "Growth steps, bound and node count must be positive."
        raise DerivationError(message)
    state = read_json(source / "run_state.json")
    checkpoint_name = f"search_checkpoints/generation_{generation:06d}.sharded"
    export_name = f"tree_exports_sharded/generation_{generation:06d}.json"
    if (
        state["generation"] != generation
        or state["latest_runtime_checkpoint_path"] != checkpoint_name
        or state["latest_tree_snapshot_path"] != export_name
    ):
        message = "Source run_state does not point to the requested runtime checkpoint/export."
        raise DerivationError(message)
    checkpoint = inside(source, checkpoint_name)
    manifest = read_json(checkpoint / "manifest.json")
    metadata = _checkpoint_metadata(checkpoint)
    export = read_json(inside(source, export_name))
    if (
        manifest["total_node_count"] != expected_nodes
        or metadata["node_count"] != expected_nodes
        or export["node_count"] != expected_nodes
        or export["generation"] != generation
    ):
        message = (
            "Checkpoint, export and requested node/generation identities disagree."
        )
        raise DerivationError(message)
    if manifest.get("generation") not in (None, generation):
        message = "Checkpoint manifest generation disagrees."
        raise DerivationError(message)
    kinds = {entry["kind"] for entry in manifest["shards"]}
    if (
        not {"metadata", "node_shells", "node_runtime", "selector", "latest_expansions"}
        <= kinds
    ):
        message = f"Incomplete checkpoint runtime/selector shards: {sorted(kinds)}"
        raise DerivationError(message)
    files = [
        source / "bootstrap_config.json",
        source / "run_state.json",
        checkpoint / "manifest.json",
        source / "pipeline/active_model.json",
    ]
    for entry in manifest["shards"]:
        path = inside(checkpoint, entry["path"])
        if entry.get("sha256") is not None and sha256(path) != entry["sha256"]:
            message = f"Checkpoint shard checksum mismatch: {path}"
            raise DerivationError(message)
        files.append(path)
    # Additive exports need their historical node shards, index and manifests.
    export_root = inside(source, "tree_exports_sharded")
    files.extend(path for path in export_root.rglob("*") if path.is_file())
    active = read_json(source / "pipeline/active_model.json")
    bundle = inside(source, active["model_bundle_path"])
    for required in (
        "param.pt",
        "morpion_regressor_args.json",
        "morpion_manifest.json",
    ):
        inside(bundle, required)
    files.extend(path for path in bundle.rglob("*") if path.is_file())
    if active["evaluator_name"] != state["active_evaluator_name"]:
        message = "Source active-model pointer and run_state disagree."
        raise DerivationError(message)
    raw = read_json(source / "bootstrap_config.json")
    if (
        raw["pipeline_mode"] != "artifact_pipeline"
        or raw["training_export_mode"] != "sharded"
        or raw["runtime"]["runtime_checkpoint_format"] != "sharded"
    ):
        message = (
            "This derivation supports only the persisted sharded artifact pipeline."
        )
        raise DerivationError(message)
    if rollout_seed != raw["search"]["rollout"]["random_seed"]:
        message = "Rollout reset seed must preserve the persisted rollout seed."
        raise DerivationError(message)
    control = source / "control.json"
    if control.exists() and any(
        value is not None for value in read_json(control).values()
    ):
        message = "Explicitly resolve source controls before deriving."
        raise DerivationError(message)
    effective, mapping = migrate_config(raw, growth_steps=growth_steps)
    hashes = {
        str(path.relative_to(source)): sha256(
            inside(source, str(path.relative_to(source)))
        )
        for path in sorted(set(files))
    }
    identity_payload = {
        "source_files": hashes,
        "effective_config": effective,
        "generation": generation,
        "search_seed": search_seed,
        "rollout_seed": rollout_seed,
        "training_seed": training_seed,
        "max_generations": max_generations,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "experiment_id": identity,
        "source_work_dir": str(source),
        "target_work_dir": str(target),
        "source_checkpoint": checkpoint_name,
        "source_training_export": export_name,
        "source_tree_generation": generation,
        "source_cycle_index": state["cycle_index"],
        "source_node_count": expected_nodes,
        "source_active_model_name": active["evaluator_name"],
        "source_active_model_generation": active["generation"],
        "source_active_model_bundle": active["model_bundle_path"],
        "source_config_sha256": hashes["bootstrap_config.json"],
        "source_checkpoint_manifest_sha256": hashes[f"{checkpoint_name}/manifest.json"],
        "source_file_sha256": hashes,
        "code_sha": code_sha,
        "reason_exact_continuation_impossible": {
            "search_rng_missing": metadata.get("rng_state") is None,
            "rollout_rng_missing": metadata.get("rollout_rng_state") is None,
        },
        "rng_reset_policy": {
            "search": {
                "implementation": "random.Random(seed).getstate() injected in copied checkpoint metadata",
                "seed": search_seed,
            },
            "rollout": {
                "implementation": "independent random.Random(seed).getstate() injected in copied checkpoint metadata",
                "seed": rollout_seed,
            },
            "training": {
                "seed": training_seed,
                "implementation": "Python/NumPy/Torch reset to training_seed + local generation immediately before full training stage; CUDA seeded if available",
                "hardware_bitwise_reproducibility_guaranteed": False,
            },
            "validation": {
                "seed": raw["validation_seed"],
                "policy": "unchanged existing split",
            },
        },
        "config_migration_policy": "compatibility implementation of historical graph_tokens_v1; no modern representation conversion",
        "config_field_mapping": mapping,
        "active_model_policy": "external source generation retained; only explicit matching derived seed uses local tree generation as publication bound",
        "scheduler_policy": "sequential_all_generations_v1",
        "reevaluation_drain_policy": "complete full pass, checkpoint each consumed patch before atomic pointer commit/deletion; no extra Growth",
        "max_generations": max_generations,
        "expected_final_generation": generation + max_generations,
        "temporary_overrides": {
            "max_growth_steps_per_cycle": {
                "from": raw["runtime"]["max_growth_steps_per_cycle"],
                "to": growth_steps,
            }
        },
        "effective_config": effective,
        "observability_dependency": "PR #70 must be integrated before real launch; no duplicate instrumentation",
        "experiment_launched": False,
    }


def prepare_workspace(plan: dict[str, Any]) -> Path:
    """Copy owned files, reset only copied RNG metadata, then commit preparation."""
    source = Path(plan["source_work_dir"])
    target = Path(plan["target_work_dir"])
    before = _inventory(source)
    target.mkdir(parents=True, exist_ok=False)
    atomic_json(target / "preparation_incomplete.json", {"status": "preparing"})
    for relative, expected_hash in plan["source_file_sha256"].items():
        origin = inside(source, relative)
        if sha256(origin) != expected_hash:
            message = f"Source changed since inspection: {relative}"
            raise DerivationError(message)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        if (
            sha256(destination) != expected_hash
            or origin.stat().st_ino == destination.stat().st_ino
        ):
            message = f"Copy verification failed: {relative}"
            raise DerivationError(message)
    checkpoint = target / plan["source_checkpoint"]
    metadata = _checkpoint_metadata(checkpoint)
    metadata["rng_state"] = random.Random(
        plan["rng_reset_policy"]["search"]["seed"]
    ).getstate()
    metadata["rollout_rng_state"] = (
        random.Random(plan["rng_reset_policy"]["rollout"]["seed"]).getstate()
        if plan["effective_config"]["search"]["rollout"]["enabled"]
        else None
    )
    encoded = json.dumps(metadata, sort_keys=True).encode()
    compressed = zstandard.ZstdCompressor().compress(encoded)
    (checkpoint / "metadata.json.zst").write_bytes(compressed)
    manifest = read_json(checkpoint / "manifest.json")
    for entry in manifest["shards"]:
        if entry["kind"] == "metadata":
            entry.update(
                compressed_bytes=len(compressed),
                uncompressed_bytes=len(encoded),
                sha256=hashlib.sha256(compressed).hexdigest(),
            )
    atomic_json(checkpoint / "manifest.json", manifest)
    atomic_json(
        target / "bootstrap_config.json",
        bootstrap_config_to_dict(bootstrap_config_from_dict(plan["effective_config"])),
    )
    active = read_json(target / "pipeline/active_model.json")
    active.update(
        source="external_seed",
        source_generation=active["generation"],
        local_trained_generation=None,
    )
    active["metadata"] = {
        **active.get("metadata", {}),
        "derived_experiment_id": plan["experiment_id"],
    }
    atomic_json(target / "pipeline/active_model.json", active)
    generation = plan["source_tree_generation"]
    atomic_json(
        target / "pipeline/training_cursor.json",
        {
            "latest_started_generation": generation,
            "latest_completed_generation": generation,
        },
    )
    state = read_json(target / "run_state.json")
    state["metadata"]["derived_experiment_id"] = plan["experiment_id"]
    atomic_json(target / "run_state.json", state)
    provenance = {
        **plan,
        "derivation_timestamp": datetime.now(UTC).isoformat(),
        "prepared_checkpoint_metadata_sha256": sha256(checkpoint / "metadata.json.zst"),
        "prepared_config_sha256": sha256(target / "bootstrap_config.json"),
    }
    if _inventory(source) != before or any(
        sha256(source / relative) != checksum
        for relative, checksum in plan["source_file_sha256"].items()
    ):
        message = "Historical source changed during preparation."
        raise DerivationError(message)
    provenance["prepared_restore_sha256"] = {
        relative: sha256(target / relative)
        for relative in plan["source_file_sha256"]
        if relative.startswith((
            plan["source_checkpoint"] + "/",
            plan["source_active_model_bundle"] + "/",
        ))
    }
    atomic_json(target / PROVENANCE_NAME, provenance)
    atomic_json(
        target / "derived_orchestration.json",
        {
            "schema": SCHEMA,
            "experiment_id": plan["experiment_id"],
            "completed_generation": generation,
            "phase": "ready",
            "generations": {},
        },
    )
    # The launcher uses the same interpreter, but the CLI independently checks pinned code.
    import shlex

    command = shlex.join([
        "env",
        f"PYTHONPATH={Path(plan['code_root']) / 'src'}",
        sys.executable,
        "-m",
        "chipiron.environments.morpion.bootstrap.derived.cli",
        "run",
        "--work-dir",
        str(target),
    ])
    (target / "run_derived_bootstrap.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec " + command + ' "$@"\n'
    )
    (target / "preparation_incomplete.json").unlink()
    return target
