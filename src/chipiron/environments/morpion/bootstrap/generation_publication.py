"""Immutable capsules with explicit scientific v1 and v2 export adapters.

All committed capsules remain durable retention roots. Only uncommitted owned
attempts can be collected, under the same Growth lease as normal publication.
"""

import hashlib
import json
import shutil
import uuid
from contextlib import ExitStack
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import ijson

from .derived.provenance import atomic_json, inside
from .generation_publication_io import (
    ATTEMPT_MARKER,
    PUBLICATION_KEY,
    artifact_digest,
    capsule_lease,
    capsule_path,
    capsule_seal,
    file_digest,
    generation_read_pin,
    growth_receipt_ownership,
    publication_namespace,
    publication_records,
    publish_directory_if_absent,
    read_committed_descriptor,
    read_object,
    require,
    sync_artifact,
    sync_directory,
    validate_capsule,
)
from .generation_publication_io import (
    copy_artifact as _copy_artifact,
)
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state
from .sharded_training_export import (
    MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
    _load_generation_manifest,
    _load_root_manifest,
    _root_manifest_to_dict,
    _write_json,
    load_morpion_sharded_training_tree_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def _is_v1_sharded(path: Path, *, allow_flat_metadata: bool = False) -> bool:
    """Classify an explicit existing export without changing legacy dispatch."""
    header: dict[str, object] = {}
    fields = {"format_kind", "format_version"}
    accepted = fields | (
        {f"metadata.{field}" for field in fields} if allow_flat_metadata else set()
    )
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix in accepted and event in {
                "string",
                "number",
            }:
                header[prefix] = value
                if fields <= header.keys():
                    break
    if allow_flat_metadata and fields.isdisjoint(header):
        # Saved flat snapshots declare their existing v1 contract in metadata.
        # An explicit top-level format always wins, including unsupported v2.
        header = {field: header.get(f"metadata.{field}") for field in fields}
    require(
        header.get("format_kind")
        in {MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND, "training_tree_snapshot"}
        and header.get("format_version") == 1,
        "Only scientific v1 exports can enter this publication protocol; no scientific v2 integration.",
    )
    return header["format_kind"] == MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND


def _require_v1_manifest(path: Path) -> None:
    """Require scientific v1 for every catalog/generation manifest, not only shards."""
    payload = read_object(path)
    require(
        payload.get("format_kind") == MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND
        and payload.get("format_version") == 1,
        "Unsupported scientific manifest version; this protocol publishes scientific v1 only.",
    )


def clone_v1_catalog(source: Path, destination: Path, *, link: bool) -> Path:
    """Seed solely from selected committed history; never trust node_index.json.

    Historical node/update/generation payload bytes are unchanged. The private
    root and index are separate inodes because the v1 writer truncates them.
    """
    _require_v1_manifest(source)
    selected = _load_generation_manifest(source)
    root = source.parent
    _require_v1_manifest(root / "manifest.json")
    catalog = _load_root_manifest(root / "manifest.json")
    require(
        catalog.node_index_path == "node_index.json", "Unsupported v1 node-index path."
    )
    mapping = {
        key: value
        for key, value in catalog.generation_manifests.items()
        if int(key) <= selected.generation
    }
    require(
        mapping.get(str(selected.generation)) == source.name,
        "Selected v1 generation is absent from its catalog.",
    )
    destination.mkdir(parents=True, exist_ok=True)
    index: dict[str, int] = {}
    copied: set[str] = set()
    for key in sorted(mapping, key=int):
        manifest_path = inside(root, mapping[key])
        _require_v1_manifest(manifest_path)
        manifest = _load_generation_manifest(manifest_path)
        require(manifest.generation == int(key), "V1 catalog generation mismatch.")
        for relative in [
            mapping[key],
            manifest.node_shard_path,
            manifest.update_shard_path,
        ]:
            if relative not in copied:
                _copy_artifact(
                    inside(root, relative), destination / relative, link=link
                )
                copied.add(relative)
        count = 0
        for row in _v1_rows(
            root / manifest.node_shard_path, "nodes", manifest.generation
        ):
            require(
                isinstance(row.get("node_id"), str)
                and row.get("creation_generation") == manifest.generation,
                "Invalid v1 creation-record provenance.",
            )
            index[str(row["node_id"])] = manifest.generation
            count += 1
        require(count == manifest.new_node_count, "V1 creation shard count mismatch.")
    _write_json(
        destination / "manifest.json",
        _root_manifest_to_dict(
            replace(
                catalog,
                latest_generation=selected.generation,
                generation_manifests=mapping,
            )
        ),
    )
    _write_json(
        destination / "node_index.json",
        {
            "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
            "format_version": 1,
            "node_id_to_creation_generation": index,
        },
    )
    result = destination / source.name
    validate_v1_export(result)
    return result


def _v1_rows(path: Path, field: str, generation: int) -> list[dict[str, Any]]:
    """Validate complete arrays with the legacy JSON decoder, including float semantics."""
    payload = read_object(path)
    require(
        payload.get("format_kind") == MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND
        and payload.get("format_version") == 1,
        "Unsupported scientific shard format.",
    )
    require(
        payload.get("generation") == generation, "Shard source generation mismatch."
    )
    rows = payload.get(field)
    require(
        isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
        "Incomplete v1 shard array.",
    )
    assert isinstance(rows, list)
    return rows


def validate_v1_export(path: Path) -> None:
    """Require a complete catalog, exact index and complete scientific v1 payloads."""
    _require_v1_manifest(path)
    _require_v1_manifest(path.parent / "manifest.json")
    manifest = _load_generation_manifest(path)
    catalog = _load_root_manifest(path.parent / "manifest.json")
    require(
        catalog.latest_generation == manifest.generation
        and catalog.generation_manifests.get(str(manifest.generation)) == path.name,
        "V1 catalog does not select its completed generation.",
    )
    expected_index: dict[str, int] = {}
    for key in sorted(catalog.generation_manifests, key=int):
        require(
            0 < int(key) <= manifest.generation,
            "Uncommitted generation in private catalog.",
        )
        previous_path = inside(path.parent, catalog.generation_manifests[key])
        _require_v1_manifest(previous_path)
        previous = _load_generation_manifest(previous_path)
        require(previous.generation == int(key), "V1 catalog generation mismatch.")
        node_shard = inside(path.parent, previous.node_shard_path)
        updates = inside(path.parent, previous.update_shard_path)
        count = 0
        for row in _v1_rows(node_shard, "nodes", previous.generation):
            require(
                isinstance(row.get("node_id"), str)
                and row.get("creation_generation") == previous.generation,
                "Invalid v1 creation-record provenance.",
            )
            expected_index[str(row["node_id"])] = previous.generation
            count += 1
        require(count == previous.new_node_count, "V1 creation shard count mismatch.")
        count = sum(1 for _ in _v1_rows(updates, "updates", previous.generation))
        require(count == previous.node_count, "V1 update shard count mismatch.")
    actual_index = read_object(path.parent / "node_index.json").get(
        "node_id_to_creation_generation"
    )
    require(
        actual_index == expected_index,
        "V1 index differs from authoritative creation shards.",
    )
    snapshot = load_morpion_sharded_training_tree_snapshot(path)
    require(
        len(snapshot.nodes) == manifest.node_count, "Incomplete v1 scientific snapshot."
    )


class DurableGenerationPublication:
    """One Growth owner publishes immutable dependencies, then run state commits."""

    def __init__(
        self,
        paths: MorpionBootstrapPaths,
        *,
        scientific_v2_producer: Mapping[str, str] | None = None,
        closed_scientific_snapshot: bool = False,
    ) -> None:
        """Bind only to the new protocol namespace, never a legacy export root."""
        self.paths = paths
        require(
            not closed_scientific_snapshot or scientific_v2_producer is not None,
            "Closed publication requires an explicit scientific producer.",
        )
        self.version = (
            3
            if closed_scientific_snapshot
            else (1 if scientific_v2_producer is None else 2)
        )
        self._source_pin = ExitStack()
        self.closed_validation: dict[str, Any] = {}
        self.producer = (
            None if scientific_v2_producer is None else dict(scientific_v2_producer)
        )
        self.root = paths.work_dir / publication_namespace(self.version)
        self.state: MorpionBootstrapRunState | None = None
        self.staging: Path | None = None
        self.identifier: str | None = None
        self.model_pins: list[dict[str, Any]] = []
        self.sharded_path: Path | None = None
        self.config_hash: str | None = None
        self.evaluator_contract = False
        self.active_evaluator: dict[str, Any] | None = None
        self.activation: dict[str, Any] | None = None

    def start(self, generation: int) -> None:
        """Reserve a unique private attempt; incomplete attempts are never discovery."""
        self.identifier = uuid.uuid4().hex
        self.staging = self.root / "staging" / self.identifier
        self.staging.mkdir(parents=True)
        (self.root / "objects").mkdir(exist_ok=True)
        (self.root / "trash").mkdir(exist_ok=True)
        self.model_pins = []
        self.sharded_path = None
        self.closed_validation.clear()
        atomic_json(
            self.staging / ATTEMPT_MARKER,
            {
                "protocol": PUBLICATION_KEY,
                "id": self.identifier,
                "generation": generation,
            },
        )
        config_copy = self.staging / "bootstrap_config.json"
        _copy_artifact(self.paths.bootstrap_config_path, config_copy)
        require(
            hashlib.sha256(config_copy.read_bytes().removesuffix(b"\n")).hexdigest()
            == self.config_hash,
            "Frozen scientific config differs from the accepted config identity.",
        )

    def validate_state(self, state: MorpionBootstrapRunState) -> dict[str, Any]:
        """Require the current head to include every immutable dependency."""
        records = publication_records(state.metadata)
        require(
            state.metadata[PUBLICATION_KEY]["schema_version"] == self.version,
            "Explicit publication version required; downgrade refused.",
        )
        head = records[-1]
        require(
            head["generation"] == state.generation
            and head["checkpoint_path"] == state.latest_runtime_checkpoint_path
            and head["tree_snapshot_path"] == state.latest_tree_snapshot_path,
            "Run state differs from its committed publication head.",
        )
        descriptor = validate_capsule(self.paths.work_dir, head)
        if state.metadata[PUBLICATION_KEY].get("evaluator_contract") is not None:
            from .generation_publication_io import committed_evaluator

            model, _ = committed_evaluator(self.paths.work_dir, state)
            require(self.version == 3, "Evaluator binding requires closed publication.")
            self.evaluator_contract = True
            if self.activation is None:
                self.active_evaluator = model
                current = head["scientific_identity"]["producer"]["identity"]
                require(
                    self.producer is not None
                    and {k: v for k, v in self.producer.items() if k != "evaluator"}
                    == {k: v for k, v in current.items() if k != "evaluator"},
                    "Scientific configuration or revision changed.",
                )
                self.producer = dict(current)
        if self.version in (2, 3):
            from .sharded_training_v2 import _producer

            assert self.producer is not None
            require(
                records[0]["generation"] == 1
                and head["scientific_identity"]["producer"]
                == _producer(
                    self.producer
                    if self.activation is None
                    else self.activation["previous_producer"]
                ),
                "Scientific publication producer or genesis changed.",
            )
        return descriptor

    def model_pin(
        self, state: MorpionBootstrapRunState, origin: str, digest: str
    ) -> Path:
        """Authenticate the original model binding and its actual frozen bytes.

        Receipt recovery validates the current head separately. Historical model
        lookup needs sealed provenance, not unrelated checkpoint/scientific files.
        """
        for record in reversed(publication_records(state.metadata)):
            descriptor = read_committed_descriptor(self.paths.work_dir, record)
            for pin in descriptor["model_pins"]:
                if (pin["origin"], pin["sha256"]) == (origin, digest):
                    path = inside(
                        capsule_path(self.paths.work_dir, record["id"], self.version),
                        pin["path"],
                    )
                    require(
                        artifact_digest(path) == digest,
                        "Pinned original model changed.",
                    )
                    return path
        require(False, "Committed receipt has no pinned original model.")
        raise AssertionError  # Unreachable; keeps the required-return type explicit.

    def freeze_model(self, model: dict[str, Any]) -> Path:
        """Snapshot exact selected bundle bytes before the runtime loads them."""
        assert self.staging is not None
        key = hashlib_model_key(model["path"], model["sha256"])
        destination = self.staging / "models" / key
        if destination.exists():
            return destination
        source = inside(self.paths.work_dir, model["path"])
        if self.state is not None:
            for record in reversed(publication_records(self.state.metadata)):
                descriptor = read_object(
                    inside(self.paths.work_dir, record["descriptor_path"])
                )
                match = next(
                    (
                        p
                        for p in descriptor["model_pins"]
                        if (p["origin"], p["sha256"])
                        == (model["path"], model["sha256"])
                    ),
                    None,
                )
                if match is not None:
                    validate_capsule(self.paths.work_dir, record)
                    source = inside(
                        capsule_path(self.paths.work_dir, record["id"], self.version),
                        match["path"],
                    )
                    break
        require(
            artifact_digest(source) == model["sha256"],
            "Selected model changed before freezing.",
        )
        _copy_artifact(
            source, destination, link=source.is_relative_to(self.root / "objects")
        )
        require(
            artifact_digest(destination) == model["sha256"],
            "Model snapshot byte identity mismatch.",
        )
        self.model_pins.append({
            **model,
            "origin": model["path"],
            "path": destination.relative_to(self.staging).as_posix(),
        })
        return destination

    def prepare(self, state: MorpionBootstrapRunState) -> MorpionBootstrapRunState:
        """Explicitly freeze a quiescent PR #79 baseline without rewriting its history."""
        self.config_hash = state.metadata.get("bootstrap_config_hash")
        require(
            isinstance(self.config_hash, str),
            "Missing accepted scientific config identity.",
        )
        if PUBLICATION_KEY in state.metadata:
            self.state = state
            if self.version == 3:
                assert state.latest_tree_snapshot_path is not None
                self._source_pin.enter_context(
                    generation_read_pin(
                        inside(self.paths.work_dir, state.latest_tree_snapshot_path)
                    )
                )
            self.validate_state(state)
            return state
        require(
            self.version == 1,
            "Scientific v2 requires explicit initialization in a new isolated workspace.",
        )
        require(
            not self.paths.pipeline_reevaluation_patch_path.exists(),
            "Publication adoption requires a quiescent baseline without a pending patch.",
        )
        assert state.latest_runtime_checkpoint_path is not None
        self.start(state.generation)
        assert self.staging is not None
        checkpoint = inside(self.paths.work_dir, state.latest_runtime_checkpoint_path)
        checkpoint_copy = self.staging / "checkpoint" / checkpoint.name
        _copy_artifact(checkpoint, checkpoint_copy)
        require(
            artifact_digest(checkpoint_copy)
            == state.metadata["checkpoint_bound_reevaluation_v1"]["checkpoint_sha256"],
            "Baseline checkpoint copy differs from its committed identity.",
        )
        if state.latest_tree_snapshot_path is not None:
            tree = inside(self.paths.work_dir, state.latest_tree_snapshot_path)
            if _is_v1_sharded(tree):
                require(
                    _load_generation_manifest(tree).generation == state.generation,
                    "Baseline sharded source generation differs from run state.",
                )
                self.sharded_path = clone_v1_catalog(
                    tree, self.staging / "tree_exports_sharded", link=False
                )
            else:
                _copy_artifact(tree, self.staging / "tree_exports" / tree.name)
        receipts = state.metadata["checkpoint_bound_reevaluation_v1"]["receipts"]
        for receipt in receipts:
            self.freeze_model({
                "path": receipt["model_bundle_path"],
                "sha256": receipt["model_sha256"],
                "generation": receipt["evaluator_generation"],
                "name": receipt["evaluator_name"],
            })
        state = self.publish(state, baseline=True)
        save_bootstrap_run_state(state, self.paths.run_state_path, durable=True)
        self.state = state
        return state

    def begin_cycle(self, model: dict[str, Any] | None) -> Path | None:
        """Create private per-cycle model state; an unsaved cycle remains an orphan."""
        assert self.state is not None
        if self.version == 3:
            self.close()
            assert self.state.latest_tree_snapshot_path is not None
            self._source_pin.enter_context(
                generation_read_pin(
                    inside(self.paths.work_dir, self.state.latest_tree_snapshot_path)
                )
            )
        self.start(self.state.generation + 1)
        return None if model is None else self.freeze_model(model)

    def checkpoint_path(self, generation: int, checkpoint_format: str) -> Path:
        """Keep the original basename and compression codec under private staging."""
        assert self.staging is not None
        canonical = self.paths.runtime_checkpoint_path_for_generation_with_format(
            generation, checkpoint_format
        )
        return self.staging / "checkpoint" / canonical.name

    def export_paths(self, *, sharded: bool = False) -> MorpionBootstrapPaths:
        """Seed this attempt only from the selected committed capsule, never an orphan."""
        assert self.staging is not None and self.state is not None
        destination = self.staging / "tree_exports_sharded"
        require(
            self.version != 3, "Closed publication has no cumulative catalog to seed."
        )
        if sharded and not destination.exists():
            for record in reversed(publication_records(self.state.metadata)):
                descriptor = read_object(
                    inside(self.paths.work_dir, record["descriptor_path"])
                )
                selected = descriptor.get("sharded_snapshot_path")
                if selected is not None:
                    validate_capsule(self.paths.work_dir, record)
                    source = inside(
                        capsule_path(self.paths.work_dir, record["id"], self.version),
                        selected,
                    )
                    if self.version == 2:
                        from .scientific_generation_publication import (
                            clone_scientific_catalog,
                        )

                        clone_scientific_catalog(source, destination)
                    else:
                        clone_v1_catalog(source, destination, link=True)
                    break
        return replace(
            self.paths,
            tree_snapshot_dir=self.staging / "tree_exports",
            sharded_tree_snapshot_dir=destination,
        )

    def publish(
        self, state: MorpionBootstrapRunState, *, baseline: bool = False
    ) -> MorpionBootstrapRunState:
        """Publish complete dependencies; only a later durable run-state write commits."""
        assert self.staging is not None and self.identifier is not None
        if (
            self.version == 1
            and not baseline
            and state.latest_tree_snapshot_path is not None
        ):
            # A future adapter can return a new layout that the v1 filename probe
            # below cannot see. Refuse it before sealing or moving any capsule.
            _is_v1_sharded(
                inside(self.paths.work_dir, state.latest_tree_snapshot_path),
                allow_flat_metadata=True,
            )
        export = (
            self.staging
            / "tree_exports_sharded"
            / f"generation_{state.generation:06d}.json"
        )
        if self.version == 1 and export.exists():
            require(
                _load_generation_manifest(export).generation == state.generation,
                "Sharded source generation differs from publication generation.",
            )
            validate_v1_export(export)
            self.sharded_path = export
        previous = (
            [] if self.state is None else publication_records(self.state.metadata)
        )
        scientific_identity = None
        canonical_validation = None
        if self.version in (2, 3):
            from .scientific_generation_publication import (
                validate_closed_export,
                validate_scientific_export,
            )

            require(
                not baseline
                and isinstance(state.generation, int)
                and not isinstance(state.generation, bool)
                and state.generation
                == (1 if not previous else previous[-1]["generation"] + 1),
                "Scientific publication requires consecutive generations.",
            )
            require(
                state.latest_tree_snapshot_path is not None
                and state.latest_runtime_checkpoint_path is not None,
                "Scientific publication requires an export and checkpoint for every saved generation.",
            )
            assert (
                state.latest_tree_snapshot_path is not None
                and state.latest_runtime_checkpoint_path is not None
            )
            assert self.producer is not None and self.config_hash is not None
            self.sharded_path = inside(
                self.paths.work_dir, state.latest_tree_snapshot_path
            )
            checkpoint = inside(
                self.paths.work_dir, state.latest_runtime_checkpoint_path
            )
            require(
                checkpoint.is_relative_to(self.staging),
                "Checkpoint escaped private scientific attempt.",
            )
            validate = (
                validate_closed_export
                if self.version == 3
                else validate_scientific_export
            )
            scientific_identity, canonical_validation = validate(
                self.sharded_path,
                staging=self.staging,
                generation=state.generation,
                producer=self.producer,
                checkpoint=checkpoint,
                config_hash=self.config_hash,
                work_dir=self.paths.work_dir,
                parent=None if not previous else previous[-1],
                **({"validated": self.closed_validation} if self.version == 3 else {}),
            )
            if previous:
                assert self.state is not None
                parent = self.validate_state(self.state)
                # Preserve the exact committed prefix, even if an adapter replaces
                # a linked inode privately instead of using the v2 writer.
                for entry in parent["files"]:
                    relative = entry["path"]
                    if (
                        self.version == 2
                        and relative.startswith("tree_exports_sharded/")
                        and relative != "tree_exports_sharded/manifest.json"
                    ):
                        require(
                            file_digest(self.staging / relative) == entry["sha256"],
                            "Scientific committed prefix changed in private attempt.",
                        )
                initial = read_object(
                    inside(self.paths.work_dir, previous[0]["descriptor_path"])
                )
                if self.evaluator_contract:
                    require(
                        self.active_evaluator == parent["active_evaluator"]
                        or self.activation is not None,
                        "Evaluator changed without explicit activation.",
                    )
                else:
                    require(
                        self.model_pins == initial["model_pins"],
                        "Scientific publication model epoch changed; explicit new lineage required.",
                    )
        parent_id = None if not previous else previous[-1]["id"]
        files = [p for p in sorted(self.staging.rglob("*")) if p.is_file()]
        require(
            not any(p.is_symlink() for p in self.staging.rglob("*")),
            "Symlinked staged dependency.",
        )
        descriptor = {
            "protocol": PUBLICATION_KEY,
            "id": self.identifier,
            "generation": state.generation,
            "parent_id": parent_id,
            "baseline": baseline,
            "bootstrap_config_hash": self.config_hash,
            "model_pins": self.model_pins,
            "sharded_snapshot_path": None
            if self.sharded_path is None
            else self.sharded_path.relative_to(self.staging).as_posix(),
            "files": [
                {
                    "path": p.relative_to(self.staging).as_posix(),
                    "size": p.stat().st_size,
                    "sha256": file_digest(p),
                }
                for p in files
            ],
        }
        if self.version in (2, 3):
            descriptor.update(
                publication_version=self.version,
                scientific_identity=scientific_identity,
                canonical_validation=canonical_validation,
            )
        if self.evaluator_contract:
            require(
                self.active_evaluator is not None, "Missing explicit active evaluator."
            )
            descriptor.update(
                evaluator_contract=1, active_evaluator=self.active_evaluator
            )
            if self.activation is not None:
                descriptor["model_activation"] = self.activation
        seal_name = capsule_seal(self.version)
        atomic_json(self.staging / seal_name, descriptor)
        sync_artifact(self.staging, self.paths.work_dir)
        final = capsule_path(self.paths.work_dir, self.identifier, self.version)
        old_staging = self.staging
        publish_directory_if_absent(old_staging, final, self.paths.work_dir)

        def relocated(path: str | None) -> str | None:
            if path is None or baseline:
                return path
            physical = self.paths.work_dir / path
            require(
                physical.is_relative_to(old_staging),
                "Result escaped its private publication attempt.",
            )
            return self.paths.relative_to_work_dir(
                final / physical.relative_to(old_staging)
            )

        state = replace(
            state,
            latest_runtime_checkpoint_path=relocated(
                state.latest_runtime_checkpoint_path
            ),
            latest_tree_snapshot_path=relocated(state.latest_tree_snapshot_path),
        )
        record = {
            "id": self.identifier,
            "generation": state.generation,
            "parent_id": parent_id,
            "descriptor_path": self.paths.relative_to_work_dir(final / seal_name),
            "descriptor_sha256": file_digest(final / seal_name),
            "checkpoint_path": state.latest_runtime_checkpoint_path,
            "tree_snapshot_path": state.latest_tree_snapshot_path,
        }
        if self.version in (2, 3):
            record.update(
                publication_version=self.version,
                scientific_identity=scientific_identity,
            )
        if self.activation is not None:
            record["activation_request_sha256"] = self.activation["request_sha256"]
        if self.version == 3:
            validate_capsule(self.paths.work_dir, record)
        metadata = dict(state.metadata)
        metadata[PUBLICATION_KEY] = {
            "schema_version": self.version,
            "head_id": self.identifier,
            "generations": [*previous, record],
            **({"evaluator_contract": 1} if self.evaluator_contract else {}),
        }
        from .learning_schedule import on_publication

        return on_publication(
            self.paths,
            self.state,
            replace(state, metadata=metadata),
            self.active_evaluator,
            self.activation,
        )

    def committed(self, state: MorpionBootstrapRunState) -> None:
        """Keep volatile context aligned only after the authoritative state commit."""
        self.state = state
        self.activation = None
        self.validate_state(state)
        self.close()

    def close(self) -> None:
        """Release the predecessor only after validation/commit or failed invocation."""
        self._source_pin.close()

    def retention_roots(self, state: MorpionBootstrapRunState) -> set[Path]:
        """Conservatively pin all committed capsules and legacy provenance inputs."""
        roots = {
            capsule_path(self.paths.work_dir, r["id"], self.version)
            for r in publication_records(state.metadata)
        }
        roots.update({
            self.paths.runtime_checkpoint_dir,
            self.paths.tree_snapshot_dir,
            self.paths.sharded_tree_snapshot_dir,
            self.paths.model_dir,
            self.paths.pipeline_active_model_path,
            self.paths.pipeline_reevaluation_patch_path,
            self.paths.pipeline_reevaluation_cursor_path,
        })
        for receipt in state.metadata["checkpoint_bound_reevaluation_v1"]["receipts"]:
            roots.add(self.paths.work_dir / receipt["checkpoint_path"])
            roots.add(self.paths.work_dir / receipt["model_bundle_path"])
        for record in publication_records(state.metadata):
            roots.add(self.paths.work_dir / record["checkpoint_path"])
            if record["tree_snapshot_path"] is not None:
                roots.add(self.paths.work_dir / record["tree_snapshot_path"])
        return roots

    def prune_uncommitted(self, state: MorpionBootstrapRunState) -> list[str]:
        """Under Growth ownership, quarantine and collect only proven owned orphans."""
        self.validate_state(state)
        persisted = read_object(self.paths.run_state_path)
        require(
            persisted["metadata"][PUBLICATION_KEY] == state.metadata[PUBLICATION_KEY],
            "Run state changed outside Growth ownership before retention.",
        )
        sync_artifact(self.paths.run_state_path, self.paths.work_dir)
        committed_ids = {r["id"] for r in publication_records(state.metadata)}
        removed: list[str] = []
        for directory in [
            self.root / "staging",
            self.root / "objects",
            self.root / "trash",
        ]:
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if path.name in committed_ids:
                    continue
                if not path.is_dir() or path.is_symlink() or len(path.name) != 32:
                    continue
                marker = path / ATTEMPT_MARKER
                if marker.is_file():
                    ownership = read_object(marker)
                    require(
                        (ownership.get("protocol"), ownership.get("id"))
                        == (PUBLICATION_KEY, path.name),
                        "Unrecognized orphan ownership evidence.",
                    )
                elif directory.name != "trash" or any(path.iterdir()):
                    continue
                try:
                    with capsule_lease(
                        self.paths.work_dir,
                        path.name,
                        exclusive=True,
                        nonblocking=True,
                        version=self.version,
                    ):
                        trash = self.root / "trash" / path.name
                        if path != trash:
                            publish_directory_if_absent(
                                path, trash, self.paths.work_dir
                            )
                        _remove_orphan(trash)
                        sync_directory(trash.parent)
                        removed.append(path.name)
                except BlockingIOError:
                    continue
        return removed


def _remove_orphan(path: Path) -> None:
    """Keep ownership evidence until all payload entries have been removed."""
    for entry in sorted(path.iterdir()):
        if entry.name == ATTEMPT_MARKER:
            continue
        require(
            not entry.is_symlink(),
            "Symlinked orphan entry; manual reconciliation required.",
        )
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    sync_directory(path)
    (path / ATTEMPT_MARKER).unlink(missing_ok=True)
    sync_directory(path)
    path.rmdir()


def hashlib_model_key(origin: str, digest: str) -> str:
    """Keep arbitrary model paths out of managed directory names."""
    return hashlib.sha256(json.dumps([origin, digest]).encode()).hexdigest()


def prune_uncommitted_generation_artifacts(paths: MorpionBootstrapPaths) -> list[str]:
    """Collect proven orphans under the same kernel Growth lease as publication."""
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        version = state.metadata[PUBLICATION_KEY]["schema_version"]
        producer = (
            None
            if version == 1
            else publication_records(state.metadata)[-1]["scientific_identity"][
                "producer"
            ]["identity"]
        )
        return DurableGenerationPublication(
            paths,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=version == 3,
        ).prune_uncommitted(state)
