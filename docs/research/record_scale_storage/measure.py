"""Read-only inventory and disposable byte-sharing experiment; no storage engine.

Inventory deliberately uses full dictionaries ONLY for the small offline oracle.
Component copy/compression uses 256 KiB buffers and existing zstd and I/O helpers.
No source inode is linked or modified. No prototype is opened as a committed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter

import zstandard
from anemone.checkpoints.sharded_io import _iter_jsonl_shard_records
from anemone.checkpoints.sharded_manifest import read_sharded_checkpoint_manifest

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
    publication_records,
    sync_artifact,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    iter_shard_rows,
)
from docs.research.atomic_activation.measure import disk
from scripts.benchmark_morpion_scientific_reader import costs, delta


def encode(value: object) -> bytes:
    """Compare complete fields without semantic normalization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def files(root: Path) -> dict:
    """Fingerprint all source files, including physical identity and link counts."""
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            s = p.stat()
            result[p.relative_to(root).as_posix()] = {
                "sha256": file_digest(p),
                "bytes": s.st_size,
                "allocated": s.st_blocks * 512,
                "inode": [s.st_dev, s.st_ino],
                "links": s.st_nlink,
                "mtime_ns": s.st_mtime_ns,
                "ctime_ns": s.st_ctime_ns,
            }
    return result


def allocation(paths: list[Path]) -> dict:
    """Count hardlinks once in the union, never subtract them twice."""
    names = [p for root in paths for p in root.rglob("*") if p.is_file()]
    unique = {(p.stat().st_dev, p.stat().st_ino): p.stat() for p in names}
    return {
        "paths": len(names),
        "inodes": len(unique),
        "logical": sum(s.st_size for s in unique.values()),
        "allocated": sum(s.st_blocks * 512 for s in unique.values()),
    }


def changes(old: dict, new: dict) -> dict:
    """Count exact keyed row/field changes and raw changed-row lower bounds."""
    common = old.keys() & new.keys()
    fields = Counter()
    for key in common:
        for f in old[key].keys() | new[key].keys():
            if (f in old[key]) != (f in new[key]) or old[key].get(f) != new[key].get(f):
                fields[f] += 1
    changed = [v for k, v in new.items() if k not in old or old[k] != v]
    compressor = zstandard.ZstdCompressor().compressobj()
    compressed_bytes = sum(len(compressor.compress(encode(v) + b"\n")) for v in changed)
    compressed_bytes += len(compressor.flush())
    return {
        "new": len(new.keys() - old.keys()),
        "removed": len(old.keys() - new.keys()),
        "changed_existing": sum(old[k] != new[k] for k in common),
        "fields": dict(fields),
        "changed_rows_json_bytes": sum(len(encode(v)) + 1 for v in changed),
        "changed_rows_zstd_bytes_lower_bound": compressed_bytes,
    }


def inventory(root: Path, out: Path) -> dict:
    """Audit small existing generated workspace without mutation or repair."""
    before = files(root)
    state = load_bootstrap_run_state(root / "run_state.json")
    records = publication_records(state.metadata)
    generations, previous, capsule_paths = [], None, []
    for record in records:
        descriptor = validate_capsule(root, record)
        checkpoint = root / record["checkpoint_path"]
        science = (root / record["tree_snapshot_path"]).parent
        capsule = (root / record["descriptor_path"]).parent
        capsule_paths.append(capsule)
        parts = {}
        manifest = read_sharded_checkpoint_manifest(checkpoint / "manifest.json")
        for kind in ("node_shells", "state_payloads", "node_runtime"):
            parts[kind] = {
                str(row["node_id"]): row
                for shard in manifest.shards
                if shard.kind == kind
                for row in _iter_jsonl_shard_records(checkpoint, shard)
            }
        for kind in ("records", "updates"):
            parts[kind] = {
                str(row["node_id"]): row
                for row in iter_shard_rows(
                    science / f"{kind}.jsonl",
                    kind,
                    scientific_generation=record["generation"],
                    closed=True,
                )
            }
        overlap = Counter()
        for key, update in parts["updates"].items():
            runtime = parts["node_runtime"][key]
            values = update["value_contract"]["values"]
            for label, checkpoint_key in (
                ("direct", "direct_value"),
                ("tree", "backed_up_value"),
            ):
                assert values[label] == runtime["evaluation"][checkpoint_key], (
                    key,
                    label,
                )
                overlap[label] += len(encode(values[label]))
            # Current scientific builder explicitly sorts string IDs, whereas
            # checkpoint action-labelled links preserve runtime iteration order.
            assert update["child_ids"] == sorted([
                str(x["child_node_id"]) for x in runtime["linked_children"]
            ])
            assert update["depth"] == parts["node_shells"][key]["depth"]
        cp_files = files(checkpoint)
        one = {
            "generation": record["generation"],
            "nodes": len(parts["records"]),
            "checkpoint": allocation([checkpoint]),
            "science": allocation([science]),
            "capsule": allocation([capsule]),
            "checkpoint_files": cp_files,
            "descriptor_bytes": (root / record["descriptor_path"]).stat().st_size,
            "overlap_value_json_bytes": dict(overlap),
            "science_files": files(science),
            "record_field_example": next(iter(parts["records"].values())),
            "update_field_example": next(iter(parts["updates"].values())),
            "checkpoint_field_examples": {
                k: next(iter(parts[k].values()))
                for k in ("node_shells", "state_payloads", "node_runtime")
            },
            "model_pins": descriptor["model_pins"],
        }
        if previous:
            old, old_files = previous
            one["changes"] = {k: changes(old[k], parts[k]) for k in parts}
            equal = [
                k
                for k, v in cp_files.items()
                if k in old_files and v["sha256"] == old_files[k]["sha256"]
            ]
            one["identical_checkpoint_files"] = equal
            one["identical_checkpoint_allocated"] = sum(
                cp_files[k]["allocated"] for k in equal
            )
        generations.append(one)
        previous = parts, cp_files
    result = {
        "root": str(root),
        "generations": generations,
        "current_plus_two": allocation(capsule_paths[-3:]),
        "all_capsules": allocation(capsule_paths),
        "whole_workspace": allocation([root]),
        "source_files": before,
        "source_unchanged": files(root) == before,
    }
    assert result["source_unchanged"]
    return result


def components(root: Path, out: Path) -> dict:
    """Copy/link exact existing files, then losslessly compress scientific bytes.

    Header/body separation is a component measurement only, not a format proposal
    accepted by any production reader. Each full original byte stream is rebuilt
    and hash-compared. Link sources are NEW private copies, never input artifacts.
    """
    state = load_bootstrap_run_state(root / "run_state.json")
    records = publication_records(state.metadata)
    seen_files, seen_bodies = {}, {}
    results = []
    for record in records:
        generation = record["generation"]
        dest = out / f"g{generation}"
        dest.mkdir(parents=True)
        measurement, before = StageMeasurement(), costs()
        copied, linked, compressed, split = (
            dest / x for x in ("copy", "whole_link", "compressed", "body_link")
        )
        prior_allocation = disk(out)["allocated_bytes"]
        sources = [
            (
                "checkpoint/"
                + p.relative_to(root / record["checkpoint_path"]).as_posix(),
                p,
            )
            for p in (root / record["checkpoint_path"]).rglob("*")
            if p.is_file()
        ]
        science = (root / record["tree_snapshot_path"]).parent
        sources += [("science/" + p.name, p) for p in science.iterdir() if p.is_file()]
        phase = {}
        for mode, target in (("copy", copied), ("whole_link", linked)):
            started = perf_counter()
            for relative, source in sources:
                dest_file = target / relative
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                digest = file_digest(source)
                identity = (relative, digest)
                if mode == "whole_link" and identity in seen_files:
                    os.link(seen_files[identity], dest_file)
                else:
                    shutil.copyfile(source, dest_file)
                    if mode == "whole_link":
                        seen_files[identity] = dest_file
                assert file_digest(dest_file) == digest
            sync_artifact(target, out)
            phase[mode + "_s"] = perf_counter() - started
        started = perf_counter()
        for relative, source in sources:
            target = compressed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".jsonl":
                target = target.with_suffix(".jsonl.zst")
                with (
                    source.open("rb") as inp,
                    target.open("wb") as raw,
                    zstandard.ZstdCompressor().stream_writer(raw) as writer,
                ):
                    shutil.copyfileobj(inp, writer, length=256 * 1024)
                with (
                    target.open("rb") as raw,
                    zstandard.ZstdDecompressor().stream_reader(raw) as reader,
                ):
                    digest = hashlib.file_digest(reader, "sha256").hexdigest()
                assert digest == file_digest(source)
            else:
                shutil.copyfile(source, target)
        sync_artifact(compressed, out)
        phase["compressed_s"] = perf_counter() - started
        started = perf_counter()
        for relative, source in sources:
            target = split / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.name not in ("records.jsonl", "updates.jsonl"):
                shutil.copyfile(source, target)
                continue
            body = target.with_suffix(".body")
            with source.open("rb") as inp:
                header = inp.readline()
                target.with_suffix(".header").write_bytes(header)
                with body.open("wb") as writer:
                    shutil.copyfileobj(inp, writer, length=256 * 1024)
            identity = (relative, file_digest(body))
            if identity in seen_bodies:
                # Only newly created, uncommitted research output is replaced.
                body.unlink()
                os.link(seen_bodies[identity], body)
            else:
                seen_bodies[identity] = body
            digest = hashlib.sha256(header)
            with body.open("rb") as inp:
                while block := inp.read(256 * 1024):
                    digest.update(block)
            assert digest.hexdigest() == file_digest(source)
        sync_artifact(split, out)
        phase["body_link_s"] = perf_counter() - started
        results.append({
            "generation": generation,
            "measurement": measurement.finish(),
            "costs": delta(before),
            "phase": phase,
            "disk": {
                k: allocation([dest / k])
                for k in ("copy", "whole_link", "compressed", "body_link")
            },
            "new_all_variants_allocated": disk(out)["allocated_bytes"]
            - prior_allocation,
        })
    generations = sorted(out.glob("g*"), key=lambda p: int(p.name[1:]))
    totals = {
        k: {
            "all": allocation([g / k for g in generations]),
            "last_three": allocation([g / k for g in generations[-3:]]),
        }
        for k in ("copy", "whole_link", "compressed", "body_link")
    }
    return {
        "root": str(root),
        "generations": results,
        "totals": totals,
        "all_output_disk": disk(out),
        "bytewise_roundtrip": True,
        "state_decode_calls": 0,
        "buffer_bytes": 256 * 1024,
        "note": "Component timing includes hashes/fsync; excludes production validators/commit. All versions stay on disk; no retention implemented.",
    }


def scan(root: Path, out: Path) -> dict:
    """Measure bounded raw scans, and demonstrate corruption on private copies."""
    samples = []
    for iteration in range(5):
        for mode in ("copy", "compressed"):
            start, before = perf_counter(), costs()
            digests = {}
            for name in ("records", "updates"):
                path = root / mode / "science" / (name + ".jsonl")
                if mode == "copy":
                    digests[name] = file_digest(path)
                else:
                    with (
                        path.with_suffix(".jsonl.zst").open("rb") as raw,
                        zstandard.ZstdDecompressor().stream_reader(raw) as stream,
                    ):
                        digests[name] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
            samples.append({
                "iteration": iteration,
                "mode": mode,
                "elapsed_s": perf_counter() - start,
                "costs": delta(before),
                "digests": digests,
            })
    assert all(s["digests"] == samples[0]["digests"] for s in samples)
    # Valid compressed bytes with incorrect scientific content still need a seal.
    corrupt = out / "valid-but-changed.zst"
    corrupt.write_bytes(zstandard.ZstdCompressor().compress(b'{"corrupted":true}\n'))
    with (
        corrupt.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as stream,
    ):
        changed = hashlib.file_digest(stream, "sha256").hexdigest()
    assert changed != samples[0]["digests"]["records"]
    original = root / "copy" / "science" / "records.jsonl"
    a, b = out / "shared-a", out / "shared-b"
    shutil.copyfile(original, a)
    os.link(a, b)
    expected = file_digest(a)
    with a.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    assert file_digest(a) == file_digest(b) != expected
    truncated = out / "truncated.zst"
    source = root / "compressed" / "science" / "records.jsonl.zst"
    with source.open("rb") as inp, truncated.open("wb") as target:
        target.write(inp.read(source.stat().st_size // 2))
    refused = False
    try:
        with (
            truncated.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw) as stream,
        ):
            refused = hashlib.file_digest(stream, "sha256").hexdigest() != expected
    except zstandard.ZstdError:
        refused = True
    assert refused
    return {
        "samples": samples,
        "valid_changed_content_refused_by_expected_hash": True,
        "truncated_refused": True,
        "shared_corruption_affects_both_links": True,
        "all_output_disk": disk(out),
        "note": "Component hash checks, not a production new-format corruption/recovery matrix.",
    }


def main() -> None:
    """Select the isolated storage inventory or component measurement."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("inventory", "components", "scan"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    args.output.mkdir(parents=True, exist_ok=False)
    before, measurement = costs(), StageMeasurement()
    result = globals()[args.mode](args.source, args.output)
    result.update(observation=measurement.finish(), costs=delta(before))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            "mode": args.mode,
            "output": str(args.output),
            "observation": result["observation"],
        })
    )


if __name__ == "__main__":
    main()
