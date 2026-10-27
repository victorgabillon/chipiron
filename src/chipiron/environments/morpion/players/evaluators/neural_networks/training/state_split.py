"""Stable held-out membership for evolving, accepted Morpion datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
)
from chipiron.environments.morpion.learning import (
    decode_morpion_state_ref_payload,
    iter_morpion_supervised_rows_from_path,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.learning import MorpionSupervisedRow

    from .args import MorpionTrainingArgs

POLICY = "canonical_state_sha256_v1"


def canonical_identity(row: MorpionSupervisedRow) -> str:
    """Use the game's rooted D4 identity, independent of node ID and replay order."""
    state = decode_morpion_state_ref_payload(row.state_ref_payload)
    payload = json.dumps(
        [state.variant.value, state.canonical_tag], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StateSplit:
    """Compact indices plus exact held-out population and target-contract evidence."""

    validation_indices: frozenset[int]
    row_count: int
    validation_population_sha256: str


@lru_cache(maxsize=2)
def _split(
    path: str, digest: str, fraction: float, seed: int, max_rows: int | None
) -> StateSplit:
    """Reuse one immutable input split across candidates without retaining states."""
    del digest  # In the cache key: a changed file can never reuse old membership.
    indices: set[int] = set()
    held_out: list[str] = []
    count = 0
    for index, row in enumerate(iter_morpion_supervised_rows_from_path(path)):
        if max_rows is not None and index >= max_rows:
            break
        count += 1
        identity = canonical_identity(row)
        bucket = hashlib.sha256(f"{POLICY}:{seed}:{identity}".encode()).digest()
        if int.from_bytes(bucket, "big") < int(fraction * 2**256):
            indices.add(index)
            held_out.append(
                json.dumps(
                    [
                        identity,
                        row.target_value,
                        row.metadata.get("target_source"),
                        row.metadata.get("target_confidence"),
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    population = hashlib.sha256("\n".join(sorted(held_out)).encode()).hexdigest()
    return StateSplit(frozenset(indices), count, population)


def prepare_state_split(
    args: MorpionTrainingArgs, max_rows: int | None = None
) -> tuple[MorpionTrainingArgs, dict[str, object]]:
    """Bind every training/evaluation kernel to the same stable membership."""
    if not args.stable_validation_split:
        return args, {}
    result = _split(
        str(args.dataset_file),
        file_digest(Path(args.dataset_file)),
        args.validation_fraction,
        args.validation_seed,
        max_rows,
    )
    return replace(args, validation_indices=result.validation_indices), {
        "policy": POLICY,
        "seed": args.validation_seed,
        "fraction": args.validation_fraction,
        "row_count": result.row_count,
        "train_count": result.row_count - len(result.validation_indices),
        "validation_count": len(result.validation_indices),
        "validation_population_sha256": result.validation_population_sha256,
    }
