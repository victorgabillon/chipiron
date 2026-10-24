"""Unmet exact-order prerequisite: intentionally failing, never a passing CI gate.

Run explicitly as documented in the report. These assertions do not establish
that changing runtime order changes scientific outcomes; that proof is unresolved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from docs.research.atomic_model_activation.probe import compare
from docs.research.model_activation.probe import graph

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("checkpoint_format", ["sharded", "json-zst"])
def test_exact_checkpoint_continuation_gate(
    tmp_path: Path, checkpoint_format: str
) -> None:
    """Retain actual field differences instead of normalizing the assertion."""
    live = graph(1000)
    live._args = replace(live._args, runtime_checkpoint_format=checkpoint_format)
    source = tmp_path / (
        "checkpoint" if checkpoint_format == "sharded" else "checkpoint.json.zst"
    )
    live.save_checkpoint(source)
    restored = AnemoneMorpionSearchRunner(live._args)
    restored.load_or_create(
        source, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    payloads = [
        checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(i._runtime, state_codec=i._state_codec)
        )
        for i in (live, restored)
    ]
    before, after = payloads
    differences = [
        {
            "node_id": a["node_id"],
            "fields": {
                key: {"before": a[key], "after": b[key]}
                for key in a
                if a[key] != b[key]
            },
        }
        for a, b in zip(before["tree"]["nodes"], after["tree"]["nodes"], strict=True)
        if a != b
    ]
    observation = compare(live, restored)
    observation["checkpoint_node_differences"] = differences
    observation["checkpoint_other_fields_equal"] = {
        k: before[k] == after[k] for k in before if k != "tree"
    }
    science = []
    for name, instance in (("live", live), ("restored", restored)):
        directory = tmp_path / name
        instance.export_closed_scientific_snapshot(
            directory,
            generation=1,
            producer={
                "revision": "atomic-activation-prerequisite-probe",
                "evaluator": "same-default-evaluator",
                "config_sha256": "0" * 64,
            },
        )
        science.append({
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*")
            if p.is_file()
        })
    observation["closed_scientific_file_hashes"] = science
    observation["closed_scientific_files_equal"] = science[0] == science[1]
    (tmp_path / "observation.json").write_text(json.dumps(observation, indent=2) + "\n")
    assert observation["scientific_hashes"][0] == observation["scientific_hashes"][1]
    assert observation["search_rng_equal"]
    assert observation["rollout_rng_equal"]
    assert observation["closed_scientific_files_equal"]
    assert not differences and not observation["order_differences"], (
        f"Exact checkpoint prerequisite remains unresolved: {len(differences)} "
        f"node records and {len(observation['order_differences'])} runtime link orders differ"
    )
