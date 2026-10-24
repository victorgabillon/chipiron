"""Strict current checkpoint boundary and the unchanged continuation witness."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    write_checkpoint_json_payload,
)

from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
)

if TYPE_CHECKING:
    from pathlib import Path


def graph() -> AnemoneMorpionSearchRunner:
    """Small legal graph, including transpositions and nontrivial selector state."""
    live = AnemoneMorpionSearchRunner(AnemoneMorpionSearchRunnerArgs(random_seed=4))
    live.load_or_create(None, None)
    live.grow(3)
    return live


@pytest.mark.parametrize("layout", ["json-zst", "sharded"])
def test_format3_continuation(tmp_path: Path, layout: str) -> None:
    """Compare complete decoded checkpoint state before and after continuation."""
    live = graph()
    live._args = replace(live._args, runtime_checkpoint_format=layout)
    path = tmp_path / ("checkpoint" if layout == "sharded" else "checkpoint.json.zst")
    live.save_checkpoint(path)
    restored = AnemoneMorpionSearchRunner(live._args)
    restored.load_or_create(path, None)
    for _ in range(3):
        assert checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(
                live._runtime, state_codec=live._state_codec
            )
        ) == checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(
                restored._runtime, state_codec=restored._state_codec
            )
        )
        live.grow(1)
        restored.grow(1)


@pytest.mark.parametrize("layout", ["json-zst", "sharded"])
def test_format2_is_rejected_without_rewriting(tmp_path: Path, layout: str) -> None:
    """Only a private generated fixture is downgraded; restoration must refuse."""
    assert CHECKPOINT_FORMAT_VERSION == 3
    live = graph()
    live._args = replace(live._args, runtime_checkpoint_format=layout)
    path = tmp_path / ("checkpoint" if layout == "sharded" else "checkpoint.json.zst")
    live.save_checkpoint(path)
    if layout == "sharded":
        manifest = path / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["checkpoint_format_version"] = 2
        manifest.write_text(json.dumps(payload))
        original = manifest.read_bytes()
    else:
        payload, _ = load_checkpoint_json_payload(path)
        payload["format_version"] = 2
        write_checkpoint_json_payload(payload, path)
        original = path.read_bytes()
    restored = AnemoneMorpionSearchRunner(live._args)
    with pytest.raises(ValueError, match=r"(?i)(format|version)"):
        restored.load_or_create(path, None)
    assert (manifest if layout == "sharded" else path).read_bytes() == original
