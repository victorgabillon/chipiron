"""Real runner streaming, write-once and complete format-3 parity regressions."""

from dataclasses import replace
from pathlib import Path

import pytest
from anemone.checkpoints import (
    build_node_payloads,
    build_search_checkpoint_payload,
    write_sharded_search_checkpoint,
)

from chipiron.environments.morpion.bootstrap.runtime import runner as module
from tests.environments.test_morpion_checkpoint_format3 import graph


def files(root: Path) -> dict[str, bytes]:
    """Read complete tiny test artifacts without normalizing any field."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.fast
@pytest.mark.parametrize("batch", [2, 256, 1024, 4096])
def test_stream_never_materializes_and_matches_every_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    """Payload-only diagnostics cannot silently rebuild the duplicate tree."""
    live = graph()
    live._args = replace(live._args, runtime_checkpoint_format="sharded")
    payload = build_search_checkpoint_payload(
        live._runtime, state_codec=live._state_codec
    )
    write_sharded_search_checkpoint(
        payload, tmp_path / "old", layout="split", node_count_per_shard=batch
    )
    del payload

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Streaming save attempted complete payload materialization")

    monkeypatch.setattr(module, "CHECKPOINT_NODES_PER_SHARD", batch, raising=False)
    for name in (
        "build_search_checkpoint_payload",
        "_checkpoint_node_counts",
        "checkpoint_selector_state_fields",
    ):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(type(live._runtime), "_all_nodes_in_tree_order", forbidden)
    live.save_checkpoint(tmp_path / "new")
    assert files(tmp_path / "old") == files(tmp_path / "new")
    metrics = live._latest_checkpoint_metrics
    assert metrics["node_count"] == live.current_tree_size()
    assert (
        metrics["payload_build_s"]
        is metrics["anchor_count"]
        is metrics["delta_count"]
        is None
    )
    before = files(tmp_path / "new")
    with pytest.raises(FileExistsError, match="empty"):
        live.save_checkpoint(tmp_path / "new")
    assert files(tmp_path / "new") == before


def test_serializer_failure_leaves_private_partial_and_requires_new_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted attempt cannot overwrite either a complete or partial save."""
    live = graph()
    live._args = replace(live._args, runtime_checkpoint_format="sharded")
    live.save_checkpoint(tmp_path / "committed")
    before = files(tmp_path / "committed")
    original = build_node_payloads._build_node_payload
    count = 0

    def fail(**kwargs: object) -> object:
        nonlocal count
        count += 1
        if count == 4:
            message = "injected serialization failure"
            raise RuntimeError(message)
        return original(**kwargs)

    with monkeypatch.context() as local:
        local.setattr(module, "CHECKPOINT_NODES_PER_SHARD", 2)
        local.setattr(build_node_payloads, "_build_node_payload", fail)
        with pytest.raises(RuntimeError, match="injected serialization"):
            live.save_checkpoint(tmp_path / "partial")
    partial = files(tmp_path / "partial")
    assert partial and "manifest.json" not in partial
    with pytest.raises(FileExistsError, match="empty"):
        live.save_checkpoint(tmp_path / "partial")
    assert files(tmp_path / "partial") == partial
    assert files(tmp_path / "committed") == before
    restored = module.AnemoneMorpionSearchRunner(live._args)
    with pytest.raises((ValueError, FileNotFoundError)):
        restored.load_or_create(tmp_path / "partial", None)
    live.save_checkpoint(tmp_path / "fresh")
    assert files(tmp_path / "fresh") == before


@pytest.mark.parametrize("phase", ["serialize", "shard", "manifest"])
def test_stream_error_cannot_publish_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Real accepted Training stays on M1 until a fresh complete M2 commit."""
    from anemone.checkpoints import sharded_encode

    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        publication_records,
        validate_capsule,
    )
    from chipiron.environments.morpion.bootstrap.model_activation import (
        activate_model,
        prepare_model_activation_request,
    )
    from chipiron.environments.morpion.bootstrap.run_state import (
        load_bootstrap_run_state,
    )
    from tests.environments.morpion_activation_helpers import setup_activation
    from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, initial = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    before = paths.run_state_path.read_bytes()
    owner, name = {
        "serialize": (build_node_payloads, "_build_node_payload"),
        "shard": (sharded_encode, "_write_jsonl_record_shard"),
        "manifest": (sharded_encode, "write_sharded_checkpoint_manifest"),
    }[phase]
    original = getattr(owner, name)
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if phase == "serialize" and calls < 4:
            return original(*args, **kwargs)
        if phase == "shard":
            original(*args, **kwargs)
        message = "injected private checkpoint error"
        raise RuntimeError(message)

    with monkeypatch.context() as local:
        local.setattr(module, "CHECKPOINT_NODES_PER_SHARD", 2)
        local.setattr(owner, name, fail)
        with pytest.raises(RuntimeError, match="injected private"):
            activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before
    recovered = load_bootstrap_run_state(paths.run_state_path)
    assert recovered.generation == initial.generation
    validate_capsule(paths.work_dir, publication_records(recovered.metadata)[-1])
    after = activate_model(paths, request)
    assert after.generation == 2
    assert activate_model(paths, request) == after
