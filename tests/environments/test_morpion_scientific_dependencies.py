"""Exact released dependency qualification without adopting older provenance."""

import importlib.metadata
import json
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    read_closed_manifest,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    ScientificExportError,
    _producer,
)
from tests.environments.test_morpion_closed_scientific_snapshot import closed
from tests.environments.test_morpion_sharded_training_v2 import PRODUCER

if TYPE_CHECKING:
    from pathlib import Path


def test_released_producer_records_exact_identity(tmp_path: Path) -> None:
    """The real release is recorded alongside unchanged codec/API identity proofs."""
    identity = _producer(PRODUCER)
    assert identity["packages"]["algorhino-anemone"] == "0.4.0"
    assert identity["packages"]["atomheart"] == "0.3.1"
    assert identity["packages"]["valanga"] == "0.3.0"
    assert identity["identity"] == PRODUCER
    assert identity["api_sha256"]
    path = closed(tmp_path / "source")
    assert read_closed_manifest(path)["producer"] == identity
    assert load_closed_scientific_snapshot(path).provenance["producer"] == identity


@pytest.mark.parametrize(
    "package,version",
    [
        ("algorhino-anemone", "0.3.1"),
        ("algorhino-anemone", "0.4.1"),
        ("algorhino-anemone", "0.5.0"),
        ("algorhino-anemone", "0.4.0rc1"),
        ("algorhino-anemone", "0.4.0+unqualified"),
        ("atomheart", "0.3.0"),
        ("valanga", "0.4.0"),
    ],
)
def test_unqualified_dependency_refused_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, version: str
) -> None:
    """Other versions cannot publish even if their imports happen to work."""
    installed_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: version if name == package else installed_version(name),
    )
    destination = tmp_path / "refused"
    with pytest.raises(
        ScientificExportError, match="unsupported scientific dependency"
    ):
        closed(destination)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["old-package", "api", "state-contract", "config"])
def test_previous_or_changed_provenance_is_not_reinterpreted(
    tmp_path: Path, change: str
) -> None:
    """Only a newly generated fixture is changed; readers never rewrite its identity."""
    path = closed(tmp_path / "source")
    manifest = json.loads(path.read_text())
    producer = manifest["producer"]
    if change == "old-package":
        # Reproduce an older recorded package identity without reading history.
        producer["packages"]["algorhino-anemone"] = "0.3.1"
    elif change == "api":
        key = next(iter(producer["api_sha256"]))
        producer["api_sha256"][key] = "0" * 64
    elif change == "state-contract":
        producer["state_ref_contract"] = "unknown"
    else:
        producer["identity"]["config_sha256"] = "invalid"
    path.write_text(json.dumps(manifest))
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    for reader in (read_closed_manifest, load_closed_scientific_snapshot):
        with pytest.raises(ScientificExportError, match=r"identity|config SHA256"):
            reader(path)
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
