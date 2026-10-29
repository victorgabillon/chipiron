"""Exercise archival integrity and force-added output rejection in scratch Git."""

import csv
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def load_script(name: str) -> ModuleType:
    """Load standalone tools without colliding with tests/scripts."""
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_script("check_generated_artifacts")
archive = load_script("research_evidence")


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    """Run an isolated Git fixture command."""
    return subprocess.check_output(["git", "-C", str(root), *args], input=data)


def test_guard_checks_index_and_allows_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignored outputs still fail when force-added; JSON/YAML fixtures pass."""
    git(tmp_path, "init", "-q")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    for name in ("tests/data/example.json", "tests/data/example.yaml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        git(tmp_path, "add", name)
    assert guard.main() == 0
    (tmp_path / ".gitignore").write_text("evidence/\n.ci-reports/\n__pycache__/\n")
    for name in (
        "docs/research/example/evidence/output.json",
        ".ci-reports/results.xml",
        "src/chipiron/__pycache__/cached.pyc",
        "morpion_runs/run/rows.jsonl",
        "src/chipiron/scripts/example/outputs/run.yaml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated")
        git(tmp_path, "add", "-f", name)
        assert guard.main() == 1
        git(tmp_path, "rm", "-f", name)
        assert guard.main() == 0


@pytest.fixture
def archived_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, str], bytes]:
    """Commit one binary research output with its real Git and SHA-256 identity."""
    git(tmp_path, "init", "-q")
    name = "docs/research/example/evidence/output.gz"
    content = b"\x00\xff\r\noriginal bytes\x00"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    git(tmp_path, "add", name)
    git(
        tmp_path,
        "-c",
        "user.name=Archive test",
        "-c",
        "user.email=archive@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    row = {
        "source_commit": git(tmp_path, "rev-parse", "HEAD").decode().strip(),
        "path": name,
        "git_blob": git(tmp_path, "rev-parse", f"HEAD:{name}").decode().strip(),
        "bytes": str(len(content)),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest = tmp_path / "manifest.csv"
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(archive, "MANIFEST", manifest)
    path.unlink()
    return manifest, row, content


@pytest.mark.parametrize("damage", [None, "git_blob", "bytes", "sha256"])
def test_restore_checks_identity_and_refuses_overwrite(
    archived_file: tuple[Path, dict[str, str], bytes], damage: str | None
) -> None:
    """Reconstruct binary bytes and reject independently corrupted manifest fields."""
    manifest, row, content = archived_file
    if damage:
        row[damage] = "0" if damage == "bytes" else "0" * len(row[damage])
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if damage:
        with pytest.raises(ValueError, match="mismatch"):
            archive.restore()
        return
    destination = manifest.parent / "restored"
    assert archive.restore(destination) == (1, len(content))
    assert (destination / row["path"]).read_bytes() == content
    with pytest.raises(FileExistsError):
        archive.restore(destination)
