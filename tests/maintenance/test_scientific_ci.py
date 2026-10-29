"""Installed scientific identity must follow an explicit immutable build candidate."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def git(root: Path, *args: str) -> str:
    """Run Git against a private fixture repository, never the real checkout."""
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Identity test",
            "-c",
            "user.email=test@example.invalid",
            *args,
        ],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, Path, Path, str]:
    """Provide a real committed source tree and a matching isolated installation."""
    path = Path(__file__).resolve().parents[2] / ".github/scripts/scientific_ci.py"
    spec = importlib.util.spec_from_file_location("scientific_ci_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "checkout"
    source = root / "src/chipiron"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('"""Scientific fixture."""\n')
    (source / "science.py").write_text("VALUE = 1\n")
    (source / "science.pyi").write_text("VALUE: int\n")
    (source / "py.typed").touch()
    (source / "README.md").write_text("Unpackaged source documentation.\n")
    (source / "ignored.yaml").write_text("unpackaged: true\n")
    for name in (
        "data/declared.yaml",
        "data/nested/manifest.yaml",
        "web/index.html",
        "weights/model.pt",
    ):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("committed package payload\n")
    (root / "pyproject.toml").write_text(
        "[tool.setuptools]\ninclude-package-data = true\n"
        "[tool.setuptools.package-data]\n"
        'chipiron = ["py.typed", "data/*.yaml", "web/*.html", "weights/*.pt"]\n'
    )
    (root / "MANIFEST.in").write_text(
        "recursive-include src/chipiron/data *.yaml\n"
        "include tests/data/fixture.json\n"
        "prune src/chipiron/**/outputs\n"
        "global-exclude *.pyc\n"
    )
    (root / ".github").mkdir()
    (root / ".github/ci-source-contracts.json").write_text("{}\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Frozen science")
    reference = git(root, "rev-parse", "HEAD")
    site = tmp_path / "site-packages"
    shutil.copytree(
        source,
        site / "chipiron",
        ignore=shutil.ignore_patterns("README.md", "ignored.yaml"),
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.sysconfig, "get_path", lambda _name: str(site))
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(site / "chipiron/__init__.py")),
    )
    return module, root, site, reference


@pytest.mark.fast
@pytest.mark.parametrize("editable", [False, True])
def test_candidate_allows_committed_evolution_but_frozen_replay_rejects_it(
    identity: tuple[ModuleType, Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
    editable: bool,
) -> None:
    """Replay stays frozen while a new installed source must match its own commit."""
    module, root, site, frozen = identity
    contract = {"packages": {}, "chipiron_reference": frozen}
    if editable:
        monkeypatch.setattr(
            module.importlib.util,
            "find_spec",
            lambda _name: SimpleNamespace(
                origin=str(root / "src/chipiron/__init__.py")
            ),
        )
    module.verify(contract, editable_chipiron=editable)
    report_path = root / ".ci-reports/scientific-installed-identity.json"
    assert json.loads(report_path.read_text())["chipiron"]["mode"] == "frozen-replay"
    for base in (root / "src", site):
        (base / "chipiron/science.py").write_text("VALUE = 2\n")
        (base / "chipiron/new.py").write_text("CAPABILITY = True\n")
    git(root, "add", "src")
    git(root, "commit", "-qm", "Candidate evolution")
    candidate = git(root, "rev-parse", "HEAD")
    module.verify(contract, candidate=candidate, editable_chipiron=editable)
    report = json.loads(report_path.read_text())["chipiron"]
    assert report["reference"] == candidate
    assert report["mode"] == "build-candidate"
    assert report["source_members"] == 5
    assert report["package_members"] == 9
    with pytest.raises(RuntimeError, match="source membership mismatch"):
        module.verify(contract, editable_chipiron=editable)


@pytest.mark.fast
@pytest.mark.parametrize("location", ["installed", "checkout"])
@pytest.mark.parametrize(
    "fault", ["changed", "missing", "unexpected", "native", "bytecode"]
)
@pytest.mark.parametrize("candidate_mode", [False, True])
def test_candidate_rejects_source_drift(
    identity: tuple[ModuleType, Path, Path, str],
    location: str,
    fault: str,
    candidate_mode: bool,
) -> None:
    """Changing either side after the immutable commit invalidates the build."""
    module, root, site, reference = identity
    source = (site if location == "installed" else root / "src") / "chipiron"
    if fault == "changed":
        (source / "science.py").write_text("VALUE = 999\n")
    elif fault == "missing":
        (source / "science.pyi").unlink()
    elif fault == "native":
        (source / "unexpected.so").touch()
    elif fault == "bytecode":
        (source / "unexpected.pyc").touch()
    else:
        (source / "unexpected.py").write_text("UNREVIEWED = True\n")
    with pytest.raises(
        RuntimeError, match=r"Chipiron (source membership mismatch|differs)"
    ):
        module.verify(
            {"packages": {}, "chipiron_reference": reference},
            candidate=reference if candidate_mode else None,
        )


@pytest.mark.fast
@pytest.mark.parametrize("location", ["installed", "checkout"])
@pytest.mark.parametrize("fault", ["changed", "missing", "unexpected"])
@pytest.mark.parametrize(
    "name", ["data/nested/manifest.yaml", "web/index.html", "weights/model.pt"]
)
def test_candidate_rejects_package_payload_drift(
    identity: tuple[ModuleType, Path, Path, str],
    location: str,
    fault: str,
    name: str,
) -> None:
    """Declared data and recursively included manifest payloads are source identities."""
    module, root, site, reference = identity
    source = (site if location == "installed" else root / "src") / "chipiron"
    payload = source / name
    if fault == "changed":
        payload.write_bytes(b"changed science payload")
    elif fault == "missing":
        payload.unlink()
    else:
        payload.with_name("unexpected" + payload.suffix).touch()
    with pytest.raises(
        RuntimeError, match=r"Chipiron (source membership mismatch|differs)"
    ):
        module.verify({"packages": {}}, candidate=reference)


@pytest.mark.fast
def test_candidate_rejects_undeclared_installed_payload(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """Installed package membership includes every payload, even unrecognized suffixes."""
    module, _, site, reference = identity
    (site / "chipiron/unexpected.data").touch()
    with pytest.raises(RuntimeError, match="source membership mismatch"):
        module.verify({"packages": {}}, candidate=reference)


@pytest.mark.fast
@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "MANIFEST.in",
        "setup.py",
        "setup.cfg",
        ".github/ci-source-contracts.json",
    ],
)
@pytest.mark.parametrize("fault", ["changed", "missing"])
def test_candidate_binds_build_input_bytes_and_membership(
    identity: tuple[ModuleType, Path, Path, str], name: str, fault: str
) -> None:
    """Both modifying packaging declarations and adding alternate build inputs fail."""
    module, root, _, reference = identity
    path = root / name
    if fault == "missing":
        if not path.exists():
            path.write_text("committed build input\n")
            git(root, "add", name)
            git(root, "commit", "-qm", "Declare alternate build input")
            reference = git(root, "rev-parse", "HEAD")
        path.unlink()
    else:
        path.write_text("uncommitted build input\n")
    with pytest.raises(
        RuntimeError, match=r"Chipiron (build input membership|differs)"
    ):
        module.verify({"packages": {}}, candidate=reference)


@pytest.mark.fast
def test_candidate_rejects_unknown_committed_manifest_rules(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """Unsupported packaging syntax cannot silently weaken payload verification."""
    module, root, _, _ = identity
    (root / "MANIFEST.in").write_text("graft src/chipiron\n")
    git(root, "add", "MANIFEST.in")
    git(root, "commit", "-qm", "Unsupported package membership declaration")
    with pytest.raises(RuntimeError, match="Unsupported Chipiron manifest rule"):
        module.verify({"packages": {}}, candidate=git(root, "rev-parse", "HEAD"))


@pytest.mark.fast
def test_frozen_replay_uses_historical_payload_declarations(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """Migration metadata may evolve while historical package payloads remain frozen."""
    module, root, site, reference = identity
    (root / "pyproject.toml").write_text("# Different migration build metadata\n")
    (root / ".github/ci-source-contracts.json").write_text(
        "{}\n# Later identity contract\n"
    )
    contract = {"packages": {}, "chipiron_reference": reference}
    module.verify(contract)
    (site / "chipiron/data/nested/manifest.yaml").write_text("changed frozen payload\n")
    with pytest.raises(RuntimeError, match="differs from reviewed Git source"):
        module.verify(contract)


@pytest.mark.fast
@pytest.mark.parametrize("reference", ["HEAD", "main", "a6987792", "", "A" * 40])
def test_candidate_requires_full_immutable_sha(
    identity: tuple[ModuleType, Path, Path, str], reference: str
) -> None:
    """Mutable refs and abbreviations cannot serve as package source identity."""
    module, _, _, _ = identity
    with pytest.raises(RuntimeError, match="full immutable commit SHA"):
        module.verify({"packages": {}}, candidate=reference)


@pytest.mark.fast
def test_candidate_must_identify_checkout_head(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """An unrelated or stale commit cannot attest to a build checkout."""
    module, root, _, frozen = identity
    git(root, "commit", "--allow-empty", "-qm", "Different build candidate")
    with pytest.raises(RuntimeError, match="differs from checkout HEAD"):
        module.verify({"packages": {}}, candidate=frozen)


@pytest.mark.fast
def test_candidate_rejects_tree_identity(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """A full tree hash is not an immutable build commit."""
    module, root, _, _ = identity
    with pytest.raises(subprocess.CalledProcessError):
        module.verify({"packages": {}}, candidate=git(root, "rev-parse", "HEAD^{tree}"))


@pytest.mark.fast
def test_candidate_rejects_unknown_commit(
    identity: tuple[ModuleType, Path, Path, str],
) -> None:
    """A syntactically valid but unavailable identity cannot silently fall back."""
    module, _, _, _ = identity
    with pytest.raises(subprocess.CalledProcessError):
        module.verify({"packages": {}}, candidate="0" * 40)


@pytest.mark.fast
@pytest.mark.parametrize("command", ["verify-candidate", "verify-static-candidate"])
def test_candidate_cli_requires_revision(
    identity: tuple[ModuleType, Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Candidate CLI commands cannot implicitly attest to mutable HEAD."""
    module, root, _, reference = identity
    contract_path = root / ".github/ci-source-contracts.json"
    contract_path.parent.mkdir(exist_ok=True)
    contract_path.write_text(
        json.dumps({
            "packages": {},
            "chipiron_reference": reference,
            "python": module.platform.python_version(),
        })
    )
    monkeypatch.setattr(module.sys, "argv", ["scientific_ci.py", command])
    with pytest.raises(SystemExit, match="FULL_COMMIT_SHA"):
        module.main()


@pytest.mark.fast
def test_candidate_rejects_import_override(
    identity: tuple[ModuleType, Path, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching wheel bytes do not authorize importing a different source tree."""
    module, root, _, reference = identity
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(root / "src/chipiron/__init__.py")),
    )
    with pytest.raises(RuntimeError, match="Unexpected Chipiron origin"):
        module.verify({"packages": {}}, candidate=reference)


@pytest.mark.fast
@pytest.mark.parametrize(
    "fault",
    ["changed", "missing", "unexpected", "version", "origin", "installed-origin"],
)
def test_candidate_retains_dependency_identity_checks(
    identity: tuple[ModuleType, Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Forward Chipiron development never relaxes frozen dependency identities."""
    module, root, site, reference = identity
    dependency = site / "dependency/__init__.py"
    dependency.parent.mkdir()
    dependency.write_text("FROZEN = True\n")
    contract = {
        "packages": {
            "dependency": {
                "version": "1.0",
                "module": "dependency",
                "files": {"dependency/__init__.py": module.digest(dependency)},
            }
        }
    }
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(site / name / "__init__.py")),
    )
    module.verify(contract, candidate=reference)
    if fault == "changed":
        dependency.write_text("FROZEN = False\n")
    elif fault == "missing":
        dependency.unlink()
    elif fault == "unexpected":
        (dependency.parent / "unexpected.py").touch()
    elif fault == "version":
        monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "2.0")
    else:
        monkeypatch.setattr(
            module.importlib.util,
            "find_spec",
            lambda _name: SimpleNamespace(
                origin=str(
                    (
                        site / "chipiron"
                        if fault == "installed-origin"
                        else root / "dependency"
                    )
                    / "__init__.py"
                )
            ),
        )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        module.verify(contract, candidate=reference)
