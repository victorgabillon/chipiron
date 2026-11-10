"""Installed independent fork acceptance, with fresh deterministic CPU processes."""

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import fork
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.fork_origin import validate
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.test_morpion_fork_contracts import (
    equal_state,
    initialize,
    invoke,
    semantic,
)


def fresh(
    function: str, *args: Path | str, expected: int = 0
) -> subprocess.CompletedProcess[str]:
    """Use the tracked fresh-process numerical contract, never a runtime plugin."""
    code = (
        "import sys,torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); "
        "from tests.environments import test_morpion_fork as t; "
        f"t.{function}(*sys.argv[1:])"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def hashes(root: Path) -> dict[str, str]:
    """Scientific and operational bytes, excluding read atime."""
    return {str(p.relative_to(root)): file_digest(p) for p in fork.inventory(root)}


def script(root: Path, mode: str) -> Path:
    """Deterministic normal supervisor work, with a generous separate native cap."""
    path = root.parent / f"{root.name}-{mode}.sh"
    path.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " -B "
        + shlex.quote(str(Path(__file__).with_name("fork_workload.py")))
        + ' "$MORPION_WORK_DIR" '
        + shlex.quote(mode)
        + "\n"
    )
    return path


def make_parent(raw: str, *, nodes: int = 5000, mode: str = "parent") -> None:
    """Produce a positively closed actual native parent without authority edits."""
    root = Path(raw)
    initialize(root, nodes=nodes)
    invoke(root, script(root, mode))


def cli(
    parent: Path,
    target: Path,
    *,
    dry: bool = False,
    expected: str | None = None,
    nodes: int = 10000,
) -> dict[str, Any]:
    """Exercise the public installed command in a fresh numerical owner."""
    command = [
        sys.executable,
        "-B",
        "-m",
        "chipiron.environments.morpion.bootstrap.operator_cli",
        "fork",
        "--from",
        str(parent),
        "--to",
        str(target),
        "--max-elapsed-seconds",
        "300",
        "--max-tree-nodes",
        str(nodes),
    ]
    if dry:
        command.append("--dry-run")
    if expected is not None:
        command += ["--expect-plan", expected]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def vertical(raw: str) -> None:
    """Full public copy, origin/Adam, next checkpoint and ancestor independence."""
    root = Path(raw)
    parent, child = root / "parent", root / "child"
    make_parent(str(parent))
    before, expected = hashes(parent), semantic(parent)
    from unittest.mock import patch

    with (
        patch("torch.load", side_effect=AssertionError("preview loaded tensors")),
        patch("torch.cuda.init", side_effect=AssertionError("preview initialized GPU")),
    ):
        preview = fork.plan_fork(parent, child, 300, 10000)
    plan = cli(parent, child, dry=True)
    assert plan == preview
    assert before == hashes(parent) and not child.exists()
    result = cli(parent, child, expected=plan["plan_sha256"])
    assert result["action"] == "published"
    equal_state(expected, semantic(child))
    assert before == hashes(parent)
    assert not {
        (p.stat().st_dev, p.stat().st_ino) for p in fork.inventory(parent)
    }.intersection((p.stat().st_dev, p.stat().st_ino) for p in fork.inventory(child))
    state = load_bootstrap_run_state(child / "run_state.json")
    origin = validate(
        MorpionBootstrapPaths.from_work_dir(child),
        load_bootstrap_config(child / "bootstrap_config.json"),
        state,
        verify=True,
    )
    assert origin is not None
    assert origin["budget"] == {"seconds": 300, "nodes": 10000}
    assert origin["incumbent"] != plan["parent"]["parents"]["linear_a"]["model"]
    for name, seed in origin["candidate_seeds"].items():
        original = parent / plan["parent"]["parents"][name]["model"]["path"]
        assert artifact_digest(original) == artifact_digest(
            child / seed["model"]["path"]
        )
    invoke(child, script(child, "child"))
    local = load_bootstrap_run_state(child / "run_state.json")
    rounds = [r for r in publication_records(local.metadata) if training.ACCEPTED in r]
    assert len(rounds) == 2 and local.generation > 1
    proposals = [
        training._proposal(
            MorpionBootstrapPaths.from_work_dir(child), r[training.ACCEPTED]
        )
        for r in rounds
    ]
    assert all(
        p["kind"] == "fork_seed" for p in proposals[0]["admission"]["parents"].values()
    )
    assert all(
        p.get("kind") != "fork_seed"
        for p in proposals[1]["admission"]["parents"].values()
    )
    for model in proposals[0]["models"].values():
        continuation = json.loads(
            (child / model["path"] / "morpion_manifest.json").read_text()
        )["metadata"]["continuation"]
        assert (
            continuation["optimizer_state_loaded"]
            and continuation["starting_optimizer_step_count"] > 0
        )
    assert before == hashes(parent)
    child_before = hashes(child)
    invoke(parent, script(parent, "child"))
    assert child_before == hashes(child) and before != hashes(parent)
    # A fork of a fork consumes only its immediate, independently verified cut.
    grandchild = root / "grandchild"
    cli(child, grandchild)
    expected_grandchild = semantic(child)
    parent.rename(root / "parent-unavailable")
    child.rename(root / "child-unavailable")
    equal_state(expected_grandchild, semantic(grandchild))
    invoke(grandchild, script(grandchild, "child"))
    assert (
        cli(
            root / "child",
            grandchild,
            expected=json.loads((grandchild / "genesis/fork/origin.json").read_text())[
                "plan_sha256"
            ],
        )["action"]
        == "already-published"
    )


@pytest.mark.exhaustive
def test_complete_independent_fork(tmp_path: Path) -> None:
    """Public preparation and ordinary native destination work preserve the cut."""
    fresh("vertical", tmp_path)


def node_completed(raw: str) -> None:
    """Retain a genuinely completed source cap while its new child grows."""
    root = Path(raw)
    parent, child = root / "parent", root / "child"
    make_parent(str(parent), nodes=2, mode="node")
    state = load_bootstrap_run_state(parent / "run_state.json")
    assert state.tree_size_at_last_save >= 2
    before = hashes(parent)
    cli(parent, child, nodes=10000)
    imported = json.loads((child / "genesis/fork/origin.json").read_text())[
        "candidate_seeds"
    ]
    assert imported and all(seed is None for seed in imported.values())
    invoke(child, script(child, "child"))
    assert (
        load_bootstrap_run_state(child / "run_state.json").tree_size_at_last_save
        > state.tree_size_at_last_save
    )
    assert before == hashes(parent)


@pytest.mark.exhaustive
def test_node_completed_parent_new_absolute_goal(tmp_path: Path) -> None:
    """Do not clear or extend the parent terminal fence."""
    fresh("node_completed", tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "claim",
        "patch",
        "activation",
        "staging",
        "adam",
        "seal",
        "symlink",
        "hardlink",
        "profile",
        "retired",
    ],
)
@pytest.mark.exhaustive
def test_no_write_refusal(tmp_path: Path, mutation: str) -> None:
    """Each invalid cut is task-owned and remains byte-for-byte untouched by refusal."""
    fresh("refusal", tmp_path, mutation)


def refusal(raw: str, mutation: str) -> None:
    """Construct the bad condition only after a genuine clean close."""
    root = Path(raw)
    parent, target = root / "parent", root / "child"
    make_parent(str(parent))
    paths = MorpionBootstrapPaths.from_work_dir(parent)
    state = load_bootstrap_run_state(paths.run_state_path)
    if mutation == "claim":
        (
            paths.pipeline_dir / "generation_00000001" / "training_claim.json"
        ).parent.mkdir(exist_ok=True)
        (paths.pipeline_dir / "generation_00000001" / "training_claim.json").write_text(
            "{}"
        )
    elif mutation == "patch":
        paths.pipeline_reevaluation_patch_path.write_text("{}")
    elif mutation == "activation":
        p = paths.pipeline_dir / "automatic_activation" / "pending.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text('{"training_generation": 999}')
    elif mutation == "staging":
        p = parent / "training_publications_v1/staging/unknown"
        p.mkdir(parents=True, exist_ok=True)
    elif mutation == "adam":
        refs = training.latest_candidate_parents(paths, state, ["linear_a"])
        (parent / refs["linear_a"]["model"]["path"] / "optimizer.pt").unlink()
    elif mutation == "seal":
        (
            parent / publication_records(state.metadata)[-1]["descriptor_path"]
        ).write_text("{}")
    elif mutation == "profile":
        config = json.loads(paths.bootstrap_config_path.read_text())
        config["metadata"]["unknown_fork_change"] = True
        paths.bootstrap_config_path.write_text(json.dumps(config))
    elif mutation == "retired":
        from chipiron.environments.morpion.bootstrap.generation_publication_io import (
            PUBLICATION_KEY,
        )

        data = json.loads(paths.run_state_path.read_text())
        data["metadata"][PUBLICATION_KEY]["generations"][-1]["retired_payload_v1"] = {}
        paths.run_state_path.write_text(json.dumps(data))
    elif mutation == "symlink":
        (parent / "unknown-link").symlink_to(root)
    else:
        os.link(paths.run_state_path, root / "outside-hardlink")

    # Use lstat inventory for deliberately invalid links; do not follow them.
    def snapshot() -> Any:
        return {
            str(p.relative_to(parent)): (
                os.readlink(p) if p.is_symlink() else file_digest(p)
            )
            for p in parent.rglob("*")
            if p.is_file() or p.is_symlink()
        }

    before = snapshot()
    with pytest.raises((ValueError, OSError, KeyError)):
        fork.plan_fork(parent, target, 300, 10000)
    assert before == snapshot() and not target.exists()


@pytest.mark.exhaustive
def test_owners_target_and_expected_plan_races(tmp_path: Path) -> None:
    """Existing owners, rather than an independent writer authority, fence copying."""
    fresh("make_parent", tmp_path / "parent")
    parent, target = tmp_path / "parent", tmp_path / "child"
    before = hashes(parent)
    with (parent / "resource_monitor/owner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((BlockingIOError, ValueError)):
            fork.plan_fork(parent, target, 300, 10000)
    with (
        growth_receipt_ownership(
            MorpionBootstrapPaths.from_work_dir(parent), create=False
        ),
        pytest.raises((BlockingIOError, ValueError)),
    ):
        fork.plan_fork(parent, target, 300, 10000)
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        capsule_lease,
    )

    with (
        fork.source_cut(parent, target, 300, 10000) as (plan, _, _),
        pytest.raises(BlockingIOError),capsule_lease(
        parent,
        plan["parent"]["head"]["id"],
        version=3,
        create=False,
        exclusive=True,
        nonblocking=True,
    )
    ):
        pytest.fail("Fork did not hold its committed-cut lifetime pin")
    with pytest.raises(ValueError, match="plan changed"):
        fork.execute_fork(parent, target, 300, 10000, expected="0" * 64)
    target.mkdir()
    (target / "foreign").write_text("preserve")
    with pytest.raises((ValueError, FileNotFoundError)):
        fork.execute_fork(parent, target, 300, 10000)
    assert (target / "foreign").read_text() == "preserve" and before == hashes(parent)


@pytest.mark.parametrize("active", [True, {"reserved": "source"}])
def test_reserved_window_refuses_before_payload_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: Any
) -> None:
    """Block an old reservation until ordinary recovery, before payload access."""
    monkeypatch.setattr(fork, "window", lambda _state: {"active": active})
    with pytest.raises(ValueError, match="Reserved learning window"):
        fork._settled(
            MorpionBootstrapPaths.from_work_dir(tmp_path), None, fork.Metadata()
        )


def collide(raw: str) -> None:
    """Create an unrelated target at the actual final rename boundary."""
    root = Path(raw)
    original = fork.publish_directory_if_absent

    def publish(stage: Path, final: Path, work_dir: Path) -> None:
        if final == root / "child":
            final.mkdir()
            (final / "foreign").write_text("preserve")
        original(stage, final, work_dir)

    fork.publish_directory_if_absent = publish
    with pytest.raises(FileExistsError):
        fork.execute_fork(root / "parent", root / "child", 300, 10000)


@pytest.mark.exhaustive
def test_no_replace_race_preserves_foreign_target(tmp_path: Path) -> None:
    """Preserve a competing target even after successful private genesis."""
    fresh("make_parent", tmp_path / "parent")
    before = hashes(tmp_path / "parent")
    fresh("collide", tmp_path)
    assert (tmp_path / "child/foreign").read_text() == "preserve"
    assert not (tmp_path / "child/run_state.json").exists()
    assert before == hashes(tmp_path / "parent")


def die(raw: str, seam: str) -> None:
    """Kill only this fixture's foreground coordinator at an existing boundary."""
    root = Path(raw)
    original = getattr(fork, seam)

    def stop(*args: Any, **kwargs: Any) -> Any:
        if seam == "publish_directory_if_absent":
            original(*args, **kwargs)
        os._exit(86)

    setattr(fork, seam, stop)
    fork.execute_fork(root / "parent", root / "child", 300, 10000)


@pytest.mark.parametrize(
    "seam",
    [
        "_copy",
        "_worker",
        "_validate_stage",
        "sync_artifact",
        "publish_directory_if_absent",
    ],
)
@pytest.mark.exhaustive
def test_death_at_durability_seam(tmp_path: Path, seam: str) -> None:
    """An exact bounded retry publishes once, never a runnable partial or replacement."""
    fresh("make_parent", tmp_path / "parent")
    parent, child = tmp_path / "parent", tmp_path / "child"
    before = hashes(parent)
    plan = fork.plan_fork(parent, child, 300, 10000)
    fresh("die", tmp_path, seam, expected=86)
    result = cli(parent, child, expected=plan["plan_sha256"])
    assert result["action"] in {"published", "already-published"}
    assert before == hashes(parent)
    assert (
        cli(parent, child, expected=plan["plan_sha256"])["action"]
        == "already-published"
    )


def die_with_helper(raw: str) -> None:
    """Exercise actual parent death while the installed genesis helper is owned."""
    root = Path(raw)
    original = fork.PreparationBudget.check

    def check(self: Any, **kwargs: Any) -> None:
        original(self, **kwargs)
        if kwargs.get("extra_rss", 0):
            import psutil

            children = psutil.Process().children()
            assert len(children) == 1
            (root / "helper.json").write_text(
                json.dumps({
                    "pid": children[0].pid,
                    "created": children[0].create_time(),
                })
            )
            os._exit(86)

    fork.PreparationBudget.check = check
    fork.execute_fork(root / "parent", root / "child", 300, 10000)


@pytest.mark.exhaustive
def test_genesis_helper_parent_death_releases_writer(tmp_path: Path) -> None:
    """PDEATHSIG plus the inherited destination lease prevent an orphan writer."""
    import psutil

    fresh("make_parent", tmp_path / "parent")
    before = hashes(tmp_path / "parent")
    fresh("die_with_helper", tmp_path, expected=86)
    identity = json.loads((tmp_path / "helper.json").read_text())
    deadline = time.monotonic() + 5
    while psutil.pid_exists(identity["pid"]):
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["created"]:
            break
        assert time.monotonic() < deadline, (
            "Owned helper still writes after coordinator death"
        )
        time.sleep(0.02)
    stage = next(tmp_path.glob(".child.fork-*"))
    with fork._stage_owner(stage):
        assert not (tmp_path / "child").exists()
    assert before == hashes(tmp_path / "parent")


def cancel_helper(raw: str) -> None:
    """Interrupt the actual foreground copy owner while its helper is running."""
    import psutil

    root = Path(raw)
    original = fork.PreparationBudget.check
    children = []

    def interrupt(self: Any, **kwargs: Any) -> None:
        original(self, **kwargs)
        if kwargs.get("extra_rss", 0):
            children.extend(psutil.Process().children())
            raise KeyboardInterrupt

    fork.PreparationBudget.check = interrupt
    with pytest.raises(KeyboardInterrupt):
        fork.execute_fork(root / "parent", root / "child", 300, 10000)
    assert children and all(not p.is_running() for p in children)
    assert not (root / "child").exists()
    with fork._stage_owner(next(root.glob(".child.fork-*"))):
        pass


@pytest.mark.exhaustive
def test_cancel_reaps_helper_before_releasing_owners(tmp_path: Path) -> None:
    """Normal cancellation leaves bounded owned evidence and no writer."""
    fresh("make_parent", tmp_path / "parent")
    before = hashes(tmp_path / "parent")
    fresh("cancel_helper", tmp_path)
    assert before == hashes(tmp_path / "parent")


@pytest.mark.parametrize(
    "seconds,nodes", [(True, 2), (0, 2), (43201, 2), (1, False), (1, 0)]
)
def test_fork_explicit_finite_budgets(tmp_path: Path, seconds: Any, nodes: Any) -> None:
    """No implicit extension or boolean budget is an independent fork request."""
    with pytest.raises(ValueError):
        fork.execute_fork(tmp_path / "parent", tmp_path / "child", seconds, nodes)
