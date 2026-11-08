"""Normal current-continual entrypoint, without new scientific authority.

Preview reads bounded metadata and file presence, never tensors or tree payloads.
Native restore/admission still verify payload bytes. Execution rechecks under the
existing Growth owner, then delegates to the unmodified native monitor owner.
"""

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chipiron.bootstrap.lifecycle import RunMode, select_action

from .bootstrap_args import MorpionBootstrapArgs
from .bootstrap_paths import MorpionBootstrapPaths
from .config import bootstrap_config_from_dict, bootstrap_config_sha256
from .derived.provenance import inside
from .generation_publication_io import publication_records, require
from .learning_policy import policy_from_metadata, validate_continual_context
from .learning_schedule import schedule_from_metadata
from .run_state import _run_state_from_dict

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

MODULE = "chipiron.environments.morpion.bootstrap.lifecycle"
LAUNCHER = "chipiron.environments.morpion.bootstrap.launcher"


class Metadata:
    """Limit the whole preview, including source fingerprints and metadata joins."""

    def __init__(self) -> None:
        """Initialize the per-preview byte allowance."""
        self.remaining = 64 * 2**20

    def data(self, path: Path, maximum: int = 4 * 2**20) -> bytes:
        """Refuse symlinks/oversize inputs before reading bounded bytes."""
        require(path.is_file() and not path.is_symlink(), f"Missing metadata: {path}")
        size = path.stat().st_size
        require(
            size <= min(maximum, self.remaining), "Preview metadata budget exceeded."
        )
        with path.open("rb") as stream:
            data = stream.read(min(maximum, self.remaining) + 1)
        require(len(data) == size, "Metadata changed during preview.")
        self.remaining -= len(data)
        return data

    def read(self, path: Path) -> dict[str, Any]:
        """Read finite object metadata only."""
        value = json.loads(self.data(path), parse_constant=_invalid_constant)
        require(isinstance(value, dict), f"Expected metadata object: {path}")
        return cast("dict[str, Any]", value)

    def digest(self, path: Path) -> str:
        """Hash small metadata, never scientific payloads."""
        return hashlib.sha256(self.data(path)).hexdigest()


def _invalid_constant(value: str) -> None:
    require(False, f"Nonfinite JSON metadata: {value}")


def _clock(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            result = float(value)
        except OverflowError:
            result = float("inf")
    else:
        result = float("inf")
    require(
        math.isfinite(result) and result >= 0,
        "Invalid or uncertain cumulative budget clock.",
    )
    return result


def _entries(path: Path, maximum: int = 2048) -> list[Path]:
    values = list(islice(path.iterdir(), maximum + 1))
    require(len(values) <= maximum, "Preview directory bound exceeded.")
    return values


@contextmanager
def _existing_lock(path: Path) -> Iterator[None]:
    """Read-only nonblocking probe, never create/unlink an owner inode."""
    require(
        path.is_file() and not path.is_symlink(), "Missing existing ownership lock."
    )
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _source_identity(producer: dict[str, Any], metadata: Metadata) -> None:
    """Use the native Python fingerprint format with explicit preview bounds."""
    import torch

    require(
        producer.get("python_version") == platform.python_version()
        and producer.get("torch_version") == str(torch.__version__),
        "Python/Torch runtime differs from the prepared workspace.",
    )
    for name in ("chipiron", "anemone", "coral", "atomheart", "valanga", "parsley"):
        module = importlib.import_module(name)
        require(module.__file__ is not None, "Missing installed source origin.")
        assert module.__file__ is not None
        root = Path(module.__file__).parent
        digest = hashlib.sha256()
        count = 0
        for _directory, _, names in os.walk(root):
            for filename in names:
                if filename.endswith(".py"):
                    count += 1
                    require(count <= 4096, "Preview source membership bound exceeded.")
        # Membership was bounded before sorting; identical to package_fingerprint.
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(metadata.data(path))
        key = "revision" if name == "chipiron" else name + "_source_sha256"
        require(
            producer.get(key) == digest.hexdigest(), f"Installed {name} source differs."
        )


def _recipe(
    root: Path, tokens: Sequence[str], saved: MorpionBootstrapArgs | None
) -> Any:
    """Reuse canonical parsing; compare only explicitly supplied scientific options."""
    from .launcher import build_launcher_argument_parser, launcher_args_from_cli

    parser = build_launcher_argument_parser()
    options = parser._option_string_actions  # pylint: disable=protected-access
    plain = {f.name for f in fields(MorpionBootstrapArgs)} - {
        "work_dir",
        "pipeline_mode",
        "diagnostic_stop_after_growth",
        "training_evaluator_names",
        "growth_save_and_exit",
        "growth_skip_training_export",
    }
    from .learning_policy import MorpionContinualLearningPolicy

    policy_fields = {f.name for f in fields(MorpionContinualLearningPolicy)}
    special = {"evaluator_family", "growth_lookahead_cycles", "ready_growth_max_cycles"}
    search = {
        "rollout_after_opening": "enabled",
        "rollout_max_extra_steps": "max_extra_steps",
        "rollout_action_selector_kind": "action_selector_kind",
        "rollout_random_seed": "random_seed",
        "rollout_stop_on_existing_node": "stop_on_existing_node",
    }
    explicit = set()
    for token in tokens:
        if token.startswith("--"):
            action = options.get(token.split("=", 1)[0])
            require(
                action is not None,
                "Unsupported start option; grants/resets/imports are follow-ups.",
            )
            assert action is not None
            require(
                action.dest in plain | policy_fields | special | search.keys(),
                "Option is not a start recipe setting.",
            )
            explicit.add(action.dest)
    parsed = launcher_args_from_cli([
        "--scientific",
        "--scientific-operation",
        "prepare-continuous",
        "--work-dir",
        str(root),
        *tokens,
    ])
    if saved is None:
        require(
            not explicit.intersection({
                "tree_branch_limit",
                "save_after_seconds",
                "runtime_checkpoint_format",
                "training_export_mode",
                "evaluator_update_policy",
                "candidate_checkpoint_load_headroom_factor",
                "candidate_checkpoint_load_min_headroom_mb",
                "skip_evaluator_diagnostics",
            }),
            "These settings are fixed by continual preparation; omit them from normal start.",
        )
    if saved is not None:
        for name in explicit:
            if name in policy_fields:
                assert saved.continual_learning is not None
                expected, actual = (
                    getattr(saved.continual_learning, name),
                    getattr(parsed.continual_policy, name),
                )
            elif name in search:
                expected, actual = (
                    getattr(saved.search.rollout, search[name]),
                    getattr(parsed.bootstrap_args.search.rollout, search[name]),
                )
            elif name in {"growth_lookahead_cycles", "ready_growth_max_cycles"}:
                expected, actual = (
                    saved.learning_schedule,
                    parsed.bootstrap_args.learning_schedule,
                )
            elif name == "evaluator_family":
                expected, actual = (
                    saved.resolved_evaluators_config(),
                    parsed.bootstrap_args.resolved_evaluators_config(),
                )
            else:
                expected, actual = (
                    getattr(saved, name),
                    getattr(parsed.bootstrap_args, name),
                )
            require(
                actual == expected, f"Explicit recipe conflicts with persisted {name}."
            )
    return parsed


def _usage(
    root: Path, metadata: Metadata, generation: int
) -> tuple[float, dict[str, Any] | None]:
    monitor = root / "resource_monitor"
    if not monitor.exists():
        require(generation == 1, "Missing usage for an already advanced workspace.")
        return 0.0, None
    names = _entries(monitor)
    usage = metadata.read(monitor / "usage.json")
    consumed = _clock(usage.get("elapsed_seconds"))
    identifier = usage.get("session")
    require(
        isinstance(identifier, str) and identifier.isdecimal(),
        "Unknown monitor session identity.",
    )
    # Native sidecars share a session ID. Any incomplete newer attempt (including
    # a decision persisted before ACK/cleanup) makes cumulative usage uncertain.
    sessions = {
        p.stem.removeprefix("session_")
        for p in names
        if p.name.startswith("session_")
        and p.suffix == ".json"
        and p.stem.removeprefix("session_").isdecimal()
    }
    observed_sessions = {
        p.name.removeprefix("session_").split(".", 1)[0]
        for p in names
        if p.name.startswith("session_")
    }
    decisions = {
        p.stem.removeprefix("decision_")
        for p in names
        if p.name.startswith("decision_") and p.suffix == ".json"
    }
    require(
        bool(sessions)
        and sessions == observed_sessions == decisions
        and identifier == max(sessions, key=int),
        "Incomplete or conflicting monitor sessions.",
    )
    receipt = metadata.read(monitor / f"session_{identifier}.json")
    decision = metadata.read(monitor / f"decision_{identifier}.json")
    require(
        _clock(decision.get("monotonic_s")) <= time.monotonic(),
        "Future/uncertain decision clock; recovery required.",
    )
    require(
        receipt.get("decision") == decision
        and receipt.get("reason") == usage.get("reason") == decision.get("reason"),
        "Conflicting decision/session/usage evidence.",
    )
    require(
        _clock(decision.get("budget_elapsed_seconds"))
        <= consumed
        == _clock(receipt.get("budget_elapsed_seconds")),
        "Conflicting cumulative clocks.",
    )
    require(
        receipt.get("shutdown_confirmed") is True
        and receipt.get("forced_group_cleanup") is False
        and receipt.get("shutdown_grace_expired") is False
        and isinstance(receipt.get("supervisor_exit_code"), int)
        and not isinstance(receipt.get("supervisor_exit_code"), bool)
        and receipt["supervisor_exit_code"] in {0, 143}
        and decision.get("worker_failures") == []
        and decision.get("control_failure") is None,
        "Prior failed/uncertain worker shutdown requires explicit recovery.",
    )
    processes = receipt.get("environment_owned_processes")
    require(isinstance(processes, dict), "Missing prior child quiescence evidence.")
    assert isinstance(processes, dict)
    for pid, ticks in processes.items():
        require(
            str(pid).isdecimal() and isinstance(ticks, str) and ticks.isdecimal(),
            "Invalid prior child identity.",
        )
        path = Path(f"/proc/{pid}/stat")
        if path.exists():
            observed = path.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
            require(observed[19] != ticks, "Prior owned child remains; resume refused.")
    require(
        receipt.get("reason")
        in {
            "supervisor_exit",
            "supervisor_deadline",
            "maximum_elapsed_time",
            "maximum_tree_nodes",
            "operator_SIGINT",
            "operator_SIGTERM",
        },
        "Prior failed/resource endpoint requires explicit recovery.",
    )
    return consumed, {
        "session": identifier,
        "reason": receipt["reason"],
        "decision": decision,
        "cleanup_elapsed_seconds": consumed,
    }


def _references(
    root: Path, state: Any, metadata: Metadata
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = publication_records(state.metadata)
    require(0 < len(records) <= 1024, "Unsupported or oversized publication history.")
    parents: dict[str, Any] = {}
    datasets = []
    for record in records:
        for kind, namespace in [
            ("accepted_dataset", "dataset_publications_v1"),
            ("accepted_training", "training_publications_v1"),
        ]:
            if kind not in record:
                continue
            ref = record[kind]
            proposal_path = inside(
                root, f"{namespace}/objects/{ref['attempt_id']}/proposal.json"
            )
            require(
                metadata.digest(proposal_path) == ref["descriptor_sha256"],
                "Accepted proposal identity differs.",
            )
            proposal = metadata.read(proposal_path)
            from .worker_result_io import workspace_epoch

            admission = proposal["admission"]
            require(
                admission["identity"] == workspace_epoch(state)
                and admission["generation"] == record["generation"]
                and admission["attempt_id"] == ref["attempt_id"],
                "Accepted result belongs to another source/workspace.",
            )
            if kind == "accepted_dataset":
                manifest = proposal["manifest"]
                require(
                    inside(root, manifest["rows_path"]).is_file(),
                    "Accepted Dataset rows are missing.",
                )
                datasets.append({
                    "generation": record["generation"],
                    "reference": ref,
                    "source": proposal["admission"]["source"],
                })
            else:
                for name, model in proposal["models"].items():
                    parents[name] = {
                        "generation": record["generation"],
                        "accepted_training": ref,
                        "model": model,
                    }
    require(len(parents) <= 128, "Candidate preview bound exceeded.")
    for parent in parents.values():
        for filename in [
            "param.pt",
            "optimizer.pt",
            "morpion_manifest.json",
            "morpion_regressor_args.json",
        ]:
            require(
                inside(root, parent["model"]["path"] + "/" + filename).is_file(),
                "Accepted candidate weights/Adam dependency missing.",
            )
    return parents, datasets


def plan_start(
    work_dir: Path,
    mode: RunMode = "auto",
    recipe: Sequence[str] = (),
    cluster_script: Path | None = None,
) -> dict[str, Any]:
    """Classify without mutation; native validators still verify payloads at use."""
    root = work_dir.expanduser().absolute()
    require(
        not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked workspace refused.",
    )
    root = root.resolve()
    require(
        not any(
            (p / "LAUNCH_READY.json").exists() or (p / "evidence/binding.json").exists()
            for p in (root, root.parent)
        ),
        "Sealed experiment: use its own launcher; never enroll/resume it here.",
    )
    empty = not root.exists() or (root.is_dir() and not _entries(root))
    committed = (root / "run_state.json").is_file() and (
        root / "bootstrap_config.json"
    ).is_file()
    action = select_action(mode, empty=empty, committed=committed)
    metadata = Metadata()
    if action == "create":
        require(
            not root.with_name(root.name + "-inputs").exists(),
            "Partial preparation inputs already exist; recover explicitly.",
        )
        parsed = _recipe(root, recipe, None)
        policy = parsed.continual_policy
        require(
            policy.training_policy == "continue_from_previous_accepted_weights"
            and policy.optimizer_state_policy == "continue_adam",
            "Normal start requires full accepted weights+Adam continuation.",
        )
        from .execution_policy import validate_supported

        validate_supported(parsed.bootstrap_args)
        status: dict[str, Any] = {
            "action": action,
            "work_dir": str(root),
            "head": None,
            "candidate_parents": {},
            "accepted_datasets": [],
            "consumed_seconds": 0.0,
            "remaining_seconds": policy.max_elapsed_seconds,
            "last_endpoint": None,
            "profile": parsed.bootstrap_args.execution_profile,
            "payload_validation": "native preparation/restore/admission",
        }
    else:
        paths = MorpionBootstrapPaths.from_work_dir(root)
        require(
            not paths.launcher_pid_path.exists(),
            "Live/stale launcher registration requires explicit recovery.",
        )
        with ExitStack() as stack:
            stack.enter_context(
                _existing_lock(paths.pipeline_dir / ".growth_patch_consumer.lock")
            )
            if (root / "resource_monitor/owner.lock").exists():
                stack.enter_context(
                    _existing_lock(root / "resource_monitor/owner.lock")
                )
            config = bootstrap_config_from_dict(
                metadata.read(paths.bootstrap_config_path)
            )
            state = _run_state_from_dict(metadata.read(paths.run_state_path))
            policy = policy_from_metadata(config.metadata)
            require(
                policy is not None
                and policy.training_policy == "continue_from_previous_accepted_weights"
                and policy.optimizer_state_policy == "continue_adam",
                "Legacy/unsupported continuation policy; explicit migration required.",
            )
            assert policy is not None
            validate_continual_context(
                paths, state, policy, schedule_from_metadata(config.metadata)
            )
            require(
                state.metadata.get("durable_training") == {"schema_version": 1}
                and state.metadata.get("fenced_dataset") == {"schema_version": 1},
                "Unsupported accepted-result protocol.",
            )
            records = publication_records(state.metadata)
            require(
                bool(records)
                and len(records) <= 1024
                and records[-1].get("publication_version") == 3,
                "Current closed publication protocol required.",
            )
            head = records[-1]
            require(
                (
                    state.generation,
                    state.latest_runtime_checkpoint_path,
                    state.latest_tree_snapshot_path,
                )
                == (
                    head["generation"],
                    head["checkpoint_path"],
                    head["tree_snapshot_path"],
                ),
                "Run state differs from committed head.",
            )
            descriptor_path = inside(root, head["descriptor_path"])
            require(
                metadata.digest(descriptor_path) == head["descriptor_sha256"],
                "Committed descriptor changed.",
            )
            descriptor = metadata.read(descriptor_path)
            require(
                descriptor["id"] == head["id"]
                and descriptor["scientific_identity"] == head["scientific_identity"],
                "Committed head identity differs.",
            )
            require(
                len(descriptor["files"]) <= 4096,
                "Head dependency preview bound exceeded.",
            )
            for entry in descriptor["files"]:
                require(
                    inside(descriptor_path.parent, entry["path"]).is_file(),
                    "Committed head dependency missing.",
                )
            from .generation_publication_io import _active_evaluator_pin  # pylint: disable=protected-access

            active = _active_evaluator_pin(descriptor)
            require(
                inside(root, str(state.latest_runtime_checkpoint_path)).is_dir(),
                "Committed runtime checkpoint missing.",
            )
            producer = head["scientific_identity"]["producer"]["identity"]
            _source_identity(producer, metadata)
            from .execution_policy import (
                KEY,
            )
            from .execution_policy import (
                policy_from_metadata as profile_from_metadata,
            )

            require(
                profile_from_metadata(config.metadata)
                == profile_from_metadata(state.metadata),
                "Execution profile differs from genesis.",
            )
            from .launcher import _scientific_args  # pylint: disable=protected-access

            saved = _scientific_args(MorpionBootstrapArgs(work_dir=root), config)
            _recipe(root, recipe, saved)
            parents, datasets = _references(root, state, metadata)
            names = list(config.evaluators.evaluators)
            require(
                len(names) <= 128 and set(parents) <= set(names),
                "Accepted candidate catalog differs or exceeds preview bound.",
            )
            consumed, endpoint = _usage(root, metadata, state.generation)
            termination = state.metadata.get("experiment_termination")
            stop = termination or (
                "maximum_tree_nodes"
                if state.tree_size_at_last_save >= policy.max_tree_nodes
                else None
            )
            if endpoint and endpoint["reason"] in {
                "operator_SIGINT",
                "operator_SIGTERM",
                "maximum_elapsed_time",
                "maximum_tree_nodes",
            }:
                stop = stop or endpoint["reason"]
            if policy.max_elapsed_seconds - consumed < 1:
                stop = (
                    stop
                    or "native budget exhausted (less than one complete second remains)"
                )
            status = {
                "action": select_action(
                    mode, empty=False, committed=True, terminal=stop is not None
                ),
                "work_dir": str(root),
                "head": {
                    "id": head["id"],
                    "generation": head["generation"],
                    "descriptor_sha256": head["descriptor_sha256"],
                },
                "configuration_sha256": bootstrap_config_sha256(config),
                "build": producer,
                "profile": config.metadata.get(KEY),
                "active_evaluator": active,
                "candidate_parents": {name: parents.get(name) for name in names},
                "accepted_datasets": datasets,
                "consumed_seconds": consumed,
                "remaining_seconds": max(0.0, policy.max_elapsed_seconds - consumed),
                "last_endpoint": endpoint,
                "termination": termination,
                "no_work_reason": stop,
                "payload_validation": "metadata/presence preview only; native restore/admission verifies payload bytes",
            }
    monitor_script = root / "resource_monitor/supervisor.json"
    recorded = metadata.read(monitor_script) if monitor_script.exists() else None
    script = cluster_script or (Path(recorded["path"]) if recorded else None)
    require(
        script is not None,
        "First start requires --cluster-script; no supervisor is guessed.",
    )
    assert script is not None
    script = script.absolute()
    digest = metadata.digest(script)
    require(
        recorded is None
        or recorded == {"path": str(script.resolve()), "sha256": digest},
        "Supervisor changed from the recorded identity.",
    )
    status["supervisor"] = {"path": str(script), "sha256": digest}
    return status


def execute_start(
    work_dir: Path, mode: RunMode, recipe: Sequence[str], cluster_script: Path | None
) -> int:
    """Fresh process only; reuse preparation then exec before configuring threads again."""
    plan = plan_start(work_dir, mode, recipe, cluster_script)
    print(json.dumps(plan, indent=2), flush=True)
    if plan["action"] == "no-work":
        return 0
    root = Path(plan["work_dir"])
    script = Path(plan["supervisor"]["path"])
    if plan["action"] == "create":
        from .launcher import main as launcher_main

        launcher_main([
            "--scientific",
            "--scientific-operation",
            "prepare-continuous",
            "--work-dir",
            str(root),
            *recipe,
        ])
        command = [
            sys.executable,
            "-I",
            "-B",
            "-m",
            MODULE,
            "--work-dir",
            str(root),
            "--mode",
            "resume",
            "--cluster-script",
            str(script),
        ]
        return os.execv(sys.executable, command)
    from .continual_operator import run_monitored_continual_experiment
    from .execution_profile import launcher_execution
    from .launcher import launcher_args_from_cli

    args = launcher_args_from_cli([
        "--scientific",
        "--scientific-operation",
        "run-continuous",
        "--work-dir",
        str(root),
        "--cluster-script",
        str(script),
    ])
    # plan_start re-observes under the existing owners; native monitor/admission
    # reacquire and validate their own authority. No preview lease crosses launch.
    with launcher_execution(args, process_start=True):
        checked = plan_start(root, "resume", recipe, script)
        if checked["action"] == "no-work":
            return 0
        run_monitored_continual_experiment(
            MorpionBootstrapPaths.from_work_dir(root), script
        )
    return 0


def main() -> int:
    """Run the internal process entry without prepared-script dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "resume", "fresh"], default="auto")
    parser.add_argument("--cluster-script", type=Path)
    parser.add_argument("recipe", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    recipe = args.recipe[1:] if args.recipe[:1] == ["--"] else args.recipe
    try:
        return execute_start(args.work_dir, args.mode, recipe, args.cluster_script)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"Start refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
