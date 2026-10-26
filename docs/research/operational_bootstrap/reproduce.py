"""Explicit bounded CLI experiment; existing workers and independent E2E oracles.

Run: python -m docs.research.operational_bootstrap.reproduce DEST INPUTS --nodes 100
No scheduler, publication replacement, retention, or implicit model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapControl,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.atomic_activation.measure import disk


def main() -> None:
    """Run real calls serially, except the explicitly bounded four-worker session."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--cluster-seconds", type=int, default=120)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--call-max-seconds", type=int, default=1800)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--session-only", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--initial-workspace", type=Path)
    parser.add_argument("--growth-steps", type=int, default=1)
    parser.add_argument("--growth-pause", type=int, default=2)
    parser.add_argument("--ownership-wait", type=float, default=60)
    parser.add_argument(
        "--validation-level", choices=("routine", "audit"), default="audit"
    )
    parser.add_argument("--audit-every-check", action="store_true")
    options = parser.parse_args()
    root = options.destination.resolve()
    if options.verify:
        from . import checks

        try:
            checks.verify(root, full=options.validation_level == "audit")
        except BaseException:
            (root / f"validation-failed-{time.time_ns()}.json").write_text(
                json.dumps({
                    "level": options.validation_level,
                    "time_ns": time.time_ns(),
                })
            )
            raise
        return
    if (
        options.cluster_seconds <= 0
        or options.growth_pause <= 0
        or options.growth_steps <= 0
        or options.max_seconds <= 0
        or options.call_max_seconds <= 0
        or not 0 <= options.ownership_wait <= 600
    ):
        parser.error("The real four-worker session requires a positive horizon.")
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / "workspace"
    paths = MorpionBootstrapPaths.from_work_dir(workspace)
    available = shutil.disk_usage(root).free
    ram = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    assert available > 20 * 2**30 and ram > 8 * 2**30
    budget = {
        "available_disk_bytes": available,
        "available_ram_bytes": ram,
        "workspace_limit_bytes": 2 * 2**30,
        "scratch_limit_bytes": 2**30,
        "minimum_available_ram_bytes": 8 * 2**30,
        "maximum_group_union_rss_bytes": 8 * 2**30,
        "minimum_disk_headroom_bytes": 20 * 2**30,
        "cluster_seconds": options.cluster_seconds,
        "stop_grace_seconds": 15,
        "growth_pause_seconds": options.growth_pause,
        "ownership_wait_seconds": options.ownership_wait,
        "audit_policy": "every-check" if options.audit_every_check else "model-changes",
        "node_target": options.nodes,
        "subsequent_steps_per_call": options.growth_steps,
        "maximum_call_seconds": options.call_max_seconds,
        "maximum_growth_calls_per_session": options.cluster_seconds
        // options.growth_pause
        + 1,
        "cycles": 2,
    }
    (root / "budget.json").write_text(json.dumps(budget, indent=2))
    experiment_started = time.monotonic()
    budget["maximum_experiment_seconds"] = options.max_seconds
    calls = []
    scratch = Path(tempfile.mkdtemp(prefix="experiment-", dir=os.environ["TMPDIR"]))
    budget["scratch"] = str(scratch)
    (root / "budget.json").write_text(json.dumps(budget, indent=2))
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TMPDIR=str(scratch),
        SQLITE_TMPDIR=str(scratch),
    )
    prefix = [
        sys.executable,
        "-m",
        "chipiron.environments.morpion.bootstrap.launcher",
        "--scientific",
        "--work-dir",
        str(workspace),
    ]

    def run(label: str, command: list[str], *, cluster: bool = False) -> None:
        assert shutil.disk_usage(root).free > budget["minimum_disk_headroom_bytes"]
        assert (
            sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())
            < budget["workspace_limit_bytes"]
        )
        log = root / f"{len(calls):02d}-{label}.log"
        metrics = log.with_suffix(".resources.json")
        env = dict(environment)
        if cluster:
            env.update(
                GROWTH_SLEEP_SECONDS=str(options.growth_pause),
                MORPION_WORKER_OWNERSHIP_WAIT_SECONDS=str(options.ownership_wait),
                MORPION_SCIENTIFIC="1",
                MORPION_USE_INSTALLED_PACKAGES="1",
                MORPION_CLUSTER_HEADLESS="1",
                MORPION_CLUSTER_MAX_SECONDS=str(options.cluster_seconds),
                MORPION_CLUSTER_STOP_GRACE_SECONDS="15",
                MORPION_CLUSTER_WORKERS="all",
                MORPION_WORK_DIR=str(workspace),
                PYTHON_BIN=sys.executable,
                MORPION_CLUSTER_OPEN_STATUS="0",
                MORPION_CLUSTER_SHOW_RECAP="0",
                MORPION_GROWTH_SHOW_RECAP="0",
                MORPION_GROWTH_WORKER_MAX_CYCLES="1",
                MORPION_REEVALUATION_MAX_NODES_PER_PATCH="7",
                MORPION_SEED_MODELS_FROM_WORK_DIR="",
            )
        with log.open("w") as stream:
            result = subprocess.Popen(
                [
                    "/usr/bin/time",
                    "-f",
                    '{"wall_s":%e,"user_s":%U,"system_s":%S,"peak_rss_kib":%M,"input_blocks":%I,"output_blocks":%O}',
                    "-o",
                    str(metrics),
                    *command,
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            observed = {}
            group_peaks = {}
            union_peak_rss = 0
            authorities = []
            previous_authority = b""
            peak_workspace = 0
            peak_staging = 0
            peak_scratch = 0
            try:
                while result.poll() is None:
                    try:
                        processes = psutil.Process(result.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        processes = []
                    if paths.run_state_path.exists():
                        authority = paths.run_state_path.read_bytes()
                        if authority != previous_authority:
                            observed_state = json.loads(authority)
                            publication = observed_state.get("metadata", {}).get(
                                "durable_generation_publication_v1", {}
                            )
                            authorities.append({
                                "elapsed_s": time.monotonic() - started,
                                "sha256": hashlib.sha256(authority).hexdigest(),
                                "generation": observed_state["generation"],
                                "nodes": observed_state["tree_size_at_last_save"],
                                "results": [
                                    {
                                        key: r[key]
                                        for key in (
                                            "generation",
                                            "accepted_dataset",
                                            "accepted_training",
                                        )
                                        if key in r
                                    }
                                    for r in publication.get("generations", [])
                                ],
                            })
                            previous_authority = authority
                    group_rss = {}
                    for process in processes:
                        try:
                            pid = process.pid
                            group = os.getpgid(pid)
                            rss = process.memory_info().rss
                            cpu = process.cpu_times()
                            previous = observed.get(str(pid), {})
                            observed[str(pid)] = {
                                "pgid": group,
                                "created": process.create_time(),
                                "command": process.cmdline(),
                                "peak_rss_bytes": max(
                                    previous.get("peak_rss_bytes", 0), rss
                                ),
                                "cpu_s": cpu.user + cpu.system,
                            }
                            group_rss[group] = group_rss.get(group, 0) + rss
                        except (psutil.NoSuchProcess, ProcessLookupError):
                            pass
                    union_peak_rss = max(union_peak_rss, sum(group_rss.values()))
                    for group, rss in group_rss.items():
                        group_peaks[str(group)] = max(
                            group_peaks.get(str(group), 0), rss
                        )
                    inodes = {}
                    staging = {}
                    for path in workspace.rglob("*"):
                        try:
                            if path.is_file():
                                stat = path.stat()
                                key = stat.st_dev, stat.st_ino
                                inodes[key] = stat.st_blocks * 512
                                if "staging" in path.parts or any(
                                    part.startswith((".closed-", ".validation-"))
                                    for part in path.parts
                                ):
                                    staging[key] = stat.st_blocks * 512
                        except FileNotFoundError:
                            pass
                    peak_workspace = max(peak_workspace, sum(inodes.values()))
                    peak_staging = max(peak_staging, sum(staging.values()))
                    try:
                        scratch_bytes = disk(scratch)["allocated_bytes"]
                    except FileNotFoundError:
                        # An owned cache may be removed between read-only samples.
                        continue
                    peak_scratch = max(peak_scratch, scratch_bytes)
                    if (
                        scratch_bytes > budget["scratch_limit_bytes"]
                        or sum(group_rss.values())
                        > budget["maximum_group_union_rss_bytes"]
                        or psutil.virtual_memory().available
                        < budget["minimum_available_ram_bytes"]
                        or time.monotonic() - started > budget["maximum_call_seconds"]
                        or time.monotonic() - experiment_started > options.max_seconds
                        or peak_workspace > budget["workspace_limit_bytes"]
                        or shutil.disk_usage(root).free
                        < budget["minimum_disk_headroom_bytes"]
                    ):
                        for child in psutil.Process(result.pid).children():
                            child.send_signal(signal.SIGTERM)
                        result.wait(timeout=30)
                        message = "Experiment resource/time budget reached; stopped."
                        raise RuntimeError(message)
                    time.sleep(0.2)
            finally:
                # Keep /usr/bin/time alive to wait for the actual command. The
                # cluster command handles TERM by stopping all owned groups.
                if result.poll() is None:
                    for child in psutil.Process(result.pid).children():
                        child.send_signal(signal.SIGTERM)
                    result.wait(timeout=30)
            log.with_suffix(".groups.json").write_text(
                json.dumps(
                    {
                        "processes": observed,
                        "authority_observations": authorities,
                        "group_peak_rss_bytes": group_peaks,
                        "process_union_peak_rss_bytes": union_peak_rss,
                        "workspace_peak_allocated_bytes": peak_workspace,
                        "staging_peak_allocated_bytes": peak_staging,
                        "scratch_peak_allocated_bytes": peak_scratch,
                        "sample_seconds": 0.2,
                    },
                    indent=2,
                )
            )
        calls.append({
            "label": label,
            "command": command,
            "returncode": result.returncode,
            "resources": metrics.read_text(),
            "disk": disk(workspace),
        })
        (root / "calls.json").write_text(json.dumps(calls, indent=2))
        assert result.returncode == 0, (label, log.read_text()[-5000:])
        if cluster:
            assert "all worker groups stopped" in log.read_text()
            for line in log.read_text().splitlines():
                if "private_reader_cache=" in line:
                    assert not Path(line.split("private_reader_cache=")[1]).exists()

    inputs = options.inputs.resolve()
    if options.initial_workspace is not None:
        source = options.initial_workspace.resolve()
        source_state = load_bootstrap_run_state(source / "run_state.json")
        assert source_state.generation == 1, (
            "Matched sessions require a task-private frozen genesis."
        )
        committed_evaluator(source, source_state)
        shutil.copytree(source, workspace)
        budget["node_target"] = source_state.tree_size_at_last_save
        budget["copied_initial_workspace"] = str(source)
        (root / "budget.json").write_text(json.dumps(budget, indent=2))
    else:
        run(
            "initialize",
            [
                *prefix,
                "--scientific-operation",
                "initialize",
                "--scientific-config",
                str(inputs / "bootstrap_config.json"),
                "--seed-model",
                str(inputs / "seed"),
                "--initial-nodes",
                str(options.nodes),
            ],
        )

    if options.initialize_only:
        return
    save_bootstrap_control(
        MorpionBootstrapControl(
            max_growth_steps_per_cycle=options.growth_steps, save_after_seconds=0.0
        ),
        paths.control_path,
    )
    if options.session_only:
        run(
            "matched-four-workers",
            ["bash", "scripts/launch_morpion_gnome_cluster.sh"],
            cluster=True,
        )
        run(
            "routine-check",
            [
                sys.executable,
                "-m",
                "docs.research.operational_bootstrap.reproduce",
                str(root),
                str(inputs),
                "--verify",
                "--validation-level",
                "routine",
            ],
        )
        return
    initial = committed_evaluator(
        workspace, load_bootstrap_run_state(paths.run_state_path)
    )[0]
    models = [initial]
    for era in (2, 3):
        before = load_bootstrap_run_state(paths.run_state_path).generation
        run(
            f"M{era - 1}-four-workers",
            ["bash", "scripts/launch_morpion_gnome_cluster.sh"],
            cluster=True,
        )
        concurrent = load_bootstrap_run_state(paths.run_state_path)
        assert concurrent.generation > before
        assert committed_evaluator(workspace, concurrent)[0] == models[-1]
        accepted = [
            record["generation"]
            for record in publication_records(concurrent.metadata)
            if "accepted_dataset" in record
            and "accepted_training" in record
            and record["scientific_identity"]["producer"]["identity"]["evaluator"]
            == models[-1]["sha256"]
        ]
        assert accepted, (
            "Four-worker session did not durably complete real Dataset and Training for this evaluator."
        )
        generation = max(accepted)
        assert (
            paths.pipeline_reevaluation_patch_path.exists()
            or concurrent.metadata["atomic_reevaluation"]["accepted_cursor"] is not None
        )
        (root / f"concurrent-M{era - 1}-authority.json").write_bytes(
            paths.run_state_path.read_bytes()
        )
        request = root / f"M{era}-request.json"
        run(
            f"M{era}-prepare",
            [
                *prefix,
                "--scientific-operation",
                "prepare-activation",
                "--pipeline-generation",
                str(generation),
                "--activation-request",
                str(request),
            ],
        )
        # This is the declared explicit authorization point in this experiment recipe.
        run(
            f"M{era}-activate",
            [
                *prefix,
                "--scientific-operation",
                "activate",
                "--activation-request",
                str(request),
            ],
        )
        run(
            f"M{era}-cold-check",
            [
                sys.executable,
                "-m",
                "docs.research.operational_bootstrap.reproduce",
                str(root),
                str(inputs),
                "--verify",
            ],
        )
        after = load_bootstrap_run_state(paths.run_state_path)
        model = committed_evaluator(workspace, after)[0]
        assert (
            model == json.loads(request.read_text())["candidate"]
            and model != models[-1]
        )
        models.append(model)
        authority = paths.run_state_path.read_bytes()
        run(
            f"M{era}-retry",
            [
                *prefix,
                "--scientific-operation",
                "activate",
                "--activation-request",
                str(request),
            ],
        )
        assert paths.run_state_path.read_bytes() == authority
    run(
        "M3-continued-growth",
        [*prefix, "--pipeline-stage", "growth", "--max-cycles", "1"],
    )
    run(
        "M3-reevaluation",
        [
            *prefix,
            "--pipeline-stage",
            "reevaluation",
            "--reevaluation-max-nodes-per-patch",
            "7",
        ],
    )
    run(
        "final-check",
        [
            sys.executable,
            "-m",
            "docs.research.operational_bootstrap.reproduce",
            str(root),
            str(inputs),
            "--verify",
            "--validation-level",
            "audit" if options.audit_every_check else "routine",
        ],
    )
    run("inspect", [*prefix, "--scientific-operation", "inspect"])
    (root / "summary.json").write_text(
        json.dumps(
            {
                "models": models,
                "calls": len(calls),
                "disk": disk(workspace),
                "head": load_bootstrap_run_state(paths.run_state_path).generation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
