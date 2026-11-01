"""Read-only, lightweight operator snapshots; no tree, model or Streamlit imports."""

import json
import math
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

JsonObject = dict[str, Any]
MAX_HISTORY_BYTES = 4 * 1024 * 1024
MAX_HISTORY_ROWS = 2000


def mapping(value: object) -> JsonObject:
    """Return an object mapping without manufacturing missing metadata."""
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON observations rather than pass them into renderers."""
    message = f"Nonfinite JSON constant: {value}"
    raise ValueError(message)


def read_artifact(
    path: Path, errors: list[str], *, limit: int = MAX_HISTORY_BYTES
) -> JsonObject:
    """Read a small JSON artifact, distinguishing absence from real damage."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            errors.append(f"{path.name}: exceeds {limit} byte observation limit")
            return {}
        value = json.loads(raw, parse_constant=_reject_constant)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, RecursionError) as exc:
        errors.append(f"{path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name}: expected a JSON object")
        return {}
    return cast("JsonObject", value)


def number(value: object) -> int | float | None:
    """Exclude booleans, nonfinite values and incorrectly typed measurements."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def display(value: object) -> str:
    """Format a truthful human-readable metric, retaining exact numeric values."""
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4g}" if math.isfinite(value) else "unknown"
    return str(value)


def timestamp(value: object) -> datetime | None:
    """Accept only explicit timezone-bearing artifact timestamps."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def age_label(value: object) -> str:
    """Show artifact freshness without mistaking it for process liveness."""
    parsed = timestamp(value)
    if parsed is None:
        return "unknown"
    seconds = max(0, int((datetime.now(UTC) - parsed).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def read_history(path: Path, errors: list[str]) -> tuple[JsonObject, ...]:
    """Bound live-monitoring history reads; original files remain accessible."""
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - MAX_HISTORY_BYTES))
            if size > MAX_HISTORY_BYTES:
                stream.readline()
            lines = stream.read().splitlines()[-MAX_HISTORY_ROWS:]
    except FileNotFoundError:
        return ()
    except OSError as exc:
        errors.append(f"{path.name}: {exc}")
        return ()
    rows: list[JsonObject] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            errors.append(f"{path.name} contains an invalid or partially written row.")
            break
        if not isinstance(row, dict):
            errors.append(f"{path.name} row must be an object.")
            break
        rows.append(cast("JsonObject", row))
    return tuple(rows)


def _live_launcher(work_dir: Path, process: JsonObject) -> bool:
    """Require a local live process with the matching launcher and exact workspace."""
    pid = process.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if process.get("start_ticks") is None or not process.get(
        "boot_id", process.get("boot")
    ):
        return False
    from .runtime_view import live_identity

    if not live_identity(process, work_dir):
        return False
    try:
        arguments = (
            (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
        )
        stat = (
            (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        )
    except OSError, UnicodeError, IndexError:
        return False
    return (
        stat[0] != "Z"
        and str(work_dir) in arguments
        and "chipiron.environments.morpion.bootstrap.launcher" in arguments
    )


def classify_run(
    work_dir: Path,
    *,
    manifest: JsonObject,
    started: JsonObject,
    finished: JsonObject,
    process: JsonObject,
    has_progress: bool,
) -> tuple[str, str]:
    """Classify only supported lifecycle evidence, without freshness heuristics."""
    if _live_launcher(work_dir, process):
        return "running", "Local launcher process and workspace identity verified."
    finished_time = timestamp(finished.get("finished_at_utc"))
    started_time = timestamp(started.get("started_at_utc"))
    if finished_time is not None and (
        started_time is None or finished_time >= started_time
    ):
        code = finished.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            status = (
                "completed"
                if code == 0
                else "stopped"
                if code in (130, 143)
                else "failed"
            )
            return (
                status,
                f"Recorded launcher outcome: {finished.get('termination_reason', code)}.",
            )
    stopped_time = timestamp(process.get("stopped_at_utc"))
    if stopped_time is not None and (
        started_time is None or stopped_time >= started_time
    ):
        return (
            "stopped",
            "Launcher recorded a stop; current worker liveness is unknown.",
        )
    if (
        manifest.get("status") == "READY_TO_LAUNCH_CANONICAL_BOOTSTRAP"
        and not started
        and not has_progress
    ):
        return (
            "prepared",
            "Preparation is recorded; no launch or progress artifact is present.",
        )
    return "unknown", "Artifacts do not establish current process liveness."


@dataclass(frozen=True)
class WorkerSummary:
    """A worker signal distinguishes reported stage state from live process state."""

    name: str
    state: str
    detail: str


@dataclass(frozen=True)
class BootstrapDashboardSnapshot:
    """Transient aggregate of authoritative small artifacts; never persisted."""

    work_dir: Path
    status: str
    status_detail: str
    config: JsonObject
    manifest: JsonObject
    run_state: JsonObject
    event: JsonObject
    history: tuple[JsonObject, ...]
    active_model: JsonObject
    model_manifest: JsonObject
    model_args: JsonObject
    dataset: JsonObject
    training: JsonObject
    record: JsonObject
    frontier: JsonObject
    workers: tuple[WorkerSummary, ...]
    errors: tuple[str, ...]
    updated_at: str | None
    dataset_history: tuple[JsonObject, ...] = ()
    training_history: tuple[JsonObject, ...] = ()
    runtime: JsonObject = field(default_factory=dict)
    activation_requests: tuple[JsonObject, ...] = ()

    @property
    def pipeline_mode(self) -> str:
        """Do not assume single-process semantics when metadata is absent."""
        return str(self.config.get("pipeline_mode", "unknown"))

    @property
    def generation(self) -> object:
        """Report search generation independently of active model generation."""
        return self.event.get("generation", self.run_state.get("generation"))

    @property
    def tree_nodes(self) -> object:
        """Use the current search event or persisted run-state count."""
        return mapping(self.event.get("tree")).get(
            "num_nodes", self.run_state.get("tree_size_at_last_save")
        )

    @property
    def evaluator_name(self) -> object:
        """Prefer the authoritative active-model pointer over historical events."""
        return self.active_model.get(
            "evaluator_name", self.run_state.get("active_evaluator_name")
        )

    @property
    def dataset_rows(self) -> object:
        """Expose persisted extraction counts when they exist."""
        metadata = mapping(self.dataset.get("metadata"))
        return metadata.get(
            "dataset_rows",
            metadata.get(
                "num_rows", mapping(self.event.get("dataset")).get("num_rows")
            ),
        )

    @property
    def active_runtime_label(self) -> str:
        """Read the prepared ceiling rather than any package default."""
        scientific = mapping(self.manifest.get("scientific_configuration"))
        seconds = number(
            mapping(scientific.get("runtime_limits")).get("active_seconds")
        )
        if seconds is None:
            return "unknown (consult the workspace launcher)"
        minutes = int(seconds) // 60
        return f"{minutes // 60}h{minutes % 60:02d}"


def _certified_record(candidate: JsonObject, errors: list[str]) -> JsonObject:
    """Never promote a frontier or uncertified value into a certified record."""
    if candidate.get("current_best_total_points") is not None and not (
        candidate.get("current_best_is_exact") is True
        or candidate.get("current_best_is_terminal") is True
    ):
        errors.append(
            "Record artifact lacks exact/terminal certification; value withheld."
        )
        return {}
    return candidate


def load_dashboard_snapshot(
    work_dir: Path, runtime_root: Path | None = None
) -> BootstrapDashboardSnapshot:
    """Read monitoring JSON only: no tree deserialization, disk walk or model load."""
    from .performance_evidence import EvidenceReader
    from .runtime_view import load_runtime, runtime_lifecycle, worker_summary

    work_dir = work_dir.resolve()
    errors: list[str] = []
    config = read_artifact(work_dir / "bootstrap_config.json", errors)
    manifest = read_artifact(work_dir / "manifest.json", errors)
    reader = EvidenceReader(work_dir, errors)
    state = reader.authority
    latest = read_artifact(work_dir / "latest_status.json", errors)
    history = read_history(work_dir / "history.jsonl", errors)
    event = mapping(latest.get("latest_event")) or (history[-1] if history else {})
    active = read_artifact(work_dir / "pipeline/active_model.json", errors)
    stage_history = {}
    for stage in ("dataset", "training"):
        if reader.publications:
            generations = sorted(reader.publications)[-64:]
            files = [
                work_dir / f"pipeline/generation_{g:06d}/{stage}_status.json"
                for g in generations
            ]
            if len(reader.publications) > 64:
                errors.append(
                    "Recent stage history limited to 64 publication generations; coverage is partial."
                )
        else:
            # Historical layouts have no authority index. Only this one directory
            # is enumerated; checkpoint/rows/model trees are never traversed.
            from itertools import islice

            files = sorted(
                islice(
                    (work_dir / "pipeline").glob(f"generation_*/{stage}_status.json"),
                    257,
                )
            )
            if len(files) > 256:
                errors.append(
                    "Legacy stage listing exceeds 256 artifacts; current stage status withheld because listing coverage is incomplete."
                )
                files = []
            files = files[-64:]
        stage_history[stage] = tuple(
            value for path in files if (value := reader.read(path))
        )
    dataset_history = stage_history["dataset"]
    training_history = stage_history["training"]
    dataset = dataset_history[-1] if dataset_history else {}
    training = training_history[-1] if training_history else {}
    activation_requests: tuple[JsonObject, ...] = ()
    training_generation = training.get("generation")
    if isinstance(training_generation, int) and not isinstance(
        training_generation, bool
    ):
        from itertools import islice

        directory = work_dir / "pipeline/automatic_activation"
        request_paths = [directory / f"generation_{training_generation:06d}.json"]
        renewals = list(
            islice(
                directory.glob(f"generation_{training_generation:06d}.recovery_*.json"),
                17,
            )
        )
        if len(renewals) > 16:
            errors.append(
                "Activation renewals exceed 16; pending status coverage is partial."
            )
        activation_requests = tuple(
            value
            for path in request_paths + renewals[:16]
            if (value := reader.read(path))
        )
    model_manifest: JsonObject = {}
    model_args: JsonObject = {}
    bundle = active.get("model_bundle_path")
    if isinstance(bundle, str) and bundle:
        bundle_path = work_dir / bundle
        model_manifest = reader.read(
            bundle_path / "morpion_manifest.json", limit=1024 * 1024
        )
        model_args = reader.read(bundle_path / "morpion_regressor_args.json")
    process = read_artifact(work_dir / "launcher_process_state.json", errors)
    if process and "pid" not in process:
        with suppress(OSError, ValueError):
            process["pid"] = int((work_dir / "launcher.pid").read_text())
    status, detail = classify_run(
        work_dir,
        manifest=manifest,
        started=read_artifact(work_dir / "launch_started.json", errors),
        finished=read_artifact(work_dir / "launch_finished.json", errors),
        process=process,
        has_progress=bool(state or event),
    )
    runtime = load_runtime(work_dir, runtime_root, errors)
    associated = runtime_lifecycle(runtime, state)
    if associated is not None:
        status, detail = associated
    workers = [
        worker_summary(work_dir, name, stage, status, runtime, errors, reader)
        for name, stage in [
            ("Growth", "growth"),
            ("Dataset", "dataset_worker"),
            ("Training", "training_worker"),
            ("Reevaluation", "reevaluation"),
        ]
    ]
    if not runtime and any(
        worker.state in {"working", "waiting", "idle"} for worker in workers
    ):
        status, detail = (
            "running",
            "Local worker process identity verified; worker activity and saved scheduling context are separate.",
        )
    times = [
        v
        for obj in [event, active, dataset, training]
        for key in ["timestamp_utc", "updated_at_utc"]
        if isinstance(v := obj.get(key), str) and timestamp(v)
    ]
    updated = (
        max(times, key=lambda value: cast("datetime", timestamp(value)))
        if times
        else None
    )
    record = (
        mapping(dataset.get("record_status"))
        or mapping(event.get("record"))
        or mapping(state.get("latest_record_status"))
    )
    frontier = (
        mapping(dataset.get("frontier_status"))
        or mapping(event.get("frontier"))
        or mapping(state.get("latest_frontier_status"))
    )
    return BootstrapDashboardSnapshot(
        work_dir,
        status,
        detail,
        config,
        manifest,
        state,
        event,
        history,
        active,
        model_manifest,
        model_args,
        dataset,
        training,
        _certified_record(record, errors),
        frontier,
        tuple(workers),
        tuple(errors),
        updated,
        dataset_history,
        training_history,
        runtime,
        activation_requests,
    )


def active_model_summary(snapshot: BootstrapDashboardSnapshot) -> JsonObject:
    """Keep bundle training provenance distinct from the next configured training."""
    model = snapshot.active_model
    metadata = mapping(snapshot.model_manifest.get("metadata"))
    training = mapping(metadata.get("training_config"))
    config = mapping(training.get("config"))
    selection = mapping(
        mapping(mapping(training.get("provenance")).get("selection")).get("selection")
    )
    validation = mapping(metadata.get("validation_metrics"))
    return {
        "Evaluator": snapshot.evaluator_name,
        "Generation": model.get("generation"),
        "Model source": model.get("source"),
        "Source generation": model.get("source_generation"),
        "Updated at": model.get("updated_at_utc"),
        "Bundle path": model.get("model_bundle_path"),
        "Architecture": snapshot.model_args.get(
            "model_kind", snapshot.model_manifest.get("model_kind")
        ),
        "Representation": snapshot.model_manifest.get("input_representation"),
        "relation_bias_scale": snapshot.model_args.get("relation_bias_scale"),
        "Parameter count": metadata.get("parameter_count", selection.get("parameters")),
        "Epochs": metadata.get("num_epochs", config.get("num_epochs")),
        "Learning rate": metadata.get("learning_rate", config.get("learning_rate")),
        "Batch size": metadata.get("batch_size", config.get("batch_size")),
        "Training rows": metadata.get(
            "num_train_samples",
            training.get(
                "train_count",
                mapping(model.get("metadata")).get("source_training_rows"),
            ),
        ),
        "Validation rows": metadata.get(
            "num_validation_samples", training.get("validation_count")
        ),
        "Validation loss": metadata.get("validation_loss"),
        "Loss name": metadata.get("loss_name"),
        "Validation MSE": metadata.get("validation_loss")
        if metadata.get("loss_name") == "mse"
        else validation.get("mse"),
        "Validation MAE": metadata.get("validation_mae", validation.get("mae")),
        "Metric provenance": "Active bundle metadata; flat current fields take precedence over legacy training_config/validation_metrics. Never latest Training result.",
    }


def render_status(snapshot: BootstrapDashboardSnapshot) -> str:
    """Render the fast terminal summary using the same truth rules as the GUI."""
    return "\n".join([
        "Morpion bootstrap",
        f"Run: {snapshot.work_dir.name}",
        f"Status: {snapshot.status}",
        snapshot.status_detail,
        f"Generation: {display(snapshot.generation)}",
        f"Tree: {display(snapshot.tree_nodes)} nodes",
        f"Certified: {display(snapshot.record.get('current_best_total_points'))} points",
        f"Frontier: {display(snapshot.frontier.get('current_best_total_points'))} points",
        f"Evaluator: {display(snapshot.evaluator_name)}",
        f"Dataset: {display(snapshot.dataset_rows)} rows",
        f"Last update: {age_label(snapshot.updated_at)}",
        *[f"Artifact error: {error}" for error in snapshot.errors],
        "",
        "Dashboard:",
        "  chipiron-bootstrap",
    ])
