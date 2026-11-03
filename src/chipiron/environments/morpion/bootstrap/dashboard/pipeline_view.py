"""Saved schedule, resource and score semantics for the five existing pages."""

import hashlib
import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .view_model import age_label, mapping, number, timestamp

if TYPE_CHECKING:
    from .view_model import BootstrapDashboardSnapshot


def schedule_summary(snapshot: BootstrapDashboardSnapshot) -> dict[str, Any]:
    """Ordinary cycles, publication generations and activation are different units."""
    metadata = mapping(snapshot.run_state.get("metadata"))
    window = mapping(
        metadata.get("learning_window_v2", metadata.get("learning_window_v1"))
    )
    active = mapping(window.get("active"))
    source_cycle = number(active.get("source_cycle"))
    target_cycle = number(active.get("limit_cycle", active.get("target_cycle")))
    cycle = number(snapshot.run_state.get("cycle_index"))
    completed = remaining = None
    if (
        source_cycle is not None
        and target_cycle is not None
        and cycle is not None
        and source_cycle <= cycle <= target_cycle
    ):
        completed, remaining = cycle - source_cycle, target_cycle - cycle
    publications = mapping(metadata.get("durable_generation_publication_v1")).get(
        "generations"
    )
    accepted = (
        [
            r
            for r in publications
            if isinstance(r, dict) and isinstance(r.get("accepted_training"), dict)
        ]
        if isinstance(publications, list)
        else None
    )
    rounds = window.get("completed")
    rounds = (
        [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else None
    )
    # completed windows record explicit activation decisions, not Training acceptance.
    activations = (
        sum(r.get("outcome") == "activated" for r in rounds)
        if rounds is not None
        else None
    )
    generation = number(snapshot.generation)
    data_generation = number(snapshot.dataset.get("generation"))
    model_generation = number(snapshot.active_model.get("source_generation"))
    selection = mapping(
        mapping(snapshot.training.get("metadata")).get("continual_selection")
    )
    candidate = snapshot.training.get("selected_evaluator_name")
    committed = (
        {
            r.get("activation_request_sha256")
            for r in publications
            if isinstance(r, dict)
            and isinstance(r.get("activation_request_sha256"), str)
        }
        if isinstance(publications, list)
        else set()
    )
    pending: object = None
    if snapshot.activation_requests:
        # Same canonical JSON identity used by saved activation request receipts.
        digests = [
            hashlib.sha256(
                json.dumps(
                    r, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            for r in snapshot.activation_requests
        ]
        if any(digest in committed for digest in digests):
            pending = "Latest Training request has a committed activation receipt"
        else:
            pending = "Recorded latest Training request has no committed completion; activation/full refresh may be pending (not live phase evidence)"

    return {
        "Lookahead L · ordinary cycles": mapping(
            mapping(snapshot.config.get("metadata")).get("learning_schedule")
        ).get("growth_lookahead_cycles"),
        "Ready policy maximum · ordinary cycles": mapping(
            mapping(snapshot.config.get("metadata")).get("learning_schedule")
        ).get("max_growth_cycles"),
        "Ordinary Growth cycle index": cycle,
        "Active window source cycle": source_cycle,
        (
            "Active window cap cycle"
            if "learning_window_v2" in metadata
            else "Active window target cycle"
        ): target_cycle,
        "Completed ordinary cycles in window": completed,
        "Remaining ordinary cycles in window": remaining,
        "Reserved source publication": mapping(active.get("source")).get("id"),
        "Reserved source generation": mapping(active.get("source")).get("generation"),
        "Dataset publication generation": data_generation,
        "Latest Training generation": snapshot.training.get("generation"),
        "Accepted Training rounds · authority references": len(accepted)
        if accepted is not None
        else None,
        "Selected candidate · latest Training": candidate,
        "Selection decision": selection.get("decision"),
        "Actually active model": snapshot.evaluator_name,
        "Active model source generation": model_generation,
        "Activation count · completed windows": activations,
        "Pending activation": pending,
        "Dataset lag · publication generations": generation - data_generation
        if generation is not None
        and data_generation is not None
        and generation >= data_generation
        else None,
        "Model lag · publication generations": generation - model_generation
        if generation is not None
        and model_generation is not None
        and generation >= model_generation
        and snapshot.active_model.get("source") == "local_training"
        else None,
        "Active training observation age": age_label(
            next(
                (
                    r.get("updated_at_utc")
                    for r in snapshot.training_history
                    if r.get("generation") == model_generation
                ),
                None,
            )
        ),
        "Dataset observation age": age_label(snapshot.dataset.get("updated_at_utc")),
        "Active pointer observation age": age_label(
            snapshot.active_model.get("updated_at_utc")
        ),
    }


def resource_summary(snapshot: BootstrapDashboardSnapshot) -> dict[str, Any]:
    """Compare saved configuration with bundle/monitor measurements without GPU probes."""
    metadata = mapping(snapshot.model_manifest.get("metadata"))
    profile = mapping(metadata.get("torch_execution_profile"))
    configured = mapping(snapshot.config.get("metadata"))
    policy = mapping(configured.get("torch_execution_policy"))
    monitor = mapping(snapshot.runtime.get("monitor"))
    receipt = mapping(snapshot.runtime.get("receipt"))
    status = mapping(snapshot.runtime.get("status"))
    usage = mapping(snapshot.runtime.get("usage"))
    cuda = mapping(
        mapping(
            mapping(snapshot.training.get("metadata")).get("round_performance")
        ).get("cuda_after")
    )
    worker_environment = mapping(snapshot.runtime.get("worker_environment"))
    environment = mapping(worker_environment.get("environment"))
    return {
        "Configured training device": snapshot.config.get("training_device"),
        "Reported active-bundle device": metadata.get(
            "resolved_device", metadata.get("model_device")
        ),
        "Reported device name": metadata.get("cuda_device_name"),
        "Reported HIP version": profile.get("hip_version"),
        "Configured profile": policy.get("profile"),
        "Reported active-bundle profile": profile.get("effective"),
        "Configured intra/inter-op threads": [
            policy.get("intra_op_threads"),
            policy.get("inter_op_threads"),
        ],
        "Reported intra/inter-op threads": [
            profile.get("intra_op_threads"),
            profile.get("inter_op_threads"),
        ],
        "Reported candidate concurrency": metadata.get(
            "candidate_workers", environment.get("CHIPIRON_CANDIDATE_WORKERS")
        ),
        "Saved worker environment · historical receipt": environment or None,
        "Worker environment observed UTC": worker_environment.get("utc"),
        "Reported flat tensor cache used": metadata.get("flat_tensor_cache_used"),
        "Reported flat tensor cache rebuilt": metadata.get("flat_tensor_cache_rebuilt"),
        "Reported entity token cache used": metadata.get("entity_token_cache_used"),
        "Reported shared preparation": metadata.get("shared_preparation"),
        "Owner elapsed seconds · startup/cleanup included": receipt.get(
            "owner_elapsed_s"
        ),
        "Observed monitor-loop elapsed seconds · startup excluded": mapping(
            status.get("sample")
        ).get("boottime_elapsed_s"),
        "Monitor elapsed seconds": monitor.get(
            "elapsed_seconds", usage.get("elapsed_seconds")
        ),
        "Runtime observation": status.get("updated_utc"),
        "Runtime observation age": age_label(status.get("updated_utc")),
        "Configured continual limits": configured.get("continual_learning"),
        "Resource limits": receipt.get(
            "limits", mapping(snapshot.runtime.get("ready")).get("limits")
        ),
        "Peak summed process RSS MiB · shared pages may repeat": mapping(
            monitor.get("peaks")
        ).get("group_rss_mb"),
        "Peak workspace MiB": mapping(monitor.get("peaks")).get("workspace_mb"),
        "Peak scratch MiB": mapping(monitor.get("peaks")).get("scratch_mb"),
        "Minimum available RAM MiB": mapping(monitor.get("minimum")).get(
            "available_ram_mb"
        ),
        "Torch peak allocated bytes · latest Training process": cuda.get(
            "max_allocated_bytes"
        ),
        "Reported GPU total bytes · not Torch allocation": cuda.get(
            "device_total_bytes"
        ),
        "GPU utilization": None,
        "Unique physical process memory": None,
    }


def score_rows(snapshot: BootstrapDashboardSnapshot) -> list[dict[str, Any]]:
    """Certified observed scores only; publication time is never called discovery time."""
    start = timestamp(mapping(snapshot.runtime.get("receipt")).get("started_utc"))
    start_origin = "Declared owner start"
    if start is None:
        created = number(mapping(snapshot.runtime.get("owner")).get("created"))
        if created is not None:
            with suppress(OverflowError, OSError, ValueError):
                start = datetime.fromtimestamp(created, UTC)
            start_origin = "Saved owner process creation (declared start unavailable)"
    if start is None:
        return []
    rows = []
    best: int | float | None = None
    observations = [
        (
            row.get("updated_at_utc"),
            mapping(row.get("record_status")),
            "Dataset publication",
        )
        for row in snapshot.dataset_history
    ] + [
        (row.get("timestamp_utc"), mapping(row.get("record")), "Growth observation")
        for row in snapshot.history
    ]
    observations.sort(
        key=lambda item: timestamp(item[0]) or datetime.min.replace(tzinfo=UTC)
    )
    for observed, record, origin in observations:
        clock = timestamp(observed)
        points = number(record.get("current_best_total_points"))
        if (
            clock is None
            or clock < start
            or points is None
            or not (
                record.get("current_best_is_exact") is True
                or record.get("current_best_is_terminal") is True
            )
        ):
            continue
        best = points if best is None else max(best, points)
        rows.append({
            "time": (clock - start).total_seconds(),
            "value": best,
            "series": "Best certified total points",
            "origin": origin,
            "observed_at": observed,
            "elapsed_origin": start_origin,
        })
    return rows


def candidate_lineages(snapshot: BootstrapDashboardSnapshot) -> list[dict[str, Any]]:
    """Inspect nine bounded bundle manifests only on the explicit Evaluator page."""
    from .performance_evidence import EvidenceReader

    warnings: list[str] = []
    reader = EvidenceReader(snapshot.work_dir, warnings)
    rows = []
    generation = snapshot.training.get("generation")
    publication = (
        reader.publications.get(generation, {})
        if isinstance(generation, int) and not isinstance(generation, bool)
        else {}
    )
    accepted_attempt = mapping(publication.get("accepted_training")).get("attempt_id")
    for name, result in list(
        mapping(snapshot.training.get("evaluator_results")).items()
    )[:32]:
        result = mapping(result)
        path = result.get("model_bundle_path")
        bundle = (
            reader.read(
                snapshot.work_dir / path / "morpion_manifest.json", limit=1024 * 1024
            )
            if isinstance(path, str)
            else {}
        )
        continuation = mapping(mapping(bundle.get("metadata")).get("continuation"))
        rows.append({
            "Candidate": name,
            "Training generation": snapshot.training.get("generation"),
            "Accepted attempt · authority reference": accepted_attempt,
            "Selected": name == snapshot.training.get("selected_evaluator_name"),
            "Active same generation": name == snapshot.evaluator_name
            and snapshot.active_model.get("source_generation")
            == snapshot.training.get("generation"),
            "Parent bundle SHA256": continuation.get("parent_bundle_sha256"),
            "Previous weights loaded": continuation.get("loaded_previous_weights"),
            "Optimizer state loaded": continuation.get("optimizer_state_loaded"),
            "Cumulative optimizer steps": continuation.get(
                "cumulative_optimizer_steps"
            ),
            "Validation loss · this dataset only": result.get("validation_loss"),
            "Evidence": "; ".join(warnings)
            or "Saved candidate bundle; acceptance is shown separately",
        })
        warnings.clear()
    return rows
