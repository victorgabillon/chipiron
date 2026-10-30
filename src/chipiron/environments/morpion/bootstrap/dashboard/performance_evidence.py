"""Bounded joins of existing performance artifacts; never invoke scientific readers."""

import hashlib
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from .performance_timeline import number
from .view_model import mapping

MAX_ARTIFACT_BYTES = 256 * 1024
MAX_AUTHORITY_BYTES = 4 * 1024 * 1024
MAX_OBSERVATIONS = 2048
MAX_ATTEMPTS = 256
MAX_RECORDS = 16384
_ID = re.compile(r"[0-9a-f]{32}\Z")


def _json_number(text: str) -> float:
    """Reject nonfinite JSON clocks/metrics before they reach a table or JSON report."""
    value = number(float(text))
    if value is None:
        message = "nonfinite JSON metric"
        raise ValueError(message)
    return value


class EvidenceReader:
    """Read metadata only, and retain acceptance separately from computation outcome."""

    def __init__(self, work_dir: Path, warnings: list[str]) -> None:
        """Read the bounded authority metadata once, independently of runtime state loaders."""
        self.root = work_dir.resolve()
        self.warnings = warnings
        self.records: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
        self.training: dict[int, dict[str, Any]] = {}
        self.attempts: list[dict[str, Any]] = []
        self.publications: dict[int, dict[str, Any]] = {}
        self._seen: dict[tuple[object, ...], dict[str, Any]] = {}
        self.authority = self.read(
            self.root / "run_state.json", limit=MAX_AUTHORITY_BYTES
        )
        publication = mapping(
            mapping(self.authority.get("metadata")).get(
                "durable_generation_publication_v1"
            )
        )
        records = publication.get("generations", [])
        for record in records if isinstance(records, list) else []:
            if isinstance(record, dict) and type(record.get("generation")) is int:
                self.publications[record["generation"]] = record

    def read(
        self, path: Path, *, limit: int = MAX_ARTIFACT_BYTES, digest: str | None = None
    ) -> dict[str, Any]:
        """Bound bytes, reject escaping paths, and check exact accepted descriptor bytes."""
        try:
            contained = path.resolve().is_relative_to(self.root)
            if not contained:
                self.warnings.append(f"{path}: metrics path escapes workspace")
                return {}
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                self.warnings.append(
                    f"{path}: metrics artifact exceeds {limit} byte limit"
                )
                return {}
            if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
                self.warnings.append(f"{path}: accepted descriptor digest mismatch")
                return {}
            value = json.loads(
                raw, parse_float=_json_number, parse_constant=_json_number
            )
        except FileNotFoundError:
            if digest is not None:
                self.warnings.append(
                    f"Missing accepted evidence: {path.relative_to(self.root)}"
                )
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            self.warnings.append(f"{path}: {exc}")
        else:
            if isinstance(value, dict):
                return value
            self.warnings.append(f"{path}: expected JSON object")
        return {}

    def paths(self, directory: Path, pattern: str, limit: int) -> list[Path]:
        """Bound observation count; report truncation rather than silently drop work."""
        paths = sorted(directory.glob(pattern))
        if len(paths) > limit:
            self.warnings.append(
                f"{directory}: truncated to {limit} artifacts; evidence is partial"
            )
        return paths[-limit:]

    def add(
        self,
        generation: int | None,
        stage: str,
        observation: dict[str, Any],
        source: str,
        *,
        identity: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> str:
        """Flatten nested intervals and deduplicate projection/private-file copies."""
        if len(self._seen) >= MAX_RECORDS:
            warning = f"Observation limit {MAX_RECORDS} reached; evidence is partial"
            if warning not in self.warnings:
                self.warnings.append(warning)
            return "truncated"
        identity = identity or {}
        record = {**identity, **observation, "generation": generation, "stage": stage}
        # A missing interval cannot safely identify duplicate measurements.
        clock = (
            number(record.get("started_unix_s")),
            number(record.get("finished_unix_s")),
        )
        key: tuple[object, ...] = (
            generation,
            stage,
            str(record.get("pid")),
            *clock,
            str(record.get("evaluator_name")),
        )
        if clock == (None, None):
            key += (source,)
        previous = self._seen.get(key)
        if (
            previous is not None
            and previous.get("attempt_id")
            and record.get("attempt_id")
            and previous["attempt_id"] != record["attempt_id"]
        ):
            key += (str(record["attempt_id"]),)
            previous = self._seen.get(key)
        if previous is not None:
            if previous.get("parent_id") is None and parent is not None:
                previous["parent_id"] = parent
            if previous.get("elapsed_s") != record.get("elapsed_s"):
                self.warnings.append(
                    f"{source}: duplicate interval has inconsistent elapsed evidence"
                )
            if source not in previous["sources"]:
                previous["sources"].append(source)
            for field, value in record.items():
                if field not in previous or previous[field] is None:
                    previous[field] = value
            return str(previous["measurement_id"])
        identifier = f"m{len(self._seen)}"
        record.update(measurement_id=identifier, parent_id=parent, sources=[source])
        record.setdefault("evidence_status", "observed; acceptance unavailable")
        self._seen[key] = record
        self.records[generation].append(record)
        if stage.count(".") >= 8:
            self.warnings.append(f"{source}: nested observation depth truncated")
            return identifier
        for name, value in observation.items():
            if (
                name != "metrics"
                and isinstance(value, dict)
                and any(
                    k in value
                    for k in ("started_unix_s", "elapsed_s", "finished_unix_s")
                )
            ):
                self.add(
                    generation,
                    f"{stage}.{name}",
                    value,
                    f"{source}#{name}",
                    identity={
                        **identity,
                        "attempt_id": record.get("attempt_id"),
                        "evaluator_name": record.get("evaluator_name"),
                        "model_generation": record.get("model_generation"),
                        "source_tree_generation": record.get(
                            "source_tree_generation", record.get("source_generation")
                        ),
                        "patch_id": record.get("patch_id"),
                        "process_identity": record.get("process_identity"),
                    },
                    parent=identifier,
                )
        return identifier

    def performance_files(
        self,
        root: Path,
        generation: int,
        identity: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> None:
        """Join bounded private or public stage files at their original identity."""
        if identity is None:
            publication = self.publications.get(generation, {})
            scientific = mapping(publication.get("scientific_identity"))
            identity = {
                "publication_id": publication.get("id"),
                "source_tree_generation": scientific.get("growth_generation"),
                "source_dataset_generation": scientific.get("source_generation"),
            }
        directory = root / "pipeline" / "performance" / f"generation_{generation:06d}"
        for path in self.paths(directory, "*.json", MAX_OBSERVATIONS):
            if len(self._seen) >= MAX_RECORDS:
                self.warnings.append(
                    "Observation limit reached; remaining stage files not read"
                )
                break
            record = self.read(path)
            if (
                record.get("schema_version") == 1
                and record.get("generation") == generation
                and isinstance(record.get("stage"), str)
            ):
                self.add(
                    generation,
                    record["stage"],
                    record,
                    str(path.relative_to(self.root)),
                    identity=identity,
                    parent=parent,
                )

    def training_status(
        self,
        generation: int,
        status: dict[str, Any],
        source: str,
        identity: dict[str, Any],
    ) -> None:
        """Retain the parent round and per-evaluator observations from a status projection."""
        metadata = mapping(status.get("metadata"))
        round_metrics = mapping(metadata.get("round_performance"))
        if not round_metrics and status.get("status") == "done":
            self.warnings.append(
                f"Generation {generation}: Training round interval unavailable ({source})"
            )
        parent = None
        if round_metrics:
            parent = self.add(
                generation,
                "training",
                round_metrics,
                source + "#metadata.round_performance",
                identity=identity,
            )
        for name, result in mapping(status.get("evaluator_results")).items():
            result = mapping(result)
            observation = mapping(result.get("performance"))
            if not observation:
                observation = {"elapsed_s": result.get("elapsed_s")}
            self.add(
                generation,
                "evaluator",
                {**observation, "evaluator_name": name, "metrics": result},
                source + f"#evaluator_results.{name}.performance",
                identity=identity,
                parent=parent,
            )

    def load_training(self) -> None:
        """Join authoritative references first, then retain incomplete/unaccepted attempts."""
        accepted: dict[str, tuple[int, dict[str, Any]]] = {
            str(mapping(r.get("accepted_training")).get("attempt_id")): (
                g,
                mapping(r.get("accepted_training")),
            )
            for g, r in self.publications.items()
            if r.get("accepted_training")
        }
        roots: dict[str, Path] = {}
        selected = sorted(accepted, key=lambda attempt: accepted[attempt][0])[
            -MAX_ATTEMPTS:
        ]
        if len(accepted) > MAX_ATTEMPTS:
            self.warnings.append(
                f"Only latest {MAX_ATTEMPTS} accepted Training attempts included"
            )
        oldest = accepted[selected[0]][0] if len(accepted) > MAX_ATTEMPTS else None
        for attempt in selected:
            if isinstance(attempt, str) and _ID.fullmatch(attempt):
                roots[attempt] = (
                    self.root / "training_publications_v1" / "objects" / attempt
                )
            else:
                self.warnings.append("Malformed accepted Training attempt identity")
        for kind in ("objects", "staging"):
            for path in self.paths(
                self.root / "training_publications_v1" / kind, "*", MAX_ATTEMPTS
            ):
                if path.is_dir() and _ID.fullmatch(path.name):
                    roots.setdefault(path.name, path)
        for attempt, root in roots.items():
            reference = accepted.get(attempt)
            proposal = self.read(
                root / "proposal.json",
                digest=reference[1].get("descriptor_sha256") if reference else None,
            )
            admission = mapping(proposal.get("admission")) or self.read(
                root / "admission.json"
            )
            generation = reference[0] if reference else admission.get("generation")
            if type(generation) is not int:
                continue
            if not proposal and not admission and reference:
                admission = {"attempt_id": attempt, "generation": generation}
            if oldest is not None and generation < oldest:
                continue
            verified = bool(
                reference
                and isinstance(reference[1].get("descriptor_sha256"), str)
                and proposal
                and admission.get("attempt_id") == attempt
                and admission.get("generation") == generation
                and mapping(proposal.get("status")).get("generation") == generation
                and mapping(proposal.get("status")).get("status") == "done"
            )
            if reference and not verified:
                self.warnings.append(
                    f"Generation {generation}: accepted Training evidence missing or inconsistent"
                )
            status = mapping(proposal.get("status"))
            outcome = (
                "accepted"
                if verified
                else "accepted reference; incomplete evidence"
                if reference
                else "unaccepted"
                if proposal
                else "incomplete"
            )
            if status.get("status") == "failed" and not reference:
                outcome = "failed"
            dataset = mapping(admission.get("dataset"))
            source = mapping(dataset.get("source"))
            scientific = mapping(source.get("scientific_identity"))
            identity = {
                "attempt_id": attempt,
                "claim_id": admission.get("claim_id"),
                "training_generation": generation,
                "model_generation": generation,
                "source_dataset_generation": scientific.get("source_generation"),
                "source_tree_generation": scientific.get("growth_generation"),
                "dataset_attempt_id": mapping(dataset.get("reference")).get(
                    "attempt_id"
                ),
                "publication_id": attempt,
                "source_publication_id": source.get("id"),
                "evidence_status": outcome,
                "descriptor_sha256": reference[1].get("descriptor_sha256")
                if reference
                else None,
            }
            self.attempts.append({
                "generation": generation,
                **identity,
                "location": str(root.relative_to(self.root)),
            })
            if verified:
                self.training[generation] = status
            self.training_status(
                generation,
                status,
                str((root / "proposal.json").relative_to(self.root)),
                identity,
            )
            parents = [
                r
                for r in self.records[generation]
                if r.get("attempt_id") == attempt and r["stage"] == "training"
            ]
            self.performance_files(
                root,
                generation,
                identity,
                parents[0]["measurement_id"] if parents else None,
            )
        for attempt_record in self.attempts:
            siblings = [
                a
                for a in self.attempts
                if a["generation"] == attempt_record["generation"]
            ]
            attempt_record["multiple_attempts_observed"] = len(siblings) > 1
            attempt_record["retry_evidence"] = (
                "multiple attempts observed; order/outcome not inferred"
                if len(siblings) > 1
                else "unavailable"
            )
            for record in self.records[attempt_record["generation"]]:
                if record.get("attempt_id") == attempt_record["attempt_id"]:
                    record["retry_evidence"] = attempt_record["retry_evidence"]
            # Absence of completion is never evidence of a failed process.
            if attempt_record["evidence_status"] == "incomplete":
                self.warnings.append(
                    f"Generation {attempt_record['generation']}: incomplete Training attempt {attempt_record['attempt_id']}; duration unknown"
                )

    def load_activation(self) -> None:
        """Read only the small committed descriptor, not its payload inventory."""
        for generation, publication in self.publications.items():
            if not publication.get("activation_request_sha256"):
                continue
            identifier = publication.get("id")
            if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
                continue
            path = (
                self.root
                / "generation_publications_v3"
                / "objects"
                / identifier
                / ".generation_publication_v3.json"
            )
            descriptor = self.read(path, digest=publication.get("descriptor_sha256"))
            activation = mapping(descriptor.get("model_activation"))
            if not activation or activation.get("request_sha256") != publication.get(
                "activation_request_sha256"
            ):
                self.warnings.append(
                    f"Generation {generation}: missing/inconsistent activation receipt"
                )
                continue
            request = mapping(activation.get("request"))
            model = mapping(request.get("candidate"))
            identity = {
                "publication_id": identifier,
                "activation_request_sha256": activation.get("request_sha256"),
                "training_generation": request.get("training_generation"),
                "model_generation": model.get("generation"),
                "attempt_id": mapping(request.get("accepted_training")).get(
                    "attempt_id"
                ),
                "source_tree_publication_id": activation.get("previous_head"),
                "evidence_status": "accepted",
            }
            parents = [
                r for r in self.records[generation] if r["stage"] == "activation"
            ]
            for parent in parents:
                parent.update(identity)
            refresh = mapping(activation.get("refresh_performance"))
            if not refresh:
                self.warnings.append(
                    f"Generation {generation}: activation refresh interval unavailable"
                )
            if not parents:
                self.warnings.append(
                    f"Generation {generation}: activation total interval unavailable"
                )
            if refresh:
                self.add(
                    generation,
                    "activation.refresh",
                    refresh,
                    str(path.relative_to(self.root))
                    + "#model_activation.refresh_performance",
                    identity=identity,
                    parent=parents[0]["measurement_id"] if len(parents) == 1 else None,
                )

    def load_runtime(self) -> None:
        """Read bounded passive records; never use them as claim or liveness authority."""
        paths = self.paths(
            self.root / "pipeline/performance/runtime", "*.json", MAX_OBSERVATIONS
        )
        for path in paths:
            record = self.read(path)
            if record.get("schema_version") != 1 or not isinstance(
                record.get("stage"), str
            ):
                continue
            generation = record.get("generation")
            if generation is not None and type(generation) is not int:
                continue
            self.add(
                generation, record["stage"], record, str(path.relative_to(self.root))
            )
        if paths:
            self.warnings.append(
                "Runtime observations use bounded retention; totals cover retained evidence only. Open sampled waits are not completed intervals."
            )
