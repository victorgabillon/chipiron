"""Verify partitioned CI evidence and conservatively select full/scoped/reused work.

This does not grant scientific publication authority. Unavailable reuse evidence
selects a new FULL run; only complete same-tree PR evidence can avoid that run.
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from .ci_test_metrics import PARTITION, SHARD_COUNT, TestMetrics, shard_for

FULL_LANES = tuple(f"full-{i}" for i in range(SHARD_COUNT))
LANES = (*FULL_LANES, "fast", "pr", "integration")
REQUIRED_JOBS = {
    "scope",
    "quality",
    "static",
    "full-evidence",
    "test",
    *(f"pytest ({lane})" for lane in LANES),
}
INPUTS = (
    "tests",
    "src/chipiron",
    "scripts",
    ".github/scripts",
    "Makefile",
    "pyproject.toml",
    ".github/ci-source-contracts.json",
    ".github/ci-cpu-constraints.txt",
    ".github/workflows/ci.yaml",
)
# Deliberately literal, reviewed prose paths: no general docs/** or suffix rule.
DOC_PATHS = {"docs/testing.md", "docs/ci_validation.md"}
MAX_REUSE_AGE = timedelta(days=7)


def require(condition: bool, message: str) -> None:
    """Refuse incomplete, ambiguous or contradictory evidence."""
    if not condition:
        raise ValueError(message)


def git(*args: str) -> str:
    """Read immutable local Git identities."""
    return subprocess.check_output(["git", *args], text=True).strip()


def source_identity() -> dict[str, Any]:
    """Describe the checkout without confusing a commit with its source tree."""
    return {
        "candidate": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "parents": git("show", "-s", "--format=%P", "HEAD").split(),
        "git_objects": {path: git("rev-parse", f"HEAD:{path}") for path in INPUTS},
    }


def digest(data: bytes) -> str:
    """Hash artifact bytes, never a filename or version string."""
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    """Decode a mandatory evidence file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    """Write only this job's private evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seal(root: Path) -> None:
    """Bind all retained runtime files, including raw coverage and distributions."""
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Evidence symlink")
        if path.is_file() and path.name != "artifact-files.json":
            files[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    write_json(root / "artifact-files.json", files)


def check_seal(root: Path) -> dict[str, str]:
    """Require exact artifact membership and bytes before interpreting receipts."""
    wanted = read_json(root / "artifact-files.json")
    actual = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Evidence symlink")
        if path.is_file() and path.name != "artifact-files.json":
            actual[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    require(bool(wanted) and wanted == actual, "Artifact membership/hash mismatch")
    return actual


def check_execution(metrics: dict[str, Any]) -> None:
    """Recheck real phases rather than trusting a green result or stored error list."""
    require(
        metrics["exit_code"] == 0 and not metrics["collect_only"], "No execution pass"
    )
    require(bool(metrics["selected_ids"]), "Empty execution")
    require(not metrics["execution_errors"], "Recorded execution error")
    recorder = TestMetrics()
    recorder.selected = metrics["selected_ids"]
    recorder.reports = metrics["reports"]
    require(not recorder.execution_errors(), "Incomplete/duplicate execution phases")
    require(
        all(
            r["phase"] in {"setup", "call", "teardown"}
            and r["outcome"] in {"passed", "skipped"}
            for r in recorder.reports
        ),
        "Failed/unknown execution outcome",
    )
    workers = metrics["worker_collections"]
    require(len(workers) == 2, "Expected both original xdist workers")
    for worker in workers.values():
        require(
            worker["selected"] == recorder.selected
            and worker["deselected"] == metrics["deselected_ids"]
            and worker.get("shard") == metrics.get("shard"),
            "Worker disagreement",
        )


def check_qualification(q: dict[str, Any], source: dict[str, Any]) -> None:
    """Validate all scientific receipts and exact FULL partition equivalence."""
    require(q["schema"] == 1 and q["source"] == source, "Wrong qualification source")
    require(set(q["lanes"]) == set(LANES), "Missing/extra runtime lane")
    contract = read_json(Path(".github/ci-source-contracts.json"))
    packages = contract["packages"] | contract.get("candidate_package_overrides", {})
    reference = q["reference"]
    require(
        reference["collect_only"] is True
        and reference["exit_code"] == 0
        and not reference["reports"]
        and reference.get("shard") is None,
        "Invalid independent unsharded reference",
    )
    full = reference["selected_ids"]
    excluded = reference["deselected_ids"]
    require(
        bool(full) and len(set(full)) == len(full), "Invalid FULL reference identities"
    )
    observed: list[str] = []
    common_identity = None
    common_dependencies = None
    for lane, record in q["lanes"].items():
        candidate = record["candidate"]
        require(
            all(candidate[key] == value for key, value in source.items()),
            "Lane source mismatch",
        )
        require(
            candidate["lane"] == lane
            and candidate["run_id"] == q["run_id"]
            and candidate["run_attempt"] == q["run_attempt"]
            and candidate["event"] == q["event"],
            "Foreign lane/attempt",
        )
        identity = record["identity-before"]
        require(
            identity == record["identity-after"], "Installed bytes changed during tests"
        )
        require(set(identity) == {*packages, "chipiron"}, "Scientific identity missing")
        chipiron = identity["chipiron"]
        require(
            chipiron["reference"] == source["candidate"]
            and chipiron["mode"] == "build-candidate"
            and chipiron["editable"] is False
            and chipiron["source_members"] > 0
            and chipiron["package_members"] > 0,
            "Wrong tested installed candidate",
        )
        if common_identity is None:
            common_identity = identity
        require(
            identity == common_identity, "Cross-runner scientific identity mismatch"
        )
        dependencies = {
            p["name"].lower().replace("_", "-"): p["version"]
            for p in record["dependencies"]
        }
        if common_dependencies is None:
            common_dependencies = dependencies
        require(
            dependencies == common_dependencies,
            "Cross-runner dependency versions changed",
        )
        for name, package in packages.items():
            require(
                dependencies[name] == package["version"]
                and identity[name]["members"] == len(package["files"])
                and identity[name]["origin"].endswith(
                    f"/site-packages/{package['module']}/__init__.py"
                ),
                "Scientific dependency mismatch",
            )
        require(
            bool(record["files"])
            and all(
                re.fullmatch(r"[0-9a-f]{64}", sha) for sha in record["files"].values()
            ),
            "Missing artifact hashes",
        )
        metric_file = "full-tests.json" if lane in FULL_LANES else f"{lane}-tests.json"
        require(
            {
                "candidate.json",
                "identity-before.json",
                "identity-after.json",
                "dependencies.json",
                metric_file,
            }
            <= record["files"].keys(),
            "Missing scientific receipt hashes",
        )
        metrics = record["metrics"]
        check_execution(metrics)
        if lane in FULL_LANES:
            index = FULL_LANES.index(lane)
            shard = metrics["shard"]
            require(
                shard
                == {
                    "index": index,
                    "count": SHARD_COUNT,
                    "algorithm": PARTITION,
                    "full_selected_ids": full,
                    "full_deselected_ids": excluded,
                },
                "FULL collection mismatch",
            )
            expected = [
                nodeid for nodeid in full if shard_for(nodeid, SHARD_COUNT) == index
            ]
            require(metrics["selected_ids"] == expected, "Wrong shard assignment")
            require(
                Counter(metrics["deselected_ids"])
                == Counter(excluded)
                + Counter(n for n in full if shard_for(n, SHARD_COUNT) != index),
                "Unaccounted shard deselection",
            )
            require(
                {".coverage", "full-coverage.json", "full.xml"}
                <= record["files"].keys(),
                "Missing FULL coverage/JUnit",
            )
            observed.extend(metrics["selected_ids"])
        else:
            require(metrics.get("shard") is None, "Unexpected auxiliary sharding")
    require(Counter(observed) == Counter(full), "FULL union is not exactly once")
    files = q["lanes"]["full-0"]["files"]
    require(
        sum(n.endswith(".whl") for n in files) == 1
        and sum(n.endswith(".tar.gz") for n in files) == 1,
        "Missing tested wheel/sdist artifacts",
    )


def api(path: str) -> Any:
    """Read GitHub provenance with a read-only token; never execute response content."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def artifacts_for(run_id: str) -> list[dict[str, Any]]:
    """Bound discovery; an unexpectedly large result is not partial evidence."""
    payload = api(f"actions/runs/{run_id}/artifacts?per_page=100")
    require(payload["total_count"] <= 100, "Ambiguous artifact pagination")
    require(isinstance(payload["artifacts"], list), "Malformed artifact list")
    return cast("list[dict[str, Any]]", payload["artifacts"])


def assemble(inputs: Path) -> None:
    """Read same-run downloaded evidence and publish a still-incomplete qualification."""
    source = source_identity()
    q: dict[str, Any] = {
        "schema": 1,
        "source": source,
        "complete": False,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "event": os.environ["GITHUB_EVENT_NAME"],
        "lanes": {},
    }
    remote = {a["name"]: a for a in artifacts_for(q["run_id"])}
    require(len(list(inputs.iterdir())) == len(LANES), "Wrong artifact directory count")
    for lane in LANES:
        name = f"ci-validation-{lane}-{source['candidate']}-{q['run_id']}-{q['run_attempt']}"
        root = inputs / name
        files = check_seal(root)
        artifact = remote[name]
        require(
            not artifact["expired"] and artifact["digest"].startswith("sha256:"),
            "Unavailable runtime artifact",
        )
        require(
            files == download_json(artifact, "artifact-files.json"),
            "Downloaded files differ from GitHub-authenticated manifest",
        )
        record = {
            key: read_json(root / f"{key}.json")
            for key in (
                "candidate",
                "identity-before",
                "identity-after",
                "dependencies",
            )
        }
        metric_name = "full-tests" if lane in FULL_LANES else f"{lane}-tests"
        record.update(
            metrics=read_json(root / f"{metric_name}.json"),
            files=files,
            artifact={key: artifact[key] for key in ("id", "name", "digest")},
        )
        q["lanes"][lane] = record
        if lane == "full-0":
            q["reference"] = read_json(root / "full-reference-tests.json")
    check_qualification(q, source)
    write_json(Path(".ci-reports/qualification.json"), q)


def check_coverage(inputs: Path, combined: Path) -> None:
    """Require the combined coverage report to retain every shard's lines and arcs."""
    q = read_json(Path(".ci-reports/qualification.json"))
    parts = [
        read_json(inputs / q["lanes"][lane]["artifact"]["name"] / "full-coverage.json")
        for lane in FULL_LANES
    ]
    final = read_json(combined)
    require(
        final["meta"]["branch_coverage"]
        and all(p["meta"]["branch_coverage"] for p in parts),
        "Lost branch coverage",
    )
    require(
        all(set(p["files"]) == set(final["files"]) for p in parts),
        "Coverage source membership mismatch",
    )
    for name, result in final["files"].items():
        fragments = [p["files"][name] for p in parts]
        for kind in ("lines", "branches"):

            def values(file: dict[str, Any], key: str) -> set[Any]:
                return {tuple(x) if isinstance(x, list) else x for x in file[key]}

            executed = set().union(*(values(f, f"executed_{kind}") for f in fragments))
            universe = values(result, f"executed_{kind}") | values(
                result, f"missing_{kind}"
            )
            require(
                all(
                    values(f, f"executed_{kind}") | values(f, f"missing_{kind}")
                    == universe
                    for f in fragments
                ),
                "Coverage statement/branch universe changed",
            )
            require(
                values(result, f"executed_{kind}") == executed,
                "Incomplete coverage union",
            )
        require(
            all(f["excluded_lines"] == result["excluded_lines"] for f in fragments),
            "Coverage exclusions changed",
        )
    q["complete"] = True
    q["coverage"] = {"sha256": digest(combined.read_bytes()), "totals": final["totals"]}
    write_json(Path(".ci-reports/qualification.json"), q)


def documentation_only(base: str, candidate: str) -> bool:
    """Allow only nonempty modifications to an explicit set of existing prose files."""
    changed = (
        subprocess
        .check_output([
            "git",
            "diff",
            "--no-renames",
            "--name-status",
            "-z",
            base,
            candidate,
        ])
        .decode()
        .split("\0")
    )
    changed = changed[:-1] if changed[-1] == "" else changed
    if not changed or len(changed) % 2:
        return False
    for status, name in zip(changed[::2], changed[1::2], strict=True):
        if status != "M" or name not in DOC_PATHS:
            return False
        for ref in (base, candidate):
            if not git("ls-tree", ref, "--", name).startswith("100644 blob "):
                return False
            text = subprocess.check_output(["git", "show", f"{ref}:{name}"]).decode(
                "utf-8"
            )
            if "\0" in text:
                return False
    return True


def check_reuse(
    main: dict[str, Any],
    pr: dict[str, Any],
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    q: dict[str, Any],
    tested: dict[str, Any],
    artifacts: list[dict[str, Any]],
    head_prs: list[dict[str, Any]],
    *,
    repository_id: int,
    workflow_id: int,
    now: datetime,
) -> None:
    """Authenticate the exact merge topology, tree, run and scientifically complete evidence."""
    require(len(main["parents"]) == 2, "Squash/rebase/direct pushes require FULL")
    require(
        pr["merged"] is True
        and pr["base"]["ref"] == "main"
        and pr["merge_commit_sha"] == main["candidate"]
        and pr["head"]["sha"] == main["parents"][1]
        and pr["head"]["repo"]["id"] == repository_id
        and pr["base"]["repo"]["id"] == repository_id,
        "Untrusted/nonmatching PR",
    )
    require(
        run["event"] == "pull_request"
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and run["workflow_id"] == workflow_id
        and run["path"] == ".github/workflows/ci.yaml"
        and run["repository"]["id"] == repository_id
        and run["head_repository"]["id"] == repository_id
        and run["head_sha"] == main["parents"][1],
        "Untrusted/failed workflow",
    )
    require(
        [p["number"] for p in head_prs] == [pr["number"]],
        "Ambiguous tested-head PR association",
    )
    # GitHub can clear a run's PR list after merge. The authenticated head-commit
    # association remains mandatory; a populated, conflicting run list is refused.
    require(
        [p["number"] for p in run["pull_requests"]] in ([], [pr["number"]]),
        "Ambiguous PR provenance",
    )
    age = now - datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
    require(timedelta(0) <= age <= MAX_REUSE_AGE, "Expired/future evidence")
    require(
        Counter(j["name"] for j in jobs) == Counter(REQUIRED_JOBS)
        and all(
            j["status"] == "completed"
            and j["conclusion"] == "success"
            and j["run_id"] == run["id"]
            for j in jobs
        ),
        "Incomplete required jobs",
    )
    source = q["source"]
    require(
        source["tree"] == main["tree"]
        and source["parents"] == main["parents"]
        and source["git_objects"] == main["git_objects"],
        "Merge inputs changed",
    )
    require(
        tested["sha"] == source["candidate"]
        and tested["tree"]["sha"] == source["tree"]
        and [p["sha"] for p in tested["parents"]] == main["parents"],
        "Unauthenticated tested commit",
    )
    require(
        q["event"] == "pull_request"
        and q["run_id"] == str(run["id"])
        and q["run_attempt"] == str(run["run_attempt"])
        and q["complete"] is True
        and re.fullmatch(r"[0-9a-f]{64}", q["coverage"]["sha256"]) is not None,
        "Incomplete/foreign qualification",
    )
    check_qualification(q, source)
    by_id = {a["id"]: a for a in artifacts}
    for record in q["lanes"].values():
        expected = record["artifact"]
        current = by_id[expected["id"]]
        require(
            not current["expired"]
            and all(current[k] == v for k, v in expected.items()),
            "Missing/expired/replaced scientific artifact",
        )


class ArtifactRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the GitHub credential to an artifact storage redirect."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Follow storage redirects without copying the API authorization header."""
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def download_json(
    artifact: dict[str, Any], member: str = "qualification.json"
) -> dict[str, Any]:
    """Check GitHub's archive digest before parsing a bounded JSON member."""
    require(
        not artifact["expired"] and artifact["size_in_bytes"] < 64 * 1024**2,
        "Expired/oversized qualification",
    )
    expected_url = (
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/artifacts/{artifact['id']}/zip"
    )
    require(artifact["archive_download_url"] == expected_url, "Foreign artifact URL")
    request = urllib.request.Request(
        expected_url,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.build_opener(ArtifactRedirect()).open(
        request, timeout=20
    ) as response:
        data = response.read(64 * 1024**2 + 1)
    require(
        len(data) <= 64 * 1024**2 and "sha256:" + digest(data) == artifact["digest"],
        "Qualification archive digest mismatch",
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(
            archive.namelist().count(member) == 1,
            "Ambiguous qualification member",
        )
        info = archive.getinfo(member)
        require(info.file_size < 64 * 1024**2, "Oversized qualification JSON")
        result = json.loads(archive.read(info))
        require(isinstance(result, dict), "Malformed qualification")
        return cast("dict[str, Any]", result)


def try_reuse(main: dict[str, Any]) -> dict[str, Any]:
    """Find one bounded latest successful PR run; any uncertainty falls back to FULL."""
    require(len(main["parents"]) == 2, "No supported two-parent merge")
    prs = api(f"commits/{main['candidate']}/pulls?per_page=100")
    matching = [
        p for p in prs if p["merge_commit_sha"] == main["candidate"] and p["merged_at"]
    ]
    require(len(matching) == 1, "No unambiguous merged PR")
    pr = api(f"pulls/{matching[0]['number']}")
    head_prs = api(f"commits/{pr['head']['sha']}/pulls?per_page=100")
    workflow = api("actions/workflows/ci.yaml")
    payload = api(
        f"actions/workflows/{workflow['id']}/runs?event=pull_request&status=success"
        f"&head_sha={pr['head']['sha']}&per_page=100"
    )
    require(payload["total_count"] <= 100, "Ambiguous run pagination")
    runs = sorted(payload["workflow_runs"], key=lambda r: r["id"], reverse=True)
    failures = []
    for run in runs[:3]:
        try:
            artifacts = artifacts_for(str(run["id"]))
            suffix = f"-{run['id']}-{run['run_attempt']}"
            candidates = [
                a
                for a in artifacts
                if a["name"].startswith("ci-qualification-")
                and a["name"].endswith(suffix)
            ]
            require(len(candidates) == 1, "No unique qualification artifact")
            q = download_json(candidates[0])
            require(
                candidates[0]["name"]
                == f"ci-qualification-{q['source']['candidate']}{suffix}",
                "Qualification name mismatch",
            )
            tested = api(f"git/commits/{q['source']['candidate']}")
            jobs = api(
                f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100"
            )
            require(jobs["total_count"] <= 100, "Incomplete job pagination")
            check_reuse(
                main,
                pr,
                run,
                jobs["jobs"],
                q,
                tested,
                artifacts,
                head_prs,
                repository_id=int(os.environ["GITHUB_REPOSITORY_ID"]),
                workflow_id=workflow["id"],
                now=datetime.now(UTC),
            )
            return {
                "mode": "reuse",
                "reason": "Authenticated equivalent PR merge tree",
                "candidate": main["candidate"],
                "parents": main["parents"],
                "run_attempt": str(run["run_attempt"]),
                "tested_candidate": q["source"]["candidate"],
                "tree": main["tree"],
                "pr": pr["number"],
                "run_id": str(run["id"]),
                "qualification_digest": candidates[0]["digest"],
            }
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
            zipfile.BadZipFile,
        ) as error:
            failures.append(str(error))
    raise ValueError("No qualifying PR evidence: " + "; ".join(failures))


def route() -> None:
    """Select work explicitly; failure to establish reuse always means new FULL."""
    result: dict[str, Any] = {"mode": "full", "reason": "Default complete validation"}
    try:
        event = os.environ["GITHUB_EVENT_NAME"]
        candidate = os.environ["GITHUB_SHA"]
        require(git("rev-parse", "HEAD") == candidate, "Wrong checkout")
        if event == "pull_request":
            payload = read_json(Path(os.environ["GITHUB_EVENT_PATH"]))
            base = payload["pull_request"]["base"]["sha"]
            if documentation_only(base, candidate):
                result = {
                    "mode": "docs",
                    "reason": "Only reviewed prose paths modified",
                    "base": base,
                    "candidate": candidate,
                }
        elif event == "push" and os.environ["GITHUB_REF"] == "refs/heads/main":
            result = try_reuse(source_identity())
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        zipfile.BadZipFile,
    ) as error:
        result = {"mode": "full", "reason": f"Evidence reuse refused: {error}"}
    result["tests_executed_by_scope_job"] = False
    write_json(Path(".ci-reports/validation-route.json"), result)
    print(json.dumps(result, indent=2))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"mode={result['mode']}\n")


def main() -> None:
    """Expose bounded workflow operations without a second test runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("route", "seal", "assemble", "coverage"))
    parser.add_argument("path", nargs="?", type=Path, default=Path(".ci-reports"))
    args = parser.parse_args()
    if args.operation == "route":
        route()
    elif args.operation == "seal":
        seal(args.path)
    elif args.operation == "assemble":
        assemble(args.path)
    else:
        check_coverage(args.path, Path(".ci-reports/full-coverage.json"))


if __name__ == "__main__":
    main()
