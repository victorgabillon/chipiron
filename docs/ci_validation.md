# CI validation scope and evidence

Local development uses focused regression tests and relevant static/scientific
checks. Qualifying PRs validate their exact GitHub merge candidate completely.
Nightly and manual runs independently perform FULL. Existing release validation
is unchanged. No local FULL is required merely to repeat the forthcoming PR run.

## Four FULL jobs and early feedback

The original `tox -e py314` marker selection, coverage, two xdist workers,
`--dist=loadfile`, timeouts and thread limits remain unchanged. The metrics plugin
partitions the **already filtered** collection by SHA-256 of each node ID's file
path (first eight bytes, big-endian, modulo four). Every file stays on one runner.
There is no timing registry, exceptional-test list or omission of costly cases.

`full-0` first collects the original unsharded selection independently. All four
jobs retain their complete pre-partition collections, selected/deselected IDs,
worker agreement and setup/call/teardown outcomes. `full-evidence` rejects missing,
extra or repeated test identities and incomplete phases. Parameterizations and
setup skips retain the existing reporter's semantics. This proves selection
against the current candidate, without freezing a historical test count.

Each shard retains original line/branch/subprocess coverage and its raw database.
The aggregate installs the tested wheel at the original environment path and
combines the four databases. It checks identical source and statement/branch
universes, exclusions, and the exact union of executed lines and branches. Partial
coverage is never labeled complete. Every runtime lane preserves its original
before/after installed scientific verification, origin check and `pip check`.
Every FULL shard also retains strict distribution checks.

FAST and PR use their unchanged selections on separate runners. FAST becomes an
early independent result; its overlap with PR and FULL is intentional. Neither
selection replaces the required complete FULL or integration evidence.

## Conservative documentation scope

Only modifications to existing regular UTF-8 prose files `docs/testing.md` and
`docs/ci_validation.md` qualify. Additions, deletions, renames, symlinks, unknown
paths, empty comparisons and unreadable inputs require FULL. Dependency,
workflow, source-qualification, test, script and application changes require FULL.
The comparison uses the event's base commit and actual checked-out merge candidate.
Quality still runs. The required `test` check reports scoped validation and says
explicitly that runtime tests did not execute.

This initial allowlist is intentionally small. A docs-only PR does **not** produce
a complete scientific qualification: its main push therefore falls back to FULL.
Broader documentation-descendant reuse is deferred rather than treating scoped
validation as scientific evidence.

## Post-merge evidence reuse

Main pushes may reuse one complete successful PR run only when all these facts
are established through the repository's read-only GitHub API and artifact bytes:

- A unique merged same-repository PR points to the actual main commit. It is a
  two-parent merge, with the tested head and base parents in the same order.
- The successful run is this repository's `ci.yaml` pull-request workflow, with
  the matching head, one associated PR, exact attempt, and all required job names
  and successful conclusions. Cancelled/partial/skipped validation is rejected.
- The recorded PR candidate commit is verified through GitHub's Git commit API.
  Its entire tree, parents and explicit source/test/build/configuration Git objects
  equal main's. Commit SHA equality is **not** required: GitHub can create a new
  merge commit with the same tree and parents.
- The qualification archive matches GitHub's SHA-256 digest. Its seven runtime
  artifacts remain present, unexpired and bound to their recorded IDs and digests.
  GitHub-digest-verified file manifests bind installed identities, dependency versions, test phases,
  coverage, the independent FULL collection, and the tested wheel and sdist.
- All runtime installed identities agree before/after and across runners, match
  the tested candidate and frozen scientific dependency contract, and pass the
  complete phase/selection checks again. The aggregate coverage must be complete.
- Evidence is at most seven days old. Discovery is bounded; ambiguous pagination,
  unavailable artifacts/API data, malformed or contradictory receipts fail closed.

A new main base requires evidence for the new equivalent merge candidate. Changed
source/configuration/dependencies cannot reuse older evidence. Squash/rebase
merges, direct pushes, fork provenance and unsupported merge topology run FULL.
The merge commit is not rewritten as the tested installed-package reference: the
receipt retains the original tested SHA, tree, workflow run, attempt and hashes.
Historical scientific contracts and frozen replay semantics remain unchanged.

The initial rollout has no qualifying historical artifacts, so its first main run
may execute FULL. A reuse run executes quality and explicit scope verification;
`test` reports the original PR evidence and that no runtime tests ran on main.
The required check name remains `test`; no protection setting is modified.

## Evidence and operational limits

Runtime uploads retain `ci-validation-LANE-SHA-RUN-ATTEMPT`. A sealed file manifest
adds exact membership and hashes, including raw coverage. The aggregate produces
`ci-qualification-SHA-RUN-ATTEMPT`, with `qualification.json` and combined coverage.
Scope artifacts retain the decision, tested candidate/run and qualification digest.
Read-only GitHub credentials are stripped on artifact storage redirects.

Four extra runtime environments trade runner usage for latency. The seven runtime
jobs still have two pytest workers each and the same per-runner resource limits.
There is no cross-runner mutable environment or test-result cache. Queue capacity,
runner speed and the remaining PR lane limit realized savings. Owner review and a
normal authorized GitHub run must validate the hosted path before relying on the
estimated improvement; focused local tests cannot establish a hosted CI speedup.

The existing protection rule requires `test` and currently has `strict: false`.
This candidate does not make an old green PR check enforce an up-to-date merge
candidate before merge. If main advances, the actual merged parents/tree must have
qualifying evidence or the main push runs FULL. Owners must still check the latest
integration candidate when pre-merge freshness is required.
