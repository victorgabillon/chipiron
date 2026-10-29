# Lean scientific CI: explicit selection, unchanged FULL contract

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Base: `f7230d13ba6dd5dfba1e5c88fd741907af8286bc`.
Implementation: `b67bd94572a1398bd82cb6187e70c0e20240a06c`.
Branch: `perf/lean-scientific-ci`.

Worktree:
`/home/pompote/oldata/victor/morpion-scratch/lean-ci-20261024/worktree`.
The environment, fixtures, logs and reports are new and use the secondary ext4
filesystem. No production code, scientific policy, checkpoint format, historical
experiment, protected worktree or release workflow is changed.

## What actually caused the delay

The retained baseline contains 2,691 selected identities: 2,689 passes and two CUDA
skips, with four standard exclusions. Its serial pytest run took 2,235.57 seconds.
Summed test phases total 2,212.86 seconds; 2,132.85 seconds (96.4%) are Morpion.
Setup accounts for 39.62 seconds and calls for 2,171.76 seconds. Construction inside
test bodies is included in calls, not reported as fixture setup.

The dominant groups are complete real process-death matrices across publication
and numeric contracts, repeated Dataset parity populations/policies, and the
large-manifest allocation regression. The 27-case Dataset scale matrix costs
125.86 seconds in that covered baseline; the new and old atomic consumer death
matrices cost 101.29 and 98.11 seconds; one large-manifest test costs 75.44 seconds.
They exercise distinct assertions/contracts and remain in FULL. There is no proof
justifying deleting them as redundant.

The largest baseline test file is 162.16 seconds, versus 2,212.86 summed seconds.
This does not demonstrate a single dominant file that needs a new sharding system.
Splitting repositories would leave almost all measured work in Morpion. No broad
new profiling study, scheduler, test database or dependency-selection framework was
created. The existing test metrics and `/usr/bin/time -v` supply the evidence.

## Implemented contract

The [developer guide](../testing.md) provides three commands:

```bash
make test-fast
make test-pr
make test-full
```

FAST selects explicit `fast` cases. PR selects `not exhaustive`; FULL applies only
the original `not integration and not external_data` exclusions. FAST is a subset
of PR, which is a subset of FULL. New unmarked tests automatically enter PR and FULL.
Neither new marker changes assertions or performs skips. All commands use the
existing reporter and two pytest processes with `--dist=loadfile`.

FAST and PR omit coverage instrumentation for iteration; FULL preserves the original
line, branch and subprocess coverage configuration. A partial/no-coverage run is
never described as complete coverage. PR must be supplemented by the explicitly
documented affected-component FULL matrices. Shared scientific/recovery/dependency
changes require whole-repository FULL at the final integration boundary.

FAST covers Values/certainty, FORCED(35)/ESTIMATE(42), all nine legacy policies on
small inputs, all five confidence-v2 policies and four gates, canonical references,
DAG scoring, both committed evaluator loaders, selector and RNG continuation,
Growth commit/retry, accepted Dataset/Training identity, exact receipts, and real
process deaths before and after current acceptance boundaries. It is not an import
smoke suite. Full process-death, fsync, concurrency, stale-worker and scale matrices
remain available under their original test identities.

The reporter now checks selected identities against setup/call/teardown counts.
Missing or duplicate execution makes a nominally successful invocation fail.
Setup skips legitimately have no call phase. Collection-only evidence is explicitly
labeled. Nine new reporter regressions exercise real collection/execution and
malformed evidence; nine new 30-node Dataset cases supply small all-policy checks.
No fixed permanent test count is embedded in the implementation.

## Input sharing and exact output parity

The scale test now constructs one immutable input per size per pytest worker/module,
instead of one per policy. Source files/directories are read-only while shared, and
their complete fingerprints are verified at fixture teardown. Permissions are then
restored solely for normal temporary-fixture cleanup. Every policy owns its mutable
SQLite index and JSONL outputs and still performs the independent reference and
indexed assertions. Corruption tests keep their original private sources.

The original 1k/5k/10k identities and assertions remain. Only the source argument
comes from a fixture. Both the old and new 27-case runs pass, despite pytest grouping
the reused inputs differently. All 27 complete legacy JSONL files and all 27 indexed
JSONL files also match before/after as hash multisets. There is no rounding or row
normalization. All 164 other existing test bodies in modified files are AST-identical.

| Matched 27-case run, serial, no coverage | Before | Shared input |
| --- | ---: | ---: |
| Pytest seconds | 80.87 | 75.34 |
| Command wall seconds | 81.96 | 76.44 |
| CPU seconds, user + system | 49.46 | 46.39 |
| Maximum process RSS MiB | 303.42 | 304.55 |
| Retained fixture allocation MiB | 293.21 | 242.46 |
| Large input constructions | 27 | 3 |

The observed savings are 5.52 command seconds, 3.07 CPU seconds and 50.75 MiB of
retained allocation. This is a modest optimization, not the main reason FAST is
fast. Peak RSS is essentially unchanged. These fresh-process comparisons use the
same synthetic populations and expectations; filesystem caches are not flushed.

## Measured validation levels

One final FULL run passed against the committed implementation `b67bd945`:
**2,707 passed, two CUDA skips, four unchanged standard exclusions, 141 warnings**.
Every original selected identity executed with the expected setup/call/teardown
accounting; no identity is missing or duplicated. Its 21m16s pytest wall time is
42.9% below the retained 37m15s serial baseline, principally from using the already
supported two-process configuration. This is not a measured GitHub speedup.

| Selection | Result | Pytest seconds | Command wall seconds | CPU seconds | Maximum process RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| FAST, loadfile | 145 passed | 83.08 | 83.78 | 102.21 | 583.43 |
| FAST, worksteal | 145 passed | 80.79 | 81.30 | 104.83 | 584.35 |
| PR, loadfile | 2,560 passed, two CUDA skips | 711.92 | 712.62 | 764.89 | 914.81 |
| FULL, loadfile, coverage | 2,707 passed, two CUDA skips | 1,276.15 | 1,276.87 | 2,051.56 | 884.72 |

FULL coverage is identical to the baseline: 80.2764% combined, 83.2749% statements,
68.4264% branches, with identical executed/missing line and branch sets in all 468
reported modules. The [complete JUnit evidence](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/lean_ci/evidence/full.xml) and
[validation summary](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/lean_ci/evidence/validation-summary.json) retain the exact
tested commit, exclusions, counters and artifact identities.

FAST meets the 1–3 minute target. PR is 11m52s, above the 5–10 minute target; this
change retains a broad 2,562-case selection and does not claim that target was met
or that it is impossible to improve further. Further filtering needs an explicit
review of the remaining expensive cases. The complete original suite is preserved.

The scheduler comparison uses the same 145 cases, separate fresh processes and two
workers, one experiment at a time. A 2.48-second command-wall difference from one
pair is insufficient evidence to change the existing scheduler. `worksteal` also
used slightly more CPU and can repeat module fixtures across workers. Keep
`loadfile`. This does not prove it is universally optimal for FULL.

The final FULL run supplies a stronger imbalance check: summed execution phases
are 2,501.51 seconds, giving a two-process lower bound of 1,250.76 seconds for
those observed task durations. Actual command wall time is 1,276.87 seconds, only
26.11 seconds above that bound, including startup, collection and reporting. There
is little demonstrated idle-time benefit available from a more complex scheduler
on this run. This conditional bound does not remove CPU/I/O contention or predict
different hardware.

The first FAST loadfile measurement preceded the implementation commit; only a
conftest line-wrap reformat and documentation changed before that commit. The
worksteal, PR and final FULL measurements ran against committed `b67bd945`.

GNU time's maximum RSS is the largest individual process maximum from the command
and its descendants, not simultaneous aggregate memory across workers. CPU time
includes descendants. Retained allocation is `du` after each invocation, not peak
temporary disk usage. The resource JSON includes I/O block counts and summed pytest
phase times; parallel phase sums are not wall time. Resource experiments were not
run concurrently. Process-local allocation assertions remain valid in parallel FULL.
Retained fixture allocation is 16.35 MiB for FAST, 870.43 MiB for PR and 1,229.63 MiB
for FULL. Summed setup times are 0.54, 30.68 and 47.98 seconds respectively; the
shared Dataset fixture moves construction from calls into setup (0.015 to 0.265
seconds in the matched matrix), so those phase labels alone are not a speedup.

Collection-only wall time is 10.75 seconds including startup/imports. A bare Python
startup takes 0.01 seconds; a fresh process importing the actual Growth runtime
takes 2.37 seconds and 245.05 MiB maximum RSS. These samples do not establish every
crash child's startup cost. Source/wheel build takes 2.01 seconds and local no-deps
reinstallation 1.09 seconds in the prepared environment. These are not uncached
GitHub dependency-install times. Static gates take 59.79 seconds and report zero
new diagnostics (206 current against the unchanged 209 baseline).

## Identity, isolation and CI/release compatibility

All original normally selected identities remain; the only additions are the 18
cases above. [Selection evidence](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/lean_ci/evidence/selection.json) records the exact
base comparison, exclusions and subsets. The full selected list and raw-report
fingerprints are retained, along with explicit execution accounting.

Anemone is still the exact `c015194533857a52d07bafdd603b762314692cfb` implementation,
wheel SHA256 `16aaa5b3a3ac339b8aef34ef75a93134136ed07010e9c2da0816d412dd109c47`.
All 289 Anemone and 476 Chipiron production Python source hashes match the reviewed
sources. The rebuilt Chipiron wheel identity is recorded separately. No dependency
was published or silently upgraded. The version-string/release prerequisite from
the previous integration remains; unspecified public `0.2.26` is not an identity.

Test outputs now honor the existing `CHIPIRON_OUTPUT_DIR` override before adding
invocation and worker identities. This keeps SQLite stores and application outputs
isolated on the selected scratch disk. `TMPDIR` and pytest basetemps are deliberately
short enough for Unix sockets. Source corruption and process-death tests retain
private directories and independent process state.

The post-FULL audit verifies all 41 pre-existing Chipiron and 12 Anemone worktree
registrations, HEADs, statuses and tracked-diff hashes unchanged. Both the new and
protected installed package source hashes remain unchanged. The final report and
README/guide updates are a documentation-only descendant of the tested commit;
executable, test, Makefile, pytest/tox and workflow content is verified unchanged.

GitHub's required `pytest`, quality, static and aggregate `test` jobs are unchanged.
The existing `tox -e py313` full command, permissions, coverage configuration,
15-minute pytest-step timeout and release workflow are unchanged. Only nightly and
manual triggers are added to CI. No required check is replaced by FAST or PR, and
no branch-protection rule is modified. The local targets accelerate development;
they do not make the mandatory GitHub PR gate a partial suite.
Bare `make` now selects FAST as its first ordinary target. Existing installation
targets remain unchanged and are explicit in the README and testing guide.

No GitHub runner was measured or activated here. Local serial timings must not be
equated with GitHub's existing two-worker execution. The local FULL comparison
primarily measures using that already-supported concurrency, not a new parallel
optimization for GitHub. The 15-minute remote limit must be verified on the actual
runner with the correctly identified dependency before claiming it is satisfied.
Local FULL is longer than that limit; this work does not demonstrate that the
GitHub timing target is met. No timeout was widened to disguise this limitation.
Keep the mandatory full gate until any proposed replacement is demonstrated and
explicitly approved.

Known representative-parent/order and legacy fallback-certainty limitations remain
visible in the [source integration report](morpion_bounded_anemone_exact_scoring.md).
This CI organization changes no scientific interpretation and enables no model.

## Size, review and next workflow

Implementation delta: **zero production lines**, 21 configuration lines, 37 lines
in the existing reporter, 265 added/12 removed test lines, and 102 guide lines.
The tests mostly add per-case markers plus the shared fixture and nine execution
guard regressions. There is no new scheduler, registry, cache or application module.
The fixture owns only immutable temporary inputs; the existing reporter owns the
execution-accounting check. The report/evidence commit is separate.

```bash
cd /home/pompote/oldata/victor/morpion-scratch/lean-ci-20261024/worktree
code docs/research/lean_ci/review.code-workspace
TMPDIR=/home/pompote/oldata/victor/ci24 git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt f7230d13 b67bd945 -- Makefile pyproject.toml .github scripts tests docs/testing.md
```

During development use FAST plus affected tests; before review use PR and the
component's complete matrices; run FULL once at the final high-risk integration
boundary. Documentation-only descendants do not require another FULL when verified
unchanged executable/test/configuration content and the tested revision are stated.
All work remains local pending review. No production workers, gen38, migration,
historical deletion, merge, publication or production activation was performed.
