# Scientific validation without repeating the full suite during development

Use Python 3.14 and the published foundational releases pinned in `pyproject.toml`.
Validate installed wheel origins and exact package bytes; a version string alone
does not establish a scientific implementation identity.

```bash
make test-fast                 # representative science and real recovery, no coverage
make test-pr                   # broader regression set, no coverage
make test-full                 # every normally selected test, with coverage
```

Set `PYTHON=/path/to/venv/bin/python` when needed. All three commands use the existing
test-identity reporter and two local pytest processes with `--dist=loadfile`.
Bare `make` now selects FAST. Use `make requirements` or `make init` explicitly
for environment setup; those existing targets are unchanged.
They do not start production pipeline workers. `TEST_COMMON` can override pytest
concurrency; use `TEST_COMMON='-n 0 --strict-markers'` for serial experiments.
`TEST_ARGS` passes ordinary pytest arguments, including explicit files and node IDs.

## Selection contract

| Level | Marker expression (plus normal exclusions) | Purpose |
| --- | --- | --- |
| FAST | `fast` | Explicit representative tests; includes real subprocess deaths before and after commit. |
| PR | `not exhaustive` | FAST plus the ordinary regression suite, representative fault boundaries and concurrency tests. |
| FULL | No new marker filtering | All existing cases, including complete scale, process-death and fsync matrices. |

The normal exclusions remain `not integration and not external_data`. Markers
change selection only; they do not skip or alter assertions. Unmarked new tests
enter PR and FULL automatically. Add `fast` deliberately when a test strengthens
the small scientific suite. Do not combine `fast` and `exhaustive` on one case.
Large matrices use parameter-level markers so their representative cases can stay
in PR or FAST. No automatic dependency inference or result cache is involved.

FAST checks actual direct/tree/effective certainty, the move-34 FORCED(35) /
ESTIMATE(42) state, all nine legacy Dataset policies on small inputs, five admitted
confidence-v2 policies and four gates, canonical references, DAG model scoring,
both committed checkpoint loaders, selector/search/rollout RNG continuation,
Growth commit/retry, Dataset and Training acceptance, exact patch deduplication,
and real deaths on both sides of current acceptance boundaries.

## Affected-component validation is additional to PR

PR is a broad development selection, not permission to omit a changed component's
expensive cases. Run that component's tests **without** `not exhaustive`:

```bash
# Dataset readers, confidence, exact JSONL and publication/recovery
make test-full TEST_ARGS='tests/environments/test_morpion_streaming_dataset.py tests/environments/test_morpion_dataset_target_confidence.py tests/environments/test_morpion_scientific_training_reader.py tests/environments/test_morpion_fenced_dataset*.py'

# Atomic proposals, receipts, score identity and real producer/consumer deaths
make test-full TEST_ARGS='tests/environments/test_morpion_atomic_reevaluation*.py tests/environments/test_morpion_checkpoint_patch_receipts.py tests/environments/test_morpion_numeric_policy.py'

# Committed evaluator, checkpoints, continuation and certainty
make test-full TEST_ARGS='tests/environments/test_morpion_committed_evaluator_restore.py tests/environments/test_morpion_anemone_runner.py tests/environments/test_morpion_value_certainty.py tests/environments/test_morpion_derived_continuation.py'

# Model acceptance and recovery
make test-full TEST_ARGS='tests/environments/test_morpion_durable_training*.py'
```

Those are component runs, not whole-repository FULL passes. Changes spanning
boundaries require the union of their groups. Changes to shared publication,
checkpoint, recovery, dependencies or CI infrastructure require whole-repository
FULL at the final integration boundary. Keep real process termination; never
replace it with mocked exceptions to make a selected suite pass faster.

During iteration run FAST and affected tests. Before review run PR and the relevant
complete matrices. Before scientific/recovery/dependency merges or releases run
FULL plus the existing quality, installed-package and static gates. GitHub's
required `pytest` job still runs unfiltered `tox -e py314`; the aggregate `test`
gate still requires quality, pytest and static success. Releases retain their
existing full gate, ancestry checks and publication permissions. The same full CI
also runs nightly and can be dispatched manually. No branch protection is changed.

## Strict source checks

`tox -e static` requires zero diagnostics from strict mypy, Pyright and Pylint.
`tox -e ruff` requires clean formatting and lint. CI and releases use these gates;
the historical Python 3.13 static baseline remains archival evidence and does not
permit inherited diagnostics in the Python 3.14 gate.

## Isolation, measurements and evidence

Use a short `TMPDIR` on the intended scratch disk for Unix sockets. Set
`CHIPIRON_OUTPUT_DIR` there too; xdist adds invocation and worker identifiers before
creating application outputs and tracking databases. Each test retains its own
mutable outputs, model paths and SQLite indexes. The Dataset parity fixture shares
only read-only input files within one module/worker, checks their final hashes, and
restores permissions for temporary-directory cleanup. Corruption tests still create
private sources. Scheduling order must not affect assertions.

Run resource/timing experiments one at a time, using serial pytest where a single
test's resources are the subject of measurement. The FULL suite's allocation
regressions use process-local `tracemalloc`, not cross-process RSS thresholds.
FAST and PR deliberately omit coverage instrumentation; FULL retains the existing
line, branch and subprocess coverage configuration. Partial runs must never be
presented as complete coverage. `/usr/bin/time -v` reports CPU and maximum process
RSS; that maximum is not the simultaneous sum of every pytest process.

The existing `.ci-reports/tests.json` records exact selected/deselected identities,
all execution phases and outcomes. A nominally successful run now fails if setup,
call or teardown evidence is missing or duplicated. Setup skips are accounted for
explicitly; collection-only evidence is labeled and is not an execution pass.
Archive this file before another invocation overwrites it. Do not hardcode a test
count: compare identities against the chosen base and account for added cases.

Do not repeat FULL for a documentation-only descendant when verified unchanged
executable, test and configuration content has already passed. Record the actual
tested commit and the documentation-only relationship. Any relevant behavior
change after FULL requires an appropriately justified new final validation.
