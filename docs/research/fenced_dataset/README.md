# Reproduction and retained evidence

Read [`../morpion_fenced_dataset_publication.md`](../morpion_fenced_dataset_publication.md)
for contract, admission/acceptance boundaries, complexity and remaining gates.
`evidence/source-manifest.json` pins the exact implementation/parent commits,
all 474 production Python source hashes, wheel hash, Python and package versions.

All work was in the new isolated worktree:

```
/home/pompote/oldata/victor/morpion-scratch/fenced-dataset-20261023/worktree
```

Scratch environment, build outputs, complete logs, failed development probes and
new generated fixtures are siblings under `fenced-dataset-20261023/`. Existing
worktrees and their environments were read only. The copied environment was
updated with the new wheel using its own `python -m pip`, with no hard linking.
The secondary ext4 filesystem had sufficient free space; no historical cleanup
was performed. Short temporary root: `/home/pompote/oldata/victor/mds-bq3524r5`.
A new reproduction should create its own short secondary-disk temp directory;
long TMPDIR paths exceed the Linux AF_UNIX pathname limit in unrelated tests.

## Actual validation commands

From the isolated worktree, using its newly installed wheel environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
export CHIPIRON_TEST_INSTALLED=1
export QT_QPA_PLATFORM=offscreen
export TMPDIR=/home/pompote/oldata/victor/mds-bq3524r5
# XDG_CACHE_HOME, MPLCONFIGDIR and CHIPIRON_OUTPUT_DIR also point to this task's scratch.

../ci-venv/bin/python scripts/check_installed_distribution.py
../ci-venv/bin/python scripts/check_toolchain_versions.py
../ci-venv/bin/python -m ruff format --check .
../ci-venv/bin/python -m ruff check .
../ci-venv/bin/python scripts/static_analysis_ratchet.py --mypy-strict --output ../evidence/static-final
../ci-venv/bin/python scripts/ci_test_metrics.py -n 0 --durations=30 --durations-min=0 \
  --cov-report=json:.ci-reports/coverage.json --junitxml=.ci-reports/pytest.xml \
  --basetemp=../tmp/full-ci -m 'not integration and not external_data' \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety
../ci-venv/bin/python -m pytest -q --no-cov docs/research/fenced_dataset/test_admission_edges.py
../ci-venv/bin/python -m pytest -q --no-cov docs/research/fenced_dataset/test_baseline.py
```

The safety plugin redirects incidental leaderboard output to the generated test
workspace. The baseline tests characterize intentionally preserved **unfenced**
behavior; they do not assert that that behavior is safe. Normal protocol tests
are in `tests/environments/test_morpion_fenced_dataset*.py`. The full local CI
runs those as well as all inherited scientific, checkpoint, receipt, publication,
reader and nine-policy tests. Seven additional research cases test the installed
wheel without changing production sources. This is local validation, not hosted CI.

Before installation, wheel and sdist were built with
`python -m build --no-isolation --sdist --wheel --outdir ../dist`, then the wheel
was installed with `python -m pip install --no-deps --force-reinstall ...` into
the new copied environment. Complete source-byte comparison independently verifies
that no old wheel or source checkout was substituted during CI.

## Serial resource comparison

`measure.py` is a thin driver around the existing
`docs.research.durable_scientific_v2.measure.measured` instruments and fixture
builders. It adds no profiling framework. Run each repetition in a fresh process
with a different **nonexistent** root, sequentially. Use the original validated
58176ff3 installed environment for `legacy`, and the new installed environment
for `fenced`, with `PYTHONPATH=.` (never `src` for the baseline):

```bash
BASE_PY=/home/pompote/oldata/victor/morpion-scratch/atomic-patch-cursor-20261023/ci-venv/bin/python
"$BASE_PY" docs/research/fenced_dataset/measure.py ../tmp/reproduce-legacy-0 --protocol legacy
../ci-venv/bin/python docs/research/fenced_dataset/measure.py ../tmp/reproduce-fenced-0 --protocol fenced
```

Repeat with new names 1 and 2. The original six measured runs finished before full
CI started. All six source-manifest/record/update fingerprints match, and all
JSONL fingerprints match exactly. Sources have the same two Morpion states and
replay complexity. These small runs measure protocol overhead, including initial
index construction and output validation. They are not large-graph benchmarks;
OS source pages were already warm after fixture creation.

`legacy-*.json` / `fenced-*.json` contain raw wall/RSS/CPU/I/O/decode/fsync/disk
observations; `resources.json` gives medians. `matched-sources.json` binds exact
source bytes. `scientific-parity.json` independently hashes every emitting real
endgame case from full CI: 29 files / 53 paired rows; full JUnit identifies all
45 scientific parameter cases including the 16 gated refusals.

## Evidence map

- `full-ci.xml`, `validation-summary.json`, `full-ci-time.txt`: complete local CI.
- `admission-edges-final.xml`: seven supplementary installed-wheel cases.
- `baseline-rows-final.xml`: four corrected baseline characterizations.
- `science.xml`: focused scientific run, followed by the same cases in full CI.
- `static-final.log`, `toolchain.log`, `installed-guard.log`: final static/tool checks.
- `production-delta.json`: exact production line additions/removals and hashes.
- `protected-worktrees.json`: 36 prior registrations/HEADs/statuses/diffs unchanged.

Full stdout, first failed fixture attempts and header-only resource probes remain
in scratch. They are not counted as passing final tests or used as nonzero-row
throughput evidence. No failed result was normalized into a passing scientific
comparison. Nothing was pushed, merged, deployed or activated.
