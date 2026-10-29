# Reproduce the epoch boundary investigation

Use the report's exact base plus this research-only commit and the dependencies
listed in `evidence/source-manifest.json`. All destinations must be new secondary
disk scratch directories. No production epoch fork implementation is included.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:.
export QT_QPA_PLATFORM=offscreen
export TMPDIR=/secondary/new-epoch-study/tmp
export XDG_CACHE_HOME=/secondary/new-epoch-study/cache
export MPLCONFIGDIR=/secondary/new-epoch-study/cache/matplotlib
export CHIPIRON_OUTPUT_DIR=/secondary/new-epoch-study/test-output
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$CHIPIRON_OUTPUT_DIR"

python -m docs.research.minimal_rolling_recovery.reproduce \
  /secondary/new-epoch-study/fresh-history
python -m docs.research.durable_closed_snapshot.measure \
  /secondary/new-epoch-study/fresh-history discover --count 4 --history
python -m docs.research.safe_recovery_epochs.summarize \
  /secondary/new-epoch-study/fresh-history > /secondary/new-epoch-study/summary.json

python -m pytest -o addopts= -q \
  docs/research/safe_recovery_epochs/test_boundaries.py \
  tests/environments/test_morpion_durable_closed_crashes.py \
  tests/environments/test_morpion_rolling_recovery_report.py \
  tests/environments/test_morpion_reevaluation_publication_worker.py \
  tests/environments/test_morpion_durable_closed_science.py \
  tests/environments/test_morpion_checkpoint_parity.py \
  tests/environments/test_morpion_checkpoint_patch_receipts.py \
  tests/environments/test_morpion_patch_receipt_concurrency.py \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety \
  --basetemp="$TMPDIR/focused"
```

The first driver refuses an existing history destination and invokes 84 fresh
processes serially. `discover` adds one fresh process after G64. The graph has
five actual nodes, constant across the history. `summarize` only reads generated
measurement JSON. It does not measure an epoch implementation; the illustrative
epoch bytes are explicitly distinguished from real run-state metadata.

The research tests characterize gaps as well as existing protections. In
particular, two tests deliberately demonstrate why relabeling a workspace's
metadata is unsafe. Their success does not certify a fork. The Training test
stubs training computation and executes the actual completion/publication path.

For CI, build the sdist/wheel and install it in a **new isolated environment**,
check `scripts/check_installed_distribution.py`, then use the repository's
`scripts/ci_test_metrics.py` with `CHIPIRON_TEST_INSTALLED=1`, `PYTHONPATH=.`,
`-n 0`, `-m 'not integration and not external_data'`, secondary-disk `--basetemp`,
and the same validation-safety plugin. Run the pinned toolchain, repo-wide Ruff
checks and strict static ratchet against the exact base. Serial execution replaces
CI's two pytest workers; it does not launch pipeline/distributed workers.
Keep `TMPDIR` short (for example a fresh `/secondary/mre-XXXXXXXX` directory):
Python multiprocessing adds Unix socket names whose complete path has a small
platform limit. The first local CI attempt used an excessively long `TMPDIR`,
causing socket errors and a stuck DataLoader; its failure log is preserved.

The committed evidence contains selected raw observations, exact source/API
fingerprints and validation summaries. All 84 command results and sampled capsule
inventories are retained. Full temporary generations, process logs, build artifacts
and the isolated CI environment remain outside Git at the location recorded in
the source manifest. No new persistence or profiling framework is introduced.
