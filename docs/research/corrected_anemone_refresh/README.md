# Corrected Anemone / Chipiron integration reproduction

This investigation starts at Chipiron
`60fdf85bb26e34f7d502122f34f3208ae2fc19d9`, whose parent is the validated durable
Training head `dc142573a1163178ae8fd5fc9feb90ff8ffee21e`.
Anemone is exactly `74bbd1250f31b72497166e624659d547324a0e5a`.
Its distribution still says **0.2.26**: that version string alone does not identify
the corrected code. The new local wheel and all 288 installed module hashes are
recorded in `evidence/dependencies.json`. No release was published.

Use a new secondary-disk checkout and environment; never update a shared environment.
The actual worktree is
`/home/pompote/oldata/victor/morpion-scratch/anemone-chipiron-integration-20261024/worktree`.
The dedicated interpreter is its sibling `../venv/bin/python`. The environment was
copied without hard links from the previous validated environment, then only its
Anemone and Chipiron wheels were replaced. Anemone source was extracted from the
exact Git object into `../build/anemone` before building; its original worktree was
not used as a mutable build directory.

```bash
python -m build --no-isolation --wheel --outdir "$REFRESH_SCRATCH/wheels" "$REFRESH_SCRATCH/build/anemone"
python -m pip install --no-deps --force-reinstall "$REFRESH_SCRATCH/wheels/algorhino_anemone-0.2.26-py3-none-any.whl"
```

Set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src:.`, `QT_QPA_PLATFORM=offscreen`.
Redirect `TMPDIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `CHIPIRON_OUTPUT_DIR`,
`ML_FLOW_URI_PATH` and `ML_FLOW_URI_PATH_TEST` into the new scratch root. Use a short
secondary-disk `TMPDIR` for existing multiprocessing tests; this run uses
`/home/pompote/oldata/victor/ar24tmp`. The MLflow values must be SQLite URIs, not
plain paths. Do not use pytest distributed workers for these reproductions.

Run research and repository tests in separate pytest invocations. Existing test
collection modifies import paths, so combining them can shadow the repository's
`scripts` namespace with `chipiron.scripts`.

```bash
python -m pytest --no-cov -q docs/research/corrected_anemone_refresh \
  --basetemp="$REFRESH_SCRATCH/research" --junitxml="$REFRESH_SCRATCH/research.xml"
python -m pytest --no-cov -q tests/environments/test_morpion_reevaluation_publication_fence.py \
  --basetemp="$REFRESH_SCRATCH/fence" --junitxml="$REFRESH_SCRATCH/fence.xml"
```

`test_integration` calls the corrected `refresh_with_evaluator(scope="all",
batch_size=...)` directly. It imports only the existing small independent oracle;
it does not execute the previous refresh algorithm. The certificate fixture
exhausts the actual one-legal-action Morpion endgame and binds its result to the
original state, objective and certificate. This is not a production proof authority.
Unknown direct FORCED values refuse without a validator.

The concurrency tests use deterministic events and the actual kernel flock. The
model case simulates identity visibility, because a durable M2 activation does not
exist. The cursor case performs a real synthetic patch acceptance. A red baseline
run must use the original installed Chipiron wheel with `CHIPIRON_TEST_INSTALLED=1`
and `PYTHONPATH=.`; otherwise `tests/conftest.py` selects the current checkout.

For matched resource measurements, generate a new accepted two-row Training result
using `python -m docs.research.durable_training.measure --root ...`. Prepare a fresh
graph once with the original dependency/environment and the previous measurement
driver, then copy its `prepare.json` and immutable `work/source-checkpoint` into two
new run directories. Run the following serially in fresh processes, one engine at a
time, with no overlapping CI or benchmarks:

```bash
python -m docs.research.corrected_anemone_refresh.measure "$REFRESH_SCRATCH/source" prepare --count 1000 --engine prior_kernel
# Copy only those newly generated source inputs into each new engine directory.
python -m docs.research.corrected_anemone_refresh.measure "$REFRESH_SCRATCH/prior_kernel" refresh --count 1000 \
  --engine prior_kernel --accepted-model-workspace "$REFRESH_SCRATCH/accepted-model/work"
python -m docs.research.corrected_anemone_refresh.measure "$REFRESH_SCRATCH/anemone" refresh --count 1000 \
  --engine anemone --accepted-model-workspace "$REFRESH_SCRATCH/accepted-model/work"
```

Use the original read-only interpreter for `prior_kernel`, the dedicated corrected
interpreter for `anemone`. Repeat at 5k only after checking the 1k results. The driver
reuses existing cProfile, StageMeasurement, codec/fsync and disk instrumentation.
In the corrected run `inference.prof` covers the entire Anemone refresh; the output
JSON calls that phase `complete_refresh`. Backup/selector are not run a second time.
The prior kernel is only a historical benchmark control, not a proposed production
implementation. Neither engine publishes a new authoritative generation.

For the installed-wheel CI gates, build/install the Chipiron sdist/wheel, set
`PYTHONPATH=.`, `CHIPIRON_TEST_INSTALLED=1`, then use the commands in
`../durable_training/README.md`, including the existing temporary-leaderboard safety
plugin and `-n 0`. Preserve the corrected Anemone wheel; do not let a clean dependency
resolution silently substitute published 0.2.26. Verify installed module hashes and
run `scripts/check_installed_distribution.py` before tests.
