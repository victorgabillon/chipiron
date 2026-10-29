# Reproducing the durable Training checks

> Historical generated outputs are available through the [evidence archive](../evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Use a new secondary-disk workspace based on the exact source/test head in
`evidence/source-manifest.json`. Do not use a historical experiment as a fixture.
The source manifest records Python, package/tool versions and all installed
production-source hashes. Local implementation ancestry is in `evidence/ancestry.txt`.

The isolated installed environment used here is
`/home/pompote/oldata/victor/morpion-scratch/durable-training-20261023/ci-venv`.
It was copied without hard links from the validated Dataset environment, then its
Chipiron wheel was replaced with a new sdist/wheel build. No previous environment
was repaired or modified. All 476 installed production sources match the recorded
checkout byte-for-byte. Use `python -m ...`, not copied console-script shebangs.

For a fresh reproduction, choose unused scratch paths. Export `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPATH=.`, `CHIPIRON_TEST_INSTALLED=1`, `QT_QPA_PLATFORM=offscreen` and redirect
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `CHIPIRON_OUTPUT_DIR` into that scratch directory.
`TMPDIR` must also be new and on the secondary disk; keep its absolute path short
because existing multiprocessing tests use AF_UNIX sockets.

```bash
python -m build --no-isolation --sdist --wheel --outdir "$TRAINING_SCRATCH/dist"
python -m pip install --force-reinstall --no-deps "$TRAINING_SCRATCH/dist/chipiron-0.2.4-py3-none-any.whl"
python scripts/check_installed_distribution.py
python scripts/check_toolchain_versions.py
python -m ruff format --check .
python -m ruff check .
python scripts/static_analysis_ratchet.py --mypy-strict --workers 1 --output "$TRAINING_SCRATCH/static"
```

Focused protocol/model/crash tests:

```bash
python -m pytest --no-cov -q \
  tests/environments/test_morpion_durable_training.py \
  tests/environments/test_morpion_durable_training_crashes.py \
  tests/environments/test_morpion_durable_training_edges.py \
  --basetemp="$TRAINING_SCRATCH/focused" \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety
```

The existing `test_morpion_fenced_dataset_science.py` compares complete reference
and bounded scientific outputs, including all nine policy names. The existing
Dataset tests and `docs/research/fenced_dataset/test_admission_edges.py` cover
shared protocol extraction and compatibility. No scientific oracle was replaced.

Full local CI, deliberately serial and using the repository's normal marker filter:

```bash
python scripts/ci_test_metrics.py -n 0 --durations=30 --durations-min=0 \
  --cov-report=json:.ci-reports/coverage.json \
  --junitxml=.ci-reports/pytest.xml \
  --basetemp="$TRAINING_SCRATCH/full-ci" \
  -m 'not integration and not external_data' \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety
```

Do not overlap the following resource measurements with CI. Each command starts a
fresh process; run commands serially, on newly generated identical tiny fixtures:

```bash
python -m docs.research.durable_training.measure --root "$TRAINING_SCRATCH/probe-1"
python -m docs.research.durable_training.measure --root "$TRAINING_SCRATCH/probe-1" --restart
```

Repeat on unused `probe-2` and `probe-3` roots. The driver reuses the existing
`durable_scientific_v2.measure.measured`, `StageMeasurement`, RSS/CPU/I/O, fsync,
cProfile and unique-inode disk instruments. It is not a new profiler framework.
Final report medians use `measure-6/7/8` and their fresh-process restarts only.
Earlier prototypes and interrupted/failed development runs remain separate in
scratch; their timings are not mixed with final results. The root source fingerprints,
row bytes and both model-weight hashes match across the final three repetitions.

The new 13 process-death points and two-process race run only on generated test
workspaces. These are synchronous local validation processes, not a deployed
bootstrap/Training worker service. No gen38 operation is part of reproduction.
