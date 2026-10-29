# Reproduction: scientific model activation investigation

Everything here is research code. There is no production activation route or new
transaction format. Base: `dc142573a1163178ae8fd5fc9feb90ff8ffee21e`.
Use a fresh isolated secondary-disk worktree and new scratch paths. Dependencies
are pinned by `evidence/source-manifest.json`; never import a historical experiment.

The interpreter used read-only was:
`/home/pompote/oldata/victor/morpion-scratch/durable-training-20261023/ci-venv/bin/python`.
Use `python -m ...` because copied environment console-script shebangs may point
elsewhere. Set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src:.`,
`QT_QPA_PLATFORM=offscreen`; redirect `TMPDIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`,
`CHIPIRON_OUTPUT_DIR` into the new scratch root. No previous environment is modified.

```bash
python -m pytest --no-cov -q docs/research/model_activation \
  --basetemp="$ACTIVATION_SCRATCH/tests" \
  --junitxml="$ACTIVATION_SCRATCH/tests.xml"
```

The existing-protocol regression command was:

```bash
python -m pytest --no-cov -q \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety \
  tests/environments/test_morpion_value_certainty.py \
  tests/environments/test_morpion_checkpoint_parity.py \
  tests/environments/test_morpion_atomic_reevaluation.py \
  tests/environments/test_morpion_atomic_reevaluation_crashes.py \
  tests/environments/test_morpion_atomic_reevaluation_science.py \
  tests/environments/test_morpion_durable_closed_snapshot.py \
  tests/environments/test_morpion_durable_closed_crashes.py \
  tests/environments/test_morpion_durable_closed_science.py \
  --basetemp="$ACTIVATION_SHORT_TMP/tests" \
  --junitxml="$ACTIVATION_SCRATCH/existing-regressions.xml"
```

Use a new writable short secondary-disk directory for `ACTIVATION_SHORT_TMP` and
`TMPDIR` in that command, since multiprocessing sockets have a path-length limit.
The successful run used `/home/pompote/oldata/victor/ma24tmp`.

`probe.reference` is an independent small default-max oracle: explicit scalar and
certainty comparisons, proof preconditions, effective-source selection and PV tie
ordering. It calls neither Anemone backup nor its effective-value/ordering helpers.
It uses O(N) test data; the measured refresh never calls it. `direct_phase` uses
bounded inference windows on the existing resident graph; the existing decoded
state cache and checkpoint payload remain separate O(N) limitations.

The stock-defect tests assert the observed incorrect behavior so a future upstream
fix makes them fail visibly and forces this report to be updated. Successful tests
are evidence of the investigation, not a claim that stock refresh is corrected.
The experimental complete pass is checked against the independent oracle, including
actual reachable partial nodes, actual Morpion terminal proofs, DAG transpositions,
repeated models, confidence gates and exact row bytes.

Create one tiny accepted Training result using the existing driver (two rows,
two candidates, one epoch); this is not a deployed worker or model activation:

```bash
python -m docs.research.durable_training.measure --root "$ACTIVATION_SCRATCH/accepted-model"
```

Run these commands serially in separate fresh processes. Finish and inspect 1k and
5k before attempting 10k. Do not overlap resource measurements with tests or other
benchmarks. Use new directories for each count.

```bash
python -m docs.research.model_activation.measure "$ACTIVATION_SCRATCH/n1000" prepare --count 1000
python -m docs.research.model_activation.measure "$ACTIVATION_SCRATCH/n1000" refresh --count 1000 \
  --accepted-model-workspace "$ACTIVATION_SCRATCH/accepted-model/work"
python -m docs.research.model_activation.measure "$ACTIVATION_SCRATCH/n1000" restart --count 1000
```

Repeat for 5000 and 10000. These are lower bounds on generated node counts: real
expansion batches produce 1003/5013/10018 nodes in this pinned implementation.
The measurement script reuses existing `StageMeasurement`, cProfile, codec/fsync
counters and disk accounting. The generated checkpoint/snapshot outputs are private
offline cost probes; they are not accepted model transitions. Source and target
model fingerprints are recorded. Model/scientific bytes are actually verified.

Graph fingerprints compare all values, certainties, provenance source, direct
version, PV and edge sets. Ordered child iteration is recorded separately, because
restore changes that representation. Neither representative-parent discrepancies
nor ordered-link differences are silently presented as byte parity.

The report distinguishes measured kernel costs, existing-protocol regression tests,
and the still-unimplemented M2 activation death/race matrix. Old scratch from initial
harness failures remains separate; final resource tables use only `n*-final` runs.
