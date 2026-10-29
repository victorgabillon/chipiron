# Reproduction and evidence

> Historical generated outputs are available through the [evidence archive](../evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Use Python 3.13 with the repository's validated dependency/toolchain versions.
All commands must use fresh scratch directories on the secondary disk. Never
point them at an experiment. Tests deliberately terminate local test processes;
no pipeline worker deployment or distributed execution is needed.

The implementation base is `8530ec6e12a6876c30aa41ce60be9c971a11c2e0`. Create
separate worktrees for that baseline and this implementation; do not reset any
existing worktree. `evidence/source-manifest.json` fingerprints production files
and records their equality with the installed wheel used for full CI.

The two old gaps can be reproduced on the baseline with:

```bash
python -m pytest -q \
  docs/research/safe_recovery_epochs/test_boundaries.py \
  tests/environments/test_morpion_reevaluation_publication_worker.py
```

New-protocol acceptance and scientific tests:

```bash
python -m pytest -q \
  tests/environments/test_morpion_atomic_reevaluation.py \
  tests/environments/test_morpion_atomic_reevaluation_crashes.py \
  tests/environments/test_morpion_atomic_reevaluation_science.py
python -m pytest --no-cov -q \
  docs/research/atomic_reevaluation/test_ack_boundaries.py
```

For the resource comparison, run each phase in a fresh process, **serially**,
with no CI/test job competing for resources. `atomic_python` is the validated
Python executable; `atomic_current` and `atomic_base` identify the worktrees;
`atomic_scratch` must be a new secondary-disk directory. Use a short secondary
`TMPDIR` (for example a fresh `/home/pompote/oldata/victor/mac-XXXXXXXX`), since
multiprocessing AF_UNIX sockets have a fixed pathname limit.

```bash
for atomic_trial in 1 2 3; do
  for atomic_protocol in legacy atomic; do
    atomic_source="$atomic_current"
    if [ "$atomic_protocol" = legacy ]; then
      atomic_source="$atomic_base"
    fi
    for atomic_stage in prepare accept recover; do
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$atomic_source/src:$atomic_source" \
      "$atomic_python" \
        "$atomic_current/docs/research/atomic_reevaluation/measure.py" \
        "$atomic_scratch/$atomic_protocol-$atomic_trial" \
        "$atomic_protocol" "$atomic_stage"
    done
  done
done
```

`prepare` generates a two-node scientific source/checkpoint and publishes a fixture
proposal; initialization is outside its measured interval. `accept` deliberately
leaves the committed proposal awaiting acknowledgement. `recover` starts fresh,
validates the current authority and finishes acknowledgement. It does not
reapply. Per-phase JSON retains actual process RSS, CPU/I/O, fsync calls/times,
file-hash calls/bytes, logical and unique-inode disk usage and scientific
fingerprints. The fixture includes missing-node filtering and blend alpha 0.5;
there is no historical input or silent target normalization.

Optional `worker` mode requires its own fresh root. It measures the actual bounded
Reevaluation producer, with the existing explicit closed-source test adapter and
a deterministic scalar evaluator. It does not deploy a worker service.

The full local CI run builds and installs a wheel in a separate environment,
checks installed distributions, and invokes the normal CI metrics entrypoint
with `-n 0`, normal markers `not integration and not external_data`, and the
existing `validation_safety` plugin to redirect incidental leaderboard output
to generated temporary artifacts. Ruff/toolchain/static-analysis ratchet are
separate checks. Static debt is compared to the exact baseline; it is not erased.

See [the report](../morpion_atomic_reevaluation_patch_cursor.md) for the contract,
crash matrix, measurements, code-size delta and unresolved epoch prerequisites.
