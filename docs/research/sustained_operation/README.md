# Bounded sustained-operation evidence

Use the installed wheel whose full Python inventory matches this worktree, exact
Anemone `37e48c40ceff430e373fd07bd00fcf41359bafda` (wheel SHA-256
`4ab0ffc88b17fb500a3729991fc4fa5c8f43d2b70be875f68a6d2d52ffe47d2f`),
Python 3.13 and the pinned toolchain. All destinations below must be new private
secondary-disk paths. Numerical library threads are fixed to one. Historical
workspaces and the unresolved earlier leaderboard preservation incident remain
protected. This task's before/after hash inventory is independent of that incident.

The existing research recipe exposes cadence/batch settings and records budget,
whole-process-group observations, authoritative acceptance observations and native
resource counters. It still uses the same four-worker shell launcher. It creates
no runtime scheduler, journal, provenance store or validation authority.

```bash
# From the isolated checkout; env.sh is retained task-private execution evidence.
source /home/pompote/oldata/victor/morpion-scratch/s26/env.sh
export PYTHONPATH=.
"$PYTHON_BIN" -m docs.research.operational_bootstrap.prepare NEW_INPUTS
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_INITIAL NEW_INPUTS \
  --nodes 5000 --initialize-only
# Run each matched case serially, without CI or another benchmark in parallel.
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_90 NEW_INPUTS \
  --initial-workspace NEW_INITIAL/workspace --session-only --cluster-seconds 180 \
  --growth-pause 90 --ownership-wait 0 --growth-steps 1
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_2 NEW_INPUTS \
  --initial-workspace NEW_INITIAL/workspace --session-only --cluster-seconds 180 \
  --growth-pause 2 --ownership-wait 0 --growth-steps 1
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_WAIT NEW_INPUTS \
  --initial-workspace NEW_INITIAL/workspace --session-only --cluster-seconds 180 \
  --growth-pause 2 --ownership-wait 60 --growth-steps 1
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_BATCH NEW_INPUTS \
  --initial-workspace NEW_INITIAL/workspace --session-only --cluster-seconds 180 \
  --growth-pause 2 --ownership-wait 60 --growth-steps 4
```

These are disposable independent copies of one newly generated frozen genesis.
They do not share pins, mutable files, worker caches or authority. No copied
history is resumed against changed executable sources. No learned model is
activated in a matched single-session benchmark. Both real learning-cycle
experiments instead initialize fresh graphs and explicitly prepare/activate the
exact accepted candidates at the two marked recipe transitions.

```bash
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_5K NEW_INPUTS \
  --nodes 5000 --cluster-seconds 180 --growth-pause 2 --ownership-wait 60 --growth-steps 4
# Recheck disk/RAM and worker quiescence before this separate larger experiment.
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce NEW_10K NEW_INPUTS \
  --nodes 10000 --cluster-seconds 180 --growth-pause 2 --ownership-wait 120 \
  --growth-steps 4 --max-seconds 3600
```

Limits include 1,000 initialization steps from the frozen example configuration,
one Growth cycle per invocation, the explicit steps per cycle, a bounded cluster
horizon, bounded retry cadence, 1,800 seconds per call, the total experiment cap,
8 GiB available RAM floor, 8 GiB sampled process-union RSS ceiling, 20 GiB disk
headroom, 2 GiB workspace and 1 GiB private scratch. Initialization can overshoot
the requested node target by one legal expansion; the step/time/resource budgets
are hard acceptance limits. Sampling can miss transient peaks. A resource stop
sends TERM through the existing supervisor and waits for whole-group shutdown.
The complete current checkpoint remains the crash-recovery source.

Routine checks reuse existing cold recovery, format-3/selector/both-RNG checks,
full current capsule/model hashes, exact accepted Dataset/Training validators and
legal terminal replay. They do not perform independent scientific reconstruction.
Both model activations still receive the existing full independent oracle. The
ordinary final Growth check uses routine validation; `--audit-every-check` restores
the previous extra full audit. A verification failure is preserved in its log and
`validation-failed-*.json`, raises before the recipe's next activation, and must be
investigated before any manual continuation or promotion. The recipe never catches
an unreadable authority to initialize another history.

```bash
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce RUN NEW_INPUTS \
  --verify --validation-level routine
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce RUN NEW_INPUTS \
  --verify --validation-level audit
"$PYTHON_BIN" -m docs.research.sustained_operation.collect RUN NEW_REPORT.json
"$PYTHON_BIN" scripts/report_morpion_rolling_recovery.py RUN/workspace > NEW_RETENTION.json
```

`decisions` cold-loads an immutable pre-activation or activated checkpoint into one
private runtime, executes at most 32 existing search steps, and discards it without
publication. Compare paired generations already proved equivalent by the activation
oracle, including identical initial search/rollout RNG hashes. Different selected
node sequences then witness an effect of model refresh; equal observed sequences
provide no evidence of changed decisions. Neither outcome establishes playing
strength. No second proof system or scientific parser is introduced.

The shell supervisor is also the manual start/restart interface. Use an already
initialized, verified workspace, the same installed sources, the persisted control
budget, `MORPION_SCIENTIFIC=1`, `MORPION_USE_INSTALLED_PACKAGES=1`,
`MORPION_CLUSTER_HEADLESS=1`, an explicit `MORPION_CLUSTER_MAX_SECONDS`,
`MORPION_GROWTH_WORKER_MAX_CYCLES=1`, `GROWTH_SLEEP_SECONDS=2`, and
`MORPION_WORKER_OWNERSHIP_WAIT_SECONDS=60` (or the measured larger-tree budget).
Set `MORPION_WORK_DIR` and invoke `bash scripts/launch_morpion_gnome_cluster.sh`.
To stop, send TERM to the retained supervisor PID and wait; TERM exits 143, normal
horizon exits 0. Require its whole-group shutdown confirmation before inspecting
or reusing private cache ownership. Restart with the identical command. Do not
run initialize on that workspace. Do not use SIGKILL as routine shutdown.

Retention remains fail closed; see `../morpion_sustained_retention.md`. The finite
experiment budgets bound storage by stopping. They do not provide indefinitely
bounded historical storage or permission to retire any original experiment.
