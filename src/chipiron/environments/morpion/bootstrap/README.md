# Morpion Bootstrap Operator

Monitor an existing Morpion Solitaire workspace, inspect its records and models,
and deliberately launch its prepared experiment.

## Continual runtime boundary

`runtime_capabilities.py` defines structural contracts for checkpoint creation,
runtime discard, bounded growth, scientific export, patch application and
certificate-checked evaluator refresh. The existing Anemone runner implements
them. Growth, receipt handling, exports and activation consume the contracts
directly; external adapters may still delegate hooks dynamically.

Continual Growth validates its required operations before setting its node
budget or invoking automatic activation. Fresh activation validates its private
runtime before preparing a publication. Runtime checks reject missing and
noncallable hooks; existing consumers retain return-value validation.

Projection repair and accepted activation retries require no live runtime.
Training acceptance continues each candidate's own lineage and does not activate
a model. Growth remains the live-tree mutation and commit authority, using the
same locks, receipts, checkpoint formats and explicit opt-in settings. The public
`search_runner_protocol.MorpionSearchRunner` compatibility path is unchanged.

## Operator commands

Install dashboard support from this checkout:

```bash
pip install -e '.[dashboard]'
```

Choose a run once, then open its dashboard from any directory:

```bash
chipiron-bootstrap use /path/to/run
chipiron-bootstrap
```

Inspect it without opening a browser:

```bash
chipiron-bootstrap current
chipiron-bootstrap status
```

Preview the prepared command, then launch it when ready:

```bash
chipiron-bootstrap run --dry-run
chipiron-bootstrap run
```

`run` asks for confirmation and delegates to the workspace's own
`run_bootstrap.sh`, including its preflight. It stays in the foreground and
preserves signals and exit status. It never reconstructs scientific settings
from package defaults. A missing launcher is an error. `--yes` is available for
deliberate noninteractive launches.

All commands share the same workspace resolution: explicit `--work-dir`, then
`MORPION_WORK_DIR`, then the saved current run, then the current directory if it
contains `bootstrap_config.json`. `use` atomically saves a versioned selection
in the platformdirs user configuration directory; it does not change the run.

The dashboard has five views:

- **Overview**: status, generation, tree, dataset, certified record, frontier and
  learning trends. It refreshes every 15 seconds by default.
- **Record**: the certified board, its source and a separate frontier summary.
- **Tree**: lazy checkpoint inspection, clickable actions, node navigation,
  stored evaluations and a bounded local graph. Manual refresh is the default.
- **Evaluator**: active bundle provenance, training metrics and example diagnostics.
- **Operations**: pipeline evidence, storage, persistence, controls and raw details.

Auto refresh supports Off, 5s, 15s and 30s. Missing liveness evidence is shown as
`unknown`; a saved artifact alone never proves that a worker is running.
Monitoring charts use a bounded recent history (2,000 events / 4 MiB and 256
pipeline generations). Operations exposes full historical analysis on request.
Runtime controls preview the requested changes before saving and retain their
existing cycle-boundary semantics. Artifact-pipeline lifecycle is controlled by
the canonical workspace launcher, with no generic Start/Restart buttons.

For a remote terminal or a different port:

```bash
chipiron-bootstrap dashboard --no-browser --port 8501
```

The server binds to loopback. Core Chipiron does not require Streamlit; dashboard
support uses the pinned, Python 3.14-validated `dashboard` extra. A packaged
installation may use `pip install 'chipiron[dashboard]'` once this change is released.

## Advanced / internal launcher

The existing low-level launcher below remains available to prepare or configure
new experiments. These options are not needed to operate an already prepared run.

```bash
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --max-cycles 100000 \
  --max-growth-steps-per-cycle 5 \
  --save-after-seconds 10 \
  --save-after-tree-growth-factor 1.2 \
  --tree-branch-limit 1000000 \
  2>&1 | tee ~/oldata/victor/morpion_runs/big_run_01/launcher_console.log
```

Use `--verbose-checkpoint-logs` only when debugging Anemone checkpoint restore
or build internals. Normal growth-worker runs keep those low-level restore
phases and delta-candidate rejection logs suppressed.

### Evaluator family presets

The launcher default remains `canonical_8_linear_mlp_subsets`, the existing
canonical linear/MLP evaluator family.

Use `--evaluator-family canonical_linear_mlp_entity_transformer_small` to opt in
to the same canonical linear/MLP evaluators plus one small
`entity_token_transformer_value_net` evaluator.

### Persistence defaults

Normal Morpion bootstrap runs now use this persistence layout by default:

* runtime checkpoints: compressed Anemone checkpoints in `search_checkpoints/`
* training export mode: `sharded`
* tree export artifact consumed by dataset extraction: per-generation sharded
  manifest in `tree_exports_sharded/`

Compatibility modes remain available:

* `flat`: legacy single-file tree export for compatibility/debug only
* `both`: writes sharded artifacts plus the flat compatibility/debug export

Old plain `.json` runtime checkpoints still load, but newly written runtime
checkpoints use the compressed default checkpoint format.

### 📌 Notes

* `--work-dir` is where all artifacts are stored (checkpoints, models, logs, etc.)
* Make sure this directory is on a **large partition** (e.g. `~/oldata`)
* Logs are saved to `launcher_console.log`

### Run modes

`single_process` is the simplest legacy mode. It runs search, dataset export,
training, and checkpointing in one in-process loop. The command above uses this
mode by default.

`artifact_pipeline` is the Phase 8 file-driven mode. It splits work into staged
launcher invocations and adds autonomous dataset, training, and reevaluation
workers. This is the recommended mode for multiprocess or large runs.

For production artifact-pipeline runs, do not use `--pipeline-stage loop`; it is only a local/debug mini-orchestrator. Use the four autonomous stages: `growth`, `dataset_worker`, `training_worker`, and `reevaluation`.

All artifact-pipeline workers must share the same `--work-dir` on a filesystem
visible to every worker.

### Recommended 4-process artifact-pipeline launch

First launch creates `<work_dir>/bootstrap_config.json`. After that, all workers
must use the same bootstrap config. To change dataset/training/growth parameters,
edit the persisted config intentionally or start a new `work_dir`.

### Recommended startup order

Start workers in this order:

1. `growth` (creates initial artifacts and bootstrap_config.json)
2. `dataset_worker`
3. `training_worker`
4. `reevaluation`

Reevaluation may idle until an active evaluator exists.
Dataset/training workers may initially report `no_pending_work`; this is normal.

Use one shared work directory visible to all workers:

```bash
export MORPION_WORK_DIR=~/oldata/victor/morpion_runs/big_run_01
```

#### 1. Growth worker

Owns the live tree/checkpoint. Usually CPU/RAM-heavy. Run only one.

```bash
while true; do
  python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir "$MORPION_WORK_DIR" \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage growth
  sleep 2
done
```

#### 2. Dataset worker

Extracts supervised rows from saved tree snapshots.

```bash
while true; do
  python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir "$MORPION_WORK_DIR" \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage dataset_worker
  sleep 2
done
```

#### 3. Training worker

Trains evaluators and updates `pipeline/active_model.json`. Prefer a GPU.

```bash
while true; do
  CUDA_VISIBLE_DEVICES=0 python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir "$MORPION_WORK_DIR" \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage training_worker
  sleep 2
done
```

#### 4. Reevaluation worker

Uses the active evaluator to produce bounded reevaluation patches. Prefer a GPU.

```bash
while true; do
  CUDA_VISIBLE_DEVICES=1 python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir "$MORPION_WORK_DIR" \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage reevaluation \
    --reevaluation-max-nodes-per-patch 10000
  sleep 2
done
```

### Important config rule

Pass your intended runtime and dataset options on the very first launch, for example:

```bash
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir "$MORPION_WORK_DIR" \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage growth \
  --max-growth-steps-per-cycle 1000 \
  --save-after-tree-growth-factor 2.0 \
  --save-after-seconds 3600 \
  --require-exact-or-terminal \
  --min-depth 2 \
  --min-visit-count 3 \
  --max-rows 100000 \
  --use-backed-up-value \
  --dataset-family-target-policy none \
  --dataset-family-prediction-blend 0.25
```

Fresh bootstrap-generated evaluator specs use one training epoch. The launcher's
default family and named family presets carry their own per-evaluator epoch counts;
the `MorpionBootstrapArgs.num_epochs` option applies only to the direct single-model
fallback. Explicit evaluator specs override either default. Once
`bootstrap_config.json` exists, its resolved evaluator epochs remain authoritative
for later workers and resumes.

`tree_branch_limit` is a growth budget and may be increased or decreased when
resuming an existing tree. Increasing it allows a run that reached the previous
branch limit to continue growing.

### Healthy pipeline signs

After startup you should observe:

- `growth` produces new `generation_XXXXXX/manifest.json`
- `growth` logs `training_export_mode`, checkpoint write format, latest runtime
  checkpoint path, and the selected tree export or sharded manifest path
- sharded exports log `[sharded-training-export] generation=... new_nodes=... reused_nodes=...`
- `dataset_worker` creates dataset rows artifacts
- `training_worker` periodically updates `pipeline/active_model.json`
- `reevaluation` occasionally creates and growth consumes
  `pipeline/reevaluation_patch.json`

Quick checks:

```bash
watch ls pipeline/
watch find pipeline -name manifest.json | wc -l
watch find pipeline -name '*claim.json'
```

### Where to place workers

* Growth worker: CPU-heavy and tree-memory-heavy; owns live tree/checkpoint
  mutation.
* Dataset worker: CPU/disk work; can run separately, ideally close to storage.
* Training worker: GPU-preferred; trains evaluators and updates the active
  model.
* Reevaluation worker: GPU-preferred when model evaluation is expensive;
  produces bounded patches and never mutates the checkpoint directly.

Example GPU placement:

* CPU node: `growth`
* CPU node or storage-close machine: `dataset_worker`
* GPU 0: `CUDA_VISIBLE_DEVICES=0 ... --pipeline-stage training_worker`
* GPU 1: `CUDA_VISIBLE_DEVICES=1 ... --pipeline-stage reevaluation`

### What each worker does

* `growth`: loads/restores runtime, consumes a pending reevaluation patch if
  present, grows the tree, then exports checkpoint/tree snapshot/manifest
  artifacts.
* `dataset_worker`: finds the latest claimable pending dataset generation and
  extracts rows once.
* `training_worker`: finds the latest claimable pending training generation and
  trains/selects the active evaluator once.
* `reevaluation`: reads `pipeline/active_model.json`, reevaluates up to N nodes,
  and writes one singleton patch if no patch is pending.

### Coordination contract

The reevaluation worker writes only one pending patch at a time:
`pipeline/reevaluation_patch.json`. If that file already exists, reevaluation
skips instead of overwriting it. Growth consumes the pending patch before tree
growth and deletes it only after successful application. If patch application
fails, the patch remains on disk for diagnosis and retry.

Dataset and training workers coordinate with claim files:
`pipeline/generation_XXXXXX/dataset_claim.json` and
`pipeline/generation_XXXXXX/training_claim.json`. Multiple workers should not
process the same generation concurrently, and expired claims can be taken over.

Training workers are monotonic by generation. Once a generation has been
started for training, older dataset generations are ignored by default.
`pipeline/active_model.json` updates are also monotonic and cannot roll back to
an older generation. Old dataset artifacts may remain on disk; pruning is
handled separately.

There is still only one live checkpoint/tree owner: the growth process. This
avoids multiple huge checkpoint copies while reevaluation produces bounded patch
artifacts.

### Reevaluation batch size

`--reevaluation-max-nodes-per-patch` bounds one patch artifact. The default is
`10000`.

* Smaller values reduce patch size and application latency.
* Larger values improve throughput but can create large JSON artifacts.
* The worker uses `pipeline/reevaluation_cursor.json`, so repeated passes should
  eventually cover the whole tree instead of looping forever on the first N
  nodes.

### Active evaluator behavior

Training writes and updates `pipeline/active_model.json`. Reevaluation uses the
current active model. If the active model changes, the reevaluation cursor
resets so the next pass starts over for that evaluator. The best selected
evaluator remains the source for both future growth and reevaluation.

### Scaling pattern

It is safe to run multiple dataset workers and multiple training workers if
needed; claim files prevent duplicate generation work. Usually start with one of
each. Growth should normally be a singleton because it owns the live
tree/checkpoint. Reevaluation should normally be a singleton because of the
singleton patch file, though multiple reevaluation loops will mostly idle or
skip while a patch exists.

## Persistence stack

Runtime checkpoints and training exports serve different purposes:

* Runtime checkpoints are whole-runtime restore artifacts. They capture the full
  Anemone search runtime needed to resume growth, selector state, and backup
  state quickly and correctly.
* Training exports are dataset-oriented artifacts. They only need the training
  snapshot view of the tree, so sharding them avoids reserializing old node
  state after the first sharded generation.

This is why runtime checkpoints remain whole-runtime while training export is
sharded.

Current real-run timings from the recent generation 138 profiling pass:

* checkpoint payload build: about `26.31s`
* checkpoint write: about `3.47s`
* total save: about `29.78s`
* blocked evaluation-payload reuse candidates: `231332 / 231332`

The remaining checkpoint build cost is acceptable for now. One future
optimization is intentionally postponed: evaluation payload reuse across
generations. That requires an explicit safe per-node evaluation payload version
such as `evaluation_payload_generation`. It is not implemented yet because there
is no safe existing dirty/version signal for the full serialized evaluation
payload.

### Manual/debug stages

Manual stage commands are useful for debugging specific generations, but they
are not the recommended normal operator pattern.

```bash
# One local/debug mini-orchestrator pass: growth, then pending dataset/training
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage loop

# One dataset stage for a known generation
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage dataset \
  --pipeline-generation N

# One training stage for a known generation
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage training \
  --pipeline-generation N
```

`loop` is a one-shot sequential mini-orchestrator for a laptop/debug run.
`dataset` and `training` with explicit generations are for debugging a specific
generation. The production autonomous workers are `growth`, `dataset_worker`,
`training_worker`, and `reevaluation`.

### Artifact files to inspect

Inside `<work_dir>`:

* `pipeline/active_model.json`
* `pipeline/reevaluation_patch.json`
* `pipeline/reevaluation_cursor.json`
* `pipeline/generation_000001/manifest.json`
* `pipeline/generation_000001/dataset_status.json`
* `pipeline/generation_000001/training_status.json`
* `pipeline/generation_000001/dataset_claim.json`
* `pipeline/generation_000001/training_claim.json`

### Troubleshooting artifact-pipeline runs

* Reevaluation keeps skipping: check whether
  `pipeline/reevaluation_patch.json` exists. Growth may not be running, or patch
  application may be failing.
* No reevaluation patch is produced: check that `pipeline/active_model.json`
  exists and that at least one manifest has a usable tree snapshot.
* Dataset/training blocked: inspect the claim files. Stale claims expire by TTL.
* Patch too large: lower `--reevaluation-max-nodes-per-patch`.
* Growth fails on patch application: do not delete the patch immediately.
  Inspect logs and the patch content first.

### Dependency expectation

Chipiron must run with an Anemone version that provides
`anemone.value_updates.NodeValueUpdate` and
`TreeExploration.apply_node_value_updates`.

---

## 🔄 3. Restart from scratch

```bash
rm -rf ~/oldata/victor/morpion_runs/big_run_01
mkdir -p ~/oldata/victor/morpion_runs/big_run_01
```

---

## Dashboard access

Use `chipiron-bootstrap use /path/to/run` and `chipiron-bootstrap` as described
above. The installed command opens the browser automatically; `--no-browser`
keeps it headless. No repository working directory or `PYTHONPATH` is required.

## 🧪 5. Run evaluator sanity checks

Use this when a bootstrap evaluator looks unable to learn from saved data. The
sanity check requires an existing tree export in `tree_exports/`.

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_sanity_check \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --generation 1 \
  --dataset-mode terminal_path \
  --evaluator-name mlp_41 \
  --num-epochs 50 \
  --batch-size 32
```

Artifacts are written under:

```text
<work_dir>/evaluator_sanity/<run_name>/
```

Use `--run-name my_debug_run` for a stable output directory, or omit it to get a
timestamped directory.

Each run also writes `target_diagnostics.json`, which compares `direct_value` and
`backed_up_value` for the extracted rows with delta quantiles, histogram,
correlation, MSE, depth summaries, visit-count summaries, and worst deltas.

Exact/terminal-only overfit probe:

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_sanity_check \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --dataset-mode bootstrap_like \
  --max-rows 200 \
  --require-exact-or-terminal \
  --evaluator-name mlp_41 \
  --num-epochs 1000 \
  --batch-size 200 \
  --learning-rate 0.001 \
  --no-shuffle
```

Fixed-tree fitted-backup probe:

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_fitted_backup_sanity \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --dataset-mode bootstrap_like \
  --max-rows 200 \
  --max-backup-nodes 5000 \
  --backup-selection exact_terminal_plus_prefix \
  --evaluator-name mlp_41 \
  --num-iterations 5 \
  --num-epochs 100 \
  --batch-size 200 \
  --learning-rate 0.001 \
  --no-shuffle \
  --run-name fitted_backup_200x5_5k
```

This freezes one tree export, recomputes backed-up labels from exact/terminal
ground truth plus evaluator predictions, trains, and repeats. Watch
`mean_abs_target_change` and `max_abs_target_change` in `summary.json`; stable
backup targets are the convergence signal. `--max-backup-nodes` limits the
frozen-tree nodes used for prediction and backup, which keeps early diagnostic
runs cheap on large snapshots. `--backup-selection exact_terminal_plus_prefix`
keeps exact/terminal anchors even if they sit beyond the prefix; use
`--backup-selection top_terminal_paths --dataset-mode top_terminal_paths` to
focus the backup universe on the selected terminal paths.

---

## ⚠️ Known issues

### Inotify watch limit (Linux)

If you see:

```
OSError: [Errno 28] inotify watch limit reached
```

Fix it with:

```bash
sudo sysctl fs.inotify.max_user_watches=524288
```

To make it permanent:

```bash
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## 🧠 Tips

* Always store runs under `~/oldata/...` (large partition)
* Use retention (latest 1–2 checkpoints) to avoid disk explosion
* The dashboard reads from the same `--work-dir`
* The new dashboard entrypoint avoids relative-import issues with Streamlit

---

## 📁 Important directories inside a run

* `search_checkpoints/` → runtime snapshots
* `tree_exports/` → training trees
* `rows/` → supervised datasets
* `models/` → trained evaluators
* `latest_status.json` → dashboard state
* `run_state.json` → bootstrap state

---

## ✅ Minimal workflow

```bash
# start run
python -m chipiron.environments.morpion.bootstrap.launcher --work-dir ...

# open dashboard
chipiron-bootstrap dashboard --work-dir /path/to/run
```

---

Enjoy exploring Morpion 🚀


### Admitted Training coordination

The scientific launcher's Training worker reaches `training_publication.run_training`,
which remains the public Morpion facade. It delegates admission-scoped execution to
`chipiron.bootstrap.attempts.run_attempt`: resolve an accepted/no-work result, or
compute privately, publish and accept before leaving admission. The in-memory
`Resolved` and `Pending` tags add no artifact schema or scheduling policy.

Morpion retains its generation pin, claim release, short ownership fences, candidate
recipes and independent accepted lineages, including losing candidates. Its round
timer still starts before configuration/admission and finishes at Training status
construction, before publication/acceptance; execution provenance is unchanged.
The computation callback keeps the existing imports outside that timed interval.
Training acceptance does not activate a model. Growth's learning barrier,
asynchronous reevaluation and explicit/opt-in activation remain unchanged.

The core owns no value semantics: Morpion supplies maximum backup, targets,
certainty, candidate comparison and refresh validation. D1 runtime capability
checks and dynamic delegation stay at the existing adapter boundaries. The
canonical scientific-launcher cycle test exercises two accepted rounds, losing
candidate continuation, accepted retries, activation and subsequent search through
the real coordination function.

### Explicit bounded asynchronous Growth

Fresh continual experiments may opt in with `--growth-lookahead-cycles L` on
`--scientific-operation prepare-continuous` (with `--scientific` and
`--automatic-activation`). The persisted `bounded_growth_window_v1` policy requires
positive L, equal Dataset/Training publication cadences, and closed sharded exports
every ordinary Growth cycle. Use the existing monitored `run-continuous` launcher.
The absent option retains historical barrier behavior and configuration identities.
An existing experiment cannot adopt this policy or change L on resume.

One source publication is reserved atomically with its Growth commit. Dataset and
Training use that immutable source while Growth commits at most L more ordinary
cycles under the incumbent. The fixed gate waits outside Growth ownership for
accepted learning. An early result still waits for the gate. Retention or full
synchronous activation resolves this single round before another source is
reserved. Activation/drain publications do not consume ordinary-cycle lookahead;
intermediate eligible cuts are recorded as skipped, never queued or silently
substituted. Training acceptance remains distinct from activation; every candidate
retains its own accepted weights/Adam lineage, including losing candidates.

The scheduling state lives in `run_state.json`'s `learning_window_v1` metadata,
bound to the immutable config by a version-2 continual context. Existing accepted
Dataset/Training references and activation receipts remain their authorities.
Atomic activation closes the slot in the same publication commit. Restart does
not reset lookahead or budgets. Input capsules remain conservative retention roots.
Claim expiry alone never authorizes another computation: stale claim takeover
requires evidence that the previous local worker session is empty, or that the
same host has rebooted. Unknown evidence refuses automatic retry. In particular,
a surviving worker shell or orphaned candidate child can require the existing
supervisor's bounded shutdown before retry. No additional process killer is added.

This policy can overlap useful search with Dataset/Training, but changes scientific
tree/data/model trajectories. It does not promise a speedup. Full refresh still
holds Growth ownership, and checkpoint/export costs remain. Device/thread/training
recipes and resource budgets are unchanged. Use J1's read-only performance report
for observed overlap, waiting and incomplete attempts; its span is not a proven
causal critical path. Background candidate inference preparation, multi-round
queues and persistent live runtimes are outside this policy slice.

### Ready-result bounded Growth (fresh opt-in)

Use `--ready-growth-max-cycles H` instead of `--growth-lookahead-cycles L` when
preparing a **new** monitored continual experiment. The options are mutually
exclusive; H must be a positive integer. The same automatic-activation, equal
Dataset/Training cadence and closed sharded publication requirements apply.
Neither historical default behavior nor the existing fixed-window format changes.
The new policy cannot be adopted by an existing workspace or changed on resume.

The `ready_bounded_growth_window_v1` policy reserves one source and permits at
most H ordinary committed cycles after it. Accepted Training is resolved at the
next owned boundary, even before the cap. A private or merely published proposal
is not ready. If Dataset/Training is late, Growth waits outside ownership at the
cap. An already owned cycle finishes before Training can commit acceptance;
Growth rechecks acceptance under ownership after its preliminary activation check.
Resolution releases/reacquires the existing ownership seam without nested locking.
Retained candidates close the slot too; acceptance does not force promotion.

For example, source cycle 10 with H=4 and acceptance visible at boundary 11
resolves at 11. Fixed L=4 waits to 14. If learning is still unaccepted at 14, both
policies stop admitting Growth there. H measures ordinary cycles, not publication
generations, nodes or seconds. Increasing steps per cycle changes the effective
work and age bound. Repeated legitimate incumbent wins may leave weights old;
the policy never promotes a worse candidate merely to reset weight age.

The new state is `learning_window_v2`, with a version-3 continual context and a
schedule hash. Its active slot binds source publication/descriptor, source cycle,
`limit_cycle`, incumbent and window sequence. Compact `decisions` live in this
same committed metadata: one per ordinary publication or resolution, bounded by
publication history plus completed windows and persisted no-save Growth cycles,
never one per poll. A no-save Growth attempt keeps the existing cycle counter and
head publication; its decision records that persisted boundary without inventing
a new scientific publication. They record
pre/post cycle, parent/current publication, reserved source (null before one is
reserved), accepted Training reference observed and action. Completed windows
retain the accepted result and retain/activation reference. Timing logs are not
authority. This history reconstructs committed admission order; it does not promise
bitwise replay across different worker completion times or hardware.

The existing RAM floor and monitored stop/resource budgets still take precedence.
Restart does not reset H, claims, counters or accepted effects. Active sources
remain retention roots; independent candidate weights/Adam parents, claim/session
quiescence, exact accepted retries and old-model patch-drain recovery are unchanged.

This scheduling choice changes tree/data/model trajectories and may increase
refresh frequency. No speed or score benefit has been demonstrated. Activation
still holds Growth ownership for the complete synchronous refresh, certificate
checks, backup/selection rebuilding and checkpoint/export. Latest-ready Dataset
queues, background candidate prediction, retained live trees and preparation-cache
changes are separate work. Compare valid certified score versus elapsed time;
CPU/GPU utilization and host overlap alone do not establish useful throughput.

## Terminal checks during evaluator updates

Terminal checks delegate to Atomheart's short-circuit legality predicate. They
still inspect the actual board and rules, including when a cached terminal flag
is incorrect. Symmetry-reduced legal-action counts and heuristic scores are
unchanged. Evaluator installation still performs the full canonical refresh,
backup, certificate checks and selector rebuild under Growth ownership.

### Monitor accounting at shutdown

The continual monitor checks elapsed limits after process/resource sampling and
the committed-state read. A deadline crossed during these reads keeps the existing
global-time precedence; later cleanup does not relabel an earlier stop decision.
Persistent job timestamps are checked against a clock read after their status
record, so a concurrent update is not mistaken for a future clock. Missing,
invalid, stale idle and stopped records still use conservative process lifetime,
measured from Linux start ticks and BOOTTIME rather than adjustable Unix time.
These operational checks do not change job/run budgets or scientific authority.

### Exact feature candidate enumeration

Handcrafted feature candidate lines are enumerated from occupied integer points,
then sorted into the original direction/x/y order. All candidate fields, numeric
legal-action counts, feature tensors and scalar inference remain unchanged. The
per-direction temporary count table has at most five starts per occupied point;
no state or prediction cache persists. Direct inputs outside the codec's plain
integer-pair frozenset retain the original bounding-box scan and its errors.


Activation phase observations use the existing post-run report and optional
`activation.phases` sidecars (at most 64 rows / 64 KiB per admitted invocation).
They separate initial/post-drain restore and preflight, patch application, each
checkpoint/export/commit, and cleanup outside the original activation total.
The original total and descriptor refresh observations retain their scopes;
children are not added to parents. Request/process/source/model/publication IDs
join the evidence. Missing clocks and observer errors remain explicit.

These host-only observations do not synchronize devices or change activation
policy. Uncommitted failures, rejected requests and exact retries emit no new
sidecar. After a known durable commit, cleanup failure can emit a committed
scientific outcome with a failed function outcome and a missing total interval.
Metrics/sink/logger errors cannot mask computational errors. No sidecar confers
scientific authority; exact low-level ownership correlation remains unavailable.

### Continuous contract foundation (not a runnable policy)

`continuous_contract` holds strict frozen v1 wire records; `continuous_transitions`
proposes the Dataset EXTRACT → accepted PREPARE → complete READY → admitted Training
path. `initialize` returns a context-bound `Frame`; each operation requires an
explicit expected `Boundary` and returns `Change` or finite `NoWork`. Coalescing
returns a mandatory `Displacement` receipt obligation and never slides oldest
learning debt. `encode_record`/`record_ref` describe canonical proposed bytes;
they do not write files or authenticate a reference. Empty held-out validation,
stale data, occupied slots and unsupported frozen-head disposal cannot silently
admit Training. Candidate parents bind in the Training Job, not preparation.

A future single owner must authenticate source/accepted publications, actual bytes,
claim fencing, parent freshness, budgets, common validation population, leases and
session status; publish records then join returned state and receipt changes in
ONE durable save. Both model-dependent comparison and activation payload validation
remain deferred. Retirement, claim takeover, Training acceptance, promotion,
stop/resume/fork wiring and all J4.1 prepared-input IO are also still required.
The current launcher/configuration dispatch continues to reject this policy; no
CLI, global guard change, new tree owner or continuous execution is provided.


### Accepted Dataset preparation (operational, opt-in)

`CHIPIRON_SHARED_PREPARATION=1` now permits a monitored Dataset worker to prepare
its accepted rows for the existing supported serial flat+relational cohort. The
same eligibility/recipe helper is used at Training. Unmonitored/default-OFF or
unsupported invocations retain private attempt inputs. No continuous scheduling
option is enabled and no parent is captured during CPU preparation.

A separate `prepared_dataset_v1` run-state entry reserves at most two operational
roots. Scientific Dataset acceptance and initial intent share the existing owned
save. Bulk row copying, I5 construction, hashing and artifact waits run outside
Growth ownership. Rows are finalized at
`pipeline/prepared_inputs_v1/<accepted-descriptor-sha256>/rows.jsonl` BEFORE I5
binds its native path; producer native ownership closes before COMPLETE commits.
Training admits normally, then borrows only that Dataset's compatible COMPLETE
record under a newly acquired process-local native lease. The lease covers all
candidate readers, diagnostics and incumbent comparison. No prediction is cached.

The operational descriptor binds existing accepted/source, workspace/configuration,
recipe and all five native files. It reuses J4.0a Accepted/Ref/Context validation;
Context policy hashes the existing schedule (or null). It supplies no invented
continuous cycle/epoch and does not instantiate the future continuous Ready/Source
contract. Future dispatch must obtain genuine publication annotations separately.

Original monitored job/claim deadlines survive recovery. TTL alone never proves
process death; takeover uses existing local session quiescence evidence and an
artifact lease. Replaced Dataset/Training claims fence late work. Native-consumer
mode refuses any rebuild. Missing/incompatible pre-consumption inputs may retire
and use the private reference path only after resource release; corruption during
consumption fails the attempt. No silent substitution follows a consumed input.

Reservations include stable rows, native staging/export and the currently retained
other root, under the smaller existing native/run scratch caps. Enabled prior-input
reuse reserves its configured upper bound once, with its existing native coexistence
checks retained. A small metadata rewrite allowance stays reserved. Capacity and
free-space refusal never truncates scientific rows. Live readers/builders provide
backpressure rather than unaccounted private fallback.

Retirement takes nonblocking artifact and native ownership, durably records a
RETIRED disposition before unlinking only operational rows/tensors, and retains
lock inodes plus descriptor/manifest/split evidence. Accepted scientific rows,
models and history remain roots. Accepted Training replay does not need retired
operational tensors. Managed close suspends unfinished preparation and releases completed input
consumption after existing native quiescence checks; original allowances remain. Fork checks reject unsettled
operations, and fresh fork genesis does not import this operational authority.
