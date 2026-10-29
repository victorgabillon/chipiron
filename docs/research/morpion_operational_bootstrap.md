# Operational scientific Morpion bootstrap

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

## Scope and source identity

This is a bounded experimental workflow for fresh Morpion 5T trees. It combines the existing four workers, real nonempty training, explicit accepted-model activation and cold continuation. It does not establish improved playing strength, authorize a record attempt, or retire any historical capsule.

The isolated branch is `feat/morpion-operational-bootstrap`, based on `7326a6f567252f6d43cfa4def7a07ee21de0e624`. That commit descends from `109757835630fa01acb23d98dcb914cf6a5767ca`; their differences are research documents/scripts/evidence only, with no production, normal-test or configuration changes. The available refs did not establish a clean remote review base containing this foundation. The work remains local to avoid duplicating the unmerged stack in a PR.

The executable implementation is `39c7abe8452bb55f76ad59b93c78533389433f52` (Python sources remain identical to `02b8c90e`; the later production change is shell pacing). Anemone is exactly `37e48c40ceff430e373fd07bd00fcf41359bafda`; all 289 installed Python files match its Git archive and wheel, whose SHA-256 is `4ab0ffc88b17fb500a3729991fc4fa5c8f43d2b70be875f68a6d2d52ffe47d2f`. Package version alone is not used as identity. Fresh genesis producer identity now includes complete loaded Chipiron and Anemone Python source fingerprints, using the existing fingerprint utility.

All new work and scratch are on the secondary ext4 disk. Raw evidence is retained at `/home/pompote/oldata/victor/morpion-scratch/operational-bootstrap-20261025/`; committed compact evidence is under `operational_bootstrap/evidence/`, with SHA-256 fingerprints of retained raw logs, complete-row comparisons and oracle payloads in `raw-manifest.json`. Protected worktrees and historical experiments are not execution targets. The separate test leaderboard isolation incident below prevents an unqualified claim that every historical artifact was untouched.

## Independently reviewable changes

| Commit | Change |
|---|---|
| `864e5411` | Confirm whole process-group quiescence after TERM/KILL before launcher success or cache removal. |
| `a23fc49b` | Feed the existing closed writer the existing lazy tree iterator; count during traversal. |
| `a25fe09f` | Expose scientific initialization, workers, inspect and explicit prepare/activate through the existing launcher; scope owned reader caches. |
| `13ba0738` | Refuse zero-sample/zero-epoch durable Training acceptance and activation. |
| `7e640c76` | Scope standalone scientific reader caches and print actual scientific configuration. |
| `9f2ccebc` | Fingerprint complete loaded producer sources for fresh genesis. |
| `d6d0a35b` | Continue bounded initialization after a productive step that added no nodes. |
| `02b8c90e` | Enter the owned cache context directly, resolving the static warning. |
| `9840ee9c` | Isolate default test leaderboard writes in the existing test-output root. |
| `39c7abe8` | Expose the existing Growth pause so fenced Dataset/Training can complete between invocations. |

There is no new coordinator, model registry, database, scientific writer, checkpoint format, retention authority or persistent cache service. The initializer is a thin bridge from the existing runtime to the existing durable genesis API. `_scientific_args` maps the existing frozen configuration to worker arguments. Standalone temporary scopes and the existing shell supervisor own only their new private caches. The experimental recipe invokes these interfaces at explicitly listed stages; it is not a scheduler.

## Operator workflow

Use the validated installed wheels in an isolated Python environment, including the exact Anemone wheel above. The repository's dependencies and toolchain remain authoritative. The optional research resource sampler requires `psutil` (recorded version in the evidence), not a new production dependency.

From this worktree, set short secondary-disk scratch paths and avoid numerical-library thread oversubscription:

```bash
export PYTHON_BIN=/home/pompote/oldata/victor/morpion-scratch/operational-bootstrap-20261025/venv/bin/python
export TMPDIR=/home/pompote/oldata/victor/mo25
export SQLITE_TMPDIR="$TMPDIR"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
```

Prepare new inputs (the seed is explicitly initialized, not a trained result), then run the bounded, monitored two-cycle example:

```bash
"$PYTHON_BIN" -m docs.research.operational_bootstrap.prepare /SECONDARY/NEW-inputs
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce \
  /SECONDARY/NEW-experiment /SECONDARY/NEW-inputs --nodes 100 --cluster-seconds 120
```

The recipe creates a fresh workspace and runs two actual four-worker sessions, under M1 and M2. Each session must advance Growth and durably accept nonempty Dataset and Training output for that evaluator before it can pass. After confirmed shutdown, the recipe **explicitly calls activation at the marked points**. It does not fall back to serial Dataset/Training calls when concurrent acceptance fails. Each call is a fresh process. It verifies accepted models and scientific outputs, retries activation without changing authority, and continues Growth under M3. Concurrent scheduling can select different accepted generations across fresh runs; each actual selected source, candidate and authority transition is recorded. Exact continuation is validated from the committed checkpoint, including selector and RNG state. It refuses to reuse an existing experiment destination. Resource limits are visible in `budget.json`: 2 GiB workspace, 1 GiB scratch, 8 GiB sampled process-group RSS, at least 8 GiB available RAM and 20 GiB disk headroom, 600 seconds per call, bounded Growth and cluster horizons. These are experimental limits, not a claim that larger runs fit them.

For manual operation, initialize explicitly once:

```bash
export WORK=/SECONDARY/NEW-workspace
"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.launcher \
  --scientific --work-dir "$WORK" --scientific-operation initialize \
  --scientific-config /SECONDARY/NEW-inputs/bootstrap_config.json \
  --seed-model /SECONDARY/NEW-inputs/seed --initial-nodes 100
```

Before running repeated Growth, set the same bounded control used by the recipe:

```bash
"$PYTHON_BIN" - <<'PYCONTROL'
import os
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.control import MorpionBootstrapControl, save_bootstrap_control
paths = MorpionBootstrapPaths.from_work_dir(os.environ["WORK"])
save_bootstrap_control(MorpionBootstrapControl(max_growth_steps_per_cycle=1, save_after_seconds=0.0), paths.control_path)
PYCONTROL
```
 The persisted configuration is the scientific authority for runtime settings; legacy shell defaults do not replace it. The example pauses Growth for 90 seconds between invocations (including failed invocations); this gives fenced learning workers an acceptance window. The launcher default remains 2 seconds. Keep the active horizon long enough for retry, cold load, publication and learning; a busy first Growth attempt can otherwise consume much of a short session. The example uses two linear candidates, one epoch, 128 rows, a 25% validation split, bounded two-step rollouts, default-max/Linoo and the `none` family policy. The four other prediction policies remain supported; the four exact-family v2 policies remain rejected.

Resume with the same verified installed source set. This workflow does not authorize a software upgrade during a run or claim arbitrary cross-version continuation. Start or resume the same committed workspace with the same command:

```bash
MORPION_SCIENTIFIC=1 MORPION_USE_INSTALLED_PACKAGES=1 \
GROWTH_SLEEP_SECONDS=90 \
MORPION_CLUSTER_HEADLESS=1 MORPION_CLUSTER_MAX_SECONDS=120 \
MORPION_CLUSTER_STOP_GRACE_SECONDS=15 MORPION_CLUSTER_WORKERS=all \
MORPION_CLUSTER_OPEN_STATUS=0 MORPION_CLUSTER_SHOW_RECAP=0 \
MORPION_GROWTH_SHOW_RECAP=0 MORPION_GROWTH_WORKER_MAX_CYCLES=1 \
MORPION_REEVALUATION_MAX_NODES_PER_PATCH=7 MORPION_WORK_DIR="$WORK" \
bash scripts/launch_morpion_gnome_cluster.sh > /SECONDARY/session.log 2>&1 &
cluster_pid=$!
```

When ready to stop early, from that shell:

```bash
kill -TERM "$cluster_pid"
wait "$cluster_pid"  # TERM returns 143; normal horizon returns 0.
```

The manual shell command has a hard time bound and uses the persisted runtime RAM settings; continuous disk/RSS budget enforcement is provided by the monitored research recipe. Check capacity before manual runs. Do not use SIGKILL as the normal stop procedure. Abrupt supervisor death can leave workers and private scratch requiring explicit inspection; there is no TTL-based cleanup claim.

Inspect authority and existing operational projections:

```bash
"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.launcher \
  --scientific --work-dir "$WORK" --scientific-operation inspect
cat "$WORK/launcher_process_state.json"
ls "$WORK/logs"
```

`inspect` validates the committed evaluator and reports current head, model identity, publications/acceptance descriptors, atomic producer state and pending-patch presence. It is intentionally a diagnostic view of existing authority, not a new compact ledger. Check logged PID/PGID liveness with `ps`; a stale status file alone does not prove a worker is alive. Training recap, worker logs, record leaderboard and the recipe's `summary.json`, `calls.json`, `*.resources.json`, `*.groups.json` and `latest-verification.json` provide the remaining evidence. Pending proposals are not deleted on inspection.

Stop the cluster before preparing activation, select an **exact accepted Training generation**, and inspect the request:

```bash
"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.launcher \
  --scientific --work-dir "$WORK" --scientific-operation prepare-activation \
  --pipeline-generation 8 --activation-request /SECONDARY/M2-request.json
cat /SECONDARY/M2-request.json
"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.launcher \
  --scientific --work-dir "$WORK" --scientific-operation activate \
  --activation-request /SECONDARY/M2-request.json
```

Replace `8` with the actual accepted generation from inspection. Request files are created exclusively and never overwritten. A changed expected head is a rejection, not permission to retarget. Activation remains a separate authorized transition; Training workers never promote their results. The existing exact receipt/cursor and complete-refresh protocol handles admissible pending old-model work before the atomic model transition. Retry the exact request after uncertain completion; do not manufacture another candidate identity.

An invalid head or failed restore does not trigger fresh initialization. Preserve interrupted genesis/staging evidence and diagnose it in place; create a different fresh workspace only when that is intentionally desired. Scientific payload, checkpoint, selector/RNG and model identity still commit together through the existing run-state authority.

## Shutdown defect and synchronization contract

The previous FULL failure recorded a descendant still observable after the test's one-second poll. Its precise historical kernel state is no longer available. We do not claim a reproduced post-SIGKILL kernel bug. The pre-change contention experiment recorded 88 observations of living `S`-state descendants after their group leader was dead/reaped: waiting for leaders is not proof of descendant quiescence. Twelve alternating TERM/horizon trials under four same-CPU burners passed, but exposed this missing synchronization contract.

The launcher now sends TERM, waits the configured grace, sends KILL to owned groups, and confirms via `ps` that no member remains in a live state. Zombies/dead tasks are not running workers. A bounded five-second confirmation interval is part of the documented total lifetime, not an extra permissive assertion delay. Failure to obtain process information or prove quiescence is a failure; private caches are preserved. The test retains real TERM-ignoring children and requires them stopped immediately when the launcher returns. Its communicate bound accounts for the explicit confirmation stage.

Twelve immutable-script trials passed under contention, followed by another **12/12** against final executable `39c7abe8` while PR CI ran, with four CPU burners pinned to the same CPU as the test. The final trials alternate TERM and horizon exits and keep the original real-process assertions. An intermediate trial accidentally overlapped a script edit and failed shell parsing; its raw evidence is retained and excluded, rather than counted as a launcher regression or success. Final FULL is reported separately below.

## Scientific and memory contracts

The original Anemone traversal was exactly `list(_iter_nodes_by_depth())`. The closed writer now consumes that iterator once, preserving depth/insertion export order and counting nodes during the same traversal. It still reuses validated unchanged canonical records, reconstructs current mutable values/topology, and validates output. It does not construct a compensating node map or second live tree.

Matched comparisons cold-restore the same legal source independently, then run the exact pre-change writer extracted from Git versus the current writer. Native timing/RSS and tracemalloc auxiliary-memory measurements are separate processes. Complete decompressed record/update bytes and headers match. The existing full scientific reader is used only as an offline oracle; production Dataset/Reevaluation continue through the bounded compact reader. Nine legacy plus five supported v2 policies compare exact entire saved JSONL and metadata. Legacy parity is relative to the same scientific source; it does not establish equivalence of historical flat and sharded-v1 exports or endorse legacy confidence semantics. Cursor windows include wrap and a missing cursor; actual neural patch contents match.

A private same-source scope reuses the existing compact index and source validation. The shell keeps Reevaluation's cache until the selected source changes; Dataset owns a synchronous per-invocation scope. The standalone CLI owns a temporary scope unless the caller supplies one. No caller-supplied cache is deleted. Scope removal follows reader closure and, for the cluster root, confirmed whole-group exit. Interrupted builds, source mutation, corruption and concurrent readers retain existing exhaustive coverage. An owner killed abruptly can leave an orphan; a later owner does not infer deletion permission.

The real operational oracles verify direct/tree/effective Values and certainty, full canonical identities, all DAG parents/children, creation generation, selector state, search RNG, rollout RNG, evaluator version, model bundles, accepted JSONL/metadata, producer cursor and pending neural patch contents. Terminal and nonterminal proofs are distinguished from frontier estimates. The separate real endgame recipe exercises FORCED(35) / ESTIMATE(42), exact legal terminal replay and independent continuation predictions.

## Measurements and validation

Raw failed preparation attempts remain evidence and are not counted as successful runs.

The initial small run completed 22 calls, reached G14 with 679 nodes / 710 edges and activated two genuine trained candidates (128 accepted rows, 96 Training / 32 validation samples per candidate). Its actual four-worker session ended with whole-group shutdown and removal of its owned cache. The separate larger exploratory run completed 22 calls, G8 with 4,227 nodes / 4,601 edges, two accepted-model transitions and M3 continuation. Its current-format checkpoint, scientific files, exact accepted JSONL, independent maximum-backup refresh and complete checkpoint oracle all agreed. These exploratory runs overlapped some development/CI activity; their timings are retained but are not the controlled final benchmark.

The real endgame recipe completed 40 fresh process steps on installed executable `02b8c90e`: initial direct ESTIMATE(42), tree/effective FORCED(35); two positive-sample Training/activation cycles; independent complete continuation predictions; exact receipt/cursor checks; current and historical activation retries. Final direct is ESTIMATE(30.484771728515625), while tree/effective/declared tree target remain FORCED(35). The neural estimate is allowed to change; the certified value is preserved. No estimated frontier score is a certified record.

Local validation of the final executable uses the repository Makefile and pinned tools; no remote GitHub workflow was launched. FAST: **152 passed**, 100.88 s. PR: **2,757 passed / 2 expected CUDA skips**, 1,142.41 s, no execution-identity errors. **FULL: 2,921 passed / 2 expected CUDA skips**, 1,814.97 s. All 2,916 previously selected FULL test identities are retained, with seven additions and no execution-identity errors. The formerly failing headless horizon test and its TERM counterpart both pass in this complete run. Strict static ratchet: baseline 209, current 206, **zero new diagnostics**. Ruff, toolchain consistency, wheel/sdist build and installed evaluator/checkpoint smoke checks passed. All 478 installed Chipiron Python sources equal Git `39c7abe8` and its executed wheel (the Python files are identical to `02b8c90e`); exact Anemone source equality is independently recorded.

The first validation sequence used normal-test/configuration head `9840ee9c`. Its FULL run was stopped before completion after the concurrency gap was identified; it is not counted as green. The completed final validation covers the pacing change at `39c7abe8`; later research changes do not alter executable package code. CI uses the checkout sources with the pinned installed dependency wheels; real worker experiments use the matching installed Chipiron wheel. The thread environment is OMP/OpenBLAS/MKL = 1, with the repository's unchanged two-worker `loadfile` CI selection. The interrupted PR attempt before test-path isolation is excluded from success counts.

The first larger probe stopped initialization at 4,040 nodes because a zero-new-node step was mistaken for search exhaustion. Later Growth extended it, demonstrating the incorrect stop condition. The initializer now uses the runtime's executed-step signal within the same fixed step budget. Its regression makes one completed step add no node and requires the next legal step to run. The original 4k probe is retained separately and is not relabeled as the corrected 5k run.

Production delta relative to the foundation: **499 added / 26 removed lines**, net +473 across eight existing files (including the shell launcher). The largest additions are existing CLI exposure and a bounded initializer that delegates commitment to the existing protocol. No new production module or persistent authority was added.

## Matched exporter comparison

Each pair restores the same frozen legal graph in a fresh process. Timing below measures the scientific exporter after restore, not the entire checkpoint/publication transaction; activation and Growth measurements include their complete publication. Native RSS is a whole-process high-water mark, including restore. Auxiliary Python allocation is measured separately with tracemalloc and must not be compared to native timing.

| Matched nodes | Old / iterator native seconds | Old / iterator peak RSS MiB | Old / iterator traced auxiliary peak bytes | Removed peak bytes |
|---|---:|---:|---:|---:|
| 1,009 | 1.614 / 1.602 | 313.625 / 313.477 | 3,689,025 / 3,678,971 | 10,054 |
| 5,142 | 9.558 / 9.532 | 343.332 / 342.953 | 12,190,879 / 12,147,243 | 43,636 |

The timing differences are too small to establish a throughput gain. The demonstrated improvement is removing the extra linear node-reference container. Cold mutable-value construction and state rematerialization still allocate memory; the traced peak is not a proof of constant exporter memory.

Both 1k runs made 172 checkpoint-anchor and 333 delta codec calls; both 5k runs made 1,180 anchor and 2,573 delta calls. Scientific state-reference decode counters were zero in all four native runs. Thus unchanged scientific records are reused, but the whole operation is not described as decode-free. Exact source digests, fsync counts, CPU and `/proc` I/O observations are in `operational_bootstrap/evidence/results.json`.

For **each graph**, the complete decompressed records, updates and manifest match before/after. Full-reference versus indexed readers match every node, value contract, current DAG link and creation generation. All nine legacy and five supported v2 Dataset policies match complete saved JSONL plus metadata (128 rows each; **1,792 paired rows per graph**). Three actual neural Reevaluation windows of 257 nodes each match exact patch rows and deterministic cursor/wrap behavior. Four unsupported exact-family v2 policies remain gated in the existing scientific test matrix.

The cache-scope benchmark initially used a symlinked workspace alias and was correctly rejected by committed-artifact validation. The research utility now accepts an explicit real workspace path; no production validator was weakened. The rejected attempt and its logs are retained separately. On the real G8 source, the compact v3 index builds in **20.098 s**, reopens in **0.02684 / 0.02655 s**, and occupies 3,796,992 logical / 3,801,088 allocated bytes. Both row tables contain 5,142 compressed BLOB payloads. One index remains inside the owned scope; after all three readers close, the scope is gone. Reused opens still read 346,319 source bytes each. Including capsule validation, all three opens hash 6,753,899 bytes in 201 calls: reuse does not mean skipping integrity checks.

To reproduce a matched pair, freeze a fresh workspace after stopping its workers, then use **new** destination directories:

```bash
"$PYTHON_BIN" -m docs.research.operational_bootstrap.export_measure "$WORK" /SECONDARY/pair/before --before
"$PYTHON_BIN" -m docs.research.operational_bootstrap.export_measure "$WORK" /SECONDARY/pair/iterator
"$PYTHON_BIN" -m docs.research.operational_bootstrap.export_measure "$WORK" /SECONDARY/pair/before-trace --before --trace
"$PYTHON_BIN" -m docs.research.operational_bootstrap.export_measure "$WORK" /SECONDARY/pair/iterator-trace --trace
"$PYTHON_BIN" -m docs.research.operational_bootstrap.compare_export "$WORK" /SECONDARY/pair/before /SECONDARY/pair/iterator /SECONDARY/pair/comparison
mkdir /SECONDARY/scope
"$PYTHON_BIN" -m docs.research.scientific_index.measure /SECONDARY/scope final scope --native --workspace "$WORK"
"$PYTHON_BIN" scripts/report_morpion_rolling_recovery.py "$WORK" > /SECONDARY/reachability.json
```

Run native and trace measurements serially without concurrent CI. The existing measurement utility's optional `--workspace` only selects the real source path; it does not create a workspace or bypass its validator.

## Controlled operational resource evidence

The corrected small run (`paced-small`) uses two real 120-second four-worker sessions and completes 15 fresh-process calls, M1 → M2 → M3, then continued Growth. Final head is G9 with 364 nodes / 372 edges. Workspace allocation is 4,239,360 bytes. Both worker sessions confirm whole-group shutdown and removal of their private cache roots.

The corrected larger run starts from **5,017 newly generated legal nodes**. M2 is the accepted G2 `linear_b` bundle `4c89b71ea810c1aaf0a8814458ba16f235b2b0f877064f18657b83a26f792cb9`; M3 is accepted G5 `linear_a`, `dcbaa0a12d1429cc2bfcf7db4cb262a7b8214a278777c5c29f0a020394119cb3`. These are actual completed positive-sample Training results, not initialized bundles. The two actual four-worker sessions finish before each separate explicit activation. The complete run finishes all 15 calls at G8 with **5,142 nodes / 5,647 edges**, two terminals and two proven nonterminal values. Its two retained terminal sequences independently replay legally to 63 moves / 99 points and 20 moves / 56 points. Complete sequences remain in the existing workspace leaderboard. Neither is claimed as a world record. Accepted Dataset generations are 2, 3, 4 and 5 (128 rows each); accepted Training generations are 2 and 5 (96 Training / 32 validation samples per candidate). Dataset-only generations are retained honestly.

| 5k stage | Wall seconds | CPU seconds (user + system) | Native process peak MiB | Output block count |
|---|---:|---:|---:|---:|
| Fresh initialization | 144.59 | 141.53 | 361.54 | 49,656 |
| Four workers under M1 | 121.33 | 217.91 | 345.80 | 163,624 |
| Explicit M2 activation | 158.68 | 149.98 | 317.53 | 136,832 |
| Independent M2 full oracle | 398.05 | 393.38 | 500.56 | 115,344 |
| Four workers under M2 | 121.34 | 217.24 | 344.94 | 167,432 |
| Explicit M3 activation | 162.47 | 154.22 | 313.11 | 138,400 |
| Independent M3 full oracle | 385.37 | 381.09 | 514.43 | 117,856 |
| Continued Growth under M3 | 22.14 | 18.01 | 311.63 | 70,656 |
| Reevaluation under M3 | 23.10 | 21.58 | 263.32 | 22,928 |
| Final complete verification | 143.23 | 139.20 | 462.01 | 119,600 |

GNU `time` measures the command and its waited descendants; its peak RSS is a single-process high-water mark, not aggregate concurrent RAM. Linux output-block counts are reported directly, not confused with retained disk bytes. Input blocks were zero except eight during M2 activation; this reflects the OS page cache, not absence of source reads.

| Worker group | Small M1 / M2 sampled peak MiB | 5k M1 / M2 sampled peak MiB |
|---|---:|---:|
| Growth | 271.10 / 273.34 | 321.08 / 318.76 |
| Reevaluation | 267.81 / 268.21 | 270.04 / 269.82 |
| Dataset | 260.91 / 261.10 | 262.94 / 263.12 |
| Training | 352.38 / 352.26 | 352.39 / 352.41 |

Per-role sampled CPU totals across each 120-second session include all observed attempts, idle-process startup and retries: Growth 26.10 / 25.08 CPU seconds, Reevaluation 69.35 / 55.63, Dataset 70.42 / 83.59, Training 46.71 / 46.71 (M1 / M2 sessions). Sampling can miss CPU after the last observation; these are distinct from GNU time's waited-child aggregate. Repeated Python startup, cold validation and lock contention are substantial relative to this deliberately tiny training workload. Candidate training itself took 3.24 / 0.010 seconds in G2 and 3.10 / 0.011 seconds in G5; the first candidate includes lazy runtime setup. These are not steady-state throughput estimates for larger models.

The existing process observability is supplemented by the research sampler's 200-ms PID/PGID observations. Sampled peaks can miss transients and group RSS sums double-count shared pages. The monitoring parent itself uses roughly 300 MiB outside the worker groups. The sum of per-group maxima is conservative with respect to concurrency, but still subject to sampling. Raw per-PID CPU, native per-call I/O, staging/scratch peaks and source fingerprints remain in the evidence. No other task CI ran during these controlled experiments.

Across the two 5k worker sessions, sampled staging peaks were 7.25 / 7.32 MiB and owned scratch peaks 7.06 / 4.20 MiB. These are not total task allocation: independent full-oracle outputs and matched benchmark fixtures are retained separately. The controlled fixture and 42-parameter linear models fit the declared budgets; these numbers are not a record-scale memory model or a benchmark of larger neural architectures. The host is an AMD Ryzen AI 9 HX PRO 370, CPU-only Torch, with numerical library threads fixed to one. Full installed distribution identities and native Zstandard fingerprints are recorded.

## Retained history and resource complexity

The final 5k workspace contains 365 files, 10,392,515 logical bytes and **11,087,872 allocated file bytes** (10.57 MiB), counted by unique inode. Generation capsules occupy 7,860,224 unique allocated file bytes across 252 names. Dataset objects occupy 1,318,912 bytes; Training objects 270,336 bytes; retained genesis inputs 794,624 bytes. Categories and per-capsule figures are not additive where hard links share inodes. Directory blocks are not included in these file-allocation figures.

| Generation | Capsule allocated file bytes | Checkpoint allocated bytes | Checkpoint files | Scientific allocated bytes |
|---|---:|---:|---:|---:|
| 1 | 819,200 | 593,920 | 19 | 184,320 |
| 2 | 827,392 | 598,016 | 19 | 188,416 |
| 3 | 827,392 | 598,016 | 19 | 188,416 |
| 4 | 1,105,920 | 684,032 | 19 | 344,064 |
| 5 | 1,097,728 | 692,224 | 19 | 348,160 |
| 6 | 1,097,728 | 692,224 | 19 | 348,160 |
| 7 | 1,138,688 | 692,224 | 19 | 352,256 |
| 8 | 1,126,400 | 712,704 | 22 | 352,256 |

G8 is current; G7 and G6 are the two preceding complete capsule boundaries. They are rollback candidates, **not executable intentional-rollback transitions**: sealed producer cursor/receipt rollback context is still missing. Crash restart continues from the current authority. Their models, selector/RNG/checkpoint state, exact receipt and activation evidence remain required by the current recovery contract. Independent reclaimability is evaluated using the existing read-only reachability tool, not by subtracting the table's rows or counting names of hard-linked files. There is **no authorization to delete committed artifacts**. Both final read-only reports validate without blockers, but even conditional candidate bytes are zero: current/previous boundaries, exact receipt sources, accepted results, pending work and conservative stale claims protect every generation. In the 5k workspace, the failed G1 Dataset claim remains a conservative dependency even though its worker is stopped; age alone does not release it. The report records two exact receipts (1,761 metadata bytes) and 1,303 bytes of proposed active-root metadata; the small workspace has four receipts (3,262 bytes) and 1,447 root bytes. These are observed finite-run sizes, not bounded-history claims.

Let N be current nodes, E edges, G committed generations and L canonical replay-history bytes. The live graph remains O(N + E + L); checkpoint serialization uses the existing bounded shards, and the removed exporter list previously added O(N) references. Complete closed scientific output is O(N + E + L) before compression. Retaining each complete generation can grow as O(G(N + E + L)); cumulative receipt/publication metadata can add aggregate O(G²) copying or validation when each new generation repeats historical metadata. These experiments cover eight or nine generations, not thousands, and do not prove history-independent operation. Exact receipts must not be replaced with an unsafe watermark.

## Test leaderboard isolation incident

During PR CI, the existing default record persistence path rewrote `/home/pompote/morpion_runs/morpion_leaderboard.jsonl`. The writer saves whenever it processes candidates, even when none enters the top 100. PR was stopped immediately on discovery. The observed 100 entries all have historical origins/timestamps, with no new-task entry; the newest observed entry is July 8. A pre-task byte snapshot was not taken, so exact pre-task byte preservation cannot be established. No speculative restoration, deletion or timestamp reset was performed. The observed file and incident manifest were copied to task-private evidence.

`9840ee9c` directs default test persistence to the already isolated application-output root. Explicit fixture leaderboard paths remain supported. All subsequent commands also receive a private leaderboard environment path, while scientific workers use their own workspace leaderboard. The closing audit confirms the observed historical file's SHA-256 **and modification timestamp** remained unchanged after isolation. This incident is distinct from the protected-worktree audit; all 47 existing protected worktrees retained their heads, status and tracked diffs; 18 pre-existing missing registrations were not recreated.

## Remaining scale limits and safe next step

The existing Growth ownership lock spans a full invocation. Concurrent Dataset/Training admission or publication can return busy; the launcher retries. Both early 45-second sessions failed to obtain concurrent Training acceptance, even though their later explicit serial cycles succeeded. This prompted the measured 90-second Growth pause and stronger recipe assertions. The revised small run completed both learning cycles through the actual four loops, reaching G9 / 364 nodes / 372 edges in 15 calls. Pacing trades search duty cycle for progress by learners; it does not remove contention or justify a high-throughput concurrency claim. Unaccepted output from interrupted or contended attempts can also remain and is included in allocation measurements.

One resident search graph remains O(N+E); legal state histories, neural features and selector metadata have their own costs. Removing an O(N) reference list does not make Growth constant-memory. Closed checkpoints and scientific snapshots still write complete current mutable state each generation; compressed bytes depend on actual state/history complexity. The offline full-reference oracle deliberately has higher memory usage than the bounded production reader.

All committed capsules in the operational experiments are retained. Complete historical capsules, accepted Dataset/Training artifacts, models and exact receipt/activation evidence remain cumulative. Run-state publication history can cause growing validation and aggregate quadratic metadata work. The existing read-only reachability report distinguishes conditional candidate bytes from actual authorization; current independent deletion permission remains zero. Abrupt-owner orphan caches and retained validation outputs also need explicit ownership inspection. None is silently reclaimed.

With local FAST/PR/FULL and quality checks green, the bounded recipe is ready for an explicitly authorized next experiment within measured resource headroom and monitored limits. The optional 10k operational stage was not run: the complete 5k double-activation workflow already exposed the lock-pacing and oracle costs, and no larger unrestricted run was needed. It is not evidence for indefinite unattended operation or record-scale capacity. Before a materially larger run, choose model/data/growth budgets, measure legal replay depth and resident graph memory, and resolve the known safe-retirement and cumulative-history costs. Frontier estimates are not certified records.

## Review in VS Code

```bash
code /home/pompote/oldata/victor/morpion-scratch/operational-bootstrap-20261025/worktree
```

Use Source Control's graph to compare `7326a6f5` with `39c7abe8`, or select each independent commit. For a concrete graphical file comparison, export the baseline into new task scratch with `git show 7326a6f5:src/chipiron/environments/morpion/bootstrap/closed_scientific_snapshot.py > /SECONDARY/closed-before.py`, then run `code --diff /SECONDARY/closed-before.py src/chipiron/environments/morpion/bootstrap/closed_scientific_snapshot.py`. Do not check out or reset a protected worktree to inspect the diff.
