# Morpion: bounded sustained operation, 26 October 2026

The real four-worker pipeline completed two positive learning cycles at 5k and
10k scale. The larger fresh experiment ended at **G16, 11,766 legal nodes and
13,099 DAG edges**, with explicit M2 and M3 activations followed by Growth.
A controlled pacing comparison reached **278.25 committed new nodes/minute**
with a 2-second pause, bounded worker ownership waits and four steps per
invocation, versus **22.95/minute** with the old 90-second pause. Final FULL:
**2,950 passed, two expected CUDA skips**. Protected historical bytes and
modification/change timestamps match their before inventory.

This establishes finite, monitored learning experiments. Indefinite unattended
search remains blocked on an explicit historical-retirement contract and growing
publication/receipt metadata. No committed recovery capsule was deleted. No
strength improvement or world-record result is claimed.

## Executable identity and review boundary

Worktree: `/home/pompote/oldata/victor/morpion-scratch/s26/w`, branch
`feat/morpion-sustained-operation`, based on
`424ee3a5aa258bab407cc948d92d008e2d7641cc`. The stable executable and final CI
source is **`99a63802c94ac865b0b908f01c8cafee633a965c`**. Subsequent changes are
this report, reproducibility prose and static evidence. The user's unrelated
original checkout is preserved. There was no merge, remote publication or
production activation; every activation below belongs to a new private experiment.

| Artifact | Exact identity |
| --- | --- |
| Final Chipiron 0.2.4 wheel | `bfe5f0350992a796e4a1a864d8d28e26bc238669c0ac52754395807a1d90bf90` |
| Anemone Git | `37e48c40ceff430e373fd07bd00fcf41359bafda` |
| Anemone 0.2.26 wheel | `4ab0ffc88b17fb500a3729991fc4fa5c8f43d2b70be875f68a6d2d52ffe47d2f` |
| Earlier baseline A wheel | `29609db2dc41258e6e00d8b0f26f96b49c8940482ce0882f2972c6ed375ac5c3` |
| Earlier sustained-5k / matched-profile B wheel | `eddf00f3077aa0a38ba1c92a02f9576df95cc7b581460c0da7ea5ae5490f546e` |
| Final four-worker shell | `c2ac56091196324a6683762edc4712f30911cf632bfdea6a711ac83e4b7ef1bb` |

[Final source inventory](sustained_operation/evidence/source-identities-final.json)
compares all **478 Chipiron and 289 Anemone Python files** across the specified
Git archives, wheels and actual installed distributions. The same checks ran
before the 10k launch and after CI/experiments. No installed package changed
during an experiment. Earlier A/B inventories are retained separately; the 5k
results are not attributed to the later guard wheel.

Python 3.13.5, Torch 2.14.0+cpu, Coral 0.1.15, Atomheart 0.2.11, Valanga 0.2.5,
Zstandard 0.25.0/libzstd 1.5.7, ijson 3.6.0 and psutil 7.2.2 were used. Tool
pins are pytest 9.1.1, Ruff 0.15.20, mypy 2.1.0, pyright 1.1.411 and pylint
4.0.6. [Dependency evidence](sustained_operation/evidence/dependency-identities.json)
records exact installed-file manifest digests, native-library hashes, interpreter
hash and the retained full inventory. No unverified Git origin is inferred from
another dependency's version. Frozen research producer `revision` labels are
inherited fixture metadata; executable identity is established by the file
inventories above.

## Reviewable commits and production size

Production means `src/` and `scripts/`, including the existing shell launchers.
Base-to-executable total is **110 added, 10 removed, net +100 lines** in existing
files. No production module, scheduler, launcher, journal, model registry, storage
engine, scientific writer or rollback engine was added.

| Commit | Change | Production + / − / net |
| --- | --- | ---: |
| `eef911a7` | Private serial test outputs; protected path admission | 29 / 0 / +29 |
| `cc9e7e84` | Bounded learning-worker ownership waits; scientific invocation default | 50 / 10 / +40 |
| `d411cd38` | Exact live-owner cleanup regression; retention admission evidence | 0 / 0 / 0 |
| `bf63dddb` | Format publication ownership call | 3 / 1 / +2 |
| `d62f546c` | Existing research recipe: routine recovery versus independent audit | 0 / 0 / 0 |
| `e7ef3f45` | Research-only observations and offline decision witnesses | 0 / 0 / 0 |
| `ec875e2d` | Apply captured effective budget in offline witness | 0 / 0 / 0 |
| `fe3c3cd1` | Shell preflight before startup persistence; explicit SQLite guard | 29 / 0 / +29 |
| `b71891ec` | Record actual copied genesis size in research budget | 0 / 0 / 0 |
| `99a63802` | Expose explicit per-call research time cap | 0 / 0 / 0 |

[Full commit SHAs and counts](sustained_operation/evidence/implementation-commits.json)
make each stage independently reviewable. The ten removed production lines replace
previous acquisition/call forms; no obsolete production subsystem was found that
needed removal. Serial/xdist fixture duplication was simplified in test code.
Research code extends the existing operational recipe and reads existing evidence;
it is not a new runtime authority.

## A: historical protection and test destinations

The before inventory was captured before any task test. It covers **109,798 files,
100,907,914,157 logical bytes**, in the historical Morpion runs, user application
data and prior operational final-5k workspace. `/home/pompote/morpion_runs` resolves
to the inventoried historical Morpion directory. Every SHA-256, size, `mtime_ns`
and `ctime_ns` matches afterward. The historical leaderboard's raw bytes also
match. All **48 pre-existing worktrees** match their captured HEAD, status and
tracked binary diff. [Protection proof](sustained_operation/evidence/protected-verification.json)
identifies the retained 39 MB before inventory and raw after report by hash.

The raw snapshot script's relative-path exclusion accidentally included this new
task checkout among its 49 entries. Its blanket assertion therefore reported only
this checkout's intended commits. That raw failure is preserved. The scoped proof
excludes this explicitly owned checkout and reports zero changes in the other 48.
The file comparison itself reported zero changes. This inventory does not claim
to cover every file in HOME. The earlier task's leaderboard byte-preservation
incident remains **unresolved**: its missing pre-task snapshot was not reconstructed
and the historical leaderboard was never rewritten.

Normal pytest runs now use a fresh output scope even without xdist. Each worker
gets private application output, bootstrap work directory, leaderboard, cache and
MLflow database/URI, overriding inherited tracking destinations. Small test-only
checks reject resolved known historical roots at workspace, leaderboard, output,
MLflow and reader-index admission. Checkpoints, scientific exports, Dataset rows,
models, status and control files inherit the guarded workspace. Explicit SQLite
URIs receive the same admission check. Production paths remain configurable when
the repository's test sentinel is absent.

A late audit found that both shell launchers could create logs before reaching a
Python guard. Their test-only realpath preflight now runs before startup side
effects, with default, explicit protected, symlink, private and production-mode
regressions. This concrete correction is why a new complete final CI run was
needed after an earlier green predecessor FULL. No global filesystem interception
was introduced. Current tests use private secondary-disk TMPDIR/SQLite/cache/model
paths; no historical cleanup was performed.

## B: ownership cause and controlled pacing result

Growth retains its existing kernel ownership lock for the whole invocation,
including restore, export and publication. Other workers previously discarded
admission/publication attempts on contention. Shortening sleep alone increased
Growth throughput but starved accepted learning in the matched run.

Dataset and Training now optionally wait at their existing admission, fenced
commit and exact-claim cleanup boundaries; Reevaluation waits at final fenced
publication. Waiting does not own or mutate authority. The same source/model/
epoch/cursor and attempt checks still run after acquisition. Growth remains
nonblocking. The helper's default wait is zero; the scientific shell supplies a
bounded 60 seconds unless explicitly configured. It defaults to one Growth cycle
per invocation, honoring an explicit override. These experiments additionally set
four steps per cycle and a 2-second pause. This is bounded polling on the existing
lock, not guaranteed FIFO scheduling.

All four matched cases copied one newly generated, task-private G1 with **5,017
nodes, 5,515 edges and selector step 137**. They ran serially for about 180 seconds,
without concurrent CI or another benchmark, using the real four-worker launcher
and the same B executable. No candidate activation occurred in these single-session
comparisons. Counts below come from committed checkpoints, not attempted log work.

| Case: pause / wait / steps | New nodes / edges | Steps/min | Nodes/min (nodes/s) | Accepted Dataset / Training | CPU cores* | Union RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p90: 90s / 0s / 1 | 69 / 72 | 0.665 | 22.95 (0.383) | 3 / 2 | 1.79 | 1,074.9 |
| p2: 2s / 0s / 1 | 235 / 243 | 1.654 | 77.75 (1.296) | 0 / 0 | 2.69 | 1,081.2 |
| pw: 2s / 60s / 1 | 296 / 309 | 1.985 | 97.93 (1.632) | 3 / 2 | 2.15 | 1,183.9 |
| pb: 2s / 60s / 4 | 841 / 908 | 6.617 | 278.25 (4.637) | 2 / 2 | 2.03 | 1,108.8 |

\* GNU-time user+system CPU seconds divided by elapsed seconds; not host CPU percent.
[Metrics, including every measured acceptance latency](sustained_operation/evidence/pacing-metrics.json)
and [committed counts](sustained_operation/evidence/pacing-comparison.json) are
retained. pb is **12.12× p90** and **3.58× p2** by durable nodes/minute. Even
one-step waiting improved Growth by 26% over p2 while restoring learning acceptance.

| Case | Completed Growth ownership seconds | Useful Growth attempts seconds | Wall outside completed ownership† | Lock failures G / D / T / R |
| --- | ---: | ---: | ---: | --- |
| p90 | 48.11 | 1.76 | 132.26 | 0 / 8 / 0 / 1 |
| p2 | 125.24 | 6.79 | 56.12 | 1 / 26 / 0 / 6 |
| pw | 145.95 | 6.34 | 35.41 | 0 / 0 / 0 / 0 |
| pb | 145.29 | 19.56 | 36.06 | 0 / 0 / 0 / 0 |

† Includes startup, pacing, retries and any unfinished invocation; it is a Growth
idle/opportunity proxy, not pure machine idle time. Shutdown can abandon attempted
search, so log compute totals are not committed step totals. pb ownership windows
were 27.13–31.37 seconds; accepted Dataset source latencies were 63.67/70.05 seconds,
Training followed Dataset by 37.83/37.73 seconds. p90 Dataset latencies were
25.30–84.09 seconds and Training followed by 3.98–4.79 seconds. Fair admission and
higher search throughput do not imply minimum learning latency.

A separate single-step cProfile of a disposable 5k copy measured ownership
**41.009 seconds**, native whole-call wall 43.80 seconds. Profiling adds overhead;
these times are not substituted into the native throughput comparison.

| Phase | Measured seconds |
| --- | ---: |
| Cold restore, log boundary | 19.378 |
| Useful Growth | 1.231 |
| Selection / expansion / evaluation / propagation | 0.313 / 0.100 / 0.693 / 0.119 |
| Checkpoint serialization | 1.429 |
| Scientific export, inclusive | 16.083 |
| Reader-index builds inside those paths, inclusive | 5.633 |
| Closed export validation, inclusive | 1.896 |
| Capsule byte validation / publication-state validation | 0.520 / 0.359 |
| Receipt-state validation, nested | 0.187 |
| Immutable directory publication / authority replacement | 0.013 / 0.014 |

[Profile evidence](sustained_operation/evidence/growth-phases.json) contains function
identities. Inclusive validation/index timings overlap restore/export paths and
must not be summed as exclusive phases. Current-head validation appears in the
publication-state/capsule validation calls. Useful selection/evaluation is a small
fraction of ownership; restore/export/index work dominates. Moving just compute
outside the lock has limited measured benefit. A safe prepare/commit split would
also need to compare the entire captured authority: Dataset/Training acceptance
can change it without advancing the Growth head. It would need model, epoch,
cursor, pins and stale-work fencing. That larger transaction change was rejected
for this task; no shared mutation was moved outside ownership.

## C: retained storage and fail-closed cleanup

The original operational G1–G8 report validates, but all committed capsules remain
protocol roots and its conditional candidates are empty. Its expired G1 Dataset
claim has null owner and no quiescence/completion evidence. Age or an empty `ps`
listing cannot release it. Exact roots and the prerequisites for retirement are
listed in [the retention analysis](morpion_sustained_retention.md).

The live-owner regression proves that an admitted worker can wait for Growth,
recheck its exact claim ID and release only that claim. An interrupted owner still
leaves conservative evidence. Actual bounded shutdown left a G7 Dataset claim and
G12 Training claim in the 10k workspace; they remain untouched. Thus a normal
supervisor stop is not treated as blanket permission to expire all claims.

The existing uncommitted-artifact collector was tested **only on a generated copy**
of p2. A real shared capsule pin blocked the owned orphan; after its release,
`prune_uncommitted_generation_artifacts` removed one protocol-owned uncommitted
attempt, reclaiming **7,659,520 allocated file bytes in that copy**. All six
committed capsules, authority bytes, exact cold payload, selector and both RNGs
were preserved. The source workspace's complete file inventory was unchanged.
No claim or committed capsule was retired. Copying separates some hardlinks, so
this reclaimed-byte count is not attributed to the original p2 allocation.
[Exact cleanup witness](sustained_operation/evidence/orphan-copy.json).

| Workspace | Files | Logical bytes | Unique allocated file bytes | Committed boundaries |
| --- | ---: | ---: | ---: | ---: |
| Baseline b5 | 366 | 10,358,118 | 11,034,624 | 8 |
| Sustained s5 | 583 | 16,141,467 | 17,305,600 | 12 |
| Sustained s10 | 1,031 | 37,299,909 | 39,383,040 | 16 |

s10 allocation separates **2,592,768 current-only**, **30,494,720 historical-only**,
**32,768 shared current/historical**, and **6,295,552 outside capsule** file blocks;
they sum exactly to its unique allocation. The last three complete capsules'
physical union is **7,462,912 bytes**, only a potential recovery-payload floor:
accepted results, models, exact receipts, control context and claims add roots.
The auditor identifies G2/4/6/9/11/13 as conditional future candidates totaling
**11,943,936 allocated bytes** under a proposed rolling contract. **Currently
permitted committed reclamation is zero.** s5 has zero conditional candidates.
pb's report explicitly refuses validation because of unresolved staging ownership;
that blocker is retained rather than converted into deletion permission.
[Retention summaries](sustained_operation/evidence/retention-summary.json) and full
reports distinguish hypothetical candidates from authorized reclamation.

Current s10 checkpoint: **1,537,605 logical bytes, 130.68 bytes/node**. Scientific
snapshot: **817,067 bytes, 69.44 bytes/node**. Per-generation capsule/checkpoint/
scientific allocation is in [s10 artifacts](sustained_operation/evidence/s10-artifacts.json).
Ordinary capsule allocation grows from 1.58 MB at G1 to 2.52 MB at G16, with larger
changes at model refresh. The whole workspace grew by 36,229,120 allocated bytes
from its initialized measurement; transient peak was 49,516,544 bytes, including
staging. There are 16 committed checkpoints and 17 checkpoint manifests in the
whole workspace; the latter includes noncommitted material and is not a recovery
boundary count. All original task artifacts remain retained.

One current plus two previous complete states requires explicit protocol admission
of absent historical payloads, authenticated descriptor ancestry and exact receipt
UUID/digest completion evidence, separation of completed-input identity from live
execution dependencies, and an owned reader-retirement fence using existing pins.
The two retained boundaries' model/cursor/receipt context must be sealed. No
watermark, delta replay, shared-block mechanism or new ledger was substituted.
Until those facts are established, finite time/step/resource limits bound storage
by **stopping**, not by indefinitely bounded historical retention.

## D: routine recovery and periodic scientific audit

Routine checks retain authoritative state/capsule/model byte validation, complete
format-3 cold restore, selector/search-RNG/rollout-RNG equality, exact accepted
Dataset and all Training candidate validation, necessary source identities and
legal terminal replay. The regression corrupts actual accepted row/candidate bytes
and required identities; routine validation rejects them. It does not trust cached
metadata in place of protocol-required hashes.

The existing independent oracle still runs after **both material model changes**,
checking all-node values/certainties, full DAG/checkpoint continuation, creation
identity, independent max refresh, terminal/forced proofs and Dataset semantics.
Ordinary final Growth uses routine recovery. `--audit-every-check` retains the
previous redundant ordinary-step audit. A failed verification is logged, leaves
`validation-failed-*.json`, and raises before the recipe's next activation; unreadable
authority never triggers a new history. Existing atomic acceptance checks were not
weakened and no second parser, oracle or validation database was created.

On the same quiescent pb workspace, serially without CI, routine validation took
**25.67 wall / 25.62 CPU seconds**, versus **115.29 wall / 112.12 CPU seconds** for
the complete independent audit: **77.7% less wall and 77.1% less CPU** for that
ordinary check. Both passed against unchanged authority. Peak process RSS was
396.6 versus 474.3 MiB. This measures changing audit frequency, not removal of
material-change audits or authoritative validation.

## E: actual two-cycle experiments and identities

All experiments initialized fresh graphs with compact scientific encoding,
streaming format-3 checkpoints and compact indexed readers. b5 reproduced the
original recipe with 90-second pauses, one step and two 120-second four-worker
sessions. s5 used 2-second pauses, 60-second waits, four steps and two 180-second
sessions. s10 used the final guard wheel and two 420-second sessions with 120-second
waits. Their correctness/audit runs overlapped CI at times, so their wall times
are engineering observations, not controlled cross-configuration speedups. The
matched pacing and validation-cost comparisons above were completed separately.

| Run | Initialized nodes / edges | Final nodes / edges | New nodes / edges | Final G | Actual accepted Dataset generations | Actual accepted Training generations |
| --- | --- | --- | --- | ---: | --- | --- |
| b5 | 5,017 / 5,515 | 5,145 / 5,644 | 128 / 129 | 8 | 2, 3, 4, 5 | 2, 5 |
| s5 | 5,017 / 5,515 | 6,407 / 7,035 | 1,390 / 1,520 | 12 | 1, 2, 6, 7 | 1, 2, 6, 7 |
| s10 | 10,005 / 11,014 | 11,766 / 13,099 | 1,761 / 2,085 | 16 | 1, 3, 5, 8, 10, 12 | 1, 3, 5, 8, 10 |

The newly reproduced b5 counts are **5,145 / 5,644**; 5,142 / 5,647 belongs to the
prior assignment and is not reused as this task's measurement. Every accepted
Dataset has **128 real rows**. Every accepted candidate trained for one epoch on
**96 training / 32 validation samples**. Seed M1 is initialized and is not evidence
of learning. In s5, G6 still has an M1 source and does not establish the second
cycle; the activated G7 candidate has an M2 source. s10's activated G10 candidate
likewise comes from the actual M2 scientific source.

M1 bundle, common to all runs:
`fa030564a592bbc10ec4eb9ea432b924b4007d43149905e9d1b85aa12ac75123`.

| Run / activation | Accepted training generation, candidate | Exact bundle SHA-256 |
| --- | --- | --- |
| b5 M2 | G2, linear_b | `cffb62e881b4c310ba9f16359ccc2c3bbf5064d02a1ef2847adc34f6bc67324a` |
| b5 M3 | G5, linear_b | `11f29e29993a79a79edc1b94771b21014795dcb2759c7068ee5766280b8d3b4d` |
| s5 M2 | G2, linear_b | `1e56108f6b87c09d4ccb936a7a7b6a1e3451b49ac4f92f9dc898b6b9b78e671b` |
| s5 M3 | G7, linear_b | `fd61f92f855161fd35ecff879981d5a74aebcbefc207dffe9b645310847d89fe` |
| s10 M2 | G5, linear_b | `2b48e2ca6048f7ad8574a873f789c93576785d512c607befc0dd806730d4befd` |
| s10 M3 | G10, linear_a | `5b82bfb48626b83b8d959e9a6b725bcecb7e3234d19ed0fc09a9532df0bb6bc0` |

[b5](sustained_operation/evidence/b5-artifacts.json),
[s5](sustained_operation/evidence/s5-artifacts.json), and
[s10](sustained_operation/evidence/s10-artifacts.json) preserve **every actual
accepted Dataset reference and rows digest, scientific source identity, accepted
Training reference, selected bundle and every alternative candidate's bundle/
parameter digest and real sample/loss metrics**. These static reports derive from
existing authoritative descriptors; they are not a model registry.

For direct lookup of s10's accepted attempts:

| G | Dataset attempt | Training attempt |
| --- | --- | --- |
| 1 | `88e7f80353a94c98899aa3d7df0dbafe` | `719d47c730d4482e952957d64c07f5e2` |
| 3 | `79a046a7e530417585a709fafa446dc0` | `cfcd5b1c9dd74c59b4f79300881fc1fc` |
| 5 | `680e54c284154100a6a8c6e744de9e63` | `d6951f46e6994c42ac45d58ca8915a2d` |
| 8 | `469cbaf47b094d57a79697befc1603ca` | `164d4b365dcf4941979450cae3f490ea` |
| 10 | `8d657cf901de4050a8435cc90c820873` | `bea28fd5592d4d1792410817379089c0` |
| 12 | `6fbe69f372244acdb520b87b31a28942` | No accepted Training |

The s10 M1 session committed +805 nodes and M2 session +801. Explicit M2 activation
was G7→G8, explicit M3 activation G14→G15; subsequent M3 Growth added 155 nodes
at G16. Checkpoint selector counts prove **48 committed session steps plus four
continued steps**, rather than counting the two abandoned final batches in logs.

[Paired 10k decision witnesses](sustained_operation/evidence/s10-decision-comparison.json)
use fresh processes on a disposable copy, eight real search steps per boundary.
With the same starting graph and identical initial search/rollout RNG hashes,
M1→M2 changed the first selected node **7199→8002**; M2→M3 changed it
**1347→1352**. The complete sequences and model identities are retained. The earlier
baseline pairs also changed decisions. This demonstrates model effects on search,
not improved playing strength. Initial empty baseline probes were rejected as
uninformative, preserved, and superseded after applying the captured effective
runtime budget; their empty output is excluded from this claim.

## Larger-run resource and record evidence

The [10k preflight](sustained_operation/evidence/s10-preflight.json) recorded a fresh
exclusive destination, 50,003,890,176 disk bytes and 79,367,512,064 available RAM
bytes, exact source/dependencies/configuration, and no living members of 32 prior
owned worker groups. Raw config SHA-256:
`d5c1a0a9e7a223d5fcdb727968b271107bdf3094de205a23f367ceabc6f938e6`.
Initialization targets 10,000 nodes and may overshoot by one legal expansion; it is
not a hard final node cap. Hard bounds include 1,000 initialization steps, four
steps/cycle, one cycle/invocation, two 420-second sessions, a 7,200-second total
cap, 2,700 seconds/call, and the finite branch budget 100,000. The conservative
cadence bound is at most 211 invocation attempts per session. Resource limits:
8 GiB available-RAM floor, 8 GiB sampled process-union RSS ceiling, 20 GiB disk
headroom, 2 GiB workspace and 1 GiB private scratch. The existing research monitor
checks them every 200 ms and sends TERM through the existing supervisor on risk.
No ceiling was reached. Sampling is not an instantaneous allocation guarantee.

[Experiment evidence](sustained_operation/evidence/experiments.json) contains all
native call resources, observations, committed per-generation nodes/edges/selector
counts, and reader allocations. Important s10 observations:

- The two worker sessions took 841.78 seconds, committing **1,606 nodes / 1,870
  edges / 48 steps**: **1.908 nodes/s, 2.221 edges/s, 0.0570 steps/s**. Including
  the separate final Growth call gives 1,761 nodes in 886.03 seconds, **1.988/s**.
  All 15 native calls, including initialization, activation and independent audits,
  total 3,523.44 wall seconds. These denominators are stated separately.
- Session CPU was 914.43 and 924.98 seconds, about **2.17 and 2.20 CPU cores**.
  Completed Growth ownership totaled 676.19 seconds. Attempted useful Growth was
  61.50 seconds; the 12 committed session batches account for 51.56 seconds, plus
  4.183 seconds in final Growth. Remaining time includes restore, export, index,
  validation, pacing and other workers; it is not all lock contention or idle.
- Dataset source-to-acceptance observations were **113.19–128.30 seconds**;
  new Training followed Dataset by **4.58–64.63 seconds**. Complete Dataset worker
  calls took **110.16–120.08 seconds**, including admission waits. Six Training
  compute attempts took **3.63–4.09 seconds**, five durably accepted; all accepted
  candidate/sample identities are checked. One attempt was interrupted at shutdown.
- Six Reevaluation windows, from snapshot loading to durable patch publication,
  took **52.49–58.20 seconds**. Seven-row inference alone logged about 0.1 seconds;
  ownership publication waits were **3.11–13.63 seconds**. Six exact proposals were
  consumed into UUID/digest receipts. Final M3 Reevaluation took **45.02 seconds**
  and left a pending proposal; it was not counted as another completed cycle.
- M2/M3 activation costs were **274.18 / 298.99 seconds**. Independent cold model
  audits took **814.22 / 852.52 seconds** and passed complete refresh/checkpoint
  comparisons. Final routine recovery took **42.54 seconds**. Initialization cost
  was 297.58 seconds; concurrent Growth cold restores logged 15.14–28.35 seconds,
  and isolated final M3 restore 12.812 seconds. Idempotent activation retries took
  2.77/2.36 seconds and did not advance authority.
- Whole process-union sampled RSS peaks were **1,213.95 / 1,211.74 MiB**; the largest
  sampled worker-group bound was about **352.2 MiB**. Native single-process maxima
  and each group's peak are retained separately; GNU-time max RSS is not a group
  sum. Independent audits reached 742.4 MiB process RSS. RSS sums can count shared
  pages; this is not physical unique memory/PSS.
- The existing recursive profiler, on another disposable G16 copy after one step
  (11,777 nodes), attributed **54.841 MiB** to exclusive reachable owned components,
  with process RSS **335.160 MiB** at that event and native process peak **403.133
  MiB** over the full profiled call. Major components were unopened links 17.531,
  evaluation runtime state 10.908 and checkpoint state roots 7.336 MiB. Exclusive
  component walks were uncapped; standalone whole-runtime/evaluation walks were
  depth capped at 64. This is graph-related Python ownership attribution plus a
  resident process bound, not an exact isolated physical graph RSS. The matching
  5k probe attributed 29.952 MiB at 302.625 MiB process RSS.
- Dataset/Reevaluation compact SQLite indexes ranged **7,364,608–8,396,800 bytes**
  for 10,005–11,383 source nodes; cold build times were **43.97–51.54 seconds**.
  Reported build temp logical bytes reached about 1.24 MB. Worker scratch peaks
  were **8,773,632 / 8,957,952 bytes**. Private reader-cache ownership ended only
  after the supervisor confirmed all worker groups stopped. No default cache was
  used. Current snapshot and historical allocations are reported separately above.

Final legal replay verified **eight retained terminal sequences** in variant 5T,
with move counts **63, 63, 60, 53, 53, 20, 20, 20** (points are moves+36).
The best is **63 moves / 99 points**. Complete fingerprints and replay results
are in the final routine/audit evidence; move coordinates remain in the retained
workspace leaderboard. Every claimed record was replayed through the existing
independent legal-move and terminal checks. Larger root/frontier estimates are
not certified records, and terminal proof counts inside the DAG need not equal
distinct retained leaderboard sequences.

A separate fresh-process continuation witness used a disposable copy of the final
G16. Only that copy's pending proposal was archived, preserving its digest, to
isolate one next Growth step. The existing oracle predicted G17 with **11,777
nodes / 13,113 edges**. Actual one-step Growth matched the **entire checkpoint
payload**, all three scientific files byte-for-byte, selector and both RNG states.
Routine recovery then passed. Source s10 file inventory and original pending
proposal were unchanged. [Continuation witness](sustained_operation/evidence/continuation-copy.json).
This is same-executable continuation; no cross-version search history was mixed.

## Scientific, crash and final CI evidence

Final checks ran against the installed guard wheel, with all selected test IDs
retained in [CI evidence](sustained_operation/evidence/final-ci.json):

| Tier/check | Result |
| --- | --- |
| FAST | 174 passed, 76.70 seconds |
| PR | 2,786 passed, two CUDA skips, 959.39 seconds |
| FULL | 2,950 passed, two CUDA skips, 1,631.61 seconds, exit 0, no execution errors |
| Test identity comparison | All 2,923 original FULL IDs preserved; 29 additions; zero missing |
| Ruff / formatting | Passed; 774 files already formatted |
| Strict static ratchet | 206 diagnostics versus 209 baseline; zero new, three removed |
| Build / installed smoke / toolchain | Wheel and sdist built; distribution/evaluator/checkpoint smoke and tool pins passed |

FULL contains exhaustive scientific values/certainties and family policy gates,
selected-value-v2 streaming Dataset parity, exact scoring, complete format-3 and
streaming checkpoint continuation, atomic Reevaluation and exact receipts,
publication/activation crash points, concurrent stale-worker rejection, live
reader pins, rolling-retention blockers, and supervisor whole-group/cache lifetime.
Relevant suites include `test_morpion_generation_publication_crashes`,
`test_morpion_model_activation_crashes`, `test_morpion_patch_receipt_concurrency`,
`test_morpion_atomic_reevaluation_science`, `test_morpion_rolling_recovery_report`,
`test_morpion_bootstrap_process_control` and the new ownership/isolation/operational
validation regressions. No scientific expectation or ratchet baseline was weakened.

Earlier green PR/FULL results belong to a predecessor before the shell preflight
correction, not the final executable. The final FULL above is one complete green
run after that correction, not a collection of isolated retries. An initial FAST
environment failure from missing installed distribution metadata, the empty offline
probes, pb's retention refusal, and the snapshot scope assertion are preserved in
raw logs. The matched raw budget field `node_target=100` was an unused parser
default when copying the 5,017-node frozen genesis; authoritative counts and equal
initial bytes establish the actual input. Those raw budgets were not rewritten;
`b71891ec` fixes future copied-budget reporting. Documentation-only descendants do
not rerun FULL.

## Reproduce, stop, recover and review

All retained paths are under `/home/pompote/oldata/victor/morpion-scratch/s26/`:
`b5`, `s5`, `s10`, `i5`, `p90`, `p2`, `pw`, `pb`, disposable probes, wheels,
private environment, caches, logs and inventories. Raw paths and hashes are listed
in [the evidence manifest](sustained_operation/evidence/raw-evidence-manifest.json).
The report and curated JSON are review artifacts; raw workspaces preserve the
accepted rows, all candidates, checkpoints, science and exact authority.

Use the installed sources identified above, from this worktree. The retained
`env.sh` directs TMPDIR, SQLite temp, MLflow, leaderboard, application output,
cache and matplotlib state into task-private paths, disables bytecode, fixes the
hash seed and limits BLAS/OpenMP to one thread. Use **new destination names**:

```bash
cd /home/pompote/oldata/victor/morpion-scratch/s26/w
source ../env.sh
export PYTHONPATH=.
"$PYTHON_BIN" -m docs.research.operational_bootstrap.prepare ../NEW_INPUTS
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce ../NEW_5K ../NEW_INPUTS \
  --nodes 5000 --cluster-seconds 180 --growth-pause 2 --ownership-wait 60 \
  --growth-steps 4 --max-seconds 7200 --call-max-seconds 2700
# Separate fresh bounded run, after headroom and previous-worker checks.
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce ../NEW_10K ../NEW_INPUTS \
  --nodes 10000 --cluster-seconds 420 --growth-pause 2 --ownership-wait 120 \
  --growth-steps 4 --max-seconds 7200 --call-max-seconds 2700
```

These commands perform explicit local M2/M3 activation at the recipe's declared
transitions and full independent audits after each. A failed audit stops the
recipe; investigate preserved evidence before further promotion. Reproduce the
controlled four-case comparison using [the cadence commands](sustained_operation/README.md).
Do not run concurrent CI when comparing timings. The normalized scientific config
identity differs from the raw config-file digest; both original bytes are retained.

For a manual bounded start/restart of a **new private initialized workspace**, use
the existing shell supervisor below. Its persisted `control.json` must already set
four steps/cycle and zero time-based intermediate save, as the recipe does. Check
available RAM/disk, exact installed identity and no living previous worker groups
first. Manual supervision provides the time horizon; the full RSS/disk monitoring
used for acceptance is the research recipe above and should be used for larger
experiments.

```bash
# Set this to your new private initialized run, never a historical evidence path.
export MORPION_WORK_DIR=/home/pompote/oldata/victor/morpion-scratch/NEW_RUN/workspace
export MORPION_SCIENTIFIC=1 MORPION_USE_INSTALLED_PACKAGES=1
export MORPION_CLUSTER_HEADLESS=1 MORPION_CLUSTER_WORKERS=all
export MORPION_CLUSTER_MAX_SECONDS=120 MORPION_CLUSTER_STOP_GRACE_SECONDS=15
export MORPION_GROWTH_WORKER_MAX_CYCLES=1 GROWTH_SLEEP_SECONDS=2
export MORPION_WORKER_OWNERSHIP_WAIT_SECONDS=120
export MORPION_REEVALUATION_MAX_NODES_PER_PATCH=7
export MORPION_CLUSTER_OPEN_STATUS=0 MORPION_CLUSTER_SHOW_RECAP=0
export MORPION_SEED_MODELS_FROM_WORK_DIR=
"$PYTHON_BIN" -m docs.research.operational_bootstrap.reproduce \
  /home/pompote/oldata/victor/morpion-scratch/NEW_RUN ../NEW_INPUTS \
  --verify --validation-level routine
bash scripts/launch_morpion_gnome_cluster.sh > ../NEW-supervisor.log 2>&1 &
supervisor_pid=$!
# Stop before the horizon if needed; expected TERM exit status is 143.
kill -TERM "$supervisor_pid"
wait "$supervisor_pid"
```

Require the `all worker groups stopped` confirmation and no living members of the
recorded owned groups before reuse or reader-cache removal. A normal deadline
exits 0. Restart by repeating routine verification and the identical shell command
with a fresh supervisor log. Never rerun initialize on that workspace, remove a
claim merely by age, ignore unresolved staging ownership, or use SIGKILL as routine
shutdown. Read/audit failures block restart; they never authorize a fresh history.
Manual restart does not activate a newly trained model automatically.

Final CI was run with `CHIPIRON_TEST_INSTALLED=1 PYTHONPATH=.` and fresh private
`--basetemp` destinations using `make test-fast`, `make test-pr`, `make test-full`,
followed by `scripts/static_analysis_ratchet.py --mypy-strict`, `ruff check .` and
`ruff format --check .`. Exact arguments and native logs are retained in
`s26/ci-guard.sh` and `s26/e/`.

For graphical review in VS Code:

```bash
code /home/pompote/oldata/victor/morpion-scratch/s26/w
# Source Control → Source Control Graph → select this branch.
# Compare each implementation commit with its parent.
# Open the ownership change as a concrete graphical file diff:
git -c diff.tool=vscode -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt 424ee3a5 HEAD -- \
  src/chipiron/environments/morpion/bootstrap/generation_publication_io.py
```

Replace the file argument to inspect another change. `git diff --stat 424ee3a5 HEAD`
lists the complete comparison. The executable comparison ends at `99a63802`;
`HEAD` additionally includes the final report and evidence. Source Control Graph
provides commit browsing; the CLI command opens a concrete file comparison.

Before a substantially larger record attempt: define and prove the retirement
contract above, handle claims left by terminated owners with authenticated lifetime
evidence, bound cumulative descriptor/receipt history, reduce measured cold restore
and index/export cost without weakening exact validation, and remeasure fairness/
throughput at the larger graph size. A stricter total-node stop, if required beyond
finite step/time limits, needs explicit admission semantics for the last expansion.
Long-run strength evaluation is separate from the demonstrated model-decision effect.
