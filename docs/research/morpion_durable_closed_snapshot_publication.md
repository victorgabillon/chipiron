# Durable closed scientific snapshot publication

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Local opt-in integration based exactly on `27b92ff342f5dea4e96dcaca25c9d313a95018c3`,
whose parent is the rolling-design commit `f05d418c42b9292410702f51de2d976fbf693059`.
The isolated branch is `feat/morpion-durable-closed-snapshots`, under
`/home/pompote/oldata/victor/morpion-scratch/durable-closed-snapshots-20261022/worktree`
on the secondary ext4 filesystem. Nothing is cherry-picked from the already
contained scientific, streaming or durability stacks.

## Contract and integration boundary

Publication version **3** is an explicit physical-layout fence:
`generation_publications_v3`, `.generation_publication_v3.json`, and
`durable_generation_publication_v1.schema_version=3`. It uses the existing
publication class, ownership lock, read pins, attempt marker, immutable rename,
seal inventory and authoritative run-state transaction. Publication v1 and v2 keep
their previous layouts and interpretation. Unknown versions fail closed.

The scientific value contract stays v2
(`anemone-values-canonical-morpion-1`), with closed kind
`morpion_closed_scientific_snapshot` and storage layout 1. Each capsule owns:

* `scientific_snapshot/generation.json`, `records.jsonl`, `updates.jsonl`;
* the complete runtime checkpoint, in its existing configured format;
* the frozen accepted configuration and fixed-epoch model dependencies;
* the existing attempt marker, seal and canonical-validation evidence.

The three scientific files are complete for the current graph. No historical
update, creation map, scientific descriptor or catalog is copied into this layout.
Original creation generations remain on individual canonical records. The capsule
identity has no `catalog_sha256`. The seal binds current scientific provenance,
producer/configuration, checkpoint digest and generation. Its inventory binds
exact bytes of every payload. Model pins are independently checked against their
specified content identities, as is the frozen configuration.

Explicit new-workspace genesis uses the existing
`initialize_scientific_v2_publication(..., closed_scientific_snapshot=True)`.
Continuation uses the existing `run_pipeline_growth_stage` with all receipt,
durable-publication and scientific-producer opt-ins plus
`closed_scientific_snapshot=True`. The real runner exposes a thin closed-writer
adapter over the same frozen runtime used for its checkpoint. Growth generation
and scientific generation are consecutive and equal, starting at one. Producer,
configuration and model epoch changes remain unsupported.

There is no CLI, launcher, Dataset worker, Training worker or Reevaluation routing
activation. Default reader classification still refuses the closed layout;
explicit bounded reading requires `allow_closed_snapshot=True`. The managed
Reevaluation parity tests inject the explicit reader adapter in test scope only.
This verifies scientific patch/cursor equivalence without activating a production
worker route. Historical publication-v2 workspaces are not silently migrated.

## Transaction, integrity and recovery invariants

1. The existing exclusive Growth lock spans authoritative-state validation,
   checkpoint restore, export, commitment and receipt acknowledgement.
2. Before restore/next-generation construction, retain the existing predecessor
   capsule read pin. Recheck its authoritative membership, inventory and identities.
   Reacquire the selected predecessor for each cycle; release after committed-state
   validation or failed-invocation cleanup. A pin protects cooperative ownership,
   not integrity.
3. Freeze accepted configuration/model dependencies in a fresh private attempt.
   Write the closed snapshot first, then use the unchanged runtime checkpoint
   serializer on that same quiescent runtime. Both precede sealing and commitment;
   there is no scientific mutation between them. Historical v1/v2 keep their
   existing operation order.
4. Copy only exact verified canonical bytes from the actual committed scientific
   predecessor. New references undergo authoritative encode/load/redump checks.
   Regenerate and validate every current value, certainty, target, edge and order.
5. The trusted closed writer can transfer its completed canonical checks through
   a private, attempt-local O(1) result sink. Before reuse, the publisher requires
   the exact descriptor digest, producer and record descriptor to match. The
   existing bounded reader rehashes records/updates and validates all mutable
   rows, origins and topology in a fresh temporary SQLite index. Missing transfer
   triggers full canonical decoding. No transfer survives process death or becomes
   a cache/ledger. The existing committed validation evidence is sealed only after
   this validation. Canonical reuse remains an integrity assertion of trusted local
   executable code, not a mathematical proof of a game's value.
6. Reverify the predecessor and fixed model epoch. Seal and fsync all dependencies,
   then publish using the existing no-replace atomic rename and namespace fsyncs.
   Validate the resulting closed capsule before returning a candidate run state.
   Its scientific descriptor's shard hashes must equal the freshly verified seal
   inventory, preventing corruption between export validation and sealing from
   becoming authoritative.
7. The existing durable `run_state.json` replacement and directory fsync are the
   sole commit barrier. Checkpoint pointers, generation root and exact receipt
   ledger advance together. An orphan's visible final name grants no authority.
8. Repair missing advisory discovery, then verify exact pending patch/cursor/model
   identities before acknowledgement. Restart adopts only the complete observed
   committed state; it never combines an old checkpoint with newer receipts.

The existing protocol retains all committed roots and receipts. This change neither
retires history nor adds a deletion policy. Existing cleanup of proven uncommitted
attempts remains the existing mechanism. A complete closed scientific snapshot is
independently readable; authoritative recovery still checks ancestry and receipts.
Those contracts must not be confused. Cooperative immutable ownership, deterministic
codec execution and SHA256 collision resistance remain explicit trust assumptions;
there is no atomic protection against arbitrary out-of-protocol writes between reads.

## Scientific invariants and checkpoint limitation

All direct/tree/effective Values, independent certainties, selected target/source,
null versus missing distinctions, current incoming/outgoing DAG edges, canonical
references, creation origins and export ordering use the existing serializers and
validators. Nine legacy Dataset policies keep their interpretation. Five enabled
confidence-v2 policies retain selected-source certainty and terminal protection;
four exact-family policies remain gated. No cross-state proof is introduced.

The actual reachable endgame keeps tree/effective FORCED(35) and direct neural
ESTIMATE(42). Direct preference does not certify 42, selected-source exact-only
filtering excludes it, and a certified terminal target 35 cannot be overwritten by
an uncertified family estimate. Source provenance and exact JSONL serialization
are compared, rather than normalized away.

Anemone checkpoint serialization is unchanged. The design's representative-parent
round-trip discrepancy remains independent: 7/70/161 raw node-shell records differed
on the first restore at N=1003/5013/10018, while scientific values/state references,
linked topology, selector and RNG checks matched and subsequent saves stabilized.
This integration does not claim raw checkpoint equality or fix that discrepancy.
The existing actual-runtime continuation assertions are exercised for both JSON-zst
and sharded checkpoint formats through the closed publication adapter.

## Validation and measurements

One complete installed-wheel CI invocation passed: **2377 passed, 2 skipped,
4 deselected**, 129 warnings, pytest elapsed **1381.49 s** (23:01). All 120 new
cases passed in that same invocation. The two skips require unavailable CUDA;
the four checkmate tests are excluded by the repository's normal
`not integration and not external_data` marker expression. The run was serial
(`-n 0`), with the existing generated-fixture safety plugin and all mutable
outputs redirected to this secondary-disk scratch root. It exceeds the current
GitHub pytest step's 15-minute limit when run serially; no hosted CI run is claimed.

Other local CI gates passed: pinned toolchain coherence, repository Ruff checks
and formatting, sdist/wheel build, installed-package guard, and the static ratchet
(**206 inherited findings, 0 new, 3 removed**, against 209 reviewed baseline
findings). This does not mean all inherited static diagnostics are resolved.
All 472 production Python source files match the installed wheel used for CI.
The independent Python 3.13.5 environment uses the pinned Anemone/Atomheart/Coral/
Valanga wheels; all 367 Python files in those four packages were byte-identical
to the previous validated environment. No old environment was repaired or changed.

Development outcomes are retained separately: initial environment imports lacked
ijson and Chipiron metadata; early fixture failures failed to trigger publication;
a 109-test focused run passed, followed by a later 164-test run with 163 passes and
one incorrect expected error-message regex. That assertion was corrected before
the single clean full-suite invocation above. No sum of reruns is presented as CI.
The benchmark's first 1k development scenario also failed to publish unchanged
cycles, and its continuation probe initially used an exhausted default branch
budget. Those attempts remain in scratch evidence and are excluded from measured
publication claims; the corrected probe asserts generation advancement and actual
selector steps.

### Measurement method

Fresh processes run serially on newly generated legal Morpion graphs, seed 4.
Initial timing includes complete scientific export, real runtime checkpoint,
copy into a private capsule, validation, sealing, fsync, run-state commit and
advisory publication; graph construction is excluded from wall time but included
in lifetime peak RSS. Repeated timing includes Growth restore, an identity-bound
no-op patch, complete snapshot/checkpoint publication and receipt acknowledgement.
The patch preserves every value; unchanged scientific digests are asserted. The
fixed valid model is declared and pinned from genesis, and reloads do not evaluate
or replace saved node values. Patch construction is outside the publication timer.

Reader timing includes opening and visiting every scientific node/value contract/
creation origin; its separate initialization observation is retained too. A fresh
process does not imply a cold OS page cache: kernel I/O counters are reported as
observed. CPU, `/proc` I/O, `ru_maxrss`, fsync counts/timing and SHA256 file-digest
calls reuse existing measurement utilities. The Python `os.fsync` counter does
not intercept SQLite's native sync calls; a separate syscall-traced pass records
those too. Profiler attribution is a separate
instrumented pass, never substituted for unprofiled wall times. Dataset comparisons
materialize a reference oracle in their own process and are not bounded-RAM claims.

| Actual nodes | Initial complete publication s | Repeat G2 / G3 s | Initial / G3 peak MiB | G3 scientific writer s | G3 checkpoint s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 3.574 | 1.841 / 1.907 | 307.2 / 303.2 | 0.749 | 0.163 |
| 5,013 | 39.449 | 5.662 / 5.837 | 386.5 / 334.5 | 2.911 | 0.585 |
| 10,018 | 94.058 | 10.883 / 10.820 | 491.1 / 362.4 | 6.006 | 0.887 |

Each initial publication encodes N new references; the writer checks them, and
unattested genesis independently checks the copied input again (2N canonical loads
in the complete initial operation). Mean/max replay lengths are 6.94/20, 15.73/62 and 17.65/62
moves respectively: node count alone does not describe canonical replay cost.
Both repeated publications encode/decode
**zero scientific state references**, copying all N canonical records. Runtime
checkpoint restore still loads its anchor; this is not a claim of zero state work.
At 10,018 nodes the repeated writer spends 0.167 s copying canonical bytes,
0.503 s building/serializing current mutable rows, 2.645 s verifying/indexing its
source and 2.629 s validating/hashing its output. These are writer component timers;
additional capsule/checkpoint validation and commitment occur outside that writer.

| Actual nodes | Cold open + full visit s | Reused open + full visit s | Full reference load s | Cold / reused / reference peak MiB | Selected discovery s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 1.206 | 0.055 | 0.881 | 290.4 / 290.9 / 294.7 | 0.021 |
| 5,013 | 15.191 | 0.229 | 13.821 | 291.1 / 289.8 / 318.2 | 0.025 |
| 10,018 | 35.077 | 0.425 | 31.326 | 291.5 / 290.5 / 350.8 | 0.028 |

The reference-load timer excludes subsequent digest computation; the indexed visit
timers include it. At 10k, initialization alone is 34.717 s cold and 0.074 s reused.
A cold consumer validates all canonical references; a repeated publisher can use
its pinned predecessor's sealed canonical evidence while rechecking actual bytes.
The oracle/reference RSS above is sampled after the first reference load; the
higher final lifetime peaks of paired-oracle processes are also retained in JSON.

| Actual nodes | Scientific files allocated MiB | Complete checkpoint allocated MiB | G3 CPU user+system s | G3 hashed MiB | G3 logical read / physical written MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 0.992 | 0.117 | 0.903 | 37.60 | 45.49 / 6.62 |
| 5,013 | 5.797 | 0.438 | 3.133 | 201.19 | 306.53 / 51.01 |
| 10,018 | 11.941 | 0.844 | 5.922 | 410.17 | 627.08 / 120.30 |

All three repeated G3 operations invoke 890 file-digest calls, 58 Python-visible
file fsyncs and 99 directory fsyncs. Python-visible fsync time totals 0.259 / 0.218 /
0.215 s respectively. Physical read counters are zero in these warm-page-cache
observations; logical reads, SQLite scratch writes and cancelled writes are all
retained in the evidence. Payload allocation excludes the separately reported seal,
config/model and run-state metadata; whole-workspace allocation is also recorded.
No claim is made that the 890 checks are minimal: existing receipt, pin and capsule
boundaries repeat validations. This task preserves those guarantees.

Across these newly generated runtime sources, **6,276 complete supervised JSONL
rows** were compared byte-for-byte: 896 at 1k (five v2 plus nine legacy policies),
1,870 at 5k and 3,510 at 10k (five v2 policies each). All scientific nodes, including
nodes without targets, were compared through the ordered full-field scientific
digest. Lazy runtime evaluation supplies targets for only 64 / 374 / 702 nodes;
missing targets are preserved, not filled to inflate row counts. The five indexed
v2 policy passes, including context construction and JSONL writing, take
0.132–0.143 / 0.695–0.722 / 1.415–1.483 s (448–485 / 518–538 / 473–496 emitted
rows/s). These are sparse-target runtime populations, not a throughput comparison
with prior fixtures that gave every node a target. Metadata equality
is separately asserted, and four exact-family policies remain covered by explicit
rejection tests. The full CI also runs the earlier large synthetic policy suites.

The 5k and 10k benchmark drivers completed all nine stages without retries. The
1k continuation stage was rerun separately after the harness branch-budget fix;
its earlier failure and successful retry remain separate evidence. The checkpoint
comparison finds the existing 7/70/161 representative-parent discrepancies; G3
also has `evaluator_version=2` versus zero at genesis, from two declared model
reloads. Other checkpoint section hashes, including selector/RNG and node-runtime
values/links, match. The regression keeps the differing raw fields visible.

### Fixed-population history

The history series uses **five actual Anemone nodes** (the first atomic expansion
exceeds the requested lower bound of four), a complete real sharded checkpoint,
and the same fixed model from genesis. Existing benchmark helpers for reduced scalar
checkpoints were deliberately not used. One identity-bound no-op patch is committed
per subsequent generation, giving R=G−1 exact receipts. The graph and scientific
values remain fixed. G8/G16/G32/G64 compare the complete reference and indexed
scientific digest with genesis. No scalar-checkpoint substitute is used.

| Generation | Publication s | Peak MiB | Current scientific bytes | Run-state bytes | Retained workspace allocated MiB | Selected discovery s / hashed bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.179 | 294.3 | 6,414 | 32,135 | 1.320 | 0.018 / 47,594 |
| 16 | 1.151 | 293.6 | 6,419 | 58,860 | 2.480 | 0.022 / 47,611 |
| 32 | 1.183 | 295.0 | 6,419 | 112,298 | 4.801 | 0.025 / 47,609 |
| 64 | 1.415 | 295.5 | 6,419 | 219,183 | 9.441 | 0.017 / 47,611 |

All 63 subsequent publications copy five records, encode no new references and
retain exactly three scientific files. The slight byte-size change is the generation
number's decimal width. There are exactly 64 committed roots and 63 exact receipts
at G64. Publication digest calls remain 890 (about 4.05 MB hashed); selected discovery
hashes 29 files, but its total logical reads rise from 160,284 to 796,047 bytes.
This directly exposes growing authoritative metadata/candidate reads despite a
fixed scientific closure. Single timings include storage noise and do not prove
constant-time discovery or publication. No committed generation was deleted.

### Instrumented cost attribution

A separate 1,003-node G4 pass under standard cProfile and syscall tracing takes
3.359 s, versus the unprofiled 1.841–1.907 s earlier. Its cumulative times overlap
and must not be summed: three existing index builds 1.568 s; closed writer 1.415 s;
capsule publication 0.714 s; closed adapter validation 0.553 s; 19 capsule validations
0.418 s; checkpoint creation 0.285 s; 157 Python-visible fsync calls 0.279 s;
890 SHA256 file-digest calls 0.045 s; authoritative run-state save 0.020 s; two
immutable directory publications 0.017 s. SQLite performs 30 commits (0.557 s),
12 schema scripts (0.495 s), 21,139 executes (0.048 s) and 6,018 executemany calls
(0.012 s). The three indexes belong to source validation, writer output validation
and capsule output validation; they are the existing disposable SQLite mechanism,
not new persistent caches. Their remaining duplication is visible for later review.

Whole-process syscall tracing also includes patch preparation and final evidence
checks, outside the publication timer. It observes 374 fsyncs, 16 renames and two
no-replace renames; this count includes native SQLite syncs that Python hooks miss.
Its CPU-time summary is retained separately from a second `strace -w` wall-time
pass. The latter observes the same 374 fsyncs, totalling **1.244 s of syscall wall
time** across the complete probe process; the first trace's 0.0039 s is syscall CPU
time and must not be read as storage latency. Instrumented repetitions are not
substituted for unprofiled benchmark rows.

## Crash and concurrency matrix

The new suite contains 120 cases, reusing the existing scientific assertions where
possible. The crash harness uses actual `os._exit(73)` in a fresh child process,
not an exception that unwinds the transaction. Filesystem-error cases inject
`OSError` at the real fsync boundary. These are process-death/filesystem-error
experiments on local ext4, not a controller power-loss test.

| Boundary or fault | Required outcome |
| --- | --- |
| Partial canonical copy/current updates, checkpoint write, source/output validation, seal, file/directory fsync | Previous committed generation and exact pending patch remain the only recovery inputs |
| Before/after immutable publication, before run-state replacement, partial temporary run-state serialization | A complete orphan is still uncommitted and unavailable to readers |
| After run-state replacement, before its directory fsync | Validate and sync the complete observed new state before adopting it; never reapply its committed receipt |
| Before/during/after discovery, before/after acknowledgement | Recover the committed checkpoint/receipt and reconstruct missing advisory metadata |
| Second process death during restart/discovery repair | Preserve the same committed head and exact receipt |
| Eight file/directory fsync failures | Resolve commitment from observed state, not from whether the call raised |
| Five genesis interruption/retry boundaries | Before commit reuse only prepared quiescent inputs; after commit use normal committed-state recovery |
| Copied/new record, update, producer/codec, checkpoint pointer, model epoch, source or transfer-identity damage | Refuse publication without changing authoritative run state |
| Damage after sealing/rename but before commit | Fresh inventory/scientific/checkpoint verification rejects the candidate |
| Committed record/update/checkpoint/model/seal corruption with an existing reader cache | Fail closed; no cache or older checkpoint fallback |
| Competing Growth owner | The existing kernel lock excludes the second owner |
| Older reader across a newer head commit | Its source remains readable under its own stable pin; exclusive retirement cannot acquire that pin |
| Missing/stale discovery | Repair missing metadata from authority; refuse conflicting identity instead of trusting or silently rewriting it |
| Patch retry | Preserve exact alpha-0.5 effects, one content-bound receipt and acknowledgement |

The matrix tests 19 publication process-death boundaries, one twice-interrupted
restart and eight fsync error targets. The scientific group additionally exercises
36 actual-endgame policy/source/filter combinations, six complete managed patch/
cursor comparisons and two real checkpoint formats with selector/RNG continuation.
The old closed-format regressions also remain in the full suite.

The full-repository lint gate exposed pre-existing research-script/test docstring,
closure-binding and annotation findings that the earlier targeted checks did not
cover. Local support-file cleanup binds immediately invoked loop callbacks,
adds missing documentation and uses the configured type/style checks; it changes
no scientific policy or saved evidence. Existing closed-reader integer checks now
reuse the same bool-rejecting `_integer` helper, and explicit annotations remove
inherited static ambiguity. The reviewed static baseline is not enlarged or
suppressed. Development logs retain the initial failures separately.

## Code and ownership delta

The production delta modifies **11 existing files: +363 / −61 lines, net +302**.
The separate local support-cleanup commit `4f6740e6` changes nine existing
research/test files (+67/−13); it is kept distinct from the integration for review.
There is no new production module, on-disk cache, database, journal or authority.
The largest addition is the 83-line closed-layout validator adapter in the existing
scientific publication module. It translates the closed manifest into the existing
seal identity and calls the existing indexed reader; it does not duplicate record,
value or topology validation. The 27-line runner method connects existing live
node ordering to the already-reviewed closed writer. Remaining changes carry the
explicit version/opt-in, pins and physical identity checks through the existing
transaction. The private O(1) validation result avoids replaying every new state
immediately after the writer has just verified the same exact output bytes.

Closed publication bypasses `catalog_paths`, historical catalog cloning and the
cumulative scientific prefix comparison. Those branches remain for historical
publication versions 1/2; deleting them now would remove required compatibility.
The new branch rejects accidental calls to cumulative `export_paths`. No retention
or model-epoch generalization has been introduced.

## Remaining scaling costs and review order

Let S be complete canonical-reference bytes, U complete current value/topology
bytes, C checkpoint bytes, M pinned model bytes, G committed generations and R exact
receipts. One closed capsule costs Θ(S+U+C+M), plus a constant number of scientific
file entries and the checkpoint's own inventory. An unchanged scientific export
still parses/copies/hashes Θ(S), serializes Θ(U), and builds bounded disk indexes;
it does not become constant-time. New-state codec work depends on K and replay
length; checkpoint writing/restoration retains its existing current-graph costs.
With bounded-size scalar/ID fields, U grows with N+E; canonical bytes S grow with
replay histories. SQLite key lookups/index construction add their logarithmic
factors. Neither N alone nor the small fixed-population run bounds those costs.

All committed snapshots/checkpoints are retained in this task: physical retention
is the sum of each generation's owned payloads (with existing immutable model
hard-link deduplication where applicable). No bounded total-disk claim is made.
Run-state roots and exact receipts still occupy Θ(G+R) RAM/disk metadata, and their
rewrites alone accumulate Θ(G²) metadata serialization/I/O when R grows
proportionally to G. Model-origin lookups, retained pins, discovery
candidate scans and historical recovery/audit reads remain. Scientific closure
verification now addresses the current snapshot, not all historical scientific
updates; it does not make the entire publication/discovery path history independent.

The indexed scientific path adds no complete Python snapshot or second tree.
The runtime adapter deliberately reuses Anemone's `_all_nodes_in_tree_order`, which
allocates an O(N) list of references to existing live nodes. It is not a copy of
scientific records, but it must not be described as constant auxiliary RAM. Growth
still owns its O(N+E+materialized-state) live runtime, checkpoint serialization retains
its existing current-graph RAM costs, and authority metadata is not bounded
independently of G/R. The row-wise writer and SQLite validation buffers are bounded by maximum row size
and configured caches; a long replay/proof or a high-degree DAG node can itself
increase row size. Whole Growth publication is not bounded independently of N. Full-reference
and paired-JSONL processes intentionally materialize an oracle; their RSS is not
used as a bounded-reader claim.

Keep this delta local until the sibling scientific/streaming/certainty/durability
stacks have an approved common base. Review the prior offline layout first, then
this adapter alone onto its exact accepted head. Later independent units should
address compact authoritative recovery roots and receipt semantics, explicit worker
routing/activation gates, and dry-run reachability retention. Scientific certainty
questions, model epoch transitions, durable stage result transactions and historical
retirement are not solved by reader or publication parity. No duplicate-code PR,
merge or production activation is part of this work.


## Evidence and reproduction

The compact [evidence manifest](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/evidence-manifest.json)
binds the [resource/scientific observations](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/benchmarks.json),
[CI result](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/ci-summary.json),
[exact source ancestry and production hashes](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/source-manifest.json),
[raw checkpoint classification](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/checkpoint-comparison.json),
[profile attribution](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_closed_snapshot/evidence/profile-summary.json),
package versions and machine details. Full logs, installed-package checks, profiles,
fixtures, JSONL files, build outputs and development failures remain under:

`/home/pompote/oldata/victor/morpion-scratch/durable-closed-snapshots-20261022/evidence`

All 32 pre-existing worktree registrations were preserved; their recorded HEAD,
status and tracked binary-diff fingerprints are unchanged, including the dirty
research checkout. Missing/prunable old `/tmp` registrations were left alone.
The new worktree's final production files retain the exact hashes tested by the
installed-wheel CI invocation. No historical experiment or existing evidence was
used as an execution input or modified.

From this isolated worktree, with the pinned environment and scratch environment
variables recorded above, reproduce in **new, absent directories**, sequentially:

```bash
TASK_ROOT=/home/pompote/oldata/victor/morpion-scratch/durable-closed-snapshots-20261022
PY="$TASK_ROOT/ci-venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.
export TMPDIR=/home/pompote/oldata/victor/morpion-scratch/dc-20261022
export XDG_CACHE_HOME="$TASK_ROOT/cache" XDG_CONFIG_HOME="$TASK_ROOT/cache/config"
export MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" QT_QPA_PLATFORM=offscreen
export CHIPIRON_OUTPUT_DIR="$TASK_ROOT/tmp/app-output"
export ML_FLOW_URI_PATH="sqlite:///$TASK_ROOT/tmp/app-output/mlflow.db"
export ML_FLOW_URI_PATH_TEST="sqlite:///$TASK_ROOT/tmp/app-output/mlflow-test.db"
EVIDENCE="$TASK_ROOT/evidence"
"$PY" -m docs.research.durable_closed_snapshot.reproduce "$EVIDENCE/reproduce-1000" --count 1000
"$PY" -m docs.research.durable_closed_snapshot.reproduce "$EVIDENCE/reproduce-5000" --count 5000
"$PY" -m docs.research.durable_closed_snapshot.reproduce "$EVIDENCE/reproduce-10000" --count 10000
"$PY" -m docs.research.durable_closed_snapshot.reproduce "$EVIDENCE/reproduce-history" --count 4 --history --last 64
```

The final driver asserts generation advancement, actual continuation steps,
unchanged scientific digests and exact JSONL bytes; the `--history` switch only
selects repeated-history reporting, not a substitute runtime. Startup and patch
preparation are outside measured publication wall times. The process peak includes
them. Benchmarks use no real saved generation and no worker concurrency.

The full installed-package test command was:

```bash
PYTHONPATH=. CHIPIRON_TEST_INSTALLED=1 "$PY" scripts/ci_test_metrics.py -n 0 \
  --durations=30 --durations-min=0 \
  --cov-report=json:.ci-reports/coverage.json --junitxml=.ci-reports/pytest.xml \
  --basetemp="$TMPDIR/ci" -m 'not integration and not external_data' \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety
```

Use the isolated environment's configured Qt offscreen, XDG cache/config, app output
and SQLite MLflow locations when reproducing CI; they are secondary-disk scratch
paths, not a production configuration. For a repeated CI run, select a fresh
`--basetemp` and distinct log/evidence paths so previous fixtures and results remain
available. Source code and installed wheel must match.
Ruff/toolchain checks and the static baseline ratchet are independent gates, not
inferred from pytest success.
