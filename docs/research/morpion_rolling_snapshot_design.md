# Rolling scientific snapshots: measured design decision

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Design review, 2026-10-22. No production implementation, format migration,
retirement, committed-artifact deletion, worker deployment or activation.

**Recommend A: a complete, immutable current scientific snapshot, published with
its complete checkpoint. Carry verified canonical record bytes forward; do not
reconstruct every unchanged state on each save.** Keep current plus two eligible
rollback roots, additional roots required by readers/work in flight and explicit
scientific retention. This bounds heavyweight history by actual liveness, not by
elapsed generation count. It does not bound a growing search, receipts, models,
or an indefinitely pinned generation.

B saves measurable state-record disk space, especially with long replay histories.
On the tested secondary disk the incremental copy cost is small compared with
scientific reconstruction. B does not eliminate complete checkpoint writes,
complete mutable updates, source hashing or index validation. Start with closed
snapshot ownership and revisit physical sharing only against an explicit capacity
budget. Existing immutable model pinning is independent of this choice.

This is a **design recommendation**, not a production-readiness claim. Exact
checkpoint shell round-trip parity has a pre-existing discrepancy documented below.
Rollback/replay semantics and versioned retirement must also be resolved before
activation. No new storage format was implemented to obtain these measurements.

## Sources and isolation

Exact reviewed implementation: `a2904d106daccce2f44d7de58d4410dc924e0c79`.
Exact report/base: `0790c596f52e4d3360a416a385b771cf70049d25`, containing that
implementation and all prior scientific, certainty and durability integrations.
No implementation was cherry-picked or applied twice.

New branch: `design/morpion-rolling-snapshots`. New worktree:
`/home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022/worktree`.
All fixtures, temporary files and evidence are on the secondary disk beneath the
same task root; the short multiprocessing temporary directory is
`/home/pompote/oldata/victor/morpion-scratch/rs-20261022`.

Read the [architecture review](morpion_lean_durable_pipeline_architecture.md) and
[selected-generation report](morpion_selected_generation_verification.md). The
former describes the earlier all-history poll; the pinned implementation now
verifies only selected capsules, but their scientific dependency closure still
contains historical updates/descriptors. This review does not attribute the old
all-capsule polling behavior to the new base.

## Comparison method and measured costs

The [driver](rolling_snapshot/measure.py) calls the existing scientific-v2 writer,
actual Anemone checkpoint codecs, `copy_artifact`, `sync_artifact`, bounded reader,
full reference reader, `StageMeasurement` and existing CPU/I/O/RSS/disk helpers.
It adds no parser, storage engine or profiling framework. Processes run serially;
1k completed before 5k, and 5k before 10k. File/directory fsync is included in the
reported scientific writes, checkpoint re-saves and component copies/links.
These are component costs, **not a benchmark of a new durable publisher**. Seal
creation, run-state commits and whole-capsule verification are not hidden inside
these timings. No OS page-cache eviction was attempted; fresh process does not
mean cold physical disk. I/O counters distinguish logical from physical reads.

Two input families prevent a misleading small-state comparison:

* Real generated search checkpoints, deterministic seed 4, no saved experiment.
  Search expands a whole frontier operation, so nominal 1k/5k/10k contain
  **1,003 / 5,013 / 10,018 nodes** and **1,114 / 5,931 / 11,846 branches**.
  A and B measurements restore the exact same newly generated checkpoint for
  each size. The checkpoint benchmark uses the actual runtime, not the scalar
  fake checkpoint used by older publication microbenchmarks.
* Exactly 1,000 / 5,000 / 10,000 bounded-degree resource fixtures, using the same
  legal deterministic boards for both choices. The second history family adds
  twenty legal replay moves before the population's depth range. Duplicate
  boards under different IDs and declared evaluation values are resource-test
  inputs, not claims of unique transpositions or solved game optima. They do not
  measure the resident state footprint of a real unique-state search.

“Complete re-encode” writes independent existing-v2 generation-1 exports of the
same quiescent graph three times. Their scientific fields, generation identity and
creation provenance match exactly. These are repeated snapshot cost samples, not
three sequential scientific generations. “Reuse” writes existing-v2 G1/G2/G3 with
K=0; current updates remain complete and actual generation provenance is retained.
Copies/links then operate on those exact immutable record bytes. No generation,
certainty or creation-origin field is normalized away.

An A byte-carry-forward estimate is the measured complete mutable write plus
measured record copy. A B estimate substitutes measured immutable links. These
are sums of separately timed operations, not measured end-to-end new formats.
Both still need current-descriptor construction and full integrity/scientific
validation. Packing records, changed identities and validation evidence must be
covered by the future versioned contract; the component timings are not proof
that those changes have already been implemented.

All times below are seconds; bytes use MiB (2²⁰). Steady times average the
second and third K=0 samples; no uncertainty interval is inferred from two samples.

| Real graph N/E | First complete export after lazy restore | Complete re-encode, steady | Complete mutable write, steady | Record copy / link, steady | State / mutable MiB | Write-process peak RSS, complete / reuse MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 / 1,114 | 2.311 | 0.824 | 0.108 | 0.025 / 0.010 | 0.198 / 0.784 | 294.7 / 294.4 |
| 5,013 / 5,931 | 21.293 | 11.618 | 0.457 | 0.024 / 0.010 | 1.833 / 3.950 | 383.6 / 383.0 |
| 10,018 / 11,846 | 47.551 | 26.071 | 0.865 | 0.025 / 0.011 | 4.032 / 7.895 | 502.3 / 502.0 |

For the real 10k graph, A’s measured component sum is **0.890 s** (0.865 s
current write + 0.025 s copy), versus roughly **0.876 s** with links. Neither sum
includes the common publication validator/commit. Full fresh canonical serialization
is 26.071 s after materialization and 47.551 s on the first restored export.

| Real graph N | Durable checkpoint re-save | Checkpoint MiB | Lazy restore | Checkpoint save peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| 1,003 | 0.204 | 0.082 | 0.125 | 295.6 |
| 5,013 | 0.515 | 0.403 | 0.478 | 352.2 |
| 10,018 | 0.894 | 0.805 | 0.906 | 422.9 |

Compressed checkpoint bytes are small here; **the checkpoint favors A through
recovery simplicity, not because it dominates scientific disk usage**. At 10k,
three complete scientific exports occupy 36.043 MiB versus 27.978 MiB for shared
records and three current update sets: 22.4% scientific-only saving. Including
three identical 0.805 MiB checkpoints gives approximately 38.46 versus 30.39 MiB,
a 21.0% saving before models/indexes. These are measured existing serializer bytes
for three retained epochs, not disk usage of an implemented rolling publisher.

| Exact N | Replay moves short / long, mean (max) | Re-encode short / long | Reused mutable write short / long | State MiB short / long | Three-export B saving short / long |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 7.99 (9) / 27.99 (29) | 1.470 / 5.554 | 0.105 / 0.106 | 0.228 / 0.610 | 13.4% / 26.8% |
| 5,000 | 10.36 (12) / 30.36 (32) | 12.134 / 33.618 | 0.419 / 0.426 | 1.366 / 3.273 | 15.5% / 28.0% |
| 10,000 | 11.36 (13) / 31.36 (33) | 27.358 / 65.332 | 0.836 / 0.843 | 2.921 / 6.736 | 16.3% / 28.5% |

The long 10k copy averages 0.027 s for 6.736 MiB. Its three-export A/B storage is
47.337 / 33.865 MiB. B’s **28.5%** saving is real, and could matter at much larger
S or many pinned roots. It does not demonstrate a necessary shared-object ownership
scheme for this workload. If S dominates U+C, the ideal saving approaches
(B−1)/B; this report does not extrapolate 10k timings to production.

For repeated exports, unchanged records require **zero canonical dumps/loads** in
the reuse/copy/link path. Complete re-encoding dumps N references every time. The
first real 5k export performed 1,372 anchor and 3,641 delta loads plus 5,013 canonical
dumps; later full re-encodes still performed 5,013 dumps. Synthetic states start
materialized, so zero `load_state_ref` calls in their writer do not mean zero
canonical processing. Existing full/reference and cold indexed reads decode N
canonical references; reused indexed reads do not.

| Actual N | Full reference load | Bounded cold open + scan + source recheck | Reused open + scan + recheck | Reference / cold / reuse peak RSS MiB | Index MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 1.378 | 1.850 | 0.042 | 288.0 / 284.7 / 283.9 | 1.293 |
| 5,013 | 20.551 | 22.206 | 0.206 | 311.2 / 286.5 / 285.1 | 7.402 |
| 10,018 | 49.466 | 51.848 | 0.416 | 340.8 / 287.4 / 285.7 | 15.188 |

The read benchmark uses the same unmanaged generated G3 source and unmodified
full/indexed readers; it excludes capsule/receipt-ledger overhead.
It measures current scientific verification/reconstruction, not a new A/B recovery
protocol. Each new generation still needs its own mutable index; shared records do
not make per-generation index construction disappear. Successful per-source indexes
also need an explicit disposable-artifact retention policy.

RSS is the actual process-lifetime high water mark, not a stage-local allocation
peak. Reference validation may deliberately materialize the full snapshot;
bounded-reader measurements run separately without that oracle in memory.
Checkpoint restore is lazy: its short initial time does not include materializing
all state handles. The first scientific export after restore pays that additional
cost. The supplied evidence separates canonical `load_state_ref`, anchor/delta
loads, canonical `dump_state_ref`, CPU, I/O and fsync counts.

## Why the checkpoint favors A, but cannot replace the scientific export

`runtime/runner.py::save_checkpoint` builds a complete typed checkpoint even for
sharded output. It already persists every current node, branch, direct/tree value,
value certainty, expansion/selector state and RNG state. Restore constructs the
current runtime from those checkpoint shards; it does not traverse earlier
scientific update files. Both designs therefore pay C_g each generation.

A complete checkpoint makes a closed current-generation recovery capsule natural:
checkpoint + current scientific snapshot + required model/config identities.
Removing historical scientific updates does not require inventing incremental
runtime reconstruction. However, the checkpoint is not the Dataset source:
selected target preference, canonical training references, creation provenance,
producer identity and the scientific export contract remain explicit snapshot
responsibilities. Recomputing a v1 flat export from checkpoint values would not
preserve the scientific-v2 contract and is not proposed.

The initial scientific extraction still needs every node's canonical reference.
Subsequent A snapshots can carry exact immutable record bytes from the verified
parent and encode only newly created states. This is distinct from modifying the
parent or trusting its hash without reading its bytes. B uses the same premise but
retains physical shared dependencies instead of making private copies.

## Resource complexity, including limits

Let N/E be current nodes/branches, K new nodes, G historical generations, H_i replay
history length, S = total canonical state-record bytes, U = complete current
mutable scientific bytes, C = full checkpoint bytes, M = required model bytes,
R = exact patch receipt count. With bounded non-state field sizes,
S = Θ(sum H_i + N), U = Θ(N+E); canonicalization cost V(N,H) is codec-dependent,
not assumed linear in replay length. Let B be the number of live roots (current,
rollback, readers, pending work, explicit retention), L the largest decoded row,
Q bounded parser/cache buffers, and F the current dependency-file count.

| Operation | A: complete private snapshot | B: shared records + current mutable snapshot |
| --- | --- | --- |
| First write | V(N,H) + serialize S+U+C; full validation | Same |
| Repeated write, immutable byte carry-forward | Read/hash/copy S; write U+C; V(K,H); traverse N/E | Read/hash S; write new-state bytes + U+C; V(K,H); traverse N/E and shared dependencies |
| Fresh integrity validation | Θ(S+U+C+M), plus F inventory operations; no historical updates | Same selected bytes, despite physical sharing |
| Fresh canonical validation without compatible committed evidence | V(N,H) in both; a parent digest alone does not remove it | Same |
| Cold indexed scientific open | Source byte scans and SQLite B-tree/index work, conservatively O((N+E) log(N+E)); canonical validation as required | Same |
| Reused indexed open | Source hashes plus cache integrity checks; not O(1) | Same |
| Current working RAM | Existing Growth graph and full checkpoint payload O(N+E+state storage); writer O(N+E); copy buffers bounded | Same, plus dependency descriptors; no new O(N) Python state cache justified |
| Selected consumer RAM | O(WL+Q+current metadata), with families keyed on disk; existing full run-state ledger adds O(R) | Same, plus shared-record dependency metadata |
| Retained heavyweight disk | Σ_live(S_i+U_i+C_i+M_i), less existing model hardlink sharing | union_live(S_i) + Σ_live(U_i+C_i+M_i), same model treatment |
| Fixed N, fixed B, no new external roots | O(B(S+U+C+M)); no dependence on elapsed G for heavy snapshots | O(S+B(U+C+M)) with reachable record blocks only |
| Linear graph growth N≈KG, fixed B | O(B·current graph bytes), necessarily grows with current N/H | Same order; lower duplicated-state constant |

For the existing writer, reciprocal adjacency membership additionally costs
O(sum d_in² + sum d_out²); these bounded-degree measurements do not establish
linear behavior on arbitrary DAGs. Full inventories currently materialize F
entries; selected-reader metadata has explicit size limits. Neither design alone
makes Growth/checkpoint RAM bounded independently of N.

The current layout instead retains S_G + Σ(U_g+D_g+C_g), and hard-links historical
scientific directories into every capsule, with Θ(G²) filenames/inventory entries.
Each D_g repeats creation keys and shard references. Selected verification removes
unselected capsule hashing, not this selected historical closure. A/B must remove
that closure, not merely unlink old top-level capsule directories.

**Residual history is intentional but not free.** Exact UUID receipts remain
Θ(R), and rewriting their current run-state list costs Θ(R) per commit, potentially
Θ(G²) cumulative metadata traffic. Lightweight metrics/provenance and stable pin
inodes also grow. Indefinitely held reader pins, pending work, scientific retention
and required old models can make B unbounded. No claim of constant total RAM/disk
for thousands of generations is justified without explicit capacity/backpressure
and separate receipt/model retention semantics.

## Scientific and checkpoint parity

Exact-byte checks passed for **240,170 paired complete scientific update rows**
across all nine population/history cases. The three complete exports' canonical
record files matched byte-for-byte, and G1 record bytes matched the reused source.
Snapshot metadata, producer and creation maps matched. G1/G2/G3 headers were checked
against their actual declared generations; the intentional epoch difference was
not stripped from provenance.

The full reference and indexed readers matched every node field, value contract
and provenance for all **16,034 real generated nodes**, in both cold/reuse oracle
checks. The separate bounded measurements never called the full reference reader.
The eight-generation fixture exercised real seal validation, receipt publication
and cold scalar recovery infrastructure; its tiny checkpoint is not used in the
runtime cost table.

The real runtime checkpoint retained identical metadata (including RNG), state
payloads, node runtime/evaluation payloads, selector state and expansion records.
However **7 / 70 / 161 node shells** changed their representative
`parent_node_id`/`branch_from_parent` pair at 1,003 / 5,013 / 10,018 nodes after the
first restore. All complete `linked_children` arrays remained exactly equal;
every replacement representative edge exists in the original graph. The three
subsequent quiescent saves of each restored graph were exactly identical across
all decompressed checkpoint components.

At 1,003 nodes, freshly exported scientific files before continuation and after
**each of three continuation steps** matched byte-for-byte between the continuous
and restored runtimes. RNG states, node counts and all compared selector report
fields matched. This proves the tested continuation cases, not arbitrary future
search equivalence; **full raw checkpoint round-trip parity is not established**.
See [exact differences](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/checkpoint-differences.json) and
[raw measurements](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/measurements.json).

The checkpoint discrepancy is not removed or normalized. Anemone
`checkpoints/build_state_payloads.py::_representative_parent_link` prefers an
existing delta state parent when available, otherwise the first runtime parent;
`node_restore.py::_link_nodes` rebuilds incoming links from every parent's
`linked_children`. Those orders can change the representative shell pair while
preserving the complete adjacency. The raw differing fields and replacement edge
membership checks are retained in evidence. A canonical representative convention
or explicit representation-equivalence contract needs separate review before
claiming **exact full checkpoint round-trip** parity; it is not a storage fix.

Invariants for both future layouts: preserve FORCED vs ESTIMATE, missing vs null,
actual direct/tree/effective values and their sources, selected target source and
certainty, all current parents/children, canonical references, node creation
origin, generation/producer/config/model identities and exact row order. Preserve
five supported Dataset-v2 policies and four proof gates, all nine legacy policies,
and the rule that uncertified family predictions cannot replace certified
same-state terminal targets. No new family proof rule is introduced here.

## Versioned publication and canonical integrity contract

Do not prune or reinterpret an existing scientific-v2 catalog in place. The current
publisher requires consecutive aligned history, a genesis-rooted publication list,
and complete capsule inventories. A sparse catalog accepted by an individual
reader would still violate that publication contract.

The smallest future change is a separately versioned **snapshot publication
layout**, with the same scientific-v2 value contract and the existing bounded
parser/SQLite implementation extended explicitly. Each generation names exactly
its current records and complete updates, a complete checkpoint, producer/config/
model identity, current inventory and committed validation evidence. No prior
update or prior generation descriptor is a read dependency. Preserve each node's
creation generation even when storing its bytes in a later snapshot; distinguish
record origin from snapshot publication generation. Source/cache identities must
include the new layout and existing scientific contracts.

For A, a private new directory owns the complete scientific byte set. Use bounded
streaming copies of verified immutable records and write complete current updates.
Avoid empty per-generation record files and cumulative catalog prefixes. If records
are repacked, copying exact validated record byte sequences must preserve all
scientific fields and creation origins; test the transformation and compute fresh
output hashes. Changed records or decoder/validator/producer identity require the
existing full canonical checks. Do not add a second persistent validation cache.

Canonical proof transfer is conditional on all of the following:

1. The source is a committed dependency selected under the existing read pin;
   freshly verify its actual bytes, exact expected identity and committed seal.
2. The same scientific/canonical contract and implementation identity apply.
   Evidence binds exact record contents and producer, not just an ancestor ID.
3. The trusted copying/packing operation preserves every record's bytes/fields;
   new or changed state references receive full canonical validation.
4. Freshly hash and seal output dependencies. Always validate current membership,
   all mutable values, order and reciprocal topology with the existing index.
5. Keep the pin through use and required output checks. Cooperative pins prevent
   compliant retirement, not external writes or bit rot; mismatches fail closed.

The same single Growth ownership lock, private attempt, fsync, no-replace immutable
publication and **run_state as the sole authoritative commit point** remain.
Discovery/status are rebuildable projections. No new transaction journal or
second authoritative catalog is needed. B would additionally need shared-record
reachability and pin-safe ownership for every live generation; no reference-count
DB should be added.

## Dry-run reachability and safe recovery

The [dry-run experiment](rolling_snapshot/reachability.py) uses eight newly
published tiny scientific generations and actual existing capsule locks. It does
not implement a collector or delete any committed artifact.

The current protocol retained **all eight committed generations**, with **zero
eligible committed deletion candidates**, seven exact receipt checkpoint references
and a required model identity. A real shared flock on G2 refused exclusive
nonblocking acquisition; acquisition succeeded after release. No timeout or stage
claim was used. The authoritative run-state bytes were unchanged by reporting.

A hypothetical future root example included G8 current, G7/G6 rollback, G2 reader,
G4 pending-source and G1 explicit scientific retention. Only the G2 lock was
actually exercised; G4/G1 are declared examples, not invented active workers.
Of 146,991 unique capsule file bytes, 28,148 were unique to the hypothetical omitted
capsules. That is **not deletion authorization**: current ancestry/receipt contracts
still protect them. The head itself retained 21 historical scientific files
(three per G1–G7), demonstrating why dropping outer directories alone does not
remove historical payloads. Results are in the `reachability` raw measurement entry.

The existing eight-generation, two-node capsule fixture restarted through the
actual receipt/publication API in **0.079 s** (283.0 MiB process peak), with identical
recovered run state, zero canonical state decodes, five file fsyncs and fourteen
directory fsyncs. This is a small **current-protocol recovery** measurement, not
an extrapolation to 10k or evidence that a future rolling recovery protocol passes.
The real 1k/5k/10k checkpoint and scientific-source costs are measured separately.

The future root calculation must start from a consistent authoritative commit and
include current/rollback recovery boundaries, actual readers, pending patches and
producer cursors, in-flight Dataset/Training/Reevaluation dependencies, selected
models/configs, required receipt/recovery evidence and explicitly retained
scientific artifacts. Reachability includes transitive files and shared inode
identity; nominal directory bytes are not reclaimable disk bytes. Unknown paths,
malformed identities and incomplete closure must produce **retain/error**, never
“unreachable”. Report reasons per root and bytes unique to candidates first.

There is a concrete distinction to review before removing old checkpoints:
`checkpoint_patch_receipts.validate_state` hashes the current checkpoint and binds
head receipts to it; duplicate acknowledgement verifies the current committed
state, exact patch bytes, cursor and original model. It does not reload an old
receipt's checkpoint. A new explicit retention contract could therefore preserve
the old checkpoint path/hash as exact receipt provenance without retaining its
bytes, **only after proving the committed head/receipt frontier incorporates that
receipt and no rollback, pending or scientific root requires the old checkpoint**.
This is not permission to weaken current v2 retention. Original model bytes are
still required by duplicate acknowledgement; retaining all such model epochs may
itself grow without bound. Neither old checkpoint identity nor an arbitrary UUID
receipt can simply be discarded on reaching a generation count.

A claim or timeout is not a reader-liveness oracle. For a later collector, use the
existing Growth lock and stable per-capsule lock, acquire exclusive nonblocking
ownership of a proposed retirement candidate, then reread roots and identities.
An active shared pin defers retirement. Commit removal from the *new version's*
active roots while still holding the exclusive lock, then atomically quarantine
with fsync and eventually remove only the proven retired object. A late reader
must acquire its pin and check current membership after acquiring it. Never unlink
and recreate a pin inode; a one-time dry-run lock observation is not a future
permission to delete. Crash testing must cover every ordering boundary. Shared
B records remain until no reachable generation or pending owner refers to them.

**Rollback is more than keeping an old checkpoint.** Current receipt validation
explicitly rejects older checkpoints paired with newer receipts. Receipts store
patch identities/hashes, not enough data to reconstruct discarded patch effects;
acknowledged patch payloads may already be gone. Reproducing the latest generation
from an older one also requires exact growth inputs/config/model/selector/RNG.
Never infer that replay capability from receipts alone.

Retain one or two *complete recovery boundaries*, including their matching
receipt frontier, producer cursor/model/config identity and necessary replay
inputs if replay is promised. A small immutable recovery descriptor can be a
committed capsule dependency, not a new authority or cumulative journal. Its
receipt frontier must bind the exact ordered subset, not a generation number
interpreted as an arbitrary-UUID deduplication watermark.

Until this contract is established, fail closed on a corrupt current head; keep
older boundaries for explicit reviewed recovery, without automatically rewinding
and reusing newer receipts. Intentional rollback that discards subsequent search
work is different from deterministic crash recovery and requires an explicit new
run/epoch and acceptance rules for old pending work and downstream outputs.
Do not silently promise lossless replay without retaining the required bytes.

## Small independent changes and code to remove

1. **Resolve/document checkpoint representative-parent round trips.** Keep the
   exact counterexample, selector/RNG continuation and full scientific comparisons.
   No serialization normalization in parity tests. This is independent of storage.
2. **Approve a closed-snapshot publication contract and tests.** Extend the existing
   reader/writer/publisher only for the explicit new layout. Implement A first,
   carrying validated records and writing complete current updates/checkpoints.
   Keep v1/v2 historical readers unchanged. Compare every scientific row, certainty,
   provenance, Dataset JSONL and reevaluation patch against the current oracle.
3. **Version active recovery roots in the existing run-state protocol.** Specify
   rollback receipt/cursor boundaries and pending/model dependencies. Keep a compact
   predecessor identity for lineage, but never treat it as content verification.
   Stop copying the whole ancestry list into the new authoritative active-root
   structure; retain lightweight scientific lineage separately from heavy read
   dependencies. Exact deduplication receipts remain authoritative and unchanged.
4. **Ship reachability reporting before collection.** Compare complete dependency
   closures, show reasons/inode bytes, conservatively retain unknowns and validate
   live-pin races. Add explicit disk-pressure backpressure; do not expire readers
   or scientific evidence merely to meet a quota.
5. **Only then implement retirement/recovery.** Prove exclusive-pin/commit/rename/
   fsync/restart ordering, interrupted copy/publication, missing/corrupt files,
   exact patch retry, model replacement and rollback/late-reader cases. Production
   remains opt-in until all those contracts pass.

For the new layout, `clone_scientific_catalog`, `catalog_paths` and the historical
prefix-validation loop in `validate_scientific_export` cease to be required on the
new hot path. Remove cumulative empty record shards, repeated historical creation
maps and prior update/descriptors from current inventories. The explicit legacy
reader remains necessary for historical artifacts; do not delete its interpretation
or mutate its data. Keep the shared seal/hash/fsync/pin primitives, SQLite reader,
Dataset workspace and scientific confidence helpers. Retire old version-specific
publication writing code only after no supported run depends on it.

B is a later physical-storage option if measured S, number of simultaneously live
roots or copy bandwidth exceed a concrete budget. It must demonstrate a benefit
large enough to justify shared-object retirement and integrity handling. Do not
add B's ownership machinery merely to save the small copy time seen here.

## Reproduction and validation evidence

**479 existing tests passed**, no failures/skips, in 326.52 s. The selected suite
covers scientific-v2 reference/indexed reconstruction, all legacy Dataset policies,
five enabled confidence-v2 policies/four gates, the real FORCED(35)/ESTIMATE(42)
regression, terminal protection, exact checkpoint selector/RNG continuation,
current DAG links, exact patch receipts, interrupted publication/crash/restart,
source corruption and reader-pin lifetimes. This is a relevant focused CI run;
no new storage implementation exists to run a new-format crash matrix against.
The full repository CI was not rerun for this documentation/measurement-only work.

The new research drivers pass Ruff 0.15.2 formatting and E4/E7/E9/F/I checks.
Production delta is **+0 / −0**: all **527 tracked source files (471 Python)** are
unchanged. All **30 pre-existing worktrees**, including dirty research and protected
PR/integration worktrees, retain their HEAD, status and tracked-diff fingerprints.
Pinned ancestry includes PRs #75/#76/#80/#82 and #78/#79/#81; exact SHAs, installed
distributions, validator implementation identity and driver hashes are in
[sources.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/sources.json). All 775 generated JSON,
JSONL and compressed-checkpoint files are fingerprinted in
[fixture-fingerprints.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/fixture-fingerprints.json).
Large fixtures and logs remain under the isolated task root; no existing evidence
was replaced.

Reproduce from this worktree with the pinned test interpreter recorded in
`sources.json` (Python 3.13, algorhino-anemone 0.2.26, Atomheart 0.2.11,
Valanga 0.2.5), selecting a **new** output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /home/pompote/oldata/victor/morpion-scratch/selected-generation-verification-20261021/ci-venv/bin/python \
  docs/research/rolling_snapshot/reproduce.py \
  /home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022/reproduction-001
```

The recipe stops on failure and runs every child serially. The environment is read
only; new runtime/cache/temp artifacts go to secondary-disk scratch. One unmeasured
read-only inspection fell back to a new Matplotlib directory under `/tmp`; that
specific temporary directory was removed and `MPLCONFIGDIR` is now explicit in
the reproduction recipe. No committed or historical artifact was deleted.

See [summary.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/summary.json),
[measurements.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/measurements.json),
[validation.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/validation.json) and
[protected-worktrees.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/rolling_snapshot/evidence/protected-worktrees.json).
The recommendation requires no new PR across the sibling implementation stacks:
this local design branch is directly based on `0790c596`. Implement the independent
contract/publication/root/retirement steps above only after their exact dependency
bases are reviewable. Nothing has been pushed, merged or activated.

All experiment failures are retained in scratch logs. The first harness call
omitted the existing measurement helper's empty `work` directory; that driver-only
error was corrected. The first checkpoint comparison deliberately failed on the
actual shell discrepancy; subsequent runs record it rather than asserting a false
pass. Neither required modifying production code or saved artifacts.
