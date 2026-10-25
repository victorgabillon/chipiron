# Record-scale Morpion storage: retain closed boundaries, remove measured overhead first

Review date: 2026-10-25. **Select A with targeted changes, not a new persistence
engine.** Preserve independently restorable closed generations and the existing
single run-state commit. First stream checkpoint construction and add an explicitly
versioned, losslessly compressed scientific encoding. Reuse identical existing
checkpoint files only if its incremental benefit warrants the code. Do not introduce
bounded delta replay or a shared object database now.

This differs from simply accepting the current cost. The current writer still
constructs a complete second checkpoint *payload*, and scientific JSONL is highly
redundant. In the measured 5k corpus, compressing the same complete scientific bytes
reduces checkpoint-plus-science allocation by **90.7%**, versus **3.9%** for sharing
byte-identical existing whole files. The format-3 1k activation pair saves **85.7%**
with compression. These are component experiments, **not implemented compressed
publication or reader timings**. They justify a small next implementation, not a
claim that arbitrary future trees compress at these ratios.

B remains the preferred replacement candidate if that smaller solution misses an
explicit capacity/write budget: complete immutable boundaries referencing shared
bounded files, with complete current mutable data and no generation replay chain.
C is not selected: change tracking, ordering changes, consolidation, recovery and
pin lifetimes add obligations without a demonstrated end-to-end advantage here.
Scientific/checkpoint consolidation is promising but lacks a complete bounded
view and provenance proof; it must not be substituted on the strength of scalar
or topology equality alone.

## Exact foundation and method

- Evidence base: `46cec0093f571ebbf9e9b4aa07bc3a8f94530177`.
- Unchanged FULL-tested executable: `fe0fe7a576f25dfdd6464060495cbd1356b03590`.
- Anemone: `90c7ef4454edf52459d7f7051b6c904ddf7f7a35`.
- Exact Anemone wheel SHA256:
  `5f3015c1a07d7a838a729896a388fad2b31f666f9abea7e82eb6f3381cb52890`.
- Branch: `research/morpion-record-scale-storage`.
- Research probes and measured evidence: `2f407b8c3500afed2e90e8fc3c2ff7f748bf0713`;
  the following report commit changes documentation/evidence only.
- New ext4 worktree:
  `/home/pompote/oldata/victor/morpion-scratch/record-storage-20261025/worktree`.

The two descendants of `fe0fe7a5` before this review contain evidence/documentation
only. All 477 installed Chipiron Python files match this source; all 289 Anemone
files match the installed package and exact pinned Git tree. See
[sources](record_scale_storage/evidence/sources.json). No source stack is reapplied.

The earlier [A/B study](morpion_rolling_snapshot_design.md), at
`f05d418c42b9292410702f51de2d976fbf693059`, explicitly favored A because copying
records was cheap, while complete current checkpoint/value writes remained in both
options. At real 10k it estimated 21.0% combined three-boundary saving with B;
long-history synthetic science alone saved 28.5%. We preserve those observations.
They used older checkpoint/dependency code; they are not new format-3 recovery
benchmarks. The new capacity objective warrants revisiting them, and motivates the
compression/file-sharing comparison below.

Inputs are the preceding task's fresh legal G1–G8 E2E workspace, the existing
format-3 1,003-node M1→M2 activation pair, and the earlier generated 5,013-node,
three-generation closed-snapshot corpus. No historical research search was loaded.
The 5k checkpoint is format 2: the current strict loader correctly refused it
([log](record_scale_storage/evidence/inventory-5k.log)). It is used only as a
fingerprinted byte corpus; its earlier scientific/continuation results retain their
original implementation identity. No compatibility bypass or migration was added.
The previous 10k/long-history results supply sufficient additional context; disk
pressure and lack of a new question did not justify generating another 10k tree.

The [inventory/component probe](record_scale_storage/measure.py) reuses existing
capsule validation, checkpoint shard iterators, scientific parser, zstandard,
`sync_artifact`, `StageMeasurement` and process counters. The inventory deliberately
uses full dictionaries for small offline comparisons; its RSS is not a bounded
production measurement. Copy/compression/scans use bounded 256 KiB buffers plus
zstd/hash buffers. There is no new production parser, index, authority or cache.
Every linked inode belongs to **new private output**; no input inode is linked,
modified or deleted. No prototype directory is a valid committed capsule.

## Current storage inventory and field-level overlap

The concrete layout is publication v3, closed scientific layout 1/value contract
v2, and Anemone checkpoint format 3 in split compressed shards. The same generation
contains both complete checkpoint and complete scientific files. Models already
use verified hardlinks where the existing publisher can reuse a committed pin.

| Fact | Checkpoint | Scientific snapshot | Unique information / mutation pattern |
| --- | --- | --- | --- |
| State and replay history | `state_payloads`: anchor reference or parent-relative delta; state summary | `records.jsonl`: canonical reference, node ID, creation generation | Same state, different access encoding. Canonical replay is not necessarily the representative runtime parent path. Existing states are stable; serialization choices require their own identity check. |
| Canonical identity | State payload/resolver and summary/tag; domain codec | Full canonical reference with pinned codec contract | Node ID and summary tag are not substitutes for canonical content identity. A tag alone is not a collision-proof state certificate. |
| Node identity/depth | Node shells, repeated ID in state/runtime rows | Record/update IDs, depth, contiguous export order | IDs are stable. Export order can shift for old nodes when Growth inserts at a depth. |
| Topology | Action-labelled ordered `linked_children`; representative parent/branch; explicit multi-parent order | Current `parent_ids`/`child_ids`, without edge actions | Checkpoint contains stronger continuation ordering. Current Anemone scientific builder sorts string IDs; that is an explicit projection, not equality of wire order. Incoming parents and old-node outgoing links can change. |
| Unopened actions/opening | Shell generated flag; runtime unopened actions and branch frontier | `all_branches_generated` only | Action lists/frontier are recovery-unique; local Growth changes them. Existing legal moves are stable for a state, opening status is not. |
| Direct evaluation | Complete serialized Value and `direct_evaluation_version` | Scalar plus complete direct Value/certainty | Same Value is duplicated. Per-node evaluator version is additional checkpoint provenance; do not discard it. |
| Backed-up tree evaluation | Complete backed-up Value | Tree/backed-up scalar aliases and complete tree Value/certainty | Duplicate scientific fact; local ancestors or most nodes may change. |
| Effective evaluation | Derived by pinned canonical evaluator from direct/tree, opening completeness and objective | Explicit effective Value, scalar, certainty and source | Derivable only under that exact rule, not v1 tree-first fallback. Partial opening compares values; full opening uses present tree value; ties use the pinned rule. |
| Selected raw target | Not a checkpoint policy | Explicit target source, selected Value/scalar/certainty | Selection policy is distinct from effective value. Preserve missing/null; never infer selected exactness from node exactness. |
| Terminal/exact proof | Serialized certainty, over-event and optional line on Values; actual state; backup data | Node flags, independent Value certainties, terminal label, scientific contract | A FORCED label is not a newly invented proof. Existing supported proof validation still applies; no cross-state transfer follows from a PV family. |
| PV and ordering caches | PV sequence/version, child-version cache, decision keys, backup best/second branch and exact-child count | No complete runtime PV cache | Dataset families are recomputed from raw targets and deterministic child selection. They are not interchangeable with the stored runtime PV. |
| Selector | Depth stats, per-node classifications, candidate queues/priorities/versions, selected node, selection step counter | Absent | Required for exact continuation, not disposable merely because some priorities are recomputable. Activation refresh must preserve defined continuation state. |
| RNG / last expansions | Search RNG, rollout RNG, evaluator version, latest expansions | Absent | Recovery-unique. Metadata shard is version-specific even with unchanged graph. |
| Creation provenance | No original scientific creation generation per node | Immutable record creation generation | Cannot reconstruct after deleting science from checkpoint alone. Keep it in any consolidated format. |
| Export metadata/producer | Runtime metadata does not replace this | Descriptor timestamp, root/count, target preference, metadata, codec/API/package/config/model identities | Explicit scientific contract; custom visit counts/metadata cannot be assumed available from generic checkpoint payloads. |
| Final Dataset target | Not present | Raw scientific inputs only | Accepted `rows.jsonl` adds family representative/prediction blending, final target certainty/provenance, filtering and counters. It is a derived artifact with its own accepted identity. |

The [small current inventory](record_scale_storage/evidence/inventory-opening.json)
and [1k inventory](record_scale_storage/evidence/inventory-1k.json) compare complete
serialized direct/tree Values, including certainty/line/over-event, for every node.
They also assert the exact documented child-ID projection and depth. They do **not**
claim that matching those fields proves a new scientific view. Full field examples,
all file hashes, actual allocation and per-field change counts are retained.

Other independently materialized information:

- **Model bundles:** weights `param.pt`, arguments/parameter/configuration and model
  manifest; accepted Training retains all candidate outputs and the selected
  candidate's exact bundle digest. Frozen capsule pins bind logical origin to
  physical bytes. Identical hardlink names are not separate physical copies.
- **Capsule and discovery descriptors:** generation/parent identity, inventory,
  checkpoint/scientific hashes, canonical-validation evidence, active evaluator,
  activation request and provenance. Small identities deliberately repeat binding
  information; they are not another copy of node Values. Discovery is advisory.
- **Run state and receipts:** cumulative publication entries, exact UUID/content
  receipt records, accepted cursor, epoch, accepted Dataset/Training bindings and
  activation completion identity. Exact receipt identity is not a generation
  watermark. Required model/evidence dependencies survive payload retirement.
- **Reader/work files:** SQLite stores complete record/update JSON payloads plus
  order index, and TEMP topology/family tables. This is substantial *derived disk
  duplication*, not a second scientific authority. The existing 5k reader cache
  alone occupies 7,761,920 logical bytes, larger than three compressed candidate
  checkpoint/science boundaries. Successful cache lifetimes need an explicit
  budget alongside roots. No cache removal is performed here.
- **Publication scratch:** source/output validation indexes, journals, full private
  checkpoint/science, temporary authority, immutable final orphan before commit.
  Checkpoint serialization additionally builds a full typed payload in RAM.
  Training can create tensor/feature caches and per-row schedules; optional cached
  training is not bounded by the Dataset reader's memory guarantees.

At current E2E G8 the checkpoint has seven files: 1,730-byte manifest, 7,002-byte
metadata, 6,350-byte states, 6,093-byte shells, 18,726-byte runtime, 2,115-byte
selector and 997-byte latest expansions. Science has three files. All current plus
two capsule boundaries occupy **1,093,632 allocated bytes / 47 unique inodes**;
the entire workspace is **3,366,912 bytes / 207 inodes**. The component comparisons
below intentionally exclude models/authority/accepted results, which remain required.

## Stable facts versus actual write amplification

| Workload | Observed change | Existing full writes and implication |
| --- | --- | --- |
| E2E Growth G1→G2 | +71 nodes; zero existing state changes; two old shells, seven old runtime rows; 24 old scientific order indexes change | 270,336 allocated checkpoint/science bytes. New canonical rows total 11,948 raw JSON bytes; changed/new science updates 92,959. Existing values/topology cannot all be called immutable. |
| E2E Growth G4→G5 | +14 nodes; four old shells/six old runtime rows; 132 old export-order indexes change | 286,720 bytes. A positional delta format can rewrite many rows despite local semantic work. |
| Fresh local Reevaluation | 50-node legal graph, one estimated direct value +1; two ancestors recomputed | Three runtime rows change (9,449 raw JSON bytes), three scientific rows (2,916); no state/shell/topology changes. Whole checkpoint/science still allocates 86,016 bytes. |
| Actual 1k M1→M2 | 1,002/1,003 runtime rows change; all canonical records, state payloads and shells unchanged | New checkpoint/science 1,150,976 bytes. Complete compressed science candidate brings that to 159,744. Old model-specific Values must remain for a complete rollback. |
| E2E exact receipt G2→G3 / G5→G6 | No scientific or checkpoint node changes | Whole checkpoint files are identical, but scientific generation headers differ. These are exact acceptance examples, **not mutating Reevaluation benchmarks**. |

The [local patch probe](record_scale_storage/local_patch.py) uses the existing
runtime apply/backup API on a newly generated search. Both independent scientific
exports use G1 intentionally: they are offline before/after content samples, not
fabricated sequential committed generations. Checkpoint writes took 0.195/0.200 s;
complete canonical scientific writes 0.261/0.265 s; process peak 294.77 MiB.
The first harness invocation incorrectly called a property and stopped before the
second save; its log/output remain. The corrected run is separate evidence.

Incremental row-only compression lower bounds are attractive but incomplete. For
the 1k activation, changed runtime rows compress to 38,767 bytes and changed science
rows to 34,294, almost the entire 33,974-byte compressed scientific update file.
That estimate excludes base, order/selector/RNG/provenance, manifests, indexing and
recovery. It is **not** a usable checkpoint delta. Empty change streams still have
frame overhead; no cumulative empty segments should be retained.

## Measured candidate comparison

All new measurements are serial fresh processes on the same ext4 device, after
focused tests finished. OS page caches were not evicted. The unchanged input
hashes and inode/link counts are recorded. Allocated file bytes use
`(st_dev, st_ino)` and `st_blocks*512`; directory/journal/filesystem metadata is
excluded. Each table row compares identical underlying states/bytes.

**A** copies checkpoint and complete scientific files. **B-file** shares identical
whole files by verified digest, including existing checkpoint shard families.
**B-body** separates the two scientific headers from their exact bodies and shares
identical bodies; no scientific field is discarded. It still copies checkpoint
files and is not combined with B-file. **A-compressed** keeps complete checkpoints
and losslessly streams science into zstd using the library already used by
checkpoints. It hashes decompressed bytes against the original exact bytes.
Header/body splitting and compression are component outputs, not valid new formats.

| Input / retained boundaries | A allocated bytes | B-file | B-body | A-compressed |
| --- | ---: | ---: | ---: | ---: |
| Actual E2E current G8 + G7/G6 | 950,272 | 929,792 (−2.2%) | 942,080 (−0.9%) | 229,376 (−75.9%) |
| Actual format-3 1k activation, **two** boundaries | 2,318,336 | 2,265,088 (−2.3%) | 2,117,632 (−8.7%) | 331,776 (−85.7%) |
| Earlier 5k byte corpus, three unchanged boundaries | 19,546,112 | 18,792,448 (−3.9%) | 7,409,664 (−62.1%) | 1,818,624 (−90.7%) |

B-body's large 5k saving shares unchanged mutable bodies too; it is not the saving
of state-only sharing during activation. A new header is still retained for each
boundary. The tiny E2E file-block/header overhead consumes most body-sharing
savings; this is not a prediction of its asymptote. B-file's first-time copies are
private; hardlink allocations are counted once across the full retained union.

| Input | Median A copy/hash/fsync s | B-file s | B-body s | A-compressed encode/check/fsync s | Component process peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2E eight boundaries | 0.089 | 0.080 | 0.102 | 0.089 | 294.07 |
| 1k two boundaries | 0.095 | 0.086 | 0.102 | 0.090 | 295.21 |
| 5k three boundaries | 0.099 | 0.075 | 0.105 | 0.097 | 298.01 |

These are fsync-heavy single component sequences, not a speedup claim for Growth.
Full indexes, canonical validation, checkpoint construction and authority commit
are excluded. No state decode occurs in these byte operations. Compression adds
small CPU here; long, high-entropy Values/metadata may change both ratio and cost.
The fresh-process bounded scan of the matched 5k scientific bytes takes median
**2.819 ms raw versus 3.362 ms decompress+hash**, five samples each, with identical
hashes and 294.77 MiB lifetime peak. This is not a complete Dataset extraction.

The retained prototype variants occupy 6,656,000 / 7,032,832 / 47,566,848 allocated
bytes for E2E/1k/5k, including every variant's payload files, before result JSON.
All completed variants are preserved. This aggregate is not a production steady-state cost.
A body-sharing attempt can transiently hold one extra full body before replacing
its private duplicate with a link; a conservative component peak bound is retained
allocation plus the largest body. No exact sampled transient maximum is claimed.
Existing actual publication staging and index peaks are reported separately below.

At 1k M2, scientific states compress from 210,848 to 9,602 bytes and updates from
827,094 to 33,974. At 5k G3, states compress from 1,921,738 to 54,510 and updates
from 4,142,287 to 110,225. These ratios reveal repeated fields/replay prefixes as
well as unchanged facts. Compression does not deduplicate physical bytes between
generations; it makes the remaining potential sharing benefit much smaller.
Sharing only canonical compressed states among three identical 5k boundaries could
save at most roughly 109 kB of logical data before descriptor/block overhead;
it would not remove the 7.76 MB derived index. Adding an object manager for that
saving is not justified by this corpus.

### Existing end-to-end costs, not interchangeable with components

Reused [durable closed measurements](morpion_durable_closed_snapshot_publication.md):

| Actual N | Initial complete publication s | Repeated G3 s / checkpoint s | Cold bounded open+visit / reuse s | Full reference load s | Dataset five-policy extraction s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 3.574 | 1.907 / 0.163 | 1.206 / 0.055 | 0.881 | 0.132–0.143 |
| 5,013 | 39.449 | 5.837 / 0.585 | 15.191 / 0.229 | 13.821 | 0.695–0.722 |

These older legal searches have only 64/374 target-bearing rows; their node counts
must not be used as emitted-row throughput. Reader peaks were approximately
290–291 MiB, versus 303/334.5 MiB repeated publication. Matching committed evidence
allowed zero unchanged scientific decodes during repeated publication; cold readers
still validated N references. At 5k G3, 890 file-digest calls hashed 201.19 MiB;
logical reads were 306.53 MiB and physical writes 51.01 MiB. The profiled 1k
publisher built three disposable indexes. File sharing/compression alone removes
none of those semantic checks or redundant index builds.

The current-format [activation evidence](morpion_atomic_model_activation.md)
measured 1,003-node M1→M2 at 14.875 s / 307.75 MiB, cold recovery plus certificate
preflight at 1.221 s / 303.40 MiB. Sampled staging peak was 3,928,064 allocated bytes;
sampling is a lower bound. Feature preparation dominated, not persistence.
Complete checkpoint write was 0.332 s and closed scientific write 1.389 s in the
separate instrumented run. Do not subtract its nested phase times from the
unprofiled total. This review implements no inference optimization.

The earlier A/B study measured lazy checkpoint restore at 0.125/0.478 s for real
1k/5k, but first complete scientific reconstruction after restore at 2.311/21.293 s.
Reused canonical bytes avoided that reconstruction. Removing the scientific state
records and deriving all canonical references from checkpoint deltas would incur
that class of replay work; no cheap bounded substitute is established here.

**Unmeasured:** full compressed publication, direct compressed bounded-reader
Dataset latency, B segment-aware recovery, C replay/consolidation and a checkpoint-
only scientific consumer. Their latency/RSS cannot honestly be filled with the
component times. They require separate implementation gates; none is advertised
as production ready.

## Can one persisted representation serve both consumers?

**In principle yes; the current checkpoint alone cannot.** For the pinned default-max
contract, most scientific mutable fields are projections of checkpoint direct/tree
Values, opening state, graph and target preference. Use the existing canonical
Value functions, including the comparator and effective source. Never invent the
lossy v1 reconstruction rules. A future single generation representation must add
original creation generations, canonical reference contract/content, explicit
scientific export policy/metadata and required identities; it must retain every
checkpoint-only field listed above.

Two bounded views could share those records: a continuation reader constructs the
one live graph; a scientific reader uses the existing SQLite machinery to index
current membership/order/incoming links and expose selected node windows. A
checkpoint contains outgoing action-labelled links, so deriving scientific incoming
links requires a disk-backed keyed pass, not an O(E) Python parent map. Reproduce
scientific sorting separately from runtime ordering. Preserve Dataset's own family
selection algorithm rather than replacing it with the checkpoint PV cache.

What consolidation could delete: the permanently stored scientific mutable row
file, duplicate direct/tree payloads and scalar aliases, and duplicate topology
projection. What it **cannot simply delete**: compact recovery state encoding,
canonical scientific identity, creation provenance, per-node model version,
selector/RNG, accepted rows/evidence or authoritative receipts. Eliminating the
canonical replay file as well requires a measured bounded checkpoint-to-canonical
transform. Keeping canonical records as a derived index is acceptable only with
explicit rebuild cost/lifetime; calling that index canonical storage would hide
rather than remove duplication.

At measured 1k, the *entire compressed* update file is only 33,974 bytes versus
827,094 raw; at 5k it is 110,225 versus 4,142,287. That is the upper bound removable
from these compressed examples before adding missing schema/view data. The current
uncompressed index and full checkpoint payload are larger demonstrated targets.
A unified format might eventually provide larger benefits for high-entropy,
proof-rich graphs, but these measurements do not yet justify replacing two mature
scientific/continuation interpretations with a new engine.

## Resource complexity and candidate contracts

Let N/E be nodes/branches, K new nodes, G generations, H_i canonical replay lengths,
S=Θ(N+ΣH_i) canonical bytes, U scientific mutable bytes, C checkpoint bytes, M unique
required models, R exact receipts, B live logical boundaries and L largest row/state.
Let Q bound parser/SQLite/compression windows, W selected nodes, D changed node
and ancestor records, P bytes per proposed immutable chunk, and d maximum delta
chain length. Actual U/C include optional proof lines, PV sequences, unopened
moves, selector queues and user metadata; they need not be Θ(N+E) when those grow.
For bounded field sizes they are O(N+E). V(N,H) denotes actual canonical codec cost;
linear time in H is not assumed. Hats below denote compressed byte sizes.

| Design | New bytes / time per generation | Retained physical disk | Auxiliary RAM / recovery |
| --- | --- | --- | --- |
| A current | Writes S+U+C; traverses graph; V(K,H) with valid committed proof reuse; hashes current selected dependencies and builds indexes | Σ retained(S_g+U_g+C_g)+M, plus authority, derived artifacts and scratch | Existing full checkpoint payload O(N+E+proof/selector data); scientific reader O(WL+Q+authority). Recovery uses one boundary, no generation replay. |
| A targeted | Writes Ŝ+Û+Ĉ; same semantic traversal/validation, compression CPU; future streaming checkpoint bounds serialization buffers | Σ retained(Ŝ_g+Û_g+Ĉ_g)+M; fixed B gives O(B·current bytes), not a bound on N | Streaming encoder O(Q+L) beyond resident graph and explicit manifest/selector structures only after those are also streamed. Same recovery semantics. |
| B complete version maps | Share verified unchanged blocks; write new/changed blocks, complete current evaluations/selector as needed. Without dirty identity proof still scans/hashes all selected data | Union of live immutable state/structural blocks + versioned mutable blocks and per-boundary maps + M | No second graph needed; O(Q+L+block-map metadata). Recovery reads one complete map, zero generation replay; shared dependencies must remain pinned. |
| C bounded deltas | Encode K and D changes plus metadata, occasionally consolidate whole base. If activation changes Θ(N) estimates then D=Θ(N); old order indexes may also force Θ(N) changes | Live bases + ≤d deltas per base, old bases needed by other roots, and consolidation scratch | Replay ≤d merges; O(Q+L) only with sorted merge/disk keys. Otherwise an O(N) map sneaks in. Worst restore reads base + all d deltas; scratch includes old base and new complete base. |

B can start with existing files, no global CAS or reference-count DB. For finer
sharing, stable node-ID pages avoid depth-order insertion shifting every later
chunk. Keep export order explicit. New Growth then rewrites at most affected pages
plus appended state pages; coarse pages trade rewrite amplification for fewer files.
Model activation changes evaluation pages broadly but never requires rewriting
unchanged canonical state pages. Current record headers include generation, so
hardlinking them as-is cannot exploit identical bodies. A new explicit layout is
needed. Logical boundaries can all reference physical immutable files, with each
capsule's own local hardlinks making lifetime ownership visible.

B's bound is a **complete map**, not an ever-growing list of historical deltas or
empty generation files. Map size O(number of live blocks) still grows with N and
can hit current manifest limits; do not claim bounded metadata merely by renaming
it a manifest. Chunking/page manifests and bounded decoder windows need a budget.
Prefer coarse file sharing before introducing more levels or an object catalog.

C must retain complete per-boundary selector/RNG and producer cursor/receipt
bindings. A hard depth cap requires deterministic consolidation under the same
single commit, including failure scratch capacity. For two rollbacks and pinned
older readers, several different bases may remain live. An activation-wide delta
approaches a full mutable rewrite anyway; receipt UUIDs cannot stand in for the
missing patch data. No measured C recovery or scientific-view proof exists.

Useful capacity examples are formulas, not extrapolated benchmarks. If unchanged
compressed state bytes are S_hat and mutable/recovery bytes U_hat+C_hat, sharing
only state across B=3 saves at most **2·S_hat**, fraction
`2·S_hat / (3·(S_hat+U_hat+C_hat))`. If state dominates, the ideal limit is 2/3;
if state is 5%, it is only 3.3%. Larger H, longer proof lines, many pinned readers
and models change the balance. Measure those distributions before selecting a
chunking implementation. Physical model redundancy and external backup budgets
are separate from this formula.

### RAM priorities at record scale

`runtime/runner.py::save_checkpoint` calls
`anemone.checkpoints.build_search_checkpoint_payload` before sharded writing.
`_build_tree_payload` creates the ordered node list and every typed node payload;
`_CheckpointBuildContext` retains whole-build atom/Value caches; split writing then
makes a tuple of nodes and buffers shards. This can exhaust RAM even though there
is only one *live* search graph. It violates the desired publication auxiliary
memory target and should be addressed before adding persistence machinery.

Closed export also obtains `_all_nodes_in_tree_order()`, an O(N) reference list,
although Anemone already provides `_iter_nodes_by_depth()`. Direct sharded restore
retains compact O(N) shells, state payload store and node lookup maps while building
the resident graph; selector JSON is loaded whole. The 256-state resolver cache
and 64-node activation batches do not bound all of these structures. Scientific
SQLite uses explicit 2 MiB page-cache settings, TEMP on disk and bounded row
reconstruction; a very high-degree single row remains large. Full reference
readers are offline oracles only.

The smallest RAM work is to stream existing typed-node construction into existing
shard writers, bound build caches per window, iterate resident order without a
second list, and separately account for selector serialization/restore lifetimes.
Preserve exact atom/value serialization, adjacency order and RNG. Do not label the
whole checkpoint writer bounded until selector and global helper structures are
also handled. This requires no state eviction or second graph implementation.

## Historical growth is still separate

Closed layout has removed old scientific update/catalog closure. Compact discovery
projections and selected verification have removed the former quadratic discovery
copies and all-history payload poll. The old G64 **55 MB / 5-second** result is a
superseded implementation, not today's hot-path measurement.

Existing closed fixed-five-node evidence at G8/G16/G32/G64 retains current science
at 6,414/6,419/6,419/6,419 bytes, while run state grows
32,135/58,860/112,298/219,183 bytes. Selected discovery hashes roughly 47.6 kB
at each point, but logical reads grow 160,284→796,047 bytes. The separate
[recovery-root study](morpion_minimal_rolling_recovery_contract.md) has three roots
at 584/587/587/587 bytes, yet exact receipts grow
4,502/9,371/19,115/38,603 bytes. These are different recorded fixtures; do not add
their totals or pretend the projection has replaced current authority.

Still present at this base:

- Publication/acceptance and exact receipts consume O(G+R) current authority RAM
  and bytes. Rewriting it each generation costs Σ(g+R_g), Θ(G²) when R_g∝g.
  Exact IDs/digests remain necessary; this task proposes no unsafe compaction.
- Retaining every complete tree costs Σ(C_g+S_g+U_g). With linear node growth this
  can be Θ(KG²), independent of the eliminated descriptor duplication. With fixed
  N it is linear in G. No committed artifact deletion is enabled.
- Discovery scans/sorts candidate metadata; model-origin lookup scans descriptors;
  freeze-model validates the matching capsule. Full audit examines all retained
  payloads. The advisory reachability tool retains per-file/inode data and its
  model-holder scans can be O(RG). Compression does not change these algorithms.
- Exact receipt ACK now authenticates old descriptors/model bytes without old
  checkpoint/science; current head validation remains complete. Accepted Dataset,
  Training, pending cursors, reader pins and explicit retention still add roots.
- Metrics/status/history, accepted rows/models and successful indexes remain
  cumulative unless a separately proved lifetime policy releases them.

## Recovery, corruption and scientific invariants

The selected design keeps one exclusive Growth owner, private staging, exact
inventory hashes, complete output verification/fsync, no-replace immutable rename,
then **one authoritative run-state replacement plus directory fsync**. A precommit
orphan is not a generation. After commit, discovery repair is advisory. Current
head corruption fails closed; do not silently load an old checkpoint with newer
receipts or M2 provenance.

Two rollback boundaries are logical complete states, including evaluator/model,
checkpoint, selected scientific contract, selector/search RNG/rollout RNG, exact
receipt frontier and accepted producer cursor. Keeping their files is not an
implementation of intentional epoch rollback. Pending M1 proposals must retain
exact admission/acknowledgement/supersession semantics across M2. The existing
receipts and accepted result records remain the authority for those transitions.

The component corruption experiment confirms that a validly recompressed but
changed payload still needs the expected content hash; truncated input is rejected
by decode/error or full-stream hash mismatch. Corrupting one **new disposable**
hardlink changes both linked names. Sharing therefore preserves process-crash
rollback but reduces independent protection against device corruption. Neither
three copies on one ext4 disk nor shared chunks establish hardware fault tolerance.
A parent digest, pin, stat tuple or successful old index cannot replace reading and
checking actual required bytes. See [scan/corruption evidence](record_scale_storage/evidence/scan-5k.json).

For compression, version encoding and cache identity explicitly; bind compressed
bytes, decoded byte count/hash, row count and contract in the existing descriptor.
Bound decoder window/decoded lengths, require complete frame/tail parsing, reject
unsupported encodings and ambiguous headers, and preserve strict duplicate-key/
canonical/value checks. No authority should point at a partly compressed stream.
The prototype does not implement that contract or claim its crash matrix passes.

For B, link only immutable committed bytes verified under the existing source pin,
retain each logical boundary's complete map and validate the selected physical
closure. No shared file may be opened for mutation after publication. Future
retirement needs existing stable pins, fenced admission, pending/accepted result
closure and inode-aware accounting; age, timeout or claim expiry cannot release a
reader. Unknown ownership fails closed. This task performs no committed deletion.

Scientific requirements remain literal: FORCED(35) is exact for its state;
ESTIMATE(42) stays estimated; selected-target certainty follows its actual source;
certified terminal 35 cannot be replaced by heuristic family 42. Preserve all nine
legacy outputs, the five selected-value-v2 policies and four exact-family gates.
Preserve direct/tree/effective/target Values and provenance, missing/nulls, current
DAG links, canonical identities, source/model/creation identities, complete Dataset
rows/metadata/order and Reevaluation patches/cursor wrap. No new proof rule is
introduced by compression, deduplication or a derived scientific view.

The older format-2 representative-parent discrepancy (7/70/161 nodes at
1k/5k/10k) stays visible in its original evidence. Current Anemone format 3 persists
representative and multi-parent ordering separately from the state delta parent.
The unchanged 1k strict witness now passes in both layouts: full node payloads,
order, both RNGs and all scientific files match. This corrects the *current status*
of the old report; it does not normalize old differences or prove every possible
future graph. A replacement must keep this strict witness and transposition cases.

## Small independent implementation sequence and removals

1. **Bound checkpoint auxiliary memory**, in Anemone first, then the Chipiron
   adapter. Reuse existing per-node encoders/shard primitives and frozen ordering;
   remove the complete tree payload/list/tuple path from current sharded saving.
   Account for selector and restore metadata separately. Gate on exact decoded
   checkpoint, format-3 order/RNG and actual next-step continuation. This changes
   the implementation, not scientific policies or automatic model activation.
2. **Version compressed closed science** in the existing writer/reader/publisher.
   One stream-opening/encoding helper is enough; use the existing SQLite index,
   strict parser and content guards. Avoid decompress-to-full-temporary-JSONL in
   production: that would reintroduce peak disk duplication. Validate five v2/nine
   legacy policies, four gates, exact JSONL/patches, malformed/truncated/bad-digest
   sources and all existing publication death boundaries. Measure one frozen
   executable on 1k/5k, high-entropy Values and long states before adoption.
3. **Remove duplicate same-operation work** only with an explicit validation
   lifetime: transfer already verified results inside the existing owner/pin,
   remove redundant full index builds or hashes where their boundary is genuinely
   identical. Preserve fresh input/output/commit checks. No persistent proof cache.
   Existing profiling, rather than speculation about SHA cost, chooses the call sites.
4. **Then revisit physical sharing** against the measured capacity budget. Start
   with exact existing checkpoint files; isolate generation framing only if savings
   after compression justify it. If coarse blocks are necessary, select B with a
   complete version map and no replay chain. Require an explicit net line/byte/CPU
   budget before approving chunking. C requires separate evidence and review.
5. **Keep retention/authority work separate.** Review exact active roots and
   accepted/pending evidence lifetimes before any collection. No format change
   releases an active worker or exact receipt dependency by itself.

Planning estimates, not measured implementation deltas: checkpoint streaming is
roughly 100–250 changed production lines plus selector-specific work discovered
by its proof audit; compressed I/O roughly 80–180 changed lines across existing
modules plus tests; whole-file reuse roughly 30–80 changed lines if it fits current
ownership; a new shared-block format or C likely costs several hundred lines and
additional reader/recovery branches. Do not treat these estimates as permission
to grow the stack or as promised net reductions. Actual prototype production delta
is **+0/−0**. Research probes are 557 lines before final formatting/evidence edits;
exact final counts are recorded with validation.

Concrete eventual removal candidates:

- Anemone `_build_tree_payload` materialization from the current sharded save path,
  whole-build serialization caches and split writer tuple-copy; preserve any
  remaining callers until the one-time transition is verified.
- Chipiron closed export's ordered node list in favor of the existing iterator.
- After an explicit one-time migration and proof of no active old readers, legacy
  publication v1/v2 branches in `generation_publication.py` and
  `generation_publication_io.py`; `clone_v1_catalog`, `clone_scientific_catalog`,
  cumulative catalog/creation-index writing and historical-prefix validation.
- Legacy `sharded_training_export.py` writer/loader and the old-layout branches of
  `sharded_training_reader.py`, once historical access is moved to the explicit
  migration/archive tool. Shared scalar, value, parser and full-oracle helpers in
  `sharded_training_v2.py` must first be retained or relocated, not blindly deleted.
- If consolidation is eventually proved, remove persistent `updates.jsonl` and its
  duplicate node serialization; retain the **single** indexed reconstruction and
  confidence/family policy implementation. Do not add a third engine alongside it.

The nine major current publication/reader modules total 6,042 lines, but this is
**not** a removable-line estimate: much is shared integrity/scientific logic. Naming
whole modules without distinguishing their shared helpers would overstate savings.

## Minimal migration and future partial reconstruction

Backward compatibility need not be permanent. For a future adopted encoding,
freeze/admit under current ownership, retain source pins, stream to a new private
versioned boundary, independently compare all scientific and checkpoint fields,
model/configuration, selector/RNG and continuation, exact receipts/cursor, accepted
results and pending-worker dependencies. Durably publish through the existing
run-state transition only after validation. Retain the only valid source until
then. Either wait for old workers to finish or retain their exact old dependency
closures; changing a format marker is not a worker lifetime fence. No migration or
retirement is implemented here.

A later on-demand graph needs stable IDs, bounded state chains/anchors, independently
decodable compressed blocks, explicit export/runtime order and keyed access to
current Values/topology. Existing checkpoint state handles and bounded scientific
SQLite windows already provide useful boundaries. Streaming whole scientific files
remains possible without making that cache authoritative. Do not require arbitrary
historical generation replay to fetch one current node. Defer eviction, a global
object index and neural feature redesign to separately demonstrated needs.

## Validation, evidence and review

**258 existing focused cases passed** in 131.83 s: 44 closed reader/writer,
48 durable closed publication, eight checkpoint continuation, 96 certainty and
62 Dataset-confidence cases. The separate unchanged strict format-3 witness passes
both checkpoint layouts in 20.05 s. These include the real endgame and literal
row/metadata/JSONL checks, all nine legacy policies, five enabled v2 policies/four
gates, corruption and publication failure/restart tests. The earlier FULL result
remains **2,798 passed, two CUDA skips at fe0fe7a5**; research-only changes do not
require another FULL/FAST run. No production behavior/test expectation changed.
See [focused JUnit](record_scale_storage/evidence/focused.xml) and
[strict checkpoint JUnit](record_scale_storage/evidence/checkpoint-format3.xml).

Byte probes compare every reconstructed scientific byte stream, not normalized
values. Original E2E/1k inputs are fully fingerprinted before/after their read-only
inventories. 5k component source fingerprints and original authority are retained
separately after its expected format refusal. All 47 pre-existing Chipiron and
13 Anemone worktrees retain their HEAD/status/diff fingerprints. No gen38, historical
modification, distributed workers, storage deployment, merge, push or PR occurred.

Reproduce in a **new secondary-disk directory**, using the interpreter and exact
source manifest recorded above, `PYTHONPATH=src:.`, `PYTHONDONTWRITEBYTECODE=1`, and
an explicit short secondary `TMPDIR` and cache/output locations:

```bash
python -m docs.research.record_scale_storage.measure inventory GENERATED_WORKSPACE NEW_OUTPUT
python -m docs.research.record_scale_storage.measure components GENERATED_WORKSPACE NEW_OUTPUT
python -m docs.research.record_scale_storage.measure scan COMPONENT_OUTPUT/g3 NEW_OUTPUT
python -m docs.research.record_scale_storage.local_patch NEW_OUTPUT
```

Review with VS Code:

```bash
code /home/pompote/oldata/victor/morpion-scratch/record-storage-20261025/worktree
```

Open this report, `record_scale_storage/measure.py`, `local_patch.py`, and
`evidence/summary.json`. Compare the earlier A/B report alongside this report using
**Select for Compare → Compare with Selected**; differences are intentional design
changes, not edits to old evidence. `git diff 46cec009..HEAD -- src tests` must be
empty. `git diff --stat 46cec009..HEAD` is the full research review. The final local
commit IDs and preservation/code-size audit are recorded in the companion evidence
and completion message; no duplicate-code PR is opened across the unmerged stack.
