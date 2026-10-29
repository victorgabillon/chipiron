# Compact derived scientific indexes

This experiment changes only the existing `ShardedTrainingGenerationView`.
Scientific JSON row bodies are stored as ordinary Zstandard-compressed SQLite
BLOBs, and a duplicate per-read field check is removed after validated index
construction/opening. Source formats, parsers, scientific reconstruction, Dataset policies,
publication, receipts, model activation and cache ownership remain unchanged.
The cache schema identity advances from 2 to 3. Existing caches are neither
migrated nor removed. This is a disposable derived representation, not a new
scientific format or authority.

## Foundation and isolation

Base: `561d906ea2b029a852c5e2c9ecffcfdfef29b556`, containing compressed-science
production `ae5415c287ea29b57e10ef8547e209305031c1ed` and normal tests
`223f76d550b433461a91ff3d41c8fd0c332c4c90`. Anemone remains exactly
`37e48c40ceff430e373fd07bd00fcf41359bafda`.

Compression implementation: `ec683aaca060ef87396f2795026ba7c8b3ae0013`.
Final production and normal tests:
`109757835630fa01acb23d98dcb914cf6a5767ca`. Compression and reconstruction
measurements use `70a6389e36f3a1eb855d763261dc9a9b9ba49a7c`; the final
integrity correction below changes only SQLite checks. The complete change adds
48 and removes 17 production lines (net +31), all in the existing reader. It adds no production
module, table or lifetime-management abstraction. The small packing function
ensures both row types use the same encoding; the decoding method centralizes
selected-row reconstruction. The version bump prevents reinterpretation of old
cache bytes. Tests add 347 and remove four lines (net +343); the two research recipes
are 292 lines and reuse existing instruments. They are separate from production.

Branch: `perf/morpion-compact-scientific-index`. The worktree, environment,
fixtures, builds and evidence are under
`/home/pompote/oldata/victor/morpion-scratch/scientific-index-20261025` on
secondary ext4. Temporary and SQLite temporary files use
`/home/pompote/oldata/victor/mi25`. Previous worktrees and their caches are
protected. No historical experiment is used as a writable fixture.
The final protection audit compares all 64 prior worktree registrations:
46 existing directories and 18 already-missing directories. Registrations,
heads, status and tracked binary-diff hashes match the before-task inventory.
It does not claim a content comparison of initially untracked files; no initial
hash inventory of those files was collected.

## What is stored and why

The existing tables and indexes remain:

| Structure | Purpose |
| --- | --- |
| `records(node_id,payload)` + primary-key B-tree | Immutable canonical state and creation generation, selected-node lookup |
| `updates(node_id,order_index,sequence,payload)` + primary-key B-tree | Complete current values, certainty, provenance and topology |
| `export_order(order_index,sequence)` | Exact deterministic export order, distinct from cursor's lexicographic node order |
| `completed(identity)` | Exact source/format/schema/producer/generation completion marker |
| `cache_digest(digest)` | Bounded scan detecting meaningful persisted payload/key/order corruption |

SQLite's existing TEXT-affinity payload columns accept BLOB bindings for
scientific v2/closed rows; legacy sharded-v1 rows remain JSON text. No SQL column,
table, secondary index or reader implementation is added. Cache filenames advance
from `v2-<identity>.sqlite3` to `v3-<identity>.sqlite3`; source scientific version
2 remains unchanged. Schema identity 3 also changes the identity hash.

The complete immutable row has node ID, creation generation and canonical state.
The complete update includes node ID, both directions of current DAG topology,
scalars, selected target source, exact/terminal flags, snapshot metadata and the
full scientific value contract. Values and certainties are intentionally present
both in snapshot-facing fields and the value contract. SQL keys/order also
duplicate small JSON fields. There is no second persistent topology table or
second serialized copy of each complete update. Dropping individual scientific
fields would require reconstructing absence/null/provenance contracts; byte
compression avoids that risk. The primary-key/order B-trees are a small fraction
of total storage and still serve their existing queries.

Cold validation uses TEMP `v2_creation`, `v2_parents`, `v2_children`,
`v2_degrees` and `v2_queue` for creation provenance, reciprocal links, closure
and acyclicity. Closed sources do not need the historical creation-index pass.
TEMP tables are dropped after validation. Dataset builds its existing disk-backed
node/family tables in the same connection and drops them on scope exit; it does
not create another source index. Main and TEMP page caches remain bounded at
the existing 2 MiB settings.

## Compact representation and integrity

The preliminary 1,003-node payload-only rebuild compared the same rows and
indexes: unchanged JSON allocated 1,347,584 bytes; whitespace removal allocated
1,265,664; per-row Zstandard allocated 663,552. Respective write times were
0.0335, 0.0487 and 0.0423 seconds. These experiments exclude completion/digest
tables and full canonical validation; production measurements below include them.

One existing-library compressor (level 3, checksum enabled, no dictionary) packs
each exact JSON body. One decoder inflates only the requested row. No state is
decoded/replayed because of compression. Historical scientific-v2's one
`json_extract` creation-generation check becomes a bounded row scan through the
same decoder; no production full reference reader is called.

The payload-digest contract described here applies to scientific v2/closed
sources; legacy sharded-v1 text payload behavior is retained.
On build and reuse SQLite `integrity_check` verifies table/B-tree consistency;
on reuse the exact completed identity must also match.
The content digest hashes the ordered SQL keys/order fields, an eight-byte payload
length and the actual compressed bytes. It verifies all payloads before any is
inflated, avoiding the previous extra JSON-escaping pass over stored JSON text.
Corrupt/truncated/type-changed payloads, altered order and oversized forged frame
headers fail the digest/type check and cause rebuilding from the source. The
standard decoder also checks its frame checksum and rejects trailing data.

The final corruption audit exposed a pre-existing gap in `quick_check`: it checks
page structure but can accept a secondary ordering B-tree inconsistent with the
table. A four-node fixture keeps the exact table/digest bytes while constructing
reversed secondary keys. The old reader reports reuse and returns
`[3, 2, 1, 0]` instead of `[0, 1, 2, 3]`; `integrity_check` detects the missing
index entries. The production correction changes only the two SQLite check
calls, at completion and reuse. The regression fails before the correction and
passes after it by requiring rebuild and exact reference order. The byte digest
remains necessary: SQLite integrity alone cannot detect altered scientific fields
inside otherwise valid rows. This is SQLite's existing verifier, not a second
validation framework.

Actual committed source bytes are still hashed. Identity includes schema,
scientific format/kind/contract, generation, producer, resolved source paths,
sizes and content hashes. Renames or changed sources cannot reuse another key.
A final four-node rename experiment keeps the manifest bytes unchanged, moves
the generated standalone source, requires a distinct freshly built cache key,
then confirms reuse and exact nodes at the new path. Both cache files remain;
[renamed-source.json](scientific_index/evidence/renamed-source.json) records them.
Existing capsule pins and committed canonical-record evidence retain exactly
their existing lifetime; there is no persistent new proof cache. Cold construction
still validates canonical states unless existing committed evidence applies.

The existing integrity assumption is cooperative immutable cache publication:
builders write private `.building-*` files and atomically replace the completed
path; readers retain their open inode. A content digest detects accidental
corruption, not a hostile actor able to rewrite both database and digest. External
in-place mutation after opening is not supported. Compression does not expand
this trust model. Memory remains proportional to a selected window, fixed parser
and SQLite buffers, and the largest already validated row; a complete scientific
snapshot is never materialized by the indexed path.

The initial compression-only candidate imposed an 18–25% Dataset cost at 5k.
The existing cProfile driver identified 25,065 node reconstructions (five passes),
56,265 row inflations and 110,540 SQLite executions for policy `none` at 5,013
nodes. Those reconstructions repeated `_validate_update` after every update had
already passed it during ingestion. The final implementation deletes just that
duplicate call: a successful build validates every update before completion; a
reused open checks the exact source identity and every persisted compressed byte.
These facts remain valid for the immutable inode held by that view. No canonical
state validation, cold scientific validation, source hashing or cache-integrity
scan is removed. A new regression counts validation at ingestion, checks repeated
reconstruction/reuse, then corrupts the cache and requires complete revalidation
on rebuilding. This reuses existing evidence only within its existing ownership
contract; it creates no proof cache or additional persistent state.

## Lifetime and safe ownership

| Caller | Existing lifetime and reuse |
| --- | --- |
| Closed writer / Growth | Caller-owned temporary source/output validation indexes inside its private attempt; removed on normal scope exit |
| Capsule validation / activation publication | Existing `.validation-index-*` temporary scope; no persistent reader index |
| Dataset | One view reused through all family passes; optional caller-supplied `reader_cache_dir`, otherwise persistent default cache |
| Reevaluation | One view per selected-window operation; optional same caller-supplied directory gives cross-operation reuse, default persists |
| Training | Consumes accepted Dataset rows; does not directly build this scientific index |

Default cache storage is under `XDG_CACHE_HOME/chipiron/sharded_training_reader`.
Every exact source identity can leave one completed index indefinitely. Multiple
builders can briefly own distinct private builds; completed readers do not create
new files. A crash before atomic completion can leave an ignored private build.
An invalid reusable index is replaced, without deleting any other cache key.

No global automatic eviction is added. A source capsule lease does not enumerate
all readers of a shared cache directory; stage claims and timeouts do not prove
process death. There is no safe way here to decide that another process has
finished using a global index. Instead, the existing `cache_dir` interface already
supports a session-owned `TemporaryDirectory`: keep it across repeated operations
on a source, close every reader and join every child before exiting the scope.
This preserves warm reads and makes allocation disappear at a proven ownership
boundary. The tests cover normal exit and both build/reader process death. If
the owner itself dies, its directory can remain; no unrelated worker cleans it.

A private directory spanning unbounded generation identities still grows until
its owner exits. Bounded retention therefore needs an explicit source/session
boundary and confirmed release of every reader, not merely a private pathname.
The measured scope holds one committed source across three operations.

Keeping the default unchanged avoids forcing a costly cold rebuild on every
one-shot Reevaluation operation. Adoption of session scope belongs to the caller
that demonstrably owns those processes. There is no new global manager, lease
database, timer, registry or cleanup authority in this change.

## Measurements and parity

Results and exact file identities are recorded in `scientific_index/evidence`.
The same newly generated compressed-science format-3 sources and model bundle
paths are read before and after installing the candidate wheel. Counts are 1,003
nodes/1,114 branches and 5,013 nodes/5,931 branches, grown with the existing seed-4
fixture. These are meaningful replay histories, not single-move index fixtures.
All benchmarks run serially in fresh processes. Allocation uses unique inodes and
`st_blocks * 512`. Existing 20 ms allocation sampling is a lower bound on transient
peaks, not an exact upper bound. CPU, process RSS, I/O, state-codec counts and hash
work use the existing measurement utilities.

The reproduction driver `scientific_index/measure.py` adds bounded random reads,
actual neural Reevaluation under both existing numeric policies, and three opens
in a caller-owned cache scope. Dataset/full scientific/publication measurements
reuse `scientific_compression.measure` and `durable_closed_snapshot.measure`.
The random sample repeats 2,048 lookups among 256 lexicographically selected IDs;
it is not claimed to model uniform whole-graph access.

The paired compression/reconstruction runs at `70a6389e` use the same committed
G5 sources; the final `10975783` opening-cost check is reported separately below. Cold means a fresh
derived index, **not** a flushed operating-system page cache. Native measurements
have no allocation-sampling thread. Three fresh processes per implementation
measure Dataset; the table reports their median. Constructor timings below come
from the unsampled session experiment (one cold open, then two warm opens).
The earlier compression-only G3 experiment and its profile are retained separately
in `evidence/compression-only.json`; its timings are not substituted for final ones.

### Actual storage

| Bytes per selected generation | 1,003 baseline | 1,003 final | 5,013 baseline | 5,013 final |
| --- | ---: | ---: | ---: | ---: |
| SQLite pages / logical file length | 1,355,776 | 671,744 | 7,761,920 | 3,608,576 |
| Settled filesystem index allocation | 1,359,872 | 675,840 | 7,766,016 | 3,612,672 |
| Checkpoint + scientific snapshot allocation | 163,840 | 163,840 | 675,840 | 675,840 |
| Combined allocation, including one index | 1,523,712 | 839,680 | 8,441,856 | 4,288,512 |

Settled `st_blocks` includes 4,096 bytes beyond SQLite's page count in each measured
index. The report counts those bytes rather than silently equating logical size
with physical allocation. Index allocation falls **50.3% / 53.5%**; combined
allocation falls **44.9% / 49.2%**. Models and cumulative run-state records are not
included in this per-generation subtotal; full workspace inventories are in the
evidence. Shared inodes are counted once in those inventories.

| SQLite page allocation at 5,013 nodes | Baseline bytes | Final bytes |
| --- | ---: | ---: |
| Records table | 2,301,952 | 1,273,856 |
| Updates table | 5,230,592 | 2,105,344 |
| Two primary-key B-trees | 139,264 | 139,264 |
| Export-order index | 77,824 | 77,824 |
| Schema, completion, digest | 12,288 | 12,288 |

The raw record/update JSON bodies total 2,020,439 / 4,500,076 bytes. Their compressed
bodies total 1,152,668 / 1,933,349 bytes. Maximum record body length falls from
1,375 to 577 bytes; maximum update length from 1,742 to 602. Small rows can grow
(the minimum record goes from 72 to 85 bytes). No free-list pages are retained.
The index is still much larger than the compressed scientific source: individual
frames preserve selected-row access and do not share a cross-row dictionary.

### Reader, Dataset and Reevaluation

| Native observation (`70a6389e`) | 1,003 baseline → compact | 5,013 baseline → compact |
| --- | --- | --- |
| Cold constructor | 1.308 → 1.284 s | 14.982 → 14.615 s |
| Warm constructor, mean of two opens | 8.30 → 6.78 ms | 35.06 → 21.28 ms |
| 2,048 selected-ID random lookups + exact serialization | 51.3 → 52.1 ms | 53.3 → 47.5 ms |
| Neural Reevaluation, raw policy | 10.910 → 10.712 s | 85.871 → 81.190 s |
| Neural Reevaluation, Growth-scaled policy | 10.581 → 10.555 s | 70.571 → 68.366 s |

Neural time varies across rounds (the earlier 5k native raw pair was 68.845 /
71.776 seconds). These are observed costs, not evidence that compression speeds
up neural inference. Final 5k traversal rates are 58.4 → 61.7 and 71.0 → 73.3
nodes/s, respectively. All patch bytes and cursor/wrap decisions agree exactly.
There are 5,005 actual state decodes at 5k Reevaluation; the eight terminal nodes
retain their existing values. Compression introduces no state replay.

| Confidence-v2 policy | 1k seconds, baseline → final | 5k seconds, baseline → final | 5k final rows/s |
| --- | --- | --- | ---: |
| `none` | 0.1359 → 0.1295 | 0.7251 → 0.6726 | 556 |
| `pv_blend_mean_prediction` | 0.1416 → 0.1322 | 0.7437 → 0.6670 | 561 |
| `pv_blend_min_prediction` | 0.1384 → 0.1369 | 0.7404 → 0.7086 | 528 |
| `pv_mean_prediction` | 0.1378 → 0.1309 | 0.7120 → 0.6882 | 543 |
| `pv_min_prediction` | 0.1422 → 0.1333 | 0.7431 → 0.6934 | 539 |

Filtering leaves 64 / 374 rows per policy. At 5k, median Dataset CPU user+system
time is 0.708–0.740 seconds before and 0.663–0.703 after. Measured Dataset process
peak RSS is 290.8–291.1 MiB before and 291.5–291.6 after; cold-reader observations
are 294.5 / 296.0 MiB. Final neural processes peak at 301.4 / 301.6 MiB. These
measurements include imports/runtime overhead and demonstrate no new whole-tree
allocation; they are not extrapolated to millions of nodes.

Hash work is identical before/after: at 5k the three-open experiment performs
183 `file_digest` calls over 4,077,330 bytes, and the random/neural experiment
performs 68 over 5,190,691 bytes. Full kernel I/O counters are retained. Native
5k raw Reevaluation reduces `rchar` through smaller SQLite reads, but physical
`read_bytes` is zero on these page-cache-warm runs; no cold-device bandwidth claim
is made. The completed scientific source/checkpoint/model file fingerprints match.

### Publication, temporary allocation and retained indexes

The existing validator identity hashes `sharded_training_reader.py`, among other
complete codec/validator sources. Consequently a reader-code change conservatively
invalidates prior canonical-record reuse evidence. It does **not** invalidate
the scientific source. This guard remains intact. Native publication costs at `70a6389e`:

| Publication after code change | 1,003 nodes | 5,013 nodes |
| --- | ---: | ---: |
| First, G6: wall / CPU | 3.053 / 1.698 s | 19.491 / 15.430 s |
| First: canonical state decodes | 1,003 | 5,013 |
| Next, G7: wall / CPU | 2.312 / 0.967 s | 6.654 / 2.846 s |
| Next: canonical state decodes | 0 | 0 |
| Next: process peak RSS | 305.7 MiB | 322.5 MiB |

Both copy every immutable record and encode zero new states. Publication hashes
951 / 1,551 actual files per operation; subsequent generations still perform
full required byte verification. Their hashed bytes are 8,985,378 / 31,104,185.
Fixture patch preparation opens a separate `patch-cache` before the publication
timer; that extra cold build is explicitly excluded from publication latency.

The sampled, same-layout publication experiment at G2/G3 versus G4/G5 observes
peak private SQLite/staging allocation falling from 2.83 to 1.46 MB at 1k and
15.75 to 7.48 MB at 5k. These are 20 ms sampled lower bounds and include the
private attempt, not just one database. At 5k the separate unlinked TEMP-file
peak is 286,720 bytes during publication and 147,456 during cold reading. Cold
graph TEMP logical size remains 585,728 bytes (135,168 at 1k). There is no plaintext
scientific-shard temporary file. Sampler timings grow with retained evidence
files, which is why native timings are reported separately.

At the end of G5 measurement the deliberately preserved baseline/candidate
experiments retain 8 / 9 SQLite files, allocating 8,142,848 / 49,127,424 bytes.
After the additional G6/G7 publications these become 10 / 11 files and
9,494,528 / 56,352,768 bytes. The exact per-file generation, schema, inode allocation
and source identity are retained in the evidence. These totals include both
schema versions and independent reader/fixture-patch/profile cache directories;
they are **not** the footprint of one production worker. All benchmark readers
have exited. Historical identities remain because this task preserves its evidence;
normal shared caches have the same absence of automatic eviction. G7's temporary
publication indexes have been removed; no persistent G7 index was created.

A session-owned cache retains one 675,840 / 3,612,672-byte index while serving
three opens (cold, reuse, reuse), then retains zero bytes after its proven scope
ends. There is no per-open accumulation. After process death, the owner waits
for the child to exit before removing its private directory. Owner death itself
still leaves a possible orphan. There is no age-based or generation-based deletion.

### Final integrity correction

After the G5 read benchmarks and G6/G7 publications, a final audit reproduced
the secondary B-tree corruption described above. The in-progress FULL run was
stopped before changing production. The final implementation is
`109757835630fa01acb23d98dcb914cf6a5767ca`; it changes only the completion/reuse
SQLite check relative to the measured compression/reconstruction implementation.

Fresh serial processes compare the baseline, compressed quick-check version and
final integrity-check version against the **same G7 source**, without publishing
another generation. Each owns one temporary directory, opens cold then twice
reused, and closes/removes it. These measurements isolate the affected operation;
the preceding G5 Dataset, payload and neural-patch comparisons remain labelled
with their actual implementation.

| Nodes | Implementation | Cold constructor s | Mean reused constructor ms | Index allocation bytes | Process peak RSS MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,003 | Baseline | 1.250258 | 8.86 | 1,359,872 | 294.32 |
| 1,003 | Compressed, quick check | 1.261740 | 10.06 | 675,840 | 294.77 |
| 1,003 | Final, integrity check | 1.229646 | 7.33 | 675,840 | 294.68 |
| 5,013 | Baseline | 14.683887 | 33.96 | 7,766,016 | 295.03 |
| 5,013 | Compressed, quick check | 14.949716 | 21.50 | 3,612,672 | 295.28 |
| 5,013 | Final, integrity check | 15.118018 | 23.42 | 3,612,672 | 295.74 |

Source manifests and hash counters match exactly across all three versions.
No additional persistent bytes or canonical-state representation is introduced.
The final check costs about 1.9 ms of warm initialization in this 5k comparison;
the final cold run is about 3% slower than baseline, while reuse is about 31%
faster. These short timings are observations, not statistical confidence bounds.
All six private scopes retain one index across three opens and return to zero
owned cache allocation after closing. Full raw observations are in
[integrity-check.json](scientific_index/evidence/integrity-check.json).

### Exact parity and review evidence

The same-G5 compression/reconstruction comparison checks **12,032 complete cached JSON bodies** against
the old TEXT payloads after lossless decompression, including SQL keys/order.
It checks every scientific node, value contract and creation provenance digest;
**2,766 complete Dataset rows plus their JSONL metadata**; and **12,032 neural
patch rows** under both numeric policies, including cursor and wrap metadata.
Nine legacy policies and five supported v2 policies are compared at 1k; all five
v2 policies are also compared at 5k. Full v2 reference extraction is the oracle.
The comparison never normalizes fields or replaces exact equality with tolerance.
Reader parity does not approve legacy confidence semantics or enable the four
gated v2 family policies.

The normal tests retain the real `FORCED(35)` / `ESTIMATE(42)` regression, both
target preferences, exact filtering, null/absent distinctions, four gated v2
exact-family policies, producer/model/receipt fencing and canonical state checks.
New process tests cover simultaneous builders, two old pinned readers across a
new publication, restart reuse, incomplete builds, active-reader death and private
scope cleanup. Corruption tests include valid compressed frames containing changed
scientific references, oversized headers, wrong storage type, altered order,
missing completion/digest records, missing indexes and inconsistent ordering
B-trees that pass SQLite quick checks. A late real M1 producer
is also tested against M2 activation with compressed scientific sources.

The first PR run found one old test that edited cached TEXT directly. Its injection
now edits/recompresses the reference with a valid frame checksum, preserving the
original semantic-corruption assertion. Production behavior did not change for
that fixture correction. The failure and successful focused rerun remain in the
scratch logs rather than being hidden.

## Validation

The final production and normal-test commit is `10975783`. Tests run against its
installed wheel with all 478 installed Chipiron Python files checked against the
worktree. The existing Makefile lanes use two local test executors; no distributed
bootstrap worker is launched. Inputs are generated temporary workspaces on the
secondary disk.

| Check | Result |
| --- | --- |
| Focused scientific/Dataset/confidence/fencing tests before integrity correction | 239 passed |
| Final cache/performance tests, including B-tree corruption | 15 passed |
| Relevant exhaustive reader/publication/crash matrices before integrity correction | 116 passed |
| Final FAST | 152 passed |
| Final PR | 2,750 passed, two CUDA-only skips; 1,084.24 s |
| Final FULL | **1 failed**, 2,913 passed, two CUDA-only skips; 1,768.81 s |
| Repository Ruff format/check and pinned toolchain checks | Pass |
| Strict static ratchet | No new diagnostics: mypy 91 → 89, pyright 73 → 73, pylint 45 → 44 |
| Wheel + source archive build and installed-distribution smoke check | Pass |

The pre-correction PR passed 2,749 tests with two CUDA-only skips. Its following
FULL attempt was explicitly interrupted after the ordering-index corruption was
demonstrated; it is not counted as a passing FULL run. The final check repeated PR
and completed one FULL after production and normal tests were frozen. Earlier
failed corruption injections and all subsequent logs remain in scratch. No
documentation-only descendant requires another FULL run.

The completed FULL is **not green**. Its sole failure is
`test_headless_cluster_stops_all_four_worker_groups[False]`: the existing shell
supervision test saw one descendant still alive after its one-second polling
deadline. Both that test file and the launcher are byte-identical to `561d906e`
(hashes in `unchanged-launcher.json`); they do not exercise the scientific reader.
They launch only generated sleeping stubs, not real Morpion workers. The process
was gone when inspected afterward. All three tests pass in a fresh serial
coverage run (9.11 s) and a fresh two-executor coverage run (10.75 s). The timing
failure's root cause is not established; a
passing retry does not erase it. The launcher and its timeouts are deliberately
unchanged in this focused reader task. This CI exception remains visible for
follow-up before integration; no speculative timeout increase or unrelated
production fix is included.

Machine-readable counts, skipped test identities and raw-log locations are in
[validation.json](scientific_index/evidence/validation.json).

## Remaining limits and next work

Compression reduces the constant factor, not asymptotic storage. One index is
O(serialized state histories + N + E); retaining all generation identities still
accumulates their sum. Existing old-schema and new-schema files can coexist until
their explicit owner safely retires them. Source hashing and cached-payload digest
scans remain linear in their respective bytes on every new open. SQLite integrity
verification additionally checks index/table consistency on every open. Cold
canonical replay, TEMP graph validation and Dataset family computation remain. Long histories can
increase the largest row; bounded selection is not constant-size input storage.

The live Growth tree, export reference list, checkpoint selector metadata, retained
capsules, exact receipts and historical metrics retain their existing global
allocations. This change neither hides those limits nor enables artifact deletion.
Source snapshots/checkpoints, receipts and model provenance remain authoritative;
derived cache bytes are excluded from scientific commit identities.

The next justified ownership improvement is to retain one existing private cache
directory for a known committed-source session across successive windows, then retire it
after the owning process group has actually stopped. It requires an explicit
session owner in callers that currently use one-shot invocations; do not infer
that lifetime from a claim timeout. Further removal of duplicate contract fields
or new shared-generation indexes is not justified by this experiment.

For Dataset CPU, the measured next target is its five reconstruction passes and
repeated contract queries. Consider passing an already-decoded update's contract
through a bounded selected window before adding any cache: the 5k profile shows
6,135 extra contract lookups. Measure the saving first; a generalized row cache
or a large interface rewrite would not be justified for that fraction alone.
Keep the conservative publication code-identity fence. A change to validation
rules/dependencies must invalidate the corresponding derived-cache contract;
this experiment proves parity with the pinned toolchain, not arbitrary future
dependency versions.

## Reproduce and review

All raw fixtures and logs remain under the isolated scratch root. Curated evidence
is in [results.json](scientific_index/evidence/results.json), including CPU, RSS,
I/O, source fingerprints, every policy timing, exact output hashes, SQLite dbstat,
all retained index identities and publication counters. The
[installed source inventory](scientific_index/evidence/installed-sources.json)
hashes actual Chipiron, Anemone, Coral, Atomheart, Valanga, ijson and Zstandard
files. All 289 Anemone Python files were independently compared with the pinned
Git archive; all 478 installed Chipiron Python files match the worktree.
The [manifest](scientific_index/evidence/source-manifest.json) records exact
commits, wheel/source-archive/native-library hashes and tool versions.

For a new reproduction, use a **fresh secondary-disk fixture directory**, leaving
the retained evidence intact. With the same interpreter/dependencies and
`PYTHONDONTWRITEBYTECODE=1`, generate each population using the existing driver:

```bash
python -m docs.research.scientific_compression.measure NEW_ROOT initial --count 1000 --encoding jsonl-zstd-1
python -m docs.research.scientific_compression.measure NEW_ROOT repeat --count 1000
python -m docs.research.scientific_compression.measure NEW_ROOT repeat --count 1000
```

Repeat with `--count 5000` in a separate directory, serially. Generate G1–G3
with the pinned baseline, G4–G5 with the compression-only implementation, then
compare baseline/`70a6389e` against the **same G5 source**. G6/G7 use that code and
demonstrate the code-change validation fence followed by steady reuse. The final
`10975783` constructor measurements use the same G7 source, with the baseline
and `70a6389e` loaded from separate exact Git archives. Keep
TMPDIR, SQLITE_TMPDIR, XDG_CACHE_HOME, plotting/output/MLflow paths on the secondary
disk, as in the isolated environment. No production routing flag is changed.

For each implementation/source, run `cold`, `reuse`, `dataset_stream` (three
fresh-process repetitions), and `dataset` with the existing driver, preserving
each result under a distinct implementation directory. Use
`durable_closed_snapshot.measure` for native timings and
`scientific_compression.measure` for allocation sampling. Run:

```bash
python -m docs.research.scientific_index.measure NEW_ROOT LABEL reads --native
python -m docs.research.scientific_index.measure NEW_ROOT LABEL scope --native
```

The baseline source override is an exact `git archive 561d906e src/chipiron`,
verified against the previous installed inventory, with only its `src` prepended
to PYTHONPATH. Final measurements load the newly built installed wheel. Everything
else uses the same isolated interpreter and dependency files. The fixture paths,
source bytes and model paths remain identical within each comparison.
`scientific_index.compare` verifies final G5 cache bodies, Dataset files and neural
patch streams and collects the two populations' measurements. The raw profiler is
the existing cProfile option, not a new profiling service.

The local implementation is reviewable directly above `561d906e`; no upstream
branch was rewritten and no PR, merge, package publication or activation occurred.
For the complete production change in VS Code:

```bash
code --diff \
  /home/pompote/oldata/victor/morpion-scratch/scientific-index-20261025/review/sharded_training_reader.before.py \
  /home/pompote/oldata/victor/morpion-scratch/scientific-index-20261025/worktree/src/chipiron/environments/morpion/bootstrap/sharded_training_reader.py
```

Open the worktree folder and compare `561d906e` with `HEAD` in Source Control Graph
for tests, recipes and evidence. No physical committed-artifact cleanup, historical
cache deletion, gen38 operation or distributed bootstrap run was performed.
