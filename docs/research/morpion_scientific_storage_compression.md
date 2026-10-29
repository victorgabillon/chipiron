# Lossless streaming scientific storage compression

This change adds an explicitly selected byte encoding to the existing closed
scientific writer and bounded reader. It preserves the scientific v2 contract,
closed layout 1, checkpoint format 3 and the existing capsule/run-state authority.
The default remains uncompressed. No historical workspace was migrated or activated.

## Foundation

Base: `f0184c767a34f17d9845f8586d446571f66183ac`. Its changes after the FULL-tested
`d8e26e6e9b21156841963e54d4e85d7f44bb7b0f` are research/evidence only; production
and normal tests are identical. Exact Anemone implementation:
`37e48c40ceff430e373fd07bd00fcf41359bafda`, wheel SHA-256
`4ab0ffc88b17fb500a3729991fc4fa5c8f43d2b70be875f68a6d2d52ffe47d2f`.
Installed files are checked independently of version `0.2.26`.

Production implementation: `ae5415c287ea29b57e10ef8547e209305031c1ed`.
Reproduction extensions: `6e52c314`; final normal tests and measurement fixes:
`223f76d550b433461a91ff3d41c8fd0c332c4c90`. Subsequent report/evidence edits do
not change executable production or normal tests. Python 3.13.5, zstandard
0.25.0 / libzstd 1.5.7, ijson 3.6.0 and Torch 2.14.0+cpu were actually loaded.
The [source manifest](scientific_compression/evidence/source-manifest.json)
records source files, tool versions, installed compression binary and package
hashes. The [installed-file inventory](scientific_compression/evidence/installed-sources.json)
records all 478 Chipiron and 289 Anemone Python files; Anemone was independently
compared against its pinned Git archive, not just its version or wheel label.

New isolated branch: `perf/morpion-compressed-science`.
Worktree: `/home/pompote/oldata/victor/morpion-scratch/science-compression-20261025/worktree`.
All fixtures, packages, caches and logs are on the secondary ext4 disk, initially
51 GiB free. TMPDIR and SQLITE_TMPDIR are `/home/pompote/oldata/victor/mz25`.
The protection audit captured 63 existing Chipiron/Anemone worktree registrations.

## Exact encoding contract

The directly readable `generation.json` retains the same scientific envelope,
producer, generation, snapshot metadata and two shard descriptors. Legacy raw
shards retain exactly `{path, generation, count, sha256}` and their filenames
`records.jsonl` / `updates.jsonl`. Absence of an encoding field has only that
existing raw meaning; unknown keys/encodings and mixed encodings are rejected.

New descriptors add exactly:

```json
{
  "encoding": "jsonl-zstd-1",
  "bytes": 123,
  "decoded_bytes": 456,
  "decoded_sha256": "<SHA-256 of every original JSONL byte>"
}
```

Their filenames are `records.jsonl.zst` / `updates.jsonl.zst`; the existing
`sha256` binds the actual compressed file. `count` still counts nodes, excluding
the one JSONL envelope. Both compressed and decoded counts are checked. The
sealed capsule inventory independently binds the actual files using its existing
hash/fsync/rename protocol. No second commit, receipt, index or model authority is
introduced. Encoding identity reaches the cache through the hashed descriptor;
scientific Dataset provenance remains identical because encoding changes no values.

`jsonl-zstd-1` specifies one ordinary Zstandard frame, compression level 3,
window log 20 (1 MiB), checksum enabled, no dictionary, no content-size field,
and no compression threads. The six-byte frame header is `28 b5 2f fd 04 50`.
There are no skippable or concatenated frames. The writer uses the pinned
zstandard library; repeated inputs/settings in that implementation produce the
same bytes. Cross-library-version compressor byte identity is not promised;
decoded scientific byte identity is mandatory.

The helper inspects only the standard frame/block envelope and delegates all
compressed-block decoding and checksum verification to libzstd. A block's
compressed and decoded data are each limited to 128 KiB. See the primary
[Zstandard block specification](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#blocks).
Feeding one bounded block to `decompressobj` provides its explicit EOF and
unused-input checks without permitting a high-ratio input chunk to concatenate
unbounded output. `stream_reader` alone does not expose a completed-frame flag;
using it alone would miss some truncated trailers. The fixed header also bounds
the decoder window independently of untrusted size declarations.

Incremental physical/decoded hashes, byte counts, newline counts and final
newline are verified at exhaustion. Declared enormous decoded sizes never size
an allocation. Truncated headers, bodies/checksums, reserved blocks, other frame
parameters, wrong hashes/counts, trailing data and extra frames fail. The existing
strict ijson parser still owns UTF-8, duplicate-key, JSON and scientific validation.
No JSON parser or scientific reconstruction has been duplicated.

## Production paths and ownership

The existing row iterator serializes each row once into the optional compressor;
it never writes a raw intermediate file. Hashes/counts accumulate as rows pass.
The frame finishes before its descriptor is written; fresh on-disk hashes must
match the bytes generated by the writer before canonical evidence can transfer.
The existing private attempt, source lifetime pin, immutable publication,
file/directory fsync and authoritative run-state replacement are unchanged.

`open_scientific_stream` adapts decoded blocks to the same parser used by the
full oracle, bounded index and canonical-record copy iterator. Readers do not
create an uncompressed temporary file. Complete indexes become reusable only
after full parsing/validation, exact source checks and existing completion marker.
Reused indexes still hash actual compressed bytes and the manifest; they do not
blindly trust a persisted digest.

Canonical record reuse consumes exact decoded source lines, then recompresses
them under the next generation's envelope. All unchanged records retain their
creation generations. They are not replayed through the Morpion codec again
when the existing committed canonical evidence is valid. Mutable values and DAG
links are always regenerated and validated in fresh existing SQLite indexes.

Only byte consumers need the helper. Capsule inventory, file hashing, pins,
Dataset/Training acceptance, patch/cursor transactions and model activation need
physical committed paths and identities; they continue to use those paths.
The publication adapter's exact file-set check now takes the two already
validated filenames from the descriptor. It accepts no other files.

## Adoption

A fresh genesis explicitly calls `export_closed_scientific_snapshot(...,
encoding="jsonl-zstd-1")`, then uses the existing fresh-workspace initializer.
Every subsequent publication inherits the encoding from its pinned committed
source. An explicit conflicting encoding is refused. No production CLI/default
is changed and no migration path is enabled.

Historical raw support remains one branch of the same stream opener and writer.
A future migration needs a fresh private copy, decoded-byte equality, full
checkpoint/selector/RNG/model/receipt/dependency validation and durable authority
commit before any original can be retired. This task authorizes no such migration
or deletion. The raw branch may be removed after separately validated adoption;
there is no permanent dual scientific implementation to maintain.

## Remaining scaling limits

Compression adds fixed compressor/decoder windows and bounded block buffers,
plus the existing largest serialized row. Long state histories still enlarge
individual rows. The existing scientific exporter obtains an O(N) list of node
references; the live search graph remains resident. SQLite scientific rows,
topology checks, Dataset family tables and cache disk remain O(N + E + scientific
bytes), even when compressed source files are much smaller. Checkpoint selector
metadata and manifests retain their existing global allocations. Retained
capsules, exact receipts, metrics and accepted artifacts still accumulate until
a separately authorized retention design is implemented.

The helper was assessed against the requested simplicity constraint: it adds no
format registry, cache, storage engine, transaction machinery or scientific
policy. The additional framing code is required for strict EOF and bounded
high-ratio decoding. Writer/reader adaptations share it rather than implementing
parallel raw/compressed scientific paths.

## Reproduction and review

Use the isolated environment and `CHIPIRON_TEST_INSTALLED=1`, with the temporary,
SQLite, plotting, cache and MLflow paths set inside the task's secondary-disk
scratch root. Benchmarks use serial fresh processes and newly generated legal
format-3 search graphs. The reproducible driver is
`docs/research/scientific_compression/measure.py`; it extends the existing durable
closed-snapshot measurements with inode-aware temporary-allocation sampling.
Samples are lower bounds, not an asserted exact transient peak. CPU/RSS/I/O use
the existing process counters and StageMeasurement. No expensive benchmark is
run concurrently with another workload from this task.

The genuine E2E recipe accepts `--scientific-encoding jsonl-zstd-1`; only genesis
selection changes. Later Growth, Dataset, Training, activation, Reevaluation,
restart and continuation checks use their actual existing durable APIs.
The inherited producer revision string in that synthetic recipe is fixture
metadata, not a claim about the executing Git revision; source hashes identify
the executed implementation separately.

For a graphical comparison, open this worktree with `code <worktree>`, use
Source Control, and compare `f0184c76` with the implementation commit. To view a
single file, save `git show f0184c76:<path>` into a new secondary-disk scratch
file, then run `code --diff <scratch-original> <worktree>/<path>`. Avoid checking
out or resetting any protected branch to inspect the difference.

The original writer and indexed-reader files have already been extracted into
the new task's `review-original` directory. For example, from this worktree:

```sh
code --diff ../review-original/closed_scientific_snapshot.py src/chipiron/environments/morpion/bootstrap/closed_scientific_snapshot.py
code --diff ../review-original/sharded_training_reader.py src/chipiron/environments/morpion/bootstrap/sharded_training_reader.py
code src/chipiron/environments/morpion/bootstrap/scientific_stream.py
```

## Exact parity evidence and independently executed pipelines

The matched 1k/5k populations are 1,003/5,013 nodes, with identical replay histories
within each raw/compressed pair. At G1–G3 every complete decoded record/update
stream compares byte for byte, including its envelope and order. All other
scientific generation-header fields match exactly. Every decoded checkpoint
component also matches exactly: topology, representative parents, openings,
Values, selector, search RNG, rollout RNG and metadata. The encoding descriptors
and their physical commitments intentionally differ; they are never normalized
into an invented common identity.

The policy comparison includes nine legacy policies and five enabled v2 policies,
direct/backed-up preferences, exact-only and row-limit filters, null/missing
Values, transpositions, partial openings, and the four gated v2 policies. On the
large matched fixtures, 2,766 paired JSONL rows (896 at 1k; 1,870 at 5k) match in
full-reference/raw, full-reference/compressed and both bounded extractions,
including complete row metadata and header counters. These sparse search graphs
have 64/374 usable target rows; node counts must not be advertised as row throughput.
The real FORCED(35)/ESTIMATE(42) regression is also tested in both encodings.
A separate test exports the same genuinely accepted, fully refreshed M2 graph
and actual producer in two fresh fixtures and compares every scientific byte.

Four genuine E2E runs completed, each with 40 serial fresh worker processes:
opening/raw, opening/compressed, forced-endgame/raw, forced-endgame/compressed.
Each reaches M1→M2→M3 using real accepted Dataset/Training artifacts and explicit
activation, checks checkpoint format 3, cold continuation, selector/both RNGs,
Reevaluation score/cursor contracts, exact receipts and historical M2 retry after
M3. Every accepted Dataset is compared byte for byte with the full oracle using
its actual scientific source, including the complete confidence/provenance header.
Every Reevaluation patch row is compared exactly with that same-source full
reference, including its actual model path. Independent terminal/FORCED proofs
remain intact through both model refreshes.

In the compressed real endgame, G1 has direct ESTIMATE(42), tree/effective
FORCED(35); after M3 at G5, direct is ESTIMATE(30.484771728515625) while
tree/effective remain FORCED(35). The full per-value contracts and source hashes
are retained in the [endgame witness](scientific_compression/evidence/endgame-witness.json).
The historical format-2 representative-parent discrepancy remains documented
in the earlier report; current format 3 persists representative and codec parents
separately. Compression changes neither interpretation nor continuation order.

Across the independently executed raw/compressed E2E workspaces, all eight
trained candidate parameter-file pairs are byte-identical. Numerical training
metrics, architectures, optimizer settings and chosen candidate names match.
Model manifests retain real workspace/cache paths and measured timings, so their
full bundle digests differ. In the second learning cycle the Dataset header
correctly differs only in the actual producer evaluator digest and the resulting
extraction identity; supervised row payloads still match exactly. First-cycle
Dataset JSONL files match completely. Cross-workspace Reevaluation rows differ
only in their real `metadata.model_bundle_path`; all remaining row fields match.
Patch envelopes also retain distinct workspace, attempt, capsule, model and
receipt identities. These differences are recorded explicitly in the evidence,
not stripped from artifacts or mislabeled as complete cross-workspace byte parity.

The original raw writer extracted verbatim from `f0184c76` was also executed
against the new raw writer with the same current dependencies and four-node
fixture. All three files compare exactly for tree, effective and direct target
selection, including explicit-null metadata and fixed creation time. See
[baseline bytes](scientific_compression/evidence/raw-baseline.json),
[matched scientific/Dataset parity](scientific_compression/evidence/parity.json)
and [independent E2E comparisons](scientific_compression/evidence/e2e-parity.json).
The E2E comparison treats a missing key differently from an explicit null.

## Implemented allocation measurements

These are newly generated legal format-3 graphs: 1,003 and 5,013 nodes, seed 4,
with 1,114 and 5,931 edges and exactly matched state histories. Canonical replay
lists average 6.94/15.73 items and reach 20/62; maximum record lines are 468/1,309
bytes and update lines 1,614/1,621 bytes. See the
[population fingerprints](scientific_compression/evidence/population.json).
These are neither identical-complexity graphs across sizes nor the earlier tiny
v1 references. G1 creates/commits the frozen graph;
G2–G4 publish exact no-op estimate receipts through the actual durable Growth
path. Publication includes restoring the committed runtime. Patch-fixture
preparation is outside its timer. G1 excludes growing the graph but includes
scientific/checkpoint writing and independent initialization validation.

The following table uses G4. Logical bytes include the readable generation
descriptor. Allocation uses `st_blocks * 512`, counting each `(device, inode)`
once, including directory blocks; hard links are not double-counted. These are
one selected checkpoint/scientific boundary, not total workspace allocation.

| Actual bytes | 1k raw | 1k compressed | 5k raw | 5k compressed |
|---|---:|---:|---:|---:|
| Scientific logical | 1,030,751 | 30,735 | 6,065,482 | 168,573 |
| Scientific allocated | 1,040,384 | 40,960 | 6,078,464 | 180,224 |
| Checkpoint logical | 86,560 | 86,560 | 436,785 | 436,785 |
| Checkpoint allocated | 122,880 | 122,880 | 495,616 | 495,616 |
| Combined allocated | 1,163,264 | 163,840 | 6,574,080 | 675,840 |
| One retained reader index | 1,355,776 | 1,355,776 | 7,761,920 | 7,761,920 |
| Combined with that index | 2,519,040 | 1,519,616 | 14,336,000 | 8,437,760 |

Checkpoint-plus-science savings are **85.9% / 89.7%**; including one derived
reader index gives **39.7% / 41.1%**. Do not apply the former percentages to
the complete run. Models, authority, receipts, old generations and additional
derived indexes are outside this pair. The scientific snapshot has three files
and four inodes including its directory in both encodings. Checkpoints have
7/19 files and 13/25 inodes including directories at 1k/5k, unchanged.

Temporary allocations were sampled every 20 ms during separate G1–G3 runs.
These are observed high-water **lower bounds**: short-lived journal allocations
can fall between samples. Private file allocation includes staging science,
checkpoint and validation indexes; directory blocks are excluded in this sampler.

| Observed private high-water bytes | 1k raw | 1k compressed | 5k raw | 5k compressed |
|---|---:|---:|---:|---:|
| G1 initialization | 2,564,096 | 1,564,672 | 14,397,440 | 8,499,200 |
| G3 repeated publication | 3,825,664 | 2,826,240 | 21,630,976 | 15,794,176 |

Connection tracing finds two private main SQLite builds for initialization and
three for each repeated publication, in both encodings. A `.building` file
renamed to a completed index is the same database, not another build. Journal
names and open-unlinked allocations are retained separately in the raw evidence.
There is **no uncompressed scientific JSONL intermediate**. Repeated publication
still pays for the existing source/output/final-capsule validation indexes; this
task does not remove those integrity boundaries.

The derived index remains necessary for selected-node lookup, deterministic
ordering, topology validation and disk-backed Dataset family context. It stores
decoded data and therefore has the same size under both source encodings. Its
default XDG cache persists per source identity/path with no eviction. Publication
uses existing temporary caches and removes them on completion; Dataset and
Reevaluation accept a caller-owned `reader_cache_dir`, which can be an existing
temporary-directory lifetime when reuse is unnecessary. That bounds cache
lifetime at the cost of rebuilding. No reader-cache garbage collector was added.

## Time, CPU, RSS and I/O

G4 runs below are fresh serial processes **without the allocation sampler or
cProfile**. They use the same installed production files as G1–G3. These are
single matched observations, not confidence intervals. “Cold” means a new SQLite
index; the OS page cache was not flushed. `/proc` reports zero physical read
bytes for these operations; logical read traffic is reported separately.

Each timing cell is raw → compressed seconds. RSS is kernel process-lifetime
peak in MiB, including imports and resident graph, not just compressor memory.

| Operation | 1k wall | 1k CPU | 1k peak RSS | 5k wall | 5k CPU | 5k peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Repeated durable publication | 2.676 → 2.320 | 1.279 → 0.970 | 303.2 → 305.5 | 8.490 → 7.022 | 3.691 → 3.128 | 319.0 → 321.9 |
| Cold index plus full indexed visit | 1.301 → 1.355 | 0.914 → 0.974 | 290.6 → 291.1 | 16.592 → 16.173 | 15.260 → 14.838 | 290.6 → 293.0 |
| Reused index plus full indexed visit | 0.066 → 0.058 | 0.049 → 0.046 | 290.1 → 290.5 | 0.232 → 0.224 | 0.214 → 0.206 | 289.7 → 289.1 |
| Decoded streams and hashing only | 0.0007 → 0.0017 | 0.0006 → 0.0016 | 289.7 → 290.0 | 0.0090 → 0.0083 | 0.0080 → 0.0082 | 288.6 → 290.6 |
| Capsule/checkpoint recovery and proof check | 1.168 → 1.131 | 1.150 → 1.113 | 300.9 → 301.3 | 7.029 → 6.749 | 7.011 → 6.731 | 315.9 → 316.8 |

Constructor-only cold initialization is 1.245 → 1.300 s / 16.411 → 15.980 s;
constructor-only reuse is 0.0105 → 0.0092 s / 0.0384 → 0.0380 s. The visit
columns above additionally reconstruct/hash all indexed scientific contracts.
The streams-only row does not perform JSON/canonical validation; cold index
construction does. Differences at millisecond scale are measurement noise,
not evidence that decompression is intrinsically faster than reading raw bytes.

Bounded Dataset extraction never loads the full reference in its timed process.
For the five supported v2 policies, median total extraction/JSONL time is
0.1386 → 0.1353 s (64 rows; 462 → 473 rows/s) at 1k and 0.7504 → 0.7135 s
(374 rows; 498 → 524 rows/s) at 5k. Dataset peak RSS is about 290–292 MiB in
both encodings. Per-policy CPU, I/O, timings and hashes are preserved in
[native observations](scientific_compression/evidence/resources-native.json).
The same process executes policies sequentially, so its RSS high-water includes
earlier policies; the full-reference parity process is separate.

G1 initialization, measured with the allocation observer, is 3.996 → 4.047 s
wall / 2.969 → 3.077 s CPU at 1k and 40.993 → 40.358 s wall / 38.276 →
37.618 s CPU at 5k. Peak RSS is 307.1 → 310.2 / 373.8 → 375.8 MiB.
The observer overhead is not mixed into the native timing table. Initial export
encodes N new states; writer and independent initialization each validate N
canonical states (2N codec loads total). Every observed repeat (G2–G4, plus
profiled G5 at 1k) encodes zero new states and copies N validated records with
**zero old state decodes**.
Cold indexes decode N states; reused indexes and indexed Dataset decode zero.
Checkpoint recovery uses the existing incremental state codec separately.

At G4 repeated publication, logical read traffic (`rchar`) is 50,419,334 →
11,245,826 bytes at 1k and 336,398,846 → 104,577,923 at 5k. Kernel write bytes
are 6,950,912 → 5,959,680 and 53,538,816 → 47,640,576. At 5k both runs
also report 10,842,112 cancelled write bytes; these are retained separately,
not subtracted to invent a net disk claim. Hash-call counts stay 951 / 1,551;
hashed physical bytes fall from 41,971,968 → 8,982,848 / 225,688,106 → 31,101,655.
Compression reduces byte traffic, not the number of historical validations.

## Profiling attribution and interpretation

Clean cProfile runs use the existing single-threaded measurement entry point,
without the allocation sampler. Their absolute runtimes include profiler
overhead and must not be compared with the unprofiled table. See the filtered
[clean profiles](scientific_compression/evidence/profiles-clean.json), which
also identify the retained original `.prof` files by path and SHA-256.

At 5k, raw/compressed index building takes 30.07/31.35 profiled seconds:
canonical load is 13.30/13.88 s and canonical redump is 14.11/14.71 s, each
5,013 calls. Existing shard iteration is 0.742/0.846 s; 45,142 SQLite executes
take 0.118/0.132 s; JSON dumps take 0.149/0.163 s. Compressed block decoding
plus its integrity checks totals 0.0165 s. These cumulative categories overlap
(especially shard iteration and its nested parsing); they are not summed into
an artificial percentage breakdown. Canonical scientific validation remains the
dominant cold cost and has not been weakened.

For the 1k repeated publication, the row writer takes 0.0071/0.0083 profiled
seconds over 2,008 calls; compressed frame finish adds 0.00016 s and decoded
block iteration 0.0089 s. Existing file hashing takes 0.0294/0.0136 s, 30
SQLite commits 0.545/0.574 s and 160 fsyncs 0.296/0.312 s. The publication
still revalidates actual dependencies and does not substitute cached hashes.

For 1k `selected_value_v2/none`, complete Dataset writing is 0.333/0.334
profiled seconds. Family context costs 0.179/0.180 s; repeated indexed iteration
and node reconstruction cost about 0.17 s, including 5,015 node reconstructions.
There are 22,085 SQLite execute calls (0.029/0.030 s). Indexed extraction does
no Morpion state replay or decompression. These existing multiple passes and
queries, rather than compression, explain its cost. They were not refactored.

Initial cProfile observations with the disk-sampling thread produced distorted
cross-thread attribution under Python 3.13. They remain in scratch for audit
but are explicitly excluded from these results; clean profiles were recollected
serially. Initial allocation observations with zero samples (an iterator's own
closed `/proc/self/fd` entry) were also discarded and recollected after fixing
the research sampler. Neither issue changed production or scientific data.

## Complexity and production size

Let S be decoded scientific bytes, C the compressed bytes, N nodes, E branches,
L the largest serialized row/history and G retained generations. Compression
and decompression are O(S) work with fixed windows plus O(L) existing row
serialization/parser memory. Output is O(C), worst-case O(S); compression ratio
depends on actual graph/history entropy. No O(N) Python structure is added by
this change. Existing export has the resident graph plus an O(N) reference list;
existing SQLite topology/selection/family data remain O(S + N + E) disk.
Cold canonical replay/redump costs depend on the sum of replay-history work,
not just N. Repeated publication still writes complete current values/topology
and reads/copies current immutable records: O(S), not O(new nodes).

Retaining every generation still costs the sum of checkpoint, C and derived
index sizes over G. Exact receipts/history and capsule validation are unchanged.
Nothing in a 5k-node result proves record-scale total RAM, bounded historical
disk or history-independent publication. Long histories, high-degree rows,
checkpoint selector payloads and current manifest allocations remain limits.

Production delta is **283 lines added, 35 removed, +248 net**, six modules.
The only new module is the 192-line stream helper. Its writer owns incremental
byte commitments/compression; its generator proves bounded framing and EOF;
its small RawIO adapter supplies the existing parser interface. The remaining
56 net lines adapt descriptors, byte opening, producer encoding inheritance and
the existing validation-code identity. No shared-block store, decompression
cache, second reader/index, scientific policy or authority was introduced.

## Reproduction commands

Run from the new worktree with its isolated installed environment. Export the
scratch TMPDIR/SQLITE_TMPDIR, XDG_CACHE_HOME, plotting/MLflow paths and
`CHIPIRON_TEST_INSTALLED=1` as recorded in the task environment. Use a new
secondary-disk destination every time; these commands create synthetic data.

```sh
PY=/home/pompote/oldata/victor/morpion-scratch/science-compression-20261025/venv/bin/python
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
PAIR=/home/pompote/oldata/victor/morpion-scratch/NEW-compression-comparison

# Run serially for each size (1000 / 5000), raw first then compressed.
# Omit --encoding only for the raw initial call.
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" initial --count 1000 --encoding jsonl-zstd-1
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" repeat --count 1000
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" repeat --count 1000
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" cold --count 1000
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" reuse --count 1000
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" dataset_stream --count 1000
"$PY" -m docs.research.scientific_compression.measure "$PAIR/1k-jsonl-zstd-1" dataset --count 1000
"$PY" -m docs.research.scientific_compression.compare "$PAIR" "$PAIR/parity.json"

# For native timings and clean profiling, use the original entry point without
# the sampler. --profile selects a separate reader cache; use a new generation
# before another cold profile. Also run streams and recover in fresh processes.
"$PY" -m docs.research.durable_closed_snapshot.measure "$PAIR/1k-jsonl-zstd-1" repeat --count 1000
"$PY" -m docs.research.durable_closed_snapshot.measure "$PAIR/1k-jsonl-zstd-1" cold --count 1000 --profile

# Repeat raw/compressed and opening/--endgame, each in a new destination.
"$PY" -m docs.research.e2e_bootstrap.reproduce "$PAIR/e2e-compressed" --scientific-encoding jsonl-zstd-1
```

Dataset output filenames now include the generation, so later native/profile
observations cannot overwrite earlier JSONL evidence. Earlier G3 oracle and
indexed outputs were retained and copied to these explicit names only after
their original recorded hashes matched. The separate bounded-only G3 runs'
hashes match those exact bytes as well. The comparison was rerun after all G4/G5
measurements and produced the identical parity report. The
[output preservation record](scientific_compression/evidence/generation-output-preservation.json)
identifies the original files and verified copies. The allocation wrapper now
rejects `--profile`, directing profiling to the existing thread-free entry point.

Raw measurements and logs remain under the task scratch root, not in historical
experiment directories. Committed evidence includes the resource observations,
source fingerprints, profile attribution, exact comparisons, E2E summaries and
test identities. Original binary profiles and full synthetic workspaces remain
on the secondary disk, with their paths and hashes recorded for review.

## Validation results

The executable production source is unchanged since `ae5415c2`; normal tests
are unchanged since `223f76d5`. All 697 production/test/script file hashes were
rechecked after validation. The rebuilt review wheel contains exactly the same
478 Python files as the installed package used by tests and experiments.

| Check | Result |
|---|---|
| `make test-fast` | 152 passed, 123.01 s |
| Final focused scientific/fault cases | 98 passed, 42 deselected, 111.64 s |
| New compression module | 29 passed, 21.37 s |
| `make test-pr` | 2,737 passed, 2 CUDA skips, 1,141.54 s |
| `make test-full` | **2,901 passed, 2 CUDA skips**, 1,825.88 s |
| Ruff check and format | Pass, 759 source/research files formatted; generated build directory excluded |
| Strict static-analysis ratchet | Baseline 209, current 206, **zero new diagnostics** |
| Wheel and sdist build | Pass; rebuilt wheel source equals tested installed source |
| Installed distribution contracts / toolchain consistency | Pass |

FULL uses the existing `not integration and not external_data` selection and
two pytest processes. Its four standard excluded checkmate cases are listed in
the evidence; the two skips require CUDA. No Morpion scientific/recovery case
was excluded. Benchmarks and genuine E2E bootstrap workers were serial, with
no distributed bootstrap execution. Overall FULL line-and-branch coverage is
80.47%; this aggregate is not a substitute for the exact scientific assertions.

The observed FAST test identities are a subset of PR, which is a subset of
FULL. FULL executes 164 additional cases. Every selected test has exactly the
expected setup/call/teardown reports, including legitimate setup skips, and both
pytest workers report identical collections. The complete affected matrices
passed: 64 closed-snapshot cases, 29 compression cases, 49 activation-crash cases
and 47 closed-publication crash/recovery cases (34 of those are FULL-only).
See [CI identities and summary](scientific_compression/evidence/ci/summary.json).

The 46 activation boundary cases span raw/compressed pre-load, inference,
backup/selector refresh, checkpoint shards/manifest, science writing, actual
partial compressed output and complete frame, validation, file/directory fsync,
immutable rename, partial/before/after authoritative commit and discovery repair.
Abrupt child termination bypasses cleanup. Pre-commit recovery retains complete
M1; post-commit recovery retains complete M2, with exact retries. The publication
matrix also checks repeated recovery interruption, fsync failures, cursor/receipt
continuation and incomplete private output. No unfinished frame becomes authority.

During development, two FAST failures exposed children using the previously
installed package; installing/verifying the current wheel fixed that harness
mismatch. One matrix failure exposed a crash hook limited to `.jsonl` filenames;
the hook now covers `.jsonl.zst` and the complete matrix passed. The partial-frame
seam now writes and flushes half the actual compressed bytes before exit. Research
setup retries (relative output path rejected by ownership checks; obsolete output
filename assumptions) and the rejected profiler/sampler runs are retained in the
scratch log inventory. None is presented as passing final evidence.

The [raw-log inventory](scientific_compression/evidence/raw-log-inventory.json)
binds retained successful and unsuccessful development logs and binary profiles.
The test execution evidence records outcomes rather than relying on collection
counts; package, static and formatting logs are retained alongside it.

## Review and adoption decision

The measured reduction justifies this narrow optional encoding: approximately
2–3 MiB higher observed publication RSS, bounded framing memory, and no material
publication/Dataset regression in matched 1k/5k observations. This does not remove
the resident search graph, canonical cold-validation cost or derived index disk.
Keep the change local on the pinned stack until that stack has a clean review
base; review the six-module production delta separately from research evidence.
No PR, merge, package publication, production activation, historical migration or
historical artifact deletion was performed.

The protection audit found zero changes across all 63 prior registrations (45
existing directories and 18 already-missing registrations): HEADs, registrations,
working-tree status and binary-diff hashes match the captured baseline. See
[protection audit](scientific_compression/evidence/protection.json). The original
dirty research checkout and all prior local commits/evidence remain intact.
