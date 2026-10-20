# Scientific v2 reader performance investigation

This investigation starts at exact local integration
`02212bc6130205b7d2ef898c7b1eb0f7d384f7aa`. Worktree:
`/tmp/chipiron-scientific-v2-reader-profile`, branch
`perf/morpion-scientific-v2-reader-profile`. The original integration worktree,
existing PR branches and dirty research checkout are unchanged. All artifacts
used as inputs were newly generated synthetic data. No gen38, historical data,
bootstrap/distributed worker, production activation, remote PR or merge is used.

The original [integration report](morpion_scientific_v2_reader_integration.md)
and its resource/parity evidence remain the baseline. Its exact sources are:

| Dependency | Commit |
| --- | --- |
| PR #75 reader | `79693ce2dc59c6d6ccb951951c9ba75fa69057f3` |
| PR #76 Dataset | `5c8906ba0d520c63d821e14a2d52530f8bed2307` |
| PR #77 investigation | `67a7cfd1955f43d5947af01dbb324fb5efd93a0d` |
| PR #80 scientific v2 | `bfd8f2ab05f017ffda5e14af5dbf6e1f1afa3fde` |
| Local dependency merge | `f09a038b2589e124ffbb96a0d9936081692b3195` |
| Integration base | `02212bc6130205b7d2ef898c7b1eb0f7d384f7aa` |

## Controls and measurement method

The existing `benchmark_morpion_scientific_reader.py`, fixture helper,
`StageMeasurement`, Linux process/I/O counters and standard cProfile/pstats are
reused. No reader, parser, persistent index or profiling framework is added.
Profile-only SQLite and existing-parser hooks follow the existing writer
benchmark's `unittest.mock` operation-counter pattern. They separate SQL literal
execution/commit costs and time spent advancing each shard iterator. They retain
only fixed SQL literals and counters, never node parameters/populations.

1k and 5k matched fixtures and controlled before/after comparisons precede 10k.
Every measured mode runs in a fresh process, serially. Fixture construction,
cProfile attribution runs, unprofiled throughput/RSS samples, and full-reference
parity processes are separate. Background validation runs do not overlap the
resource benchmark sequence. Peak RSS is actual process-lifetime `getrusage`
high water, includes runtime imports, and excludes no baseline by subtraction.
No neural model weights are loaded; the existing deterministic evaluator double
exercises actual state decoding, cursor wrap and patch serialization.

The same live fixture nodes feed both existing export writers. Actual immutable
reference bytes, node order and current child order are compared with the
existing bounded parser. Ordered node/reference SHA256, source-file SHA256,
producer identity and replay-length histograms are retained. Reference equality
means identical deterministic JSON encoding of parsed reference payloads;
whole shards have different framing/whitespace. No replay-order, value or
provenance transformation makes them match. For this controlled fixture one
valid real board exists per graph depth, repeated under distinct
IDs. It is a resource/contract fixture rather than a claim about transposition
uniqueness, action-labelled search edges or certified game optima. No performance
claim depends on memoizing those duplicate states.

V1 and v2 have different wire schemas and scientific fields. In particular v1
still has its historical lossy effective/target/provenance rules and stale
incoming parents in generation 2. This investigation preserves and records those
differences. Matching replay references/populations controls geometric work; it
does not assert cross-format scientific equivalence. Earlier tiny-reference v1
benchmark throughput is excluded from the comparison.

Cold means an absent index for the selected generation; OS page caches are not
flushed. Generation 1 and 2 use the same cache directory per implementation and
format, with both source generations published before measurements. Generation
2 changes a direct value and adds an incoming edge, with zero new immutable
nodes. This measures generation-specific rebuilds without a concurrent writer.
Full-reference processes are the explicitly requested oracle comparison; only
those offline processes may materialize the full snapshot. Indexed cold, reused
and Dataset measurements forbid the full v2 loader.

## Attribution

cProfile cumulative timings include nested work and must not be summed across
parent and child functions. SQL operation timings cover execute/DDL/explicit
commit calls; index maintenance is part of insertion, implicit DDL commits are
part of `executescript`, and SELECT cursor fetching also appears separately in
cProfile. Parser timings exclude consumer work between generator yields.
Profile overhead is excluded from the before/after throughput tables.

Initial baseline profiles establish the bottleneck at both controlled sizes:

| Profile | 1k total s | Dominant work | 5k total s | Dominant work |
| --- | ---: | --- | ---: | --- |
| Cold v2 constructor | 6.844 | decode 2.980 + canonical dump 2.932 (86.4%) | 50.611 | decode 23.799 + canonical dump 23.464 (93.4%) |
| Indexed Dataset | 6.071 | codec load 5.234 (86.2%) | 41.196 | codec load 37.518 (91.1%) |

Cold decoding and canonical redumping each replay legal moves through Atomheart.
The redump computes the authoritative canonical played-move order rather than
copying or sorting encoded actions heuristically. Each cold generation checks
all immutable records, including records created in older generations. The
current reader does not import their previous generation's validation proof.

Dataset has no canonical redump, even before this optimization. Its generic
normalizer first validates the replay by decoding, then performs
`checkpoint_payload_to_jsonable`. The latter is JSON normalization, not a new
scientific or canonical replay policy. The disk-backed Dataset summary and final
writer each traverse rows, causing **2 × saved row count** replay decodes:
1,632 at 1k and 8,154 at 5k. Matched v1 Dataset profiles show the same work:
4.626s/5.257s at 1k and 38.725s/42.175s at 5k in codec load.

Full function/caller profiles and SQL/parser operation counters retain the
following attribution, without assigning overlapping costs twice:

- Source hashing: `_source_identity`, `hashlib.file_digest`, and the existing
  surrogate preflight; cache reuse also verifies `_cache_digest`/`quick_check`.
- Manifest validation: `_source_kind`, `_scientific_manifest`,
  `_prepare_scientific_source`, and streamed creation-index checks. No complete
  creation map is materialized by the indexed path.
- JSONL parsing: advancing existing `iter_shard_rows` for `records` and `updates`
  separately, including schema envelopes, duplicate-key checks and full tails.
- Immutable indexing: canonical load/dump plus `INSERT INTO records`, record JSON
  encoding and creation-index insert/checks. There is no cross-generation skip.
- Update indexing: `_validate_update`, `INSERT INTO updates`, current-edge TEMP
  inserts and reciprocal/closed-DAG validation. Every mutable field is read anew.
- SQLite insertion/index maintenance: exact SQL-literal call counts/times;
  schema/order-index creation and topology TEMP construction appear in DDL.
- Transaction commits: explicit `Connection.commit` plus separately labelled DDL
  scripts. Commit cadence and durability/publication behavior are unchanged.
- Dataset reconstruction: five export-order node passes (family inputs, selected
  children, metadata, summary and output), `_load_node`, `_scientific_node`,
  per-node value-contract validation, JSON conversion and keyed SQL lookups.
- Family/target computation: `DiskDatasetFamilyContext.__init__`, representative
  traversal, `_build_family`, `family_node_target`, `adjust_row` and `summary`.
  Policy arithmetic and iteration/summation order are unchanged.
- Serialization: the writer's inclusive time contains row generation. Actual
  JSON encoding, `_row_to_dict` and `_write_jsonl_record` are reported separately.

The detailed generation-2 observations quantify the smaller components. These
are overhead-bearing cProfile function samples and separate lighter SQL/parser
timer samples, not an additive partition of unprofiled wall time:

| Component | 1k seconds | 5k seconds | Scope |
| --- | ---: | ---: | --- |
| Source identity hashing | 0.0082 | 0.0374 | Two `_source_identity` calls in cold constructor; cProfile |
| Manifest validation | 0.0020 | 0.0069 | Both scientific manifests; cProfile, nested in source preparation |
| Immutable JSONL parsing | 0.0144 | 0.0736 | Existing parser only, one row at a time; operation timer |
| Update JSONL parsing | 0.0559 | 0.2640 | Existing parser only; operation timer |
| Record SQL insertion/index maintenance | 0.0059 | 0.0333 | N complete record inserts; operation timer |
| Update SQL insertion/index maintenance | 0.0076 | 0.0356 | N complete update inserts; operation timer |
| Creation-map TEMP insertion | 0.0049 | 0.0088 | N keyed entries; operation timer |
| Main schema/order-index DDL | 0.1485 | 0.1471 | Full initial DDL script, including implicit commits |
| Topology degree/queue DDL | 0.0310 | 0.0374 | Full construction script |
| Explicit commits | 0.2547 (13 calls) | 1.3053 (77 calls) | Same transaction cadence and durability |
| Per-value update validation | 0.0369 | 0.1857 | N `_validate_update` calls; cProfile |
| Full topology validation | 0.0428 | 0.1014 | Includes its nested SQL/DDL; cProfile |

Canonical decode/redump still dominate. Full generation-1 and generation-2
profiles each observe exactly N loads and N dumps; the second generation
repeats the entire immutable validation population. JSONL header-only generation
2 record shards contain zero new rows. Input fingerprinting and parsing remain
bounded; no complete shard or creation map is loaded to obtain these counters.

| Dataset component (cProfile) | 1k before/after seconds | 5k before/after seconds |
| --- | ---: | ---: |
| Codec load | 4.7403 / 0 | 37.7348 / 0 |
| JSON normalization | 0.0190 / 0.0168 | 0.1057 / 0.0979 |
| Node/value reconstruction | 0.2362 / 0.2110 | 1.1535 / 1.0616 |
| Family context construction | 0.2250 / 0.2160 | 1.1296 / 1.0888 |
| Target-policy function | 0.0027 / 0.0020 | 0.0127 / 0.0099 |
| JSONL record serialization | 0.0297 / 0.0203 | 0.1451 / 0.1120 |

These nested components overlap. Normalization retains exactly 1,632/8,154
primitive calls before and after (26,898/163,461 including recursion). Target
function call counts also remain 1,632/8,154. Keyed node lookups remain
5,000/25,000, with execute-only timer totals 0.0276/0.1482 seconds before and
0.0241/0.1320 after; cursor fetching and JSON/schema work are separate. Thus
reconstruction/query improvements are possible remaining work, rather than
part of this measured patch. The raw profiles retain complete caller data.

## Small implemented optimization and integrity contract

Only the experimental indexed v2 Dataset path supplies a private normalization
callback to the existing row iterator/converter. It retains the exact existing
JSON conversion and omits only a codec replay whose validity was already proved.
No new cache, tree, schema, snapshot, row policy or persistent table is added.

The proof is established before that callback is selected:

1. The existing view accepts only explicit supported v2 kind/version/contract,
   producer/publication identity and pinned canonical decoder contract.
2. A cold build hashes its sources and performs canonical decode/redump checks
   on every immutable reference. Unsupported, noncanonical, malformed or changed
   sources cannot publish a completed index.
3. Reuse validates exact source identities, SQLite integrity and the completed
   record/update payload digest. A corrupt cached reference causes a rebuild.
   Published indexes are immutable to consumers: the main database is read-only
   and builders publish complete replacement files rather than editing an open
   generation index in place.
4. Nodes come from that verified index. Full per-value update/contract validation
   still runs during reconstruction. The complete JSON payload normalization
   remains; no target/value/source/certainty field is guessed or replaced.
5. The existing source guard still runs at the final row boundary before atomic
   output replacement. Interrupted output and source changes preserve the prior
   complete output. The callback is code-owned, not an input metadata flag.

Full/flat/v1 sources retain the default validating normalizer. The authoritative
full v2 reference loader is unchanged. The sharded reader itself, all canonical
cold checks, source verification and cache format are unchanged. Tests verify
four cold records cause exactly four codec loads and four dumps, then all nine
Dataset policies and repeated passes cause zero additional loads/dumps. They
also verify corrupted SQLite reference payloads rebuild and unverified invalid
replays still fail even when metadata claims validation.

## Results and parity

Unprofiled single samples on identical newly generated inputs:

| Nodes | Saved rows | Baseline Dataset s | Optimized s | Baseline rows/s | Optimized rows/s | Speedup | Peak MiB before/after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 816 | 2.506 | 0.336 | 325.7 | 2426.7 | 7.45× | 253.32 / 253.37 |
| 5,000 | 4,077 | 20.055 | 1.677 | 203.3 | 2431.6 | 11.96× | 255.47 / 255.58 |
| 10,000 | 8,154 | 46.863 | 3.274 | 174.0 | 2490.7 | 14.31× | 256.85 / 256.93 |

Dataset measurements include reopening/validating the existing index, family
context, metadata/summary, atomic JSONL output and final source guard. Cold
initialization is a separate measurement. The Dataset speedup is not a cold
initialization speedup or an end-to-end cold-worker throughput claim.

| Nodes | V1 indexed cold g1/g2 s | V2 indexed cold g1/g2 s | V1 reuse g2 s | V2 reuse g2 s | Optimized v2 cold/reuse s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.398 / 0.399 | 3.162 / 3.293 | 0.007 | 0.029 | 3.240 / 0.030 |
| 5,000 | 1.521 / 1.522 | 24.198 / 23.646 | 0.025 | 0.128 | 24.212 / 0.131 |
| 10,000 | 3.917 / 3.392 | 54.545 / 55.953 | 0.048 | 0.257 | 57.244 / 0.255 |

Generation 2 has zero new immutable nodes. It nevertheless rebuilds a full
index. Both profile sizes record N canonical loads and N canonical dumps for
each generation, even with the preceding index in the same directory. Source
hashing and cache reuse checks remain mandatory. The cold/reuse reader code is
byte-identical to `02212bc6`; timing differences between repetitions are noise,
not a claimed optimization.

| Nodes | V1 full init s | V2 full init s | V2 full Dataset s | V1 indexed Dataset s | Full v2 init/lifetime peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.024 | 2.678 | 1.116 | 2.518 | 261.16 / 261.29 |
| 5,000 | 0.128 | 22.965 | 9.070 | 20.556 | 305.64 / 306.51 |
| 10,000 | 0.239 | 52.864 | 21.097 | 45.404 | 361.83 / 363.46 |

Matched v1 and baseline v2 indexed Dataset costs track each other. The full
reference Dataset creates rows once; the bounded summary/output implementation
has two row passes. V1 full initialization does not provide v2's canonical and
scientific validation guarantees. The smaller full-v1 init time therefore
cannot be presented as an equivalent validation algorithm.

| Nodes | Replay length min/max | Mean replay moves | Distinct real references |
| ---: | ---: | ---: | ---: |
| 1,000 | 0 / 9 | 7.9870 | 10 |
| 5,000 | 0 / 12 | 10.3644 | 13 |
| 10,000 | 0 / 13 | 11.3631 | 14 |

Both formats have exactly this distribution and deterministic encoded reference
bytes. New v2 source-file fingerprints also reproduce the original integration
fixture fingerprints; comparison never reads those older fixture directories.
The original source/export bytes remain unmodified.

The 27 full-reference/indexed policy-size output pairs pass literal buffered
byte comparison: **117,423 paired rows**, including complete JSONL headers and
metadata. Every scientific node field, value contract/certainty, current DAG
link, export order, source reference, full provenance and wrapped Reevaluation
window/patch digest matches. Single-policy before/after measurements also have
identical output SHA256. The 1k/5k/10k fixtures retain 41/214/435 incompatible
backup-certainty nodes; nothing is normalized away to obtain parity.

The complete installed-wheel local CI suite passes **1,714 tests, 2 skipped**
in 438.55 seconds. Four workflow marker exclusions and two CUDA skips match the
baseline selection rules. The full static ratchet has **new 0** (baseline 209,
current 206, removed 3). Toolchain, Ruff check/format, sdist/wheel build,
installed-distribution smoke guard and diff checks pass. The inherited safety
fixture redirects only optional default leaderboard writes into pytest temp
folders; HOME, record logic, assertions and test selection are unchanged.
No hosted PR workflow is claimed.

Reproducible evidence:

- [Resources, input fingerprints, CPU, I/O, disk and RSS](benchmarks/morpion_scientific_reader_performance_20261020/resources.json).
- [Function-level cProfile attribution](benchmarks/morpion_scientific_reader_performance_20261020/profiles.json).
- [SQL/parser phase counters and timing](benchmarks/morpion_scientific_reader_performance_20261020/operations.json).
- [Every policy's exact byte parity and scientific/window digests](benchmarks/morpion_scientific_reader_performance_20261020/parity.json).
- [CI command, selection, source commits and evidence hashes](benchmarks/morpion_scientific_reader_performance_20261020/validation.json).

Only final successful samples are used. Setup corrections and the initial empty
operation counters on Python 3.13 are recorded in the resource evidence; valid
cProfile function results were retained and operation timers rerun separately.
No timing from failed benchmark setup is reported as a successful sample.


## Reproduction

The benchmark CLI is the same tool used by the integration, extended with
matched-format preparation, generation/output/cache selection, full reference
measurement and standard profiling. For a new root, begin at 1k then 5k:

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
export QT_QPA_PLATFORM=offscreen
python scripts/benchmark_morpion_scientific_reader.py --action prepare --matched-v1 --size 1000 --root /tmp/new-morpion-perf-1k
python scripts/benchmark_morpion_scientific_reader.py --action audit-inputs --size 1000 --root /tmp/new-morpion-perf-1k
python scripts/benchmark_morpion_scientific_reader.py --action reference --size 1000 --root /tmp/new-morpion-perf-1k --output-dir /tmp/new-morpion-perf-1k/reference-v2-g2
python scripts/benchmark_morpion_scientific_reader.py --action cold --size 1000 --root /tmp/new-morpion-perf-1k --output-dir /tmp/new-morpion-perf-1k/current
python scripts/benchmark_morpion_scientific_reader.py --action reuse --size 1000 --root /tmp/new-morpion-perf-1k --output-dir /tmp/new-morpion-perf-1k/current
python scripts/benchmark_morpion_scientific_reader.py --action dataset --size 1000 --root /tmp/new-morpion-perf-1k --output-dir /tmp/new-morpion-perf-1k/current
python scripts/benchmark_morpion_scientific_reader.py --action parity --size 1000 --root /tmp/new-morpion-perf-1k --output-dir /tmp/new-morpion-perf-1k/current
```

For v1 add `--format v1` and use separate output/cache/reference directories.
For per-generation measurements add `--generation 1` then `2`, using the same
explicit `--cache-dir` and distinct output directories. To run the exact
baseline, point PYTHONPATH at a read-only `02212bc6` source checkout, retaining
the benchmark helper on `.`; disable bytecode writing as above. Baseline and
optimized measurements need separate fresh cache directories.

Use `--profile` only in fresh, separate profile output directories; inspect the
`.prof` files with `python -m pstats`. Use `--operation-counts` for the lighter
SQL/parser timers, also separately. Reference output location can be explicit
with `--reference-dir`. CLI refuses existing resource JSON outputs and limits
roots/output/cache/reference locations to the newly prepared synthetic root.
Exact invocation lists are retained with the evidence archive. Measure 10k only
after the controlled 1k/5k comparisons pass.

## Deferred cold-build design

Cold initialization remains dominated by mandatory canonical validation. A
bounded in-memory cache of repeated fixture boards would disproportionately
improve this artificial population and would not establish benefit for a search
with unique states; it is not implemented. Removing validation on the strength
of an unsigned producer declaration or file stat would weaken verification.

A separately reviewed extension can reuse validated immutable records **inside
the existing index framework**. A certificate must bind the explicit format and
scientific contract, complete producer identity, canonical validator/decoder
version, exact immutable shard content SHA256, count/creation generation and
cached record payload digest. Source shard hashes must still be checked. An
imported completed index must pass integrity/digest checks, and copying must be
bounded or SQL-backed. New/changed shards require full canonical validation.
Mutable generation updates, current parents/children, values, certainties,
creation-index consistency and graph topology must always be revalidated.

Catalog publication changes must not be confused with immutable shard changes:
the old generation's full fingerprint may be stale while an exact immutable
shard remains reusable. This requires an explicit validation/schema epoch and
failure tests for mixed producers/contracts, forged or corrupted certificates,
partial imports, concurrent publication, new/deleted nodes, source mutation,
changed decoder versions and valid SQLite payload corruption. V1 caches cannot
certify v2. No second SQLite indexing framework is needed. No speed estimate for
that unimplemented design is presented as a measurement.

After the Dataset change, remaining reconstruction, value-contract checks,
repeated keyed SQL and family processing are visible in the profiles. Joining
export-order payloads directly or safely staging reusable row representations
may be worth a separate measured change. Value/contract checks and target
arithmetic are retained in this patch.

## Integration and activation

The pinned base still joins sibling #75/#76 and #77/#80 stacks. All performance
commits stay local; no misleading PR against one sibling is opened. Review the
small production change separately from profiling/evidence. The profiling/tools
commit is `90360dd3f91f1c339111e1760ee5adf6212629be`; the production optimization
and regression tests are `ae35addb85ce75bd2438214fbadfe22ad677c307`. Once both stacks
have a clean shared base, integrate the existing v2 reader change, then apply
these performance commits and rerun parity/CI on that exact ancestry. Do not
rewrite or rebase the existing PR branches.

All scientific v2 routing remains explicitly opt-in and disabled. Node exactness
versus certainty of the selected training target remains a hard production
activation prerequisite. A faster reader and exact oracle parity do not solve
that policy question or certify production scientific readiness.
