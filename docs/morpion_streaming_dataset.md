# Bounded sharded Dataset extraction (opt-in)

This change is stacked directly on Chipiron PR #75 at
`79693ce2dc59c6d6ccb951951c9ba75fa69057f3`, targeting
`perf/morpion-streaming-reevaluation`. The isolated worktree is
`/tmp/chipiron-streaming-dataset`, branch `perf/morpion-streaming-dataset`.
The research checkout and PR #75 worktree/history are unchanged.

## Reader reuse and allocation audit

`run_pipeline_dataset_stage(..., use_streaming_sharded_reader=True,
reader_cache_dir=...)` uses the existing `ShardedTrainingGenerationView`.
`node_count()` supplies counts, `load_nodes()` reconstructs the root, and
`iter_nodes_in_export_order()` supplies every repeatable pass. There is no new
shard parser, reader index, index schema, or independent SQLite database.
The completed PR #75 index remains read-only and reusable by Reevaluation.

`IndexedDatasetSource` only adapts that view to the existing extraction APIs:
its `nodes` are repeatable and sized, never a tuple of all nodes. No
`TrainingTreeSnapshot` is constructed on this path. Existing node conversion,
state-reference validation/canonicalization, filter and target-source selection
remain shared with legacy extraction. The metadata pass deliberately retains
legacy counting order, which differs from the emitted-row validation order near
`max_rows`.

The reader lends its existing connection for three disposable SQLite TEMP
tables: keyed node scalars/selected children/representatives, one scalar summary
per family, and emitted-row deltas. `temp_store=FILE`, a 2 MiB TEMP page cache,
a 2 MiB main page cache and disabled mmap bound SQLite memory. Scratch tables
are dropped on success, unsupported-case fallback and exceptions; abrupt
process death discards SQLite TEMP storage. They are never a persistent source
cache and never contain state payloads.

Passes populate raw targets/predictions, choose the greatest raw child with
legacy sorted-ID tie breaking, resolve representatives with disk-backed path
links and cycle detection, and aggregate families in sorted node-ID order.
No Python traversal path, family list, family dictionary or all-deltas list grows
with the generation. The selected-child forest is memoized on disk, so even one
whole-generation family does not require Python recursion or an all-node set.
Policy arithmetic is shared through `family_node_target`; per-row metadata uses
the original assembly helper with singleton maps. Missing children are ignored
exactly as before.

Scalar JSON storage preserves signed zeros that SQLite REAL could normalize.
Family means and delta means use Python `sum` over ordered scalar iterators,
retaining the same compensated summation as the legacy Python 3.13 list sums.
SQL floating-point SUM/AVG and approximate reductions are not used. Exact
anchors, conflicting exact-anchor max rules, prediction blending and all nine
policy names are preserved.

Record selection consumes certified candidates incrementally with the existing
decoder and ranking; frontier selection keeps at most 100 candidates using
stable bounded selection. Leaderboard persistence consumes candidates in export
order and retains its existing compact top-100-per-variant ranking, fingerprints
and duplicate rules. Legacy snapshot record scans retain their original logs.

Before completing the summary pass and again at the end of the write iterator,
the existing SHA-256 source identity is checked. The existing JSONL writer has
not renamed its temporary file at that second boundary. Changed sources cannot
publish a new completed rows artifact. Published exports must still remain
immutable; this is not a filesystem transaction against simultaneous rewrites.

## Compatibility and activation boundary

The default remains disabled. The launcher exposes the ephemeral
`--dataset-streaming-sharded-reader` switch for `dataset` and `dataset_worker`.
It is not saved in scientific bootstrap configuration. The next-generation
worker forwards the opt-in explicitly; default call shapes remain compatible.

Only configured sharded generation manifests with JSONL output use the indexed
path. Flat snapshots, `both` exports, non-JSONL output, unrecognized family policies and unsupported missing or
nonfinite family scalars retain legacy processing. Fallback reasons and reader
mode are logged. Malformed indexed input fails explicitly rather than invoking
a permissive whole-shard loader. Existing PR #75 format rejection rules apply.
Single-process cycle dataset construction remains legacy.

## Exact parity evidence

The differential tests cover every existing policy at **1k, 5k and 10k** nodes:

- `none`
- `pv_mean_prediction`, `pv_min_prediction`
- `pv_blend_mean_prediction`, `pv_blend_min_prediction`
- `pv_exact_then_mean_prediction`, `pv_exact_then_min_prediction`
- `pv_exact_then_blend_mean_prediction`, `pv_exact_then_blend_min_prediction`

They compare every row field and metadata field, row order, canonical state
references, target values and sources, terminal/exactness flags, selected-child
and family assignments, counts, summaries, repeated passes and exact JSONL
bytes. Additional policy-wide cases cover direct/backed-up preference, compound
filters, zero/one/interior/oversized `max_rows`, empty results, blend endpoints,
empty generations, a whole-population chain, catastrophic-cancellation inputs
and signed zeros. Worker tests compare exact JSONL, pipeline manifest, Dataset
status (including record/frontier status) and leaderboard bytes with a fixed
clock. Full loaders, snapshot construction and whole-source-shard JSON reads are
forbidden during the indexed worker test.

Failure tests cover source generation changes and same-generation identity
changes, completed cache reuse, deterministic multi-pass output, malformed JSON,
cycles, interrupted writes, stale partial output, preserved prior completed
output, failed-manifest publication and claim release. The existing PR #75
reader suite additionally covers corrupt caches, interrupted builds, source
changes during indexing, duplicate IDs, missing records, malformed envelopes,
Unicode ordering and parser memory.

The benchmark independently runs the **unmodified PR #75 source** (read-only
`git archive` of the exact SHA) as its legacy oracle. All four scientific
artifact SHA-256 values match between legacy, cold indexed and reused indexed
workers at each size. No timestamp, counter, metadata or row field is removed
from those comparisons. The additional [frozen-source all-policy parity report](benchmarks/morpion_streaming_dataset_all_policies.json)
records **3 sizes × 9 policies × 4 complete scientific artifacts**. That pass
uses zero padding and serial per-policy workers within each oracle/indexed
process; it is parity evidence, not fresh-process per-policy memory evidence. Performance sidecars contain process/time/RSS
observations and are intentionally separate from scientific manifests.

## Process RSS, throughput and index reuse

All measurements are serial, fresh Python processes over generated scratch
fixtures, with 4096 bytes of synthetic update metadata per node and real
Morpion state-reference decoding. Imports contribute approximately 269 MiB.
Peak RSS is the actual process-lifetime `resource.getrusage` high water mark
through the existing `StageMeasurement`; it includes native SQLite allocations.
Timings are single samples on a shared host, not statistical speed claims.
No neural model inference is performed by Dataset; stored predictions drive
family policies. No historical experiment or real gen38 operation was run.

The standard fixture contains UTF-8 Unicode IDs and short PV families:

| Nodes | Rows | Legacy peak MiB | Cold peak MiB | Reuse peak MiB | Legacy s | Cold s | Reuse s | Reuse rows/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 823 | 289.77 | 272.29 | 271.06 | 0.094 | 0.696 | 0.388 | 2,121 |
| 5,000 | 4,109 | 377.61 | 273.25 | 272.63 | 0.484 | 2.994 | 1.799 | 2,284 |
| 10,000 | 8,214 | 489.14 | 273.79 | 273.70 | 1.040 | 5.967 | 3.759 | 2,185 |

| Nodes | Cold reader initialization s | Reused initialization s | Source MiB | Index MiB | TEMP logical MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.3036 | 0.0073 | 4.30 | 4.61 | 0.20 |
| 5,000 | 1.1724 | 0.0247 | 21.50 | 22.98 | 0.92 |
| 10,000 | 2.3952 | 0.0538 | 43.00 | 45.94 | 1.81 |

See [complete observations and artifact hashes](benchmarks/morpion_streaming_dataset.json).
Index reuse avoids reconstruction but still hashes every source file and
validates the index. Family TEMP context is recomputed deterministically on
every attempt. The end-to-end cold worker performs approximately six logical
source passes (the reader's four plus two output identity checks); reuse performs
approximately three. `reader_source_bytes_read` captures those bytes; metadata
reads cause small deviations from integer multiples. TEMP logical bytes are
sampled page_count × page_size before cleanup, not a peak disk measurement.

The same generated fixtures with escaped surrogate IDs deliberately exercise
PR #75's conservative Python-parser fallback. At 10k, legacy/cold/reuse peak RSS
is 363.31/271.34/266.97 MiB, cold initialization is 36.37 s and reused
initialization is 0.0566 s. A UTF-8 astral character makes the legacy complete
JSON string representation wider, explaining its larger transient peak in the
standard fixture. These input-encoding costs are reported, not hidden. See the
[escaped-surrogate observations](benchmarks/morpion_streaming_dataset_escaped_surrogates.json).

The additional [single whole-generation PV family benchmark](benchmarks/morpion_streaming_dataset_chain.json)
checks the family-size dimension separately; it compares the same four complete
artifact hashes and records process RSS at 1k/5k/10k. At 10k, legacy/cold/reuse peak RSS is 488.29/274.16/273.43 MiB; reused extraction takes 3.684 s (2,230 emitted rows/s).

Reproduce in an environment with the project dependencies installed:

```bash
PYTHONPATH=src python scripts/benchmark_morpion_streaming_dataset.py \
  --legacy-source /path/to/read-only-pr75-source/src \
  --output /tmp/dataset-benchmark.json
PYTHONPATH=src python scripts/benchmark_morpion_streaming_dataset.py \
  --chain --legacy-source /path/to/read-only-pr75-source/src \
  --output /tmp/dataset-chain-benchmark.json
PYTHONPATH=src python scripts/benchmark_morpion_streaming_dataset.py \
  --escape-unicode --legacy-source /path/to/read-only-pr75-source/src \
  --output /tmp/dataset-escaped-benchmark.json
PYTHONPATH=src python scripts/benchmark_morpion_streaming_dataset.py \
  --all-policies --payload-bytes 0 \
  --legacy-source /path/to/read-only-pr75-source/src \
  --output /tmp/dataset-all-policy-parity.json
```

The script generates its own temporary fixtures and runs one worker process at
a time. It accepts no existing experiment fixture. Raw source paths in the
reports identify deleted generated temporary directories, not historical data.

## Validation

Local focused/regression validation passed 224 tests across Dataset, the shared
reader, Reevaluation, record status and row extraction. A separate 11-test tail
checks floating-point ordering, whole-generation families, empty input and the
launcher default. The installed wheel passed 22 focused worker/fallback/policy
tests and the repository's installed-distribution/evaluator/checkpoint contract
check. Wheel and sdist builds, repository-wide Ruff and toolchain consistency
passed. The serial static-analysis ratchet reported zero new diagnostic
identities (mypy 89 vs baseline 91, Pyright 73 vs 73, Pylint 44 vs 45); its
reviewed baseline was not changed.

The broader local legacy pipeline run encountered the existing sandbox's
read-only global leaderboard path on two tests. Hosted full CI runs the complete
installed-package suite, formatting/toolchain and static gates in its normal
isolated environment; the final results are linked in the stacked PR description.
No local test was granted write access to the real global leaderboard.

## Remaining limits and eventual activation

Extraction memory is O(largest JSON row/state reference + parser buffers + fixed
SQLite caches + bounded frontier/leaderboard entries + generation manifests),
not O(total nodes) or O(largest family). One huge node's state, child list or
metadata can still be expensive; manifests retain PR #75's 4 MiB cap. Scratch
and the shared cache use O(nodes) disk. Root-generation reference metadata is
O(number of generations). Existing external leaderboard file loading assumes
the compact writer-produced top-100-per-variant file; manually oversized files
remain proportional to that independent file size.

The sharded reader still has full-file identity hashing, per-generation cache
duplication and no automatic retention policy. Dataset adds multiple indexed
reconstruction passes and disk writes: the memory reduction costs throughput.
The existing JSONL writer provides atomic rename visibility but not fsync-based
durability or transactional rows/manifest/leaderboard publication. Interrupted
completion after rows publication retains the existing pipeline recovery rules;
this is not exactly-once publication. Downstream training and single-process
cycles retain their separate global-memory limitations.

Keep the switch off by default. Before any activation: require exact parity and
green full CI at the final stacked commit; review TEMP/cache disk capacity and
retention; profile additional generated graph shapes and the intended resource
budget; review artifact publication/replay durability; then explicitly authorize
a bounded canary against an immutable copied generation with artifact-hash
comparison and rollback to legacy. Real generation operations, four-worker
activation, default configuration changes and merge require separate decisions.
