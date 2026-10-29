# Bounded sharded Reevaluation

## Baseline audit (50fb67a4)

`load_morpion_sharded_training_tree_snapshot` reads the selected generation
manifest and root manifest. For each referenced generation in 1..G (missing
references are skipped), it loads a JSON document containing an immutable
`nodes` array into an all-node dictionary. Later duplicate immutable IDs replace
earlier entries. It loads G's entire `updates` array, deserializes every update,
stably sorts by `order_index`, merges every record/update pair, then constructs
a complete `TrainingTreeSnapshot`. The node-creation index is not used by this
loader. The merge helper determines all effective/tree/target value fields.

Reevaluation picks the newest pipeline manifest whose snapshot path exists,
loads the active-model identity, blocks on a pending patch and checks available
RAM before loading the complete snapshot. It sorts string IDs lexicographically,
then selects an inclusive, circular window. An absent cursor starts at the first
ID. A request covering the population returns each ID once, rotates from the
cursor and sets next cursor to the first sorted ID. Reaching the end, including
landing exactly on it, counts one completed pass. Cursor ownership is evaluator
generation, evaluator name and bundle path; tree generation does not reset it.
Both built-in and snapshot-value evaluators build an all-node dictionary.

The built-in adapter loads the bundle lazily, preserves finite stored terminal
backups as direct values and otherwise calls the existing training-state codec,
model and score extractor. It emits `backed_up_value=None`. Snapshot fallback
uses stored direct values, then backups, then zero and retains stored backups.
After inference the worker checks for a pending patch again, exclusively creates
the final JSON patch, then saves the cursor. These publication steps remain
unchanged by this optimization.

## Facilities and parser decision

The repository already uses SQLite for a canonical JSONL state cache, with a
2 MiB page cache and full-file SHA-256 identities. That cache decodes states and
has prefix-resume semantics, so it cannot serve immutable training export rows.
The new reader reuses the SQLite/cache conventions and `StageMeasurement`,
`log_pipeline_memory` and the existing node deserializers/merge helper.

The existing stdlib JSON and orjson interfaces materialize complete JSON values;
the existing JSONL reader is inapplicable to JSON document arrays. PyArrow's
line-oriented JSON input does not expose these nested arrays incrementally.
`ijson` supplies a maintained incremental parser and object builder, avoiding a
new handwritten JSON parser. See its [event interception and items APIs](https://github.com/ICRAR/ijson).

## API and supported format

`ShardedTrainingGenerationView(manifest_path, cache_dir=...)` is a context
manager exposing `generation`, `root_node_id`, `node_count()`,
`select_sorted_node_window(cursor, max_nodes)`, `load_nodes(ids)` and
`iter_nodes_in_export_order()`. The window result is the same three-tuple as the
legacy selector. Export iteration uses `(order_index, input_sequence)` to match
Python's stable sort, including ties. Reconstruction calls the original
`_node_record_from_dict`, `_node_update_from_dict` and
`_merge_node_record_and_update`; state references remain opaque until inference.

Supported input is the existing v1 Morpion sharded export: one root manifest,
referenced immutable `nodes` JSON document shards from 1..G and G's `updates`
JSON document shard. Sparse generation references follow the full loader's
selection. The node-creation index is unnecessary and is never loaded. Small
legacy manifests/shards omitting format headers are accepted. Flat snapshots
retain the complete loader. Dataset is unchanged.

SQLite has one immutable-record table, one generation-specific update table
referencing records, and a completion/identity table. State payloads are present
once per immutable row within an index, never copied into update rows or decoded
into states during indexing. The update primary key uses explicit `BINARY`
collation. UTF-8 byte order agrees with Python code-point order for Unicode
scalar strings, including NUL and astral characters; randomized differential
tests validate that ordering. Lone surrogates are rejected.

## Bounds, cache identity and limitations

Working memory is O(K + largest JSON row/token + 64 KiB parser buffers + 2 MiB
SQLite page cache + generation-manifest metadata). The parser may allocate event
objects for its current input chunk. JSON row conversion/reconstruction may
hold several copies of that row, never all rows. One huge state reference or
metadata row can therefore still be expensive. Manifest documents have a 4 MiB
limit and shard envelope keys a 64-key limit in bounded mode. The root manifest
contains O(number of generations), not O(number of nodes), references.
SQLite sorts spill to disk; mmap is disabled. Model weights, framework imports,
state decoding and inference have their own fixed/window-dependent memory.
No expensive object-graph profiling is enabled.

Disk is O(historical immutable records + selected-generation updates), including
unused historical immutable records needed to preserve loader selection.
Different generation indexes can duplicate immutable records across caches;
there is no automatic eviction policy in this initial disposable cache. Cache
cleanup/retention and cross-generation incremental reuse remain follow-ups.
The default cache is `platformdirs.user_cache_path("chipiron") /
"sharded_training_reader"`, outside exports. An override inside the export is
rejected; use a separate disposable directory. The reader never writes exports
or historical scientific artifacts.

Every open hashes all relevant source files in fixed buffers, including the
selected/root/referenced manifests, immutable shards and selected update shard.
Identity includes canonical paths, sizes, SHA-256 contents, export version and
index schema version. Reuse avoids parsing/index reconstruction, but still has
O(source bytes + index validation) I/O. Small manifest reads are included in the
logical source-byte counter; it does not represent physical device reads.
Root-manifest changes conservatively invalidate a cache, even if unrelated to G.
Published exports must be immutable during consumption. The builder checks
fingerprints again before completing, and detects changes during hashing/build.
This is not a filesystem snapshot against hostile simultaneous source rewrites.

Construction uses unique temporary files and commits every 256 rows. Foreign
keys, uniqueness, node count, root presence and SQLite integrity are checked;
completion identity is committed before atomic rename. Interrupted temporary
files are never reused. A corrupt completed cache is rebuilt. Concurrent builders
can publish identical complete indexes without exposing partial indexes. Cache
publication is disposable, independent of scientific patch publication.
`index_build_peak_disk_bytes` samples main/journal bytes at transaction
boundaries; it is a sampled high water mark, not continuous filesystem polling.

Malformed inputs fail explicitly. Compared with the permissive full loader,
bounded mode rejects duplicate immutable/update IDs, missing/wrong arrays,
duplicate envelope keys, unsupported explicit versions, missing immutable
records, count/root inconsistencies, nonstandard JSON NaN/Infinity literals,
lone surrogate IDs and order indexes outside SQLite's signed 64-bit range.
It does not silently overwrite, deduplicate or repair malformed records.
Empty generations with an arbitrary root ID remain supported.

## Reevaluation opt-in and compatibility

Pass `use_streaming_sharded_reader=True` to the one-shot worker or use
`--reevaluation-streaming-sharded-reader` on the canonical launcher with the
existing Reevaluation stage. This is an ephemeral launcher setting, default off;
no persisted scientific configuration or distributed-run defaults change.
`reader_cache_dir` on the worker can override the disposable cache directory.
No real asynchronous deployment is recommended before durability review.

Built-in inference and snapshot-value fallback consume only selected nodes.
`MorpionWindowNodeReevaluationEvaluator.evaluate_node_window(nodes)` is the
separate explicit opt-in contract for custom adapters. Existing
`MorpionNodeReevaluationEvaluator.evaluate_patch_rows(snapshot, ids)` keeps its
complete-snapshot semantics. Custom adapters without the window method fall
back to the legacy loader. A custom window adapter must not require ancestors,
children or global snapshot metadata to be loaded implicitly.

Logs and `StageMeasurement` observations identify `sharded_indexed` versus
`legacy_full_snapshot`, fallback reasons, source generation/count, index
build/reuse and size, initialization, sorted-ID selection, selected-node
reconstruction, model load/inference, row/patch generation and cursor
publication. Existing RSS/peak RSS/available RAM and throughput fields are reused.
These observations never enter scientific patch/cursor metadata.

Cursor ownership, selection/wrap/pass counting, terminal stored-value behavior,
training-state codec rejection of delta refs, score extraction, finite-value
validation and both patch guards remain unchanged. Differential tests compare
complete serialized scientific patch and cursor contents with fixed model,
patch ID, timestamp, cursor and generation. Only the shared test bundle path is
fixed identically across fixtures; scientific metadata is not dropped.

## Publication durability: separate high-priority follow-up

`_save_reevaluation_patch_exclusive` still opens the final path using `"x"`,
then writes JSON, then the worker saves its cursor. Readers can observe an
incomplete final patch; a crash can leave a partial patch or a published patch
with an older cursor. Exclusive creation is not atomic complete visibility,
transactional patch/cursor publication or exactly-once crash recovery.
This PR deliberately preserves those semantics. Review atomic complete patch
publication, fsync/rename or link semantics, concurrent consumers and replay
rules separately before real asynchronous deployment.

## Future Dataset reuse and remaining bottlenecks

Dataset can reuse the view's exact generation selection, one-node reconstruction
and deterministic `iter_nodes_in_export_order()` in a separate PR. Its row
assembly/training input path still needs its own bounded-memory review.
Before four-worker operation: publication durability, Dataset memory, per-model
inference/framework memory, full-file reuse hashing, cache retention/I/O
contention and the independent Growth checkpoint lifecycle work remain open.
This reader requires no new Anemone functionality and enables no warm Growth.

## Synthetic benchmark evidence (2026-10-20)

Run serially with fresh processes for each mode/size, no parallel test workers:

```bash
PYTHONPATH=src python scripts/benchmark_morpion_streaming_reevaluation.py \
  --output /tmp/reevaluation-benchmark.json
PYTHONPATH=src python scripts/benchmark_morpion_streaming_reevaluation.py \
  --sizes 10000 --window 1000 --output /tmp/reevaluation-window1000.json
```

Fixtures contain compact `[0, []]` state refs and 4096 bytes of synthetic update
metadata per node. The built-in adapter decodes real Morpion states and uses a
constant-score model double. This measures reader/adapter costs, not real neural
model throughput. Wall times include other concurrent local activity and each
mode is one sample, not a statistical speed claim. Imports/frameworks contribute
roughly 271 MiB before reading. RSS peaks are process-lifetime high water marks.

| Nodes (K=100) | Legacy peak MiB | Cold peak MiB | Reuse peak MiB | Cold init s | Reuse init s | Index MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 283.20 | 273.32 | 272.39 | 0.292 | 0.0073 | 4.57 |
| 5,000 | 331.83 | 273.30 | 272.04 | 1.137 | 0.0267 | 22.78 |
| 10,000 | 369.42 | 273.32 | 272.04 | 2.247 | 0.0544 | 45.52 |

| Nodes (K=100) | Legacy worker s | Cold worker s | Reuse worker s | Reuse selection ms | Reuse reconstruction ms | Reuse rows/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.198 | 0.455 | 0.200 | 0.146 | 2.940 | 500.4 |
| 5,000 | 0.287 | 1.335 | 0.214 | 0.137 | 2.730 | 466.9 |
| 10,000 | 0.421 | 2.426 | 0.242 | 0.178 | 3.313 | 413.4 |

The cold end-to-end worker rebuilds its cache; the reused worker reuses it.
Phase-only initialization/build, selection, reconstruction, built-in evaluation,
end-to-end worker, fresh-process wall time, RSS boundaries, source/index bytes,
sampled temporary index/journal size and total fixture disk are in the
[100-node-window raw results](benchmarks/morpion_streaming_reevaluation_100.json).
Legacy reconstruction is included in complete initialization, not the separate
selected-node reconstruction timer. Constant model load takes approximately a
microsecond; no trained-model/device load is measured.

At 10k nodes source size is 41.97 MiB and the index is 45.52 MiB. Cold indexing
reads about four source passes (initial fingerprint, Unicode preflight, parse,
final fingerprint),
whereas reuse hashes about one source pass and validates the SQLite index.
The extra disk and cold CPU/I/O are real costs. The parser-only Python allocation
peak is about 0.195 MiB at all three sizes, measured separately with tracemalloc.
The focused indexing memory test separately checks its Python allocation peak;
process RSS captures SQLite/native allocations in these fresh-process runs.

At 10k nodes, increasing K to 1000 gives roughly 276 MiB indexed peak RSS
versus 396 MiB legacy, and a 1.838 s reused worker pass (cold: 4.488 s).
See the [1000-node-window raw results](benchmarks/morpion_streaming_reevaluation_1000.json).
The fixed-K series stays flat as N grows; larger K increases reconstruction and
inference work. This supports the allocation audit, rather than deriving a
memory claim merely from returning K rows.

Validation uses Python 3.13.5, ijson 3.6.0, SQLite from stdlib and released installed
Anemone 0.2.26 at
`/home/pompote/oldata/victor/chipiron-warm-growth-runtime/.tox/py313/lib/python3.13/site-packages/anemone`.
Its 288 installed package file hashes match distribution RECORD, with no
`direct_url.json`. This is not Anemone PR #77 and no source checkout was injected.
The isolated environment reuses installed dependencies read-only and installs
ijson into `/tmp/chipiron-streaming-venv` only.

The native parser's unpaired-surrogate substitution is avoided by a bounded,
conservative preflight for possible surrogate escapes. Such shards use ijson's
stdlib-string-decoder Python backend, preserving opaque values and explicitly
rejecting surrogate node IDs. That backend receives one byte per parser input
chunk: its lexer otherwise can retain consumed prefixes when successive chunk
boundaries fall inside strings. The underlying file remains buffered. This
rare cold-build path trades throughput for a provable token-sized bound. When
no native backend is installed, all shards use that bounded Python path. Normal
numeric-ID/compact-state shards use native 64 KiB parsing. The preflight is a
backend selector, not a handwritten JSON parser; false positives only slow the
library parser. Cold builds now read four logical source passes including this
preflight, while reused indexes still hash one pass.

## Validation status

The final focused reader/worker/export tests pass (84 tests), as do the six
existing Reevaluation dispatcher tests and 24 tests after the unrelated stalled
chess training test. Full repository Ruff formatting/lint, toolchain consistency,
wheel/sdist build and the serial static-analysis ratchet pass. The ratchet
reports zero new diagnostic identities (209 baseline, 206 current).

The full serial non-integration/non-external-data attempt reached 1388 passing
and two skipped tests before interruption at the chess supervised-learning test
(`multiprocessing` warned about fork after threads). It reported 23 failures:
four launcher stub failures were fixed and rerun successfully; all 19 remaining
failures reproduce on an untouched archive of protected main 50fb67a4 using the
same environment. Fourteen attempt to write the global user Morpion leaderboard
and are blocked by the sandbox; the others involve unavailable multiprocessing
facilities/child processes. The 24 tests after the stalled test pass separately.
This is not a claim of a green full suite or hosted CI. No sandbox protection was
removed for these tests and no global leaderboard was modified.

Safety: the dirty research working files, Anemone investigation, Chipiron #74
and Anemone #77 are untouched. No historical scientific export or experiment was
modified. No real gen38 operation, bootstrap continuation, distributed worker,
dependency release, warm Growth activation or merge was performed. All benchmark
fixtures and operational worker calls were synthetic and isolated under `/tmp`.
