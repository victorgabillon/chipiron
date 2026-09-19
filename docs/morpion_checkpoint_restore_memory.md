# Morpion checkpoint restore memory plan

This note records the C1 investigation after A1/A2 checkpoint payload compaction
and B3 restore-time memory instrumentation. The A2 42k checkpoint showed that
RSS rises mostly while decoding JSON and building the typed checkpoint payload,
then remains high after raw and typed payload references are dropped. The likely
cause is Python allocator arena retention and fragmentation after large temporary
object graphs coexist during restore.

## Current restore pipeline

1. Compressed file read/decompress
   - `anemone.checkpoints.io.load_checkpoint_json_payload()` opens the plain,
     gzip, or zstd JSON stream.
   - Large live objects: compressed file buffers and decompressor/text stream
     buffers only. No checkpoint object graph exists yet.

2. Raw JSON object creation
   - `json.load()` materializes the whole checkpoint as nested Python dicts,
     lists, strings, ints, and floats.
   - Large live objects: full raw decoded checkpoint. On the A2 42k checkpoint
     this graph was about 208 MB recursively.

3. Normalization / tuple restoration
   - Chipiron normalizes the raw mapping for dacite, including checkpoint atom
     tuple restoration.
   - Large live objects: raw decoded checkpoint plus normalized mapping/list
     objects. Many unchanged leaves may still be shared, but containers for the
     transformed structure add another temporary graph.

4. Dacite typed payload construction
   - `dacite.from_dict()` builds `SearchRuntimeCheckpointPayload` and nested
     dataclass payload objects.
   - Large live objects: raw decoded checkpoint, normalized payload, and full
     typed checkpoint payload. On the A2 42k checkpoint the typed payload was
     about 141 MB recursively.

5. State payload store construction
   - Anemone builds `DenseCheckpointPayloadStore` or `DictCheckpointPayloadStore`
     from the typed node payloads.
   - Large live objects: typed checkpoint payload plus payload-store references
     to each node's state payload. The store itself is intentionally retained by
     `CheckpointStateResolver` for lazy state resolution.

6. State handles creation
   - One `CheckpointBackedStateHandle` is created for each restored node.
   - Large live objects: typed payload, state payload store, resolver, state
     handles, and the raw/normalized graphs if the caller has not dropped them.

7. Tree nodes creation
   - Runtime `AlgorithmNode` and `TreeNode` objects are created with lazy state
     handles and no eager state materialization.
   - Large live objects: typed payload, resolver/store, state handles, runtime
     nodes, node evaluation shells, and tree-node shells.

8. Link nodes
   - Parent-child edges are restored from `linked_children`.
   - Large live objects: same as step 7, with populated tree graph links.

9. Restore node runtime states
   - Unopened branches, evaluation values, decision ordering, PV, branch
     frontier, backup runtime, and exploration-index state are restored.
   - Large live objects: typed payload plus complete runtime graph and retained
     checkpoint payload store.

10. Drop raw/typed payloads
   - Chipiron deletes local raw, normalized, and typed payload references after
     runtime restore.
   - Large live objects: runtime graph, lazy state resolver, retained checkpoint
     payload store, selector state, and any allocator arenas already reserved
     for dropped temporary objects.

11. GC
   - `gc.collect()` reclaims unreachable Python objects, but process RSS can
     remain high when CPython keeps arenas for later reuse.
   - Large live objects: final runtime graph and retained checkpoint state
     payload store. The A2 42k reachable profile was about 102.6 MB while RSS
     stayed about 847 MB.

## Lower-peak design options

### Option A: Drop raw decoded payload earlier

This is only partially feasible with the current whole-file `json.load()` plus
dacite pipeline. We can normalize and type smaller sections, then delete raw
subsections, but ordinary nested dict/list objects still exist until the raw
top-level container and references are gone. It would also require replacing
dacite's one-shot top-level construction with a custom staged builder.

Expected benefit: moderate if node sections are processed in chunks.
Risk: medium, because section-level conversion must preserve every payload
default and checkpoint atom encoding rule.

### Option B: Avoid full dacite typed payload for all nodes

This is the most promising medium-term path. A direct restore builder could
read normalized node records and construct:

- the lazy state payload store,
- state handles,
- runtime nodes,
- topology links,
- evaluation/runtime state,
- latest expansions and selector state,

without materializing every `AlgorithmNodeCheckpointPayload` dataclass first.
The retained state payload store still needs typed or typed-like state payload
objects because `CheckpointStateResolver` is generic and codec-facing, but many
node structural/evaluation payload dataclasses could be short-lived or skipped.

Expected benefit: high for typed-payload coexistence and allocator churn.
Risk: medium-high unless implemented behind an opt-in flag with equivalence
tests. The direct builder must share validation and restoration helpers with the
current path rather than reimplementing semantics in a fork.

### Option C: Sharded runtime checkpoint restore

This is likely the right long-term format for million-node trees. A metadata
file plus node shards, selector shard, and latest-expansion shard would allow
incremental node processing and smaller temporary decode graphs. It also opens
the door to line-oriented or per-shard streaming.

Expected benefit: highest for peak RSS.
Risk: high for one PR. It changes checkpoint format and writer/reader
orchestration, requires migration and compatibility decisions, and needs broader
tests. This should be a separate design and implementation series.

### Option D: Subprocess restore handoff

This could isolate raw/typed decode arena retention in a child process, but the
parent still needs the live runtime graph. Passing that graph back would require
pickle or a custom compact handoff format, which resembles building another
checkpoint format.

Expected benefit: uncertain.
Risk: high complexity and likely poor portability. Not recommended for C1.

### Option E: JSON library or streaming parser

Using `orjson` for reads might reduce CPU time and possibly temporary churn, but
it still materializes the whole raw graph. A true streaming parser such as
`ijson` could reduce raw graph coexistence, but it adds a dependency and works
best with a format designed for streaming arrays or shards.

Expected benefit: low-to-moderate for faster JSON library, high for streaming
with format work.
Risk: dependency and parser complexity. Do not add this dependency casually.

## Recommended PR split

C1a should stay instrumentation/design focused:

- keep the current restore path as default,
- add precise I/O read-phase instrumentation around stream open vs `json.load`,
- record this design note,
- rerun B3 profile to verify the exact allocation boundary.

C1b should prototype an opt-in direct lightweight restore builder in Anemone:

- keep it generic and domain-agnostic,
- share validation and restore helpers with the current typed-payload path,
- consume a normalized mapping/sequence representation rather than a full
  `SearchRuntimeCheckpointPayload`,
- retain only the generic checkpoint state payload store required for lazy
  state resolution,
- add equivalence tests against the current restore path on small checkpoints.

C2 should design and implement sharded runtime checkpoints if C1b still leaves
peak RSS dominated by raw JSON decode or allocator retention.
