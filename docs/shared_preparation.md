# Serial shared Morpion preparation

`CHIPIRON_SHARED_PREPARATION=1` opts the continual streaming Training worker into
one certified-row pass for the canonical flat plus relational `transformer_v1`
family. Absence or `0` retains the historical path. This operational environment
setting does not enter training arguments, configuration identities or scientific
hyperparameters. It does not change Torch threads, execution profiles, candidate
seeds, batches, epochs, precision, optimizer state, selection or parent lineages.

The first slice supports a **serial** cohort containing both flat (`linear`, `mlp`)
and relational-v1 models, using JSONL, stable canonical splitting, one common
source/fraction/seed/row limit/chunk size, and the existing relational1536 input
representation. Other valid cohorts log a fallback and use their existing path.
There is no preprocessing pool, GPU concurrency or cross-generation feature index.

## Exact producer and consumers

The private JSONL reader retains the state decoded while validating each row's
checkpoint and confidence contract. It exposes that state only after the existing
row, target, certificate and envelope checks succeed. The producer shares it with
the original split identity/fold and the unchanged flat and relational converters.
The converters receive the immutable Chipiron state wrapper. Rows, duplicate
positions, targets, ordered features, tokens, relation triples and lengths retain
the native algorithms. A finite row limit still validates the first excluded row,
as the native split does; it never encodes that excluded row.

Bounded chunks are staged and packed into the existing tensor-cache formats. The
streaming trainer uses those formats through its existing read-only readers and
training loop. A private source/recipe-bound, context-local split handoff seeds
`prepare_state_split` through its ordinary LRU or persistent builder, avoiding a
second full replay on the first miss. Its scope resets on failure as well as success.
`CHIPIRON_STATE_SPLIT_CACHE_DIR` remains independently optional; I5.1's key,
storage, eviction and metadata contracts are unchanged.

## Ownership and retry

Preparation uses `.rows.jsonl.shared-preparation-v1` beside the Training attempt's
private `rows.jsonl` input, never inside an accepted immutable Dataset. Existing
Training admission, source/model pins, publication and acceptance remain in charge.
One stable exclusive lock covers building **and consumption**. Concurrent callers
therefore wait, including readers; this initial serial contract prevents replacement
beneath a consumer. No helper children or orphan writers are created.

A versioned completion binds exact source content/order, implementation/runtime
semantics, representation settings, split settings and checksums of every native
payload/manifest and split artifact. The complete directory is validated with the
original readers before atomic publication. An interrupted private build is cleaned
under ownership before retry. A damaged completed bundle fails closed. A failed
publication or consumer cannot return a successful training round. The attempt
owner deletes inputs only after its work has finished, using its existing lifecycle.

Valid legacy native caches retain their native hits. If either native cache is
already valid, preparation records `native-warm-fallback` and the original readers
handle that cohort, including materializing a missing sibling. Their path/stat
identities are insufficient to certify a new exact-content bundle, so they are not
silently adopted into one. A valid completed shared bundle is a shared hit. Retained
native manifests still require the original source path/stat; this is not a portable
feature store. Moving or externally touching an attempt's input is unsupported.

## Resource limits and observations

Defaults cap packed tensors at 2048 MiB and staging/export disk at 6144 MiB:

- `CHIPIRON_SHARED_PREPARATION_MAX_PACKED_MB`
- `CHIPIRON_SHARED_PREPARATION_MAX_SCRATCH_MB`

Both accept positive integer MiB values. These are byte limits, not row limits;
exhaustion fails the attempt instead of truncating its scientific dataset. Chunk
buffers are capped at 64 MiB and at most `min(row_chunk_size, 128)` rows. Compatible
export allocates one complete packed tensor payload. It is included in the memory
checks, together with chunk workspace and 512 MiB headroom, using the existing
host/cgroup memory probe. Disk headroom is checked before staging and export.
The original compact split membership/population fold also uses memory proportional
to rows. This slice does not claim constant total RSS or an unbounded packed export.

`operation.json` and the `shared_preparation` log record the resolved outcome,
source/semantics key, validation/encoding/split/staging/packing/export timings and
byte counts. They are operational evidence, separate from model quality and
scientific configuration. Measure full preparation and process peak RSS: chunk-only
memory or conversion timing excludes compatible export and is not an end-to-end
bootstrap speedup. Raw benchmark evidence belongs outside Git.
