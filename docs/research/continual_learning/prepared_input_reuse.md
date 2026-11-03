# Serial prepared-input reuse (I5.3)

Default behavior is unchanged. After review, opt in for the existing serial flat
plus relational transformer_v1 shared-preparation cohort with:

```sh
CHIPIRON_SHARED_PREPARATION=1
CHIPIRON_SHARED_PREPARATION_REUSE=1
CHIPIRON_SHARED_PREPARATION_REUSE_DIR=/absolute/private/operational-inputs
CHIPIRON_SHARED_PREPARATION_REUSE_MAX_MB=2048
```

The directory must be dedicated, outside disposable attempt inputs and immutable
publications, and contain no symlink components. There is one retained entry.
The positive cache limit is independent of the existing positive shared packing
(2048 MiB) and staging (6144 MiB) limits. Previous inputs, the current index,
chunks, compatible native export and new input export coexist under those bounds.
Memory admission includes prior mapped inputs, current allocations and chunk
headroom against available host/container RAM, which includes driver residency.
Resource exhaustion stops the attempt without changing its row limit.

The entry stores full canonical flat features, entity features, static int32
relation triples and int64 lengths. It contains no targets, confidence, splits,
models or optimizers. Lookup requires the exact complete finite JSON state
payload, including orientation and replay order, and a conservative code/runtime/
codec/representation identity. Node IDs and D4 split equivalence grant no hit.
CPU and AMD builds have distinct identities. Nothing initializes a GPU context.

Every current row is still validated and decoded. Current labels, confidence,
order, limits and canonical split/population evidence are rebuilt. Input hits
consume no RNG. Native consumers, candidate seeds, batching, numerical settings,
selection and accepted own-parent continuation are unchanged.

The existing stable owned lock excludes concurrent builders and pruning while a
prior mmap is in use. Both file digests and per-row payload/tensor bindings are
verified; the manifest is checked again against its initial value after gathering.
Missing, incompatible or corrupt derived entries rebuild from the current source.
Foreign or symlinked storage is refused. New entries are private until complete,
verified and durably published. A crash during replacement may lose the previous
cache and cause a miss. Restart cleans owned staging only after kernel lock release.
Native attempt files are independent; the global lock ends before model training.

Cache presence supplies no source admission or claim authority. Existing Dataset
pins, source checks and publication/retry contracts remain authoritative. No
publisher-only reader proof is copied. Fresh and persistent workers use the same
path; per-job cleanup retains no models, tensors, optimizers or mutable readers.

Small `operation.json` receipts record hits, reencoded rows, bytes and phase
times. Same-generation shared/native warm hits retain their existing path and do
not reseed the prior store. Parsing, legal replay, canonical split identity,
packing, export and integrity checks remain real costs. High input turnover can
make this slower; preparation savings alone establish no bootstrap throughput or
playing-strength gain. Independent integration review and quiet timing remain
rollout requirements.
