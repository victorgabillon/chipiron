# Optional persistent canonical splits

Set `CHIPIRON_STATE_SPLIT_CACHE_DIR` to an operational directory outside dataset
and immutable publication directories before starting the normal Training worker
or bootstrap launcher:

```sh
CHIPIRON_STATE_SPLIT_CACHE_DIR=/absolute/private-runtime/split-cache \
  python -m chipiron.environments.morpion.bootstrap.launcher <existing arguments>
```

Keep this environment setting on worker restarts. With it unset, the original
two-entry process LRU and canonical split algorithm are unchanged. Empty values
are rejected when stable splitting is used. The setting is not a scientific
configuration field, model policy, or manifest addition. The returned training
arguments and split metadata are identical. Non-stable splitting ignores it.

A fresh worker can reuse an exact-content split, including when the source file
has moved. A new dataset, row order, label, target source/confidence, split seed,
fraction, row limit, or semantic fingerprint gets a different key. The existing
canonical algorithm computes every miss. This improves identical-input retries
and restarts; it does not eliminate first-build preparation or establish a
complete-bootstrap speedup. Shared decoding and cross-generation feature reuse
are separate work.

The fingerprint hashes actual Chipiron source, installed codec/foundation and
row-reader dependency code/native libraries, Python and JSON implementation, and
runtime/version evidence. This is intentionally conservative: unrelated code
updates may miss. Shadowed foundation installations are rejected. Torch's build
label and version module deliberately separate CPU and AMD/CUDA installations;
cross-backend reuse is not qualified. The helper does not initialize a GPU,
change Torch flags/threads, or alter G5 model/parent guards. Installations must
remain immutable for a worker lifetime; restart after code/dependency updates.

The helper hashes the input before and after every lookup, including memory
hits, and again before publication after computation. Existing J2/Training
source/read pins remain held by the caller through preparation and consumption.
No Growth ownership, claim, reservation, acceptance, or candidate scheduling
rule changes. Auxiliary cache files never certify an accepted scientific result.

Storage belongs exclusively to the `chipiron-state-split-v1` child of the
configured directory. It has sixteen fixed slots and stable kernel locks. A key
always uses the same slot/lock; collisions serialize and replace a valid older
entry, trading hit rate for a hard retention bound. Each slot retains one
completed payload (at most 16 MiB and one million held-out indices), one latest
operational record, and at most one private build. Thus completed payloads occupy
at most 256 MiB, plus fixed small manifests/locks/records; concurrent private
payloads can temporarily double that bound. No training row cap is introduced:
results exceeding the storage bounds are returned exactly, without durable
caching. The opt-in path also has a two-entry in-process LRU.

Readers fully validate and load independent immutable values under the same
exclusive slot lock as builders/eviction. They retain no file-backed views, so
replacing an entry after lock release cannot invalidate a reader. Lock files are
never unlinked. Acquisition waits at most 600 seconds, then raises. Threads and processes use the same slot exclusion; lookup observations are
context-local. The helper creates no children or background cleanup workers.

Builds use unique private directories. Payload and completion files are fsynced,
validated, the private directory is fsynced, then renamed atomically and its
parent fsynced. Publication/I/O failures propagate. Missing output is a miss;
incomplete private builds are cleaned only by the next owner of that slot lock.
Corrupt, malformed, oversized, incompatible or unknown completed entries fail
closed. Unknown files, foreign namespaces and symlinks are not silently adopted
or removed. An intact in-process result does not reread an evicted/corrupted disk
artifact; it remains an independently validated value under the same exact key.

`slot-XX/operation.json` is replaced atomically after successful lookups; at most
sixteen records are retained. They distinguish `miss`, `hit`, `memory-hit` and
`uncached-size-limit`, recording source hashing/verification, semantic checking,
lock wait, native build, load and publication times. `elapsed_before_record_s`
excludes writing that diagnostic record; measure the complete invocation for
total overhead. The same bounded-size record is emitted through the module
logger. Configure ordinary log rotation separately. Scientific metadata is not
extended with cache/resource observations.

Automatic cleanup touches only recognized private staging and a valid replaced
entry within the marked namespace while owning its slot lock. Historical data,
I4/I5 evidence and other cache roots are never eviction targets. To remove the
entire namespace or its stable lock files, first stop and reap every worker that
could use that root; otherwise retain it. Do not manually remove a live lock.
For a corrupt completed artifact, stop users and preserve it for diagnosis before
choosing a fresh operational root. There is no automatic corruption fallback.
