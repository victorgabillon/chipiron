# Optional MLP41 activation inputs

`CHIPIRON_EVALUATOR_INPUT_CACHE=1` enables derived raw-input reuse during owned
candidate activation's full refresh. It defaults to `0`. This is execution-only
operational storage: it does not alter bootstrap configuration/scientific identity,
learning cadence, accepted weights, claims, source pins, either preflight/restore,
full refresh, canonical backups, scalar forwards or atomic activation publication.
Exact accepted retries never open or require this cache.

The initial supported recipe is Linux, the exact standard full 41-feature converter
and dynamics, a CPU FP32 MLP in evaluation mode, CPU FP32 tensor defaults and the
explicit `strict_math` profile. Other models, feature subsets and unqualified
recipes use the original converter. Runtime/profile/thread/code identities are
fenced separately; cache contents are never a way to adopt another scientific run.
No GPU context is needed for this scalar CPU conversion, including in a ROCm build.

Optional execution settings:

| Environment variable suffix after `CHIPIRON_EVALUATOR_INPUT_CACHE` | Default | Meaning |
|---|---|---|
| `_DIR` | `<work>/pipeline/evaluator_inputs` | Absolute dedicated owned root outside immutable artifacts |
| `_MAX_ROWS` | `20000` | Maximum refresh requests/tree nodes, reducible only |
| `_MAX_MB` | `256` | SQLite entry MiB ceiling, reducible only |

A private root is marked once. Existing unmarked/foreign directories, any symlink
component or child, hardlinked payloads, foreign ownership and unknown children
are refused without deleting their targets. Do not place unrelated data in this
root. A root creation killed before its owner marker is durable needs manual
inspection; it is deliberately not adopted as owned storage.

The stable exclusive kernel lease is nonblocking. A competing refresh performs
fully uncached work without pruning or changing entries. The successful owner
holds the lease across prior verification, all consumption, reader closure and
replacement. There is one complete prior and at most one private `building` entry.
The adapter detaches to the original converter before closing storage, so the
same live runtime can keep evaluating, exporting and growing afterward.

Each row binds the exact complete state (all eight fields including orientation,
history, variant, turn and terminal flag) to raw contiguous finite FP32 inputs.
Sorted sets do not discard orientation or ordered history. Compact marshal2 bytes
are **never decoded from storage**. Full payload equality accompanies SHA lookup.
No model weights, predictions, targets, optimizer state or certificates are cached.
Unsupported state types/oversized keys use ordinary conversion for that state;
current certificate validation and every model forward still run.

The full code/runtime/ordered-feature recipe, completion SHA/size/schema and every
payload/raw binding are verified before use. Prior completion/data and new rows
are verified again after consumption. Missing, incompatible, incomplete or
accidentally corrupted prior data is rebuilt before use; unsafe filesystem
objects are refused. Independent late corruption, I/O or resource failure aborts
and discards the private runtime before scientific publication. There is no
mid-refresh fallback with partly updated values. Digests are integrity checks on
trusted owned storage, not authentication against an actor rewriting data and
all its bindings coherently.

SQLite uses a hard page ceiling, bounded 2 MiB page caches per connection, no mmap
and no staging journal/WAL. Disposable staging is fsynced together with its small
completion before atomic directory publication. Crash recovery discards owned
staging only after acquiring the released lease. A crash in the prior-removal /
replacement gap leaves no authoritative cache; the next owner rebuilds. The
scientific head is always governed by the unchanged activation transaction.

Admission reserves the complete prior plus replacement and bounded metadata,
not just tensor bytes (13k inputs have roughly 98 MB of state/index content).
It checks current process RSS plus 32 MiB headroom against 4 GiB, 64 MiB available
host RAM, and 20 GiB free disk plus full storage reserve. Bounds/resources are
rechecked during consumption. There is no 200k-node capacity claim. Native
checkpoint/export outputs follow cache finalization and remain under their
existing run/resource controls; this cache does not change run budgets.

An accepted opted-in activation adds bounded `evaluator_inputs` metadata to its existing
`activation` observation in the `pipeline/performance` sink: opt-in, outcome/bypass reason,
limits, prior/entry bytes, hits, misses, unsupported-state count and preparation /
verification durations. Failure of this reporting sink cannot change acceptance.
There is no additional timeline interval, per-node log, scheduler or cache admission authority. Default-off
and exact-retry observation behavior remain unchanged.
