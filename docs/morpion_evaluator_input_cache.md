# Optional canonical MLP20/41 activation inputs

`CHIPIRON_EVALUATOR_INPUT_CACHE=1` enables derived raw-input reuse during owned
candidate activation's full refresh. It defaults to `0`. This is execution-only
operational storage: it does not alter bootstrap configuration/scientific identity,
learning cadence, accepted weights, claims, source pins, either preflight/restore,
full refresh, canonical backups, scalar forwards or atomic activation publication.
Exact accepted retries never open or require this cache.

The supported recipes are Linux, the exact standard `handcrafted_41` or
`handcrafted_20_core` converter and dynamics, a matching CPU FP32 MLP in evaluation
mode, CPU FP32 tensor defaults and the
explicit `strict_math` profile. Other models, feature subsets and unqualified
recipes use the original converter. Runtime/profile/thread/code identities are
fenced separately; cache contents are never a way to adopt another scientific run.
No GPU context is needed for this scalar CPU conversion, including in a ROCm build.
Custom names, reordered features, converter/model subclasses and other widths
remain unsupported. The absent row setting remains 20,000 for both recipes.
An explicit valid row request up to 35,000 can increase admission only for the
qualified canonical MLP20. MLP41's effective ceiling remains 20,000, even when
35,000 is requested. Reductions apply to either recipe. Larger trees use the
ordinary uncached path; the 256 MiB SQLite ceiling is unchanged.

Optional execution settings:

| Environment variable suffix after `CHIPIRON_EVALUATOR_INPUT_CACHE` | Default | Meaning |
|---|---|---|
| `_DIR` | `<work>/pipeline/evaluator_inputs` | Absolute dedicated owned root outside immutable artifacts |
| `_MAX_ROWS` | `20000` | Positive decimal request up to `35000`; effective maximum `35000` for qualified MLP20, `20000` for MLP41; reductions win |
| `_MAX_MB` | `256` | SQLite entry MiB ceiling, reducible only |

Invalid or above-35,000 row requests are refused, not silently clamped. Default-off,
unsupported-recipe and exact accepted-retry behavior stays unchanged. Admission
uses the total current tree size, including terminal nodes, and an independent
per-refresh request counter. Unique keys or nonterminal counts cannot replace
either bound. The effective and requested limits appear in bounded observations.

35,000 is an upper admission bound, **not a fit guarantee**. The saved B5
32,884→34,426-node adjacency fit with a 240.201 MiB complete entry and 15.801 MiB
SQLite margin in J3.20's private investigation. A separate large-payload fixture
exhausted the same byte guard after only 2,039 requests. Late byte/request/resource
failure still aborts and discards before publication; it cannot continue uncached
in place. This is no universal 35k/50k/200k, whole-bootstrap speed or playing-strength
qualification. Existing frozen launchers, including J3.21's 20k recipe, are unchanged.

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

The recipe explicitly binds the canonical representation name, ordered features
and tensor width. Storage validates 20 or 41 finite FP32 values against that
recipe; a correctly checksummed row of the other width is still invalid. The
owned directory/SQLite layout is unchanged. Earlier recipes without the explicit
representation field, or entries for another representation/runtime/code identity,
are incompatible priors and are rebuilt through the existing path. They are never
reinterpreted or used to cache predictions. Row-limit opt-in does not change the
storage schema or representation identity.

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
