# Optional canonical flat linear/MLP5/10/20/41 activation inputs

`CHIPIRON_EVALUATOR_INPUT_CACHE=1` enables derived raw-input reuse during owned
candidate activation's full refresh. It defaults to `0`. This is execution-only
operational storage: it does not alter bootstrap configuration/scientific identity,
learning cadence, accepted weights, claims, source pins, either preflight/restore,
full refresh, canonical backups, scalar forwards or atomic activation publication.
Exact accepted retries never open or require this cache.

Supported consumers are the exact canonical `handcrafted_5_core`,
`handcrafted_10_core`, `handcrafted_20_core` and `handcrafted_41` converters with
standard dynamics and a matching scalar linear or MLP MorpionRegressor in
CPU FP32 evaluation mode. The explicit `strict_math` profile, one intra-op CPU
thread and CPU FP32 tensor defaults are required. Runtime/profile/thread/code
identities remain fenced; no GPU context is needed, including in a ROCm build.
Custom names, reordered features, converter/model subclasses, altered network
structures, normalized targets and token/GPU/batched consumers remain unsupported.

The absent row setting remains 20,000 for every consumer. An explicit valid row
request up to 35,000 can increase admission only for canonical MLP20. Every other
flat consumer, including linear20, retains a 20,000 ceiling. Caller reductions win.
Larger trees use the ordinary uncached path; the 256 MiB SQLite ceiling is unchanged.

Optional execution settings:

| Environment variable suffix after `CHIPIRON_EVALUATOR_INPUT_CACHE` | Default | Meaning |
|---|---|---|
| `_DIR` | `<work>/pipeline/evaluator_inputs` | Absolute dedicated owned root outside immutable artifacts |
| `_MAX_ROWS` | `20000` | Positive decimal request up to `35000`; effective maximum `35000` for qualified MLP20, `20000` for every other flat consumer (including linear20); reductions win |
| `_MAX_MB` | `256` | SQLite entry MiB ceiling, reducible only |

Invalid or above-35,000 row requests are refused, not silently clamped. Default-off,
unsupported-recipe and exact accepted-retry behavior stays unchanged. Admission
uses the total current tree size, including terminal nodes, and an independent
per-refresh request counter. Unique keys or nonterminal counts cannot replace
either bound. The effective and requested limits appear in bounded observations.

35,000 is an upper admission bound, **not a fit guarantee**. The saved B5
32,884→34,426-node adjacency previously fit with a 240.201 MiB complete entry
when MLP20 stored only 20 features. Canonical41 now stores 164 rather than 80
raw bytes per narrow input; that old margin is not current-build fit evidence. A separate large-payload fixture
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

One recipe (`canonical-flat-41-v1`) stores canonical ordered 41-feature FP32
vectors for every admitted flat consumer, independently of its weights/subset.
The original converter already computes all features before selecting a subset.
The adapter performs that full conversion once per miss, then projects by the
explicit canonical name indices. Training and evaluation share the same small
index helper; no Training service is imported by this adapter. Fresh scalar
forwards still execute for every evaluated state. Only exact canonical converter,
dynamics, MorpionRegressor args and linear/Linear-ReLU structures qualify.
Custom/subclass/token/GPU/batch/normalized consumers remain unsupported.

Storage validates exactly 41 finite FP32 values plus the explicit new recipe.
Legacy20 and legacy41 entries rebuild before consumption; a20 row can never
reconstruct41. The ownership marker, directory and SQLite layout are unchanged,
and model/subset switches use the SAME entry. No hidden per-model retained caches.
Consumer representation/model kind/ordered indices and projection time are reported
separately from stored representation and encoding time. Returned tensors are fresh
and detached. The explicit35k node allowance depends on the MLP20 consumer, not the
41-feature storage recipe; linear20 remains20k. Caller reductions still win.

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
