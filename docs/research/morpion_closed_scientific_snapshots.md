# Closed scientific snapshots: offline implementation

This implements option A from `f05d418c42b9292410702f51de2d976fbf693059`,
whose parent is the exact reviewed implementation
`0790c596f52e4d3360a416a385b771cf70049d25`. No implementation was cherry-picked
or duplicated. The local branch is `feat/morpion-closed-scientific-snapshots` in
`/home/pompote/oldata/victor/morpion-scratch/closed-scientific-snapshots-20261022/worktree`.
All new scratch/evidence is on the secondary ext4 disk under that task root.

The writer creates an independently readable immutable-by-ownership directory.
It does **not** commit a run, publish a capsule, change discovery, retire history,
change a model epoch, or activate a worker. Current pipeline classifiers reject
this new format. Closed output is deliberately not accepted as a committed reuse
source yet: the next review unit must teach the existing capsule protocol about
its versioned identity and exact dependency inventory.

## Storage and scientific contracts

The explicit `format_kind` is `morpion_closed_scientific_snapshot`,
`format_version=2`, `scientific_contract=anemone-values-canonical-morpion-1`,
and `storage_layout_version=1`. These fields occur in the descriptor and both
JSONL headers, with explicit generation and role. Neither historical format is
reinterpreted. A directory contains exactly the writer's three files:

* `generation.json`: current generation, root, count, timestamp, metadata,
  full producer identity (including any declared model identity), and exactly two
  local descriptors with byte hashes/counts/generation.
* `records.jsonl`: exactly one immutable canonical state record per current node;
  the original `creation_generation` stays inside that record. File generation
  identifies the owning snapshot, not the creation date of every enclosed node.
* `updates.jsonl`: exactly one complete current scientific row per current node,
  including all parents/children, depth, metadata, flags, scalar aliases,
  actual serialized Values, independent certainties and selected-source contract.

No catalog, creation-index JSON map, prior descriptor, old mutable file, or
historical path is needed to read this directory. Record order is deterministic:
new nodes in current export order, then retained source records in source order.
The complete mutable file's contiguous `order_index` defines scientific export
and Dataset order. Both readers preserve it exactly. Provenance explicitly adds
the new layout identity; comparisons across formats must report this intentional
identity difference rather than erase it. Full and indexed consumers of the
same closed source must agree in literal JSONL bytes, including metadata order.

## Verification and ownership

The caller freezes the live graph and must never recycle a node ID for a different
state. This is the existing writer's identity contract. Genesis requires generation
1. Later writes require the exact scientific head dependency of an actually
committed v2 capsule, with consecutive generation and identical full producer/API/
codec identity. An arbitrary file or matching parent hash does not qualify.

The existing `generation_read_pin` protects source ownership throughout copying.
It checks authoritative run-state membership and hashes the capsule dependencies.
The existing indexed reader freshly hashes its sources and validates current
mutable rows/topology. Matching committed canonical evidence may avoid replaying
old states only when its exact record descriptors, producer, contract, interpreter,
parser and implementation identity match. Missing/stale evidence falls back to full
canonical validation. Every source byte is still checked; no new persistent proof
cache or transaction state exists.

New states are dumped, loaded and canonically redumped using the authoritative
codec. Existing record bytes are copied exactly, using the existing strict lossless
parser to frame single-line JSONL emitted by the historical writer. Historical
multiline JSON is still readable by the ordinary reader; the offline copy path
explicitly rejects it instead of silently changing record bytes. The writer hashes
validated bytes as it emits them and requires the completed file's independent hash
to match before transferring canonical validation in a private scoped context.
This prevents output corruption from acquiring a fresh canonical certificate.
Mutable update bytes are also hashed while regenerated and checked independently
before indexing; valid-looking altered values or metadata cannot inherit that identity.

The output uses the same SQLite index and TEMP graph checks. All mutable rows,
values/certainties, membership, reciprocal edges and DAG acyclicity are checked
again, even if every immutable record was copied. Disposable validation indexes
are outside the source/output and are removed after use. There is no full snapshot
in this writer or indexed path; the explicitly named reference reader is the only
new operation that deliberately materializes the population.

Files and directories are synced with existing primitives. A no-replace atomic
rename exposes the complete output. An interruption before rename leaves no final
snapshot; an abrupt process death can leave a private staging directory, and an
error after rename can leave a complete offline output. Neither is a committed
run. Existing paths are never overwritten and recovery/adoption are not guessed.

## Compatibility and scientific invariants

| Input/consumer | Behavior |
| --- | --- |
| Historical scientific v1 | Existing writer/reader/interpretation and nine Dataset v1 policies retained |
| Historical scientific v2 | Existing envelope, cumulative layout, provenance and value interpretation retained |
| Closed snapshot, default indexed reader | Refused; requires `allow_closed_snapshot=True` |
| Closed snapshot, existing production discovery | Refused; no routing added |
| Closed snapshot, explicit full oracle | Full canonical and scientific validation, original creation origins exposed |
| Closed snapshot, explicit bounded reader | Same parser/index, cursor wrap/order, node reconstruction and source checks |
| Dataset confidence v2 | Same five supported policies; only explicit source provenance recognition is extended |
| Four exact-family policies | Still gated; no cross-state proof rule is introduced |
| Closed snapshot as reuse source | Refused until committed capsule integration exists |

The real endgame retains effective/tree FORCED(35), direct ESTIMATE(42),
selected-source exact-only filtering and terminal target 35. Dataset v1 is not
silently repaired or reinterpreted. Node flags do not certify the selected direct
estimate in confidence v2. Checkpoint serialization, patch receipts, selector and
RNG code are untouched.

The design study's checkpoint representative-parent discrepancy remains open:
7/70/161 node-shell records differed after restore at actual N=1,003/5,013/10,018.
State payloads, Values/runtime, linked children, metadata/RNG and selector matched;
subsequent saves stabilized. This implementation does not normalize those shells
or claim to repair exact raw checkpoint round-trip parity.

## Complexity and remaining work

Let S be total canonical reference bytes (including replay histories), U the
complete mutable bytes, L the largest row/state, and H the dependencies in the
selected historical capsule. A closed snapshot retains Θ(S+U) bytes and O(1)
file descriptors with respect to generations G. The header's integer widths grow
logarithmically; no cumulative entries are serialized. R retained snapshots cost
Θ(R(S+U)); this patch introduces no retention bound or deletion policy.

The writer owns no additional O(N) Python map or snapshot. Beyond the caller's
live graph and supplied metadata, its working memory is O(L + parser buffers +
bounded SQLite page caches). The inherited source pin additionally materializes
the authoritative run-state/receipt metadata and selected capsule inventory;
that older protocol is not bounded independently of G or receipt history. The
closed scientific payload path removes no such authority checks. The full reference reader intentionally uses O(S+U+E)
RAM. The indexed path stores O(S+U+N+E) temporary/cache disk and reads windows with
bounded RAM. SQLite keyed insertion/topological operations add approximately
O((N+E) log N) work for bounded-degree fixtures; large individual adjacency lists
and codec replay complexity remain explicit limits.

A repeated write reads/hashes source dependencies, copies Θ(S) bytes, serializes
Θ(U), builds source/output indexes and verifies topology. It encodes and checks K
new states, not N unchanged states when committed evidence matches. Costs are not
independent of current N, E or state history. Before capsule integration, source
verification still sees H from the historical layout and its cumulative authority;
this offline implementation does not claim to eliminate that historical cost.

Next review unit: add a distinct closed-layout scientific identity/inventory to
the existing capsule adapter, route it only through explicit opt-in, bind the
same complete checkpoint and exact receipt/model dependencies, and keep the
single atomic `run_state.json` commit point. Test rename/fsync/commit interruption,
old/current selected pins, corruption and restart before accepting closed output
as a committed copy source. Only afterward consider reachability reporting and
separate reviewed retirement. Do not use age, claim expiry or generation counts
as proof that readers/dependencies are dead.

Existing cumulative catalog creation, creation-index JSON materialization,
historical descriptor copying and cumulative source-closure assembly can leave
the new-format hot path once that adapter exists. Historical readers remain for
reproducibility. No broad compatibility wrapper, second index, persistent ledger,
or speculative cache is needed. Keep this implementation local until the sibling
scientific/streaming/durability stacks have an approved common base; then review
this offline layout independently from its capsule integration.

## Reproduction and validation evidence

Use the existing Python 3.13 scientific dependencies from the pinned integration.
Run the research driver from this worktree with `PYTHONPATH=src:.`; set `TMPDIR`,
application output, and Python/tool caches to new secondary-disk scratch. For
example, after the smaller size succeeds:

```sh
python -m docs.research.closed_scientific_snapshot.reproduce /new/secondary/scratch/1000 --count 1000 --datasets --legacy
python -m docs.research.closed_scientific_snapshot.reproduce /new/secondary/scratch/5000 --count 5000 --datasets
python -m docs.research.closed_scientific_snapshot.reproduce /new/secondary/scratch/10000 --count 10000 --datasets
```

Each stage runs serially in a fresh process. The driver calls the existing
`StageMeasurement`, codec counters, process peak-RSS/CPU/I/O/disk helpers and
scientific fixture generator. No page-cache eviction is claimed. Declared Values
and duplicate boards under different node IDs are controlled resource inputs, not
proofs of game optima or a realistic unique-state runtime memory footprint.
The source publication fixture uses a tiny scalar checkpoint only to exercise
actual committed membership/pins; it is not a runtime-checkpoint cost benchmark.

The indexed reader measurements retain no full oracle. Dataset parity processes
intentionally retain the full reference alongside the indexed result; their RSS
must not be reported as a bounded-only memory measurement. The original update
rows are compared literally with the committed v2 source; full/indexed digests
include every serialized node field, complete value contract and original creation
identity. Dataset comparisons check each complete JSONL line and header bytes.
The source format/layout provenance change is explicit, not normalized away.

## Measured resource costs

Seconds and MiB; one fresh process per stage, warm OS page cache possible.

| N / extra replay moves | Initial complete write | Unchanged write | Write with K=N/100 new nodes | Snapshot MiB | Unchanged writer peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 / 0 | 4.991 | 1.177 | 1.281 | 1.113 | 285.8 |
| 5,000 / 0 | 37.859 | 4.761 | 5.068 | 5.795 | 290.6 |
| 10,000 / 0 | 87.426 | 8.384 | 9.389 | 11.781 | 296.0 |
| 1,000 / 20 | 17.073 | 1.191 | 1.368 | 1.496 | 286.4 |

Every unchanged write decoded and encoded **zero** states. Growing writes encoded/validated exactly 10/50/100 new states (10 in the longer-history case). Initial writes validate every new canonical reference; they perform N loads and 2N dumps (encode plus canonical redump).

| N / extra moves | Source verification/index s | State copy/parse/check s | Current update serialization s | Output validation/hash s |
| --- | ---: | ---: | ---: | ---: |
| 1,000 / 0 | 0.542 | 0.029 | 0.077 | 0.482 |
| 5,000 / 0 | 2.153 | 0.157 | 0.398 | 1.971 |
| 10,000 / 0 | 3.737 | 0.279 | 0.713 | 3.550 |
| 1,000 / 20 | 0.555 | 0.037 | 0.073 | 0.480 |

These sequential component timers exclude some setup, SQL membership and final sync work. Copy time includes strict framing, membership selection and the source end check, so it is not equivalent to the design study’s bare file-copy measurement. Source verification includes inherited capsule checks and its fresh disposable index.

| N / extra moves | Cold initialization s | Reused initialization s | Cold open + full scan/check s | Reused open + full scan/check s | Full reference load s | Cold / reuse / reference peak RSS MiB | Index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1,000 / 0 | 3.409 | 0.016 | 3.483 | 0.094 | 2.984 | 285.1 / 284.3 / 290.5 | 1.645 |
| 5,000 / 0 | 25.264 | 0.058 | 25.631 | 0.418 | 23.681 | 286.6 / 284.4 / 319.1 | 8.520 |
| 10,000 / 0 | 55.772 | 0.114 | 56.523 | 0.850 | 53.162 | 288.1 / 284.4 / 355.0 | 17.172 |
| 1,000 / 20 | 11.382 | 0.016 | 11.459 | 0.093 | 10.788 | 285.2 / 284.2 / 291.5 | 2.129 |

The bounded full scan also obtains every value contract and creation origin by selected-node queries. Reused initialization verifies hashes/cache contents; it is not stat-only acceptance. Cold and full readers still perform N canonical checks. Writer and bounded-reader RSS include imports; writer RSS also includes the caller-owned synthetic graph. These are actual process high-water measurements, not allocation estimates.

| N | Confidence-v2 Dataset throughput across five policies, rows/s | Paired complete JSONL rows |
| --- | ---: | ---: |
| 1,000 | 1,844–1,954 | 5,000 |
| 5,000 | 2,045–2,100 | 25,000 |
| 10,000 | 1,942–2,018 | 50,000 |

Dataset timing includes family construction, summaries, streaming output and source checks, on an already-open reused view. Its parity process holds the full oracle and is not used for bounded-RSS claims. Nine legacy policies add 9,000 exact 1k-row comparisons. Total: **89,000 paired JSONL rows**, **34,170 complete scientific node comparisons** (unchanged and growing populations, including long histories), and **17,000 literal current-update row comparisons against the committed historical-v2 writer**. Small regression fixtures additionally exercise changed certainties, late DAG parents, nulls, terminal proofs, gates, metadata, cursor wrap and patch rows.

At 10k, unchanged-write user+system CPU was **3.717 s**, logical bytes read **294.084 MiB**, physical bytes read **12 KiB**, and physical bytes written **73.578 MiB** (including disposable SQLite work). Raw syscall, CPU, I/O, codec, Python fsync counters, disk inventories and source fingerprints are retained in [measurements](closed_scientific_snapshot/evidence/measurements.json). Python fsync counters do not count SQLite’s internal C-level sync calls.

The design study’s matched short-history complete re-encoding component was 1.470/12.134/27.358 s at 1k/5k/10k. The new unchanged end-to-end offline writes are 1.177/4.761/8.384 s, **including validation absent from that old component measurement**. Conversely, new initial writes cost 4.991/37.859/87.426 s because they include authoritative load/redump checks and bounded topology/index validation. Do not present initial writes as faster, or compare the design’s 0.890 s real-10k copy-plus-update component directly with a complete validated write. The original real-search 502 MiB writer RSS is not a matched memory comparison: these controlled graphs share board objects.

At 1k, twenty extra replay moves increased initial write time from 4.991 to 17.073 s and cold initialization from 3.409 to 11.382 s; unchanged writes stayed at 1.177/1.191 s. This demonstrates avoiding repeated canonical reconstruction, not history-independent total cost at arbitrary S, N, E or G. Every retained complete snapshot still costs Θ(S+U).

Zero unchanged decodes apply when the committed source carries evidence from the **current** validator. Older/stale/missing receipts conservatively trigger N canonical validations. The regression suite checks this fallback explicitly. The measurements use only newly generated sources with matching evidence; they do not promise the same fast path when first converting a historical capsule.

## Final regression and static validation

The final relevant regression run passed **523 tests in 354.54 seconds**:
44 closed-snapshot regressions plus 479 existing scientific reader/writer,
durable-v2 publication/crash, checkpoint/receipt/parity, selected-generation
verification and Dataset confidence/value-certainty tests. See the exact
[pytest result](closed_scientific_snapshot/evidence/regressions-final.txt).
This is the relevant regression suite, not a claim that every installed-package
tox environment was run.

Production Ruff checks [passed](closed_scientific_snapshot/evidence/ruff-final.txt).
The new test and research scripts passed the
[E4/E7/E9/F/I checks](closed_scientific_snapshot/evidence/ruff-support.txt).
Targeted mypy with `--follow-imports=silent` passed for the writer, indexed reader
and shared validation evidence ([result](closed_scientific_snapshot/evidence/mypy-final.txt));
the existing unused PySide6 configuration note remains. `git diff --check` passed.

The [source manifest](closed_scientific_snapshot/evidence/source-manifest.json)
records exact ancestry and SHA256 hashes of the six production files used by
these measurements. All 31 pre-existing worktrees retained their original HEAD,
status and tracked diff, including the dirty research checkout and design/PR
worktrees ([comparison](closed_scientific_snapshot/evidence/protected-worktrees.json)).
Only newly generated synthetic fixtures were used. There was no production
activation, real continuation, historical deletion, distributed worker, push or
PR creation.

## Production code delta and necessary interfaces

Relative to the exact design base, six production files contain **786 added and
81 removed physical lines**, a net **+705**. This includes the new 396-line offline
writer/reference module; five existing files contribute 390 additions and 81
removals. Tests, measurement scripts and this report are excluded. No historical
reader is removed because historical artifacts retain their original meaning.

| Interface/change | Why this unit needs it |
| --- | --- |
| Explicit closed envelope and strict three-file manifest reader | Distinguish the independently readable layout without reinterpreting v1/v2 |
| Shared scientific header and evidence validators | Reuse existing semantics and remove duplicated validation bodies |
| Raw-record byte iterator over the existing parser | Copy verified canonical bytes exactly, without another parser or re-encoding |
| Selected creation-generation lookup | Expose original origins using the existing SQLite index, without a population-sized Python map |
| Private scoped canonical-validation transfer | Reuse completed validation only for exact verified bytes; no persistent proof cache or new ledger |
| Offline writer and full reference oracle | Separate bounded writing/indexing from the intentionally materializing correctness oracle |
| Closed-provenance recognition in Dataset confidence | Apply the existing selected-source contract to the explicit new layout without changing policy |

The new functionality is a local review unit, not a generalized storage framework.
Future durable integration can remove cumulative catalogs, creation-map
materialization and historical descriptor/closure assembly from this layout's hot
path. Those removals require the next capsule adapter and do not justify deleting
historical compatibility now.
