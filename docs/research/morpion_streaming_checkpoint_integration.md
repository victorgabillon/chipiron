# Morpion streaming checkpoint integration

This focused integration replaces the sharded save branch with Anemone's live-search
writer. JSON saves retain their materialized implementation. Scientific export,
activation, receipt, publication, fsync and recovery code are unchanged.

## Foundation and isolation

The base is `d744b4b7d2947d9e8fe20de02e9bdc3266cbb401`, descending from the
validated production commit `fe0fe7a576f25dfdd6464060495cbd1356b03590`.
The integration implementation is `d8e26e6e9b21156841963e54d4e85d7f44bb7b0f`.
Every intervening change is under docs/research; src and normal tests are identical.
The new branch is `perf/morpion-streaming-checkpoints`, in
`/home/pompote/oldata/victor/morpion-scratch/streaming-checkpoint-20261025/worktree`.
The ext4 filesystem had 51 GiB available initially. Temporary fixtures use
`/home/pompote/oldata/victor/mc25`. All existing worktrees and evidence are protected.

Anemone is exactly `37e48c40ceff430e373fd07bd00fcf41359bafda`. A new wheel was
built from a clean git archive of that commit, without editing its existing worktree.
Its SHA-256 is
`4ab0ffc88b17fb500a3729991fc4fa5c8f43d2b70be875f68a6d2d52ffe47d2f`.
All 289 installed Anemone Python files match that archive. The package version
`0.2.26` alone does not identify this API; adoption requires the reviewed artifact
or a separately released dependency containing the exact implementation.
All 477 installed Chipiron Python files also match the implementation. The benchmark
wheel and final wheel differ only in formatting in runner.py; an AST comparison of
every Python file verifies identical executable semantics. Both full file-hash sets
and wheel/sdist hashes are retained in [sources.json](streaming_checkpoint/evidence/sources.json).

## Small production change

`AnemoneMorpionSearchRunner.save_checkpoint` branches before building any payload.
The sharded branch passes the existing live runtime and codec to
`write_sharded_search_checkpoint_from_search`, with split layout and an explicit
constant shard size. Encoding and generation remain the previous defaults.
There is no second serializer, graph, cache, format, persistent metadata or authority.

Node totals come from the returned manifest. Separate typed-payload build time,
anchor/delta counts and payload-only selector telemetry are unavailable for streaming
saves. Aggregate Anemone codec logs remain, alongside operation time, RSS, node count,
compressed bytes and shard counts. The JSON branch retains its existing diagnostics.
No complete payload is reconstructed after writing or during metrics cleanup.

The existing publisher holds exclusive Growth ownership, allocates a unique private
attempt, validates and fsyncs dependencies, performs no-replace immutable publication,
then replaces run_state and fsyncs its directory. The new writer rejects every
nonempty output directory, including incomplete attempts. Failure must retry through
a fresh attempt. No old manifest or partial output is overwritten.
Standalone or non-durable callers that reuse a nonempty path now receive the
intentional write-once error and must choose a fresh empty path. This integration
does not add automatic overwrite or cleanup to those callers. Durable Growth
already supplies the required UUID attempt directory under its existing owner.

Tests make the materialized builder, payload-count helper, payload-selector helper
and full ordered-node-list helper fail if reached by a sharded save. The same test
compares every output byte with the materialized API at matching partitions.

## Measurement method

[measure.py](streaming_checkpoint/measure.py) uses existing StageMeasurement, process
CPU/I/O counters, 2 ms RSS sampling and optional tracemalloc. It does not run alongside
CI or other benchmarks. Each save generates the same seeded legal graph in a fresh
Chipiron process; the 1k population is completed before beginning 5k.
The populations are 1,003 nodes / 1,114 branches and 5,013 nodes / 5,931 branches,
using the existing default max/Linoo runner with seed 4. They use real legal states
and transpositions, not tiny invented state references. The measured checkpoint
encoding is stdlib JSON with jsonl_zst split shards and generation=None, exactly as
the former caller selected. Neural bundles and rollouts are exercised separately
by the actual bootstrap recipes.
The old save method is extracted verbatim from the pinned Git source; only its
physical shard size is overridden for matched comparisons. Both use the exact new
Anemone wheel, whose materialized encoder remains the correctness reference.

Measurement ends before fingerprints and complete offline oracles are constructed.
Save times cover the actual runner method, including its cleanup, but exclude the
outer capsule transaction's fsync/rename/authority commit. Those operations are
unchanged and are exercised in the real publication and death tests. No end-to-end
publication speedup is inferred from the checkpoint component timings.
Fresh separate processes measure complete split-shard decoding plus supported
restore/certificate validation, and cold runtime restore alone. Full validation
parses every persisted field, including selector/RNG/expansions, using the existing
research decoder, then restores and invokes the existing node-certificate validator.
It is an offline correctness cost, not a new production parser; its full in-memory
objects are excluded from save RSS. Capsule inventory/source checks remain unchanged.

RSS includes Torch and domain imports, the necessary resident graph, allocator
arenas, shard buffers and selector work. Sampling and kernel high-water readings
can differ. Tracemalloc runs are separate from production timing/RSS runs. OS page
caches are not evicted; cold restore means a fresh process/runtime, not cold disk.
Allocated bytes count unique file inodes via st_blocks and exclude filesystem
journal/directory overhead. This is a synthetic workload, not a record-scale forecast.

The first pass loaded the frozen old method only in the old process. Its AST left
allocator residency before measurement. The final pass compiles and retains that
same method in **both** processes before GC, choosing the actual caller afterward.
All final files equal the first-pass files exactly. The first-pass observations
remain in the evidence, but the tables below use only the symmetric final pass.
There is no evidence of a reliable RSS benefit at 1k; its sub-MiB differences are
sensitive to allocator residency and sampling.

## Final matched measurements and shard selection

Full raw values, CPU/I/O counters, lifetime high-water RSS, source fingerprints and
all file hashes are in [measurements.json](streaming_checkpoint/evidence/measurements.json).
Each old/new pair uses identical newly generated states and physical partitioning.
Times below are seconds; RAM is MiB. These are single controlled pairs, not
statistically established speedups.

| Nodes | Nodes/shard | Old/new save wall | Old/new CPU | Old/new extra RSS | Old/new total sampled RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 256 | 0.411 / 0.405 | 0.415 / 0.410 | 0.55 / 0.55 | 307.04 / 306.92 |
| 1,003 | 1,024 | 0.426 / 0.412 | 0.430 / 0.413 | 1.38 / 0.50 | 307.84 / 307.37 |
| 1,003 | 4,096 | 0.407 / 0.410 | 0.412 / 0.415 | 1.25 / 1.06 | 307.82 / 307.50 |
| 5,013 | 256 | 2.984 / 2.935 | 3.002 / 2.943 | 6.94 / 2.49 | 372.41 / 368.06 |
| 5,013 | **1,024** | **2.976 / 3.065** | **2.988 / 3.102** | **9.57 / 4.64** | **375.36 / 370.18** |
| 5,013 | 4,096 | 3.016 / 2.929 | 3.035 / 2.941 | 13.92 / 11.25 | 379.25 / 376.99 |

The selected constant is **1,024 nodes/shard**: 51.6% less extra RSS than the
matched materialized path at 5k, with 3.0% more wall time / 3.8% more CPU in this
pair. The kernel lifetime peak is 373.52 / 368.73 MiB. Physical files fall from
64 at B=256 to 19; B=4096 saves another nine files but more than doubles streaming
extra RSS. No automatic controller or second Chipiron buffer is introduced.

For the actual previous setting, **100,000 nodes/shard**, the same 5,013-node
materialized caller uses 16.40 MiB extra / 382.08 MiB total sampled RSS, 2.950 s wall
and 2.964 s CPU. The new setting reduces extra RSS by **71.7%**, with 3.9% more wall
time / 4.7% more CPU. Its seven files occupy 426,720 logical / 438,272 allocated
bytes. The selected 19-file output increases allocation 7.5%, while retaining the
complete decoded checkpoint. The extra old-default run measures the real previous
configuration; it is not a proposed larger streaming batch.

The following byte and file counts are identical within every matched old/new pair.
Physical shard files exclude manifest.json; all files include it.

| Nodes | B | Shard files / all files | Logical / allocated bytes | Manifest bytes | Full validation old/new | Cold restore old/new |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,003 | 256 | 15 / 16 | 86,101 / 122,880 | 3,964 | 1.145 / 1.105 | 0.151 / 0.148 |
| 1,003 | 1,024 | 6 / 7 | 86,560 / 98,304 | 1,737 | 1.102 / 1.111 | 0.156 / 0.151 |
| 1,003 | 4,096 | 6 / 7 | 86,560 / 98,304 | 1,737 | 1.133 / 1.104 | 0.156 / 0.160 |
| 5,013 | 256 | 63 / 64 | 439,876 / 577,536 | 15,886 | 6.801 / 6.545 | 0.319 / 0.312 |
| 5,013 | 1,024 | 18 / 19 | 436,785 / 471,040 | 4,748 | 6.553 / 6.684 | 0.463 / 0.480 |
| 5,013 | 4,096 | 9 / 10 | 426,546 / 442,368 | 2,490 | 6.687 / 6.626 | 0.504 / 0.500 |

Six matched partition pairs are byte-identical, including manifests. Across all
partitions, complete decoded component digests agree; the old 100k default also has
the same full decoded digest. Ten independent pair/cross-partition oracles compare
complete restored payloads and scientific files, then **30 paired future Growth
steps**. Arrays, node/parent/child order, certainty, provenance and RNG are never
filtered out. JSON object comparisons do not discard any fields.

## Allocation attribution

At B=1024/5,013 nodes, separate traced runs retain at most **5,013 old versus 1,024
streaming typed node payloads**. Peak auxiliary Python allocations are 12,671,251
versus 6,118,521 bytes (12.08→5.84 MiB, 51.7% lower).
The node-phase traced high-water is 9,976,632 / 6,118,521 bytes. Selector-payload
allocation increments are 781,401 / 713,206 bytes. Node/cache/shard buffers still
overlap; these categories are not disjoint values to sum.

- Imports/Torch/domain runtime: roughly 294.5 MiB before graph generation.
- Necessary live graph and domain/allocator residency: roughly 365.5 MiB before
  the final 5k saves, about 71 MiB above imports, including the symmetric measurement
  setup. This is not an object-by-object graph size.
- Checkpoint auxiliary serialization/metadata: sampled increments in the table;
  separate tracing establishes the bounded retained typed-node count.
- Selector/expansion/manifest work: remains explicit above and below; it is not
  hidden behind a constant-memory claim.
- Scientific export: a separate cold-restored 5k runtime starts at 314.49 MiB and
  peaks at 320.05 MiB, an additional 5.56 MiB over 27.96 s. This cold graph uses
  checkpoint-backed state handles and is not the same resident allocation as a
  freshly grown graph. This existing scientific writer/canonical-validation cost
  is unchanged; its memory is not charged to checkpoint serialization or claimed
  to disappear. Publication additionally retains its existing SQLite/source
  verification costs.

## Remaining memory and resource costs

Let B be nodes/shard, L_B the complete serialized size of that batch, S=ceil(N/B),
M selector classifications, F candidates, D depths and X latest expansions.
Anemone's node construction/caches/projections are bounded by O(L_B+B), with one
node's branch workspace. Individual state, proof, PV or high-degree rows can be large.
The actual live search remains O(N+E+scientific field data).
Save time still traverses all N nodes and E links, with the existing codec,
selector, encoding and compression costs. Disk retains one complete checkpoint
plus O(S) file/manifest overhead; this is not incremental checkpoint storage.

Selector refresh can still create a temporary O(N) node-ID map. Selector payloads,
sorting and whole-JSON encoding retain O(M+F+D) data; latest expansions retain O(X).
Manifest/shard references cost O(S), with three split families. Restore retains
O(N) shells, state payloads and lookup maps while constructing the necessary graph.
The outer publisher also inventories these files in O(S) memory and performs
per-file hashing/fsync. More physical shards therefore affect publication beyond
the save-call measurements; no history- or node-independent total cost is claimed.
Closed scientific publication still takes its existing O(N) ordered reference list
and performs canonical validation and bounded SQLite work. None of these costs is
made constant by this checkpoint change.

At record scale, one million nodes require 11,725 / 2,935 / 739 total files at
B=256/1024/4096, assuming metadata, selector, expansions and manifest are present.
At 100 million nodes these become 1,171,879 / 292,975 / 73,249. These are calculations,
not enormous generated fixtures. If such populations require large physical shards
and smaller auxiliary buffers simultaneously, the narrow next Anemone change is to
separate serialization_batch_size from node_count_per_shard in its shared writer.
Chipiron must not implement a second buffering layer. Selector and manifest scaling
also require their own demonstrated bottlenecks before further changes.

## Scientific and durability contract

Format 3, sharded format 1 and scientific closed layout/value contract are unchanged.
Exact file equality at the same partition covers ordered nodes, codec parents,
representative parents, incoming/outgoing links, unopened branches, direct/tree Values,
certainties, evaluator versions, PV, selector, both RNG states and expansions.
Different partitions intentionally change filenames, compression blocks and manifest
bytes; complete decoded records and subsequent continuation must agree.

Scientific fields are never normalized away. The old format-2 representative-parent
discrepancy remains in its original evidence; current format 3 persists it separately
from the codec parent and retains the strict continuation witness.

Training completion still does not activate a model. The generated E2E recipe uses
explicit requests for M1→M2→M3, accepted Dataset/Training, atomic patch/cursor receipts
and cold restarts. Actual terminal/FORCED proofs, the FORCED(35)/ESTIMATE(42) witness,
five enabled selected-value-v2 policies, four exact-family gates and nine legacy
policies remain obligations, not claims inferred from serialization alone.

Both original recipes passed **40 fresh processes each**. The opening reaches G8,
277 nodes and 281 edges; the endgame reaches G5 with two nodes and one edge because
its proven tree has no further legal Growth. Both perform two actual accepted
Training transitions, exact pending-patch/cursor commits, M2/M3 activation, repeated
cold restarts and idempotent historical M2 retry after M3. Every check reconstructs
the complete checkpoint, both RNGs, selector and all scientific files. Dataset
reference/streamed JSONL and Reevaluation patch contents are independently compared.

The endgame starts with direct ESTIMATE(42), tree/effective FORCED(35), and one
terminal child. After M3 its direct value is ESTIMATE(30.484771728515625), while
tree/effective remain FORCED(35) and the terminal certificate remains intact.
This is expected explicit neural refresh, not an accidental target-certainty change.
See [initial endgame](streaming_checkpoint/evidence/endgame-initial.json),
[final endgame](streaming_checkpoint/evidence/endgame-summary.json) and
[opening result](streaming_checkpoint/evidence/opening-summary.json).

The reused recipe deliberately retains its original supplied logical producer
revision label `4ada2471...`; it does not discover the executing Git revision.
The actual new execution is d8e26e6e/37e48c40 and is independently bound by the source
manifest. Full producer/API/model/config identities are preserved and verified inside
the generated fixtures. E2E correctness runs overlap CI; their wall times are not
used as performance comparisons. All checkpoint benchmark pairs ran serially.

## Error handling and validation evidence

Four new materialization guards fail against the exact old caller, and pass against
the installed integration. The 20 focused checkpoint/error cases pass, as do both
unchanged strict 1k format-3 witnesses. FAST passes 152 tests. New errors during
node serialization, shard writing and manifest publication use a genuinely accepted
Training workspace: authority remains byte-identical M1, capsule validation succeeds,
and retry on a fresh attempt commits M2 exactly once.

The existing actual process-death helper now additionally dies after writing half
a compressed shard, before manifest completion, and immediately after the private
streamed checkpoint completes. Existing death boundaries cover model load/inference,
backup/selector, complete first shard, science, output/hash validation, file/directory
fsync, before/after capsule rename, before/after authority replacement, discovery
repair and producer restart. Recovery must select complete M1 or complete M2 and
preserve exact receipts. No production fault seam or persistence primitive was added.

Production delta is **+63/−45**, net **+18**, in one existing module. Most changed
lines move existing diagnostics into the JSON branch. Tests add 185 lines across a
new focused file and the existing death helper. The sole new production concept is
an explicit fixed shard-size constant; all serialization and durability abstractions
already exist. Two inherited research entrypoints receive one missing docstring
each so the repository-wide Ruff gate passes; they change no execution.

Ruff checks and all 754 format checks pass. The strict static ratchet reports 206
inherited diagnostics versus baseline 209, **zero new**. Wheel/sdist build, installed
distribution contracts, toolchain and source-hash checks pass. Twine validates both
artifacts using the existing Anemone tool environment; it does not import runtime
dependencies. No dependency is upgraded or published for that metadata check.

PR passes **2,662 tests with two CUDA-unavailable skips**. All **90 additional
exhaustive fault cases** pass. Their union covers the complete affected process-death
matrices; the new three streamed death boundaries remain in PR and FULL.
The PR run includes all 196 existing certainty/Dataset/atomic-science cases and all
25 activation death/restart cases. FAST is a strict subset of PR; the existing
execution recorder reports no missing or duplicate test phases.
The single frozen-implementation **FULL run passes 2,809 tests**, with the same
two CUDA-unavailable skips and **80.42% coverage**, in 1,470.89 seconds. PR takes
978.44 seconds; the separate exhaustive matrix takes 494.50 seconds. Both may
overlap independent correctness work, so these are validation durations rather
than CI performance comparisons. FAST⊂PR⊂FULL and exhaustive⊂FULL are verified.
All installed source hashes still match after validation. These are local
equivalents of the required CI gates, not a remote GitHub workflow run.
See [validation.json](streaming_checkpoint/evidence/validation.json).

The implementation is followed only by report/evidence changes; FULL is not rerun
for those. Pytest's existing failed-only retention policy applies to new disposable
CI temporary directories; all experiment workspaces, raw logs, failures, source
fingerprints and reports remain. It does not retire any historical pipeline artifact.
The preservation audit matches all **48 pre-existing Chipiron and 14 Anemone**
worktree registrations, HEADs, statuses and tracked-diff fingerprints.
See [protected.json](streaming_checkpoint/evidence/protected.json).

Early harness errors are retained in scratch logs: wrong expected exception class
for nonempty output, the typed loader's unsupported split layout, relative export
paths, and a mixed test/research collection's scripts-package collision. They were
corrected in tests/measurement invocation without relaxing assertions or changing
scientific code. An initial before-test selected this checkout through pytest's source
path setup; the verified rerun explicitly selects the pinned old source and fails
at its full-payload builder in all four cases. Only verified final runs support the
reported parity. No scientific mismatch was normalized away.

## Reproduction and graphical review

Use the pinned local wheels and source-hash manifest, PYTHONDONTWRITEBYTECODE=1,
PYTHONPATH=., CHIPIRON_TEST_INSTALLED=1, and a short new secondary TMPDIR.
All destinations must be new private directories.

```bash
python -m docs.research.streaming_checkpoint.measure save NEW_DIRECTORY \
  --mode stream --nodes 5000 --batch 1024
python -m docs.research.streaming_checkpoint.measure validate NEW_DIRECTORY
python -m docs.research.streaming_checkpoint.measure restore NEW_DIRECTORY
python -m docs.research.e2e_bootstrap.reproduce NEW_E2E_DIRECTORY
python -m docs.research.e2e_bootstrap.reproduce NEW_ENDGAME_DIRECTORY --endgame
make test-fast
make test-pr
make test-full
```

Open this isolated worktree in VS Code. For graphical production/test review:

```bash
git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt d744b4b7 HEAD -- src tests
```

Close each diff tab to advance. This changes no global Git configuration.
The implementation commit is separately reviewable against d744b4b7. Before
integration elsewhere, establish the existing production stack and the reviewed
Anemone artifact containing 37e48c40; version 0.2.26 alone is insufficient.
Keep the code commit separate from this research evidence. No sibling stack is
reapplied and no duplicate-code PR is opened. The default remains JSON checkpoints;
the existing explicit sharded setting selects the new writer.
No package publication, production activation, historical mutation, gen38 operation,
distributed bootstrap worker, merge or release is part of this task.
