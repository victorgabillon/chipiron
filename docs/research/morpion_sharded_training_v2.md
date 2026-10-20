# Morpion scientific sharded export v2

This is an offline writer and full reference reader. No runner, Dataset,
Reevaluation, launch argument, production setting or existing v1 implementation
selects v2. The bounded reader integration is a separate PR.

Base: Chipiron PR #77, `67a7cfd1955f43d5947af01dbb324fb5efd93a0d`.
The retained source audit is `/tmp/morpion-scientific-audit-20261020-1ZvDXi/report.html`
and its evidence bundle SHA256 is
`785968e9fed3154ba4d2ebe34ad647a871b73c7f9503e2f21934e09aa12d0f65`.
The historical format audit SHA256 is
`3ff6868603cd29f13620811917ec3d0e46d1fd90f51f6d73d95083dd8ff1107e`.
These identify evidence, not a production source generation.

## Scientific definitions and invariants

The pinned authoritative APIs are Anemone 0.2.26
`training_export.builders.build_training_node_snapshot`,
`node_evaluation.common.value_snapshot.snapshot_node_values`,
`canonical_value.validate_value_semantics`, and the live evaluation's
`get_effective_value_candidate`. Valanga 0.2.5 supplies `Value` and certainty;
Atomheart 0.2.11 supplies the canonical state-reference codec. No upstream
Anemone or Atomheart code is changed.

* Direct D is the local evaluator's raw `Value.score`.
* Tree T is the child/subtree-derived raw score. The compatibility backup alias
  equals T, including null. No initial-pattern point offset is added.
* Effective E and its source are sampled from the actual live candidate API,
  which accounts for opening completeness, objective ordering and certainty.
  The exporter does not reimplement that selection. Valid sources are `none`,
  `direct_self`, `tree_child`.
* The declared target is strictly D, T or E. Missing T means a null tree target,
  even with D present. No implicit fallback is inserted. Target selection is
  declared independently for every node, allowing an explicit change between
  generations. Each generation's metadata declares its default.
* Full serialized `Value` payloads accompany all four scalar views using
  Anemone's public checkpoint value serializer. Score, certainty, over-event
  outcome/termination/winner and optional tagged line atoms remain associated.
  Effective and target payloads must equal their explicitly selected inputs.
* `is_exact` and `is_terminal` retain the authoritative builder's observed
  **node** flags. They do not certify every stored scalar. FORCED and TERMINAL
  are exact; only TERMINAL asserts own-state terminality. Parent backups must
  not inherit a child's terminality. No new scientific target policy is added.
* Current parent IDs, child IDs, depth, visit count, flags and metadata are
  generation updates. Parent/child edge sets must be reciprocal in a closed
  DAG. The supplied ID/link order and parallel child-link multiplicity are
  preserved exactly. Depth is sampled, never inferred from topology.
* Root membership, unique node IDs and contiguous export order are required.
  Old generations remain independently readable after later publications.
* The caller exports a quiescent graph, with a single writer, and must not
  recycle a node ID for a different immutable state within this export root.
  Current references are created once, in the creation generation. Checking
  every reused node's state would defeat lazy reuse and is not performed.

### Certainty association gate

The controlled partial-node case D=FORCED(9), T=ESTIMATE(3) has E=FORCED(9),
`direct_self`, node `is_exact=true`, and tree target ESTIMATE(3). V2 represents
all of these independently. It records `backup_exactness_compatible=false`.
This is an observed compatibility diagnostic, not a new certainty inference.
For a fully opened node E=T and the node flag becomes false in this fixture.
Equal-score direct-exact/tree-estimated ties also select the actual candidate
according to Anemone's objective.

Current Dataset code can label backup 3 as a hard exact anchor using the node
flag. All existing nine policies are preserved here. The risk is **not proved
unreachable in all production searches**; the small actual-search tests have
no incompatible nodes. Before activation, prove a producer invariant over the
supported evaluation families or separately review a versioned Dataset
confidence policy using the per-value payload. Export parity alone does not
settle that scientific question.

## Replay-reference contract: option A

For **new nodes only**, obtain the actual Morpion state, unwrap Chipiron's
MorpionState if necessary, and call exactly Atomheart 0.2.11
`dump_state_ref`. Normalize tuple/list containers only through the same JSON
serialization boundary used by flat saved snapshots. This is the deterministic
legal canonical replay encoding used by the flat runner, with no D4/orientation
normalization. No entire flat snapshot is constructed by the writer.

For the commuting witness moves A=`9222105397312061435`,
B=`9222668330085744635`, both raw checkpoint orders are legal and fully
state-equivalent. V1 preserves B,A; v2 and the authoritative saved flat snapshot
both encode A,B, **byte-identically in serialized row references**. The reference
reader decodes/redumps each immutable ref and refuses noncanonical/malformed
references even if a modified shard is rehashed. It checks concrete state, not
only a symmetry-invariant tag.

The initial canonicalization cost is intentional and measured. Reused states
are not accessed for reference creation in later generations. New state
resolution can populate the input graph's checkpoint resolver cache; the writer
makes no bounded-total-memory claim. The existing efficient v1 writer remains
unchanged. No conversion of old artifacts happens automatically.

## Exact on-disk schema

Every envelope has these mandatory identities:

```
format_kind = "morpion_scientific_training_export"
format_version = 2                     # integer, never coerced
scientific_contract = "anemone-values-canonical-morpion-1"
role = "catalog" | "generation" | "records" | "updates"
```

A v2 root must be a new empty directory. Its layout is:

```
manifest.json
generation_000001/{generation.json,records.jsonl,updates.jsonl}
generation_000002/{generation.json,records.jsonl,updates.jsonl}
...
```

| File/row | Required fields in addition to envelope |
| --- | --- |
| Catalog | `latest: int`, `v2_generations: {decimal generation: relative generation.json path}`, `producer` |
| Generation descriptor | `generation: int`, `snapshot`, `producer`, `creation_index: {node_id: creation generation}`, `record_shards: [shard descriptor]`, `update_shard: shard descriptor` |
| `snapshot` object | `root_node_id: str`, `node_count: int`, `created_at_unix_s: number|null`, `metadata: object` |
| Shard descriptor | `path: relative str`, `generation: int`, `count: int`, `sha256: str` |
| First JSONL line | Corresponding `records` or `updates` envelope plus `generation`; not counted as a row |
| Immutable record | `node_id: str`, `creation_generation: int`, `state_ref_payload: canonical non-null JSON reference` |
| Mutable update | All fields below; no state ref, no creation-time parent fallback |

Every update has **exactly** these keys:

```
node_id: str
order_index: nonnegative int
parent_ids: list[str]                  # current incoming links, ordered
child_ids: list[str]                   # current outgoing links, ordered; duplicates retained
depth: nonnegative int
direct_value_scalar: finite number|null
tree_value_scalar: finite number|null
effective_value_scalar: finite number|null
effective_value_source: "none"|"direct_self"|"tree_child"
target_value_scalar: finite number|null
backed_up_value_scalar: finite number|null
is_terminal: bool
is_exact: bool
over_event_label: str|null
visit_count: nonnegative int|null
metadata: object
value_contract: {
  target_source: "tree_value"|"direct_value"|"effective_value",
  values: {direct, tree, effective, target: SerializedValue|null},
  certainties: {direct, tree, effective, target: "ESTIMATE"|"FORCED"|"TERMINAL"|null},
  all_branches_generated: bool,
  backup_exactness_compatible: bool
}
```

`SerializedValue` is the public Anemone checkpoint serialization:
`{score, certainty, over_event, line}`. `over_event` is null or
`{outcome, termination, winner}`; enum and tuple atoms use Anemone's tagged
encoding. The score and certainty must agree with the corresponding scalar and
certainty view. No present scalar may lose its certainty; absence is explicit
null, never an omitted mandatory field. There is no guess from another field.

`producer` records explicit caller `identity` (nonempty `revision`, `evaluator`,
64-character `config_sha256`, plus optional lineage), installed package versions
for Chipiron/Anemone/Atomheart/Valanga, SHA256 fingerprints of the four authoritative
API source modules, and
`state_ref_contract="atomheart-0.2.11-dump-state-ref"`. The writer requires the
pinned scientific dependency versions and a stable identity throughout a root.
The full reference result exposes the exact flat-compatible `snapshot`, separate
`value_contracts`, and `provenance` including source kind/version/contract,
generation and producer. Source lineage is not hidden inside node metadata or
silently inserted into the flat-compatible snapshot projection.

Generation numbers start at 1 and must be consecutive. Every descriptor includes
all earlier immutable record-shard descriptors, plus its own (possibly empty)
record shard. The creation index is cumulative; generation membership and order
come only from that generation's updates. Paths must stay inside the export
root. Hashes, shard envelopes, counts, creation generations and update order are
validated. A shard is rehashed after reading to detect concurrent changes.

Publication stages rows and descriptor in a private directory, renames it to a
new immutable generation directory, then atomically replaces the catalog last.
Failures before directory rename remove staging. Failure after rename leaves an
**unpublished orphan**: reference loading refuses it, and retry refuses to
overwrite it. Recovery requires explicit inspection/removal of newly created
scratch artifacts. No automatic recovery/migration or historical mutation is
implemented. This is atomic process-interruption publication, not an fsync-backed
power-loss durability claim. Concurrent writers are outside this offline API's
contract.

## Compatibility

| Artifact | Existing v1 loader / PR #75/#76 | New v2 full reference reader |
| --- | --- | --- |
| Historical sharded v1 | Unchanged legacy interpretation, including direct target fallback, derived E/source and creation parents | Refuses; no silent repair |
| Historical flat v1 | Existing loader behavior, including meaningful absent/null distinctions | Refuses |
| Explicit scientific v2 | Refuses: different kind, namespace and envelope; legacy loader cannot obtain a v1 generation/root/shards | Strict v2 validation; exact snapshot plus value contracts/provenance |
| Future/mixed/unknown contract/version | No new fallback introduced | Refuses |

V2 deliberately does not expose the v1 `generation_manifests`, `node_shard_path`,
`update_shard_path`, or top-level generation `root_node_id` fields. A version bump
alone would not protect against the old loader, which ignores v1 version
headers. The incompatible envelope makes its attempted read fail. No dispatcher
is widened to route v2 into that loader.

No v1 migration is implemented. Lost E selection/provenance, opening completeness,
per-value certainty and original current-parent ordering cannot generally be
reconstructed from saved v1. Inverting children recovers an edge set only under
an independently established closed-world assumption. Any future recovery must
create a separate artifact with original source hashes and explicit unknowns;
regeneration from the matching saved search checkpoint is preferable. Training
exports remain distinct from continuation checkpoints and do not contain full
selector/RNG/runtime state.

## Validation and before/after evidence

`tests/environments/test_morpion_sharded_training_v2.py` retains the actual v1
failures alongside the corrected v2 results:

| Counterexample | Frozen v1 | New v2 / authoritative contract |
| --- | --- | --- |
| Old join gains second parent | parent `[1]`, despite second incoming edge | current parents `[1,2]` |
| Partial D=9,T=3 | E=3, source `tree_value` | E=9, source `direct_self` |
| Direct-only exact D=4, tree target | target 4 | target null, direct certainty preserved |
| Checkpoint replay B,A | serialized B,A | deterministic canonical A,B |
| Direct exact / tree estimated | node flag cannot associate every scalar | explicit D/T/E/target values and certainties; compatibility flag false |

The tests compare every serialized scientific field, root, timestamp and complete
snapshot metadata against the **saved/reloaded authoritative flat snapshot**.
They do not drop source, target or parents. Cases cover all three target sources,
partial/full opening, absent D/T, score ties and certainty ties, declared terminal
and forced values, current depth/metadata/visits, three generations, deterministic
repeat bytes and retained prior generations. Declared terminal/forced fixture
values are contract inputs, not claims that the tiny fixture boards are solved.

648 actual Dataset comparisons cover two generations × nine policies × both
backup preferences × blends 0/.25/1 × six filter/truncation choices. Every row
field, counter/summary and complete JSONL byte matches. Unlike v1's 324 replay
mismatches, **all 648 v2 JSONL pairs match**. The real-search/checkpoint tests also
exercise all nine policies on original and restored small live graphs, preserve
RNG state/tree sizes through bounded continuation, and compare all snapshot
fields. They start a local deterministic runner, never a bootstrap worker.

Negative tests cover explicit old/future/coerced/mixed versions, null versus
absent targets, provenance/certainty corruption, nonreciprocal topology, order,
source hashes, rehashed noncanonical refs and interrupted publication on both
sides of directory rename. V1 writer/loader source files are unchanged.

## Performance evidence

[Raw measurements and exact validation evidence](benchmarks/morpion_sharded_training_v2_20261020.json).

| Fixture / format | Initial G1 (s) | Same G2 (s) | Growth G3 (s) | Peak RSS (MiB) | Three-generation bytes |
| --- | --- | --- | --- | --- | --- |
| 1k flat | 2.059–2.150 | 0.681–0.742 | 0.680–0.852 | 252.24–252.34 | 2,459,288 |
| 1k v1 | 0.045–0.051 | 0.035–0.039 | 0.048–0.053 | 243.11–243.38 | 1,420,128 |
| 1k v2 | 2.154–2.395 | 0.083–0.085 | 0.095–0.104 | 249.92–250.03 | 2,725,367 |
| 5k flat | 19.526–19.659 | 10.099–10.591 | 10.174–11.246 | 361.57–361.91 | 16,400,760 |
| 5k v1 | 0.235–0.241 | 0.175–0.257 | 0.189–0.207 | 278.82–279.14 | 8,470,503 |
| 5k v2 | 19.115–20.296 | 0.417–0.419 | 0.407–0.422 | 344.34–344.41 | 14,506,212 |

Actual G1/G2 counts are 1,003 and 5,013; G3 adds 22 and 18 nodes respectively.
Ranges are two ordered/reverse fresh-process trials on a shared host. Static
analysis overlapped early trials; these are observations, not a controlled
speedup estimate. The initial 5k scientific v2 export costs 19.115–20.296s versus
0.235–0.241s for the lossy v1 contract. Initial v2 canonicalization alone costs
approximately 19s; the full snapshot is never built internally. This cost is
not hidden to match v1 performance.

At 5k, v2 G1 decodes 1,372 anchors + 3,641 deltas (5,013 reconstructed
checkpoint states) and performs 5,013 canonical dumps. Internal codec replay
work is separately counted: 418,374 `_apply_move` calls across decode/dump.
G2 performs **zero state requests, decodes or canonical dumps**. G3 dumps only
18 new already-live states (182 codec replay steps), without checkpoint decodes.
V1 does no G1/G2 state materialization, but has the demonstrated scientific
losses. Flat dumps every state every generation. The 1k counts show the same
pattern (1,003 initial states, 22 new G3 states).

V2 5k throughput is about 247–262 nodes/s initially and 11,953–12,029 nodes/s
for identical G2. Peak process RSS reaches 344.34–344.41 MiB, versus
361.57–361.91 MiB for flat and 278.82–279.14 MiB for v1. This is a total
process high-water mark, including retained input states, **not bounded-memory
Dataset evidence**. Across three generations v2 stores 14,506,212 bytes, flat
16,400,760 and v1 8,470,503. V2 carries additional scientific Value payloads
and certainty/provenance declarations; identical-byte size parity is not claimed.

All six generated generation snapshots match every flat scientific field and
all live Value payloads. Another 54 default-filter/all-nine-policy row pairs
match complete JSONL bytes, beyond the 648 small-fixture cases. Forty-eight
export files repeat byte-for-byte across the two trials. The v1 comparison
retains, rather than normalizes away, its differences: in 5k G3, 5,031 source
labels, 4,664 targets, 3,230 replay refs and **three current-parent lists** differ.

Reproduce in new scratch roots with the pinned dependencies and this source:

```bash
PYTHONPATH=src python scripts/benchmark_morpion_sharded_v2.py \
  --operation prepare --nodes 1000 --root /tmp/NEW-v2-1k
# Run each in its own fresh process, serially; repeat in reversed format order
# using a separate root containing copies of only checkpoint.json/fixture.json.
PYTHONPATH=src python scripts/benchmark_morpion_sharded_v2.py \
  --operation measure --nodes 1000 --root /tmp/NEW-v2-1k --format flat
PYTHONPATH=src python scripts/benchmark_morpion_sharded_v2.py \
  --operation measure --nodes 1000 --root /tmp/NEW-v2-1k --format v1
PYTHONPATH=src python scripts/benchmark_morpion_sharded_v2.py \
  --operation measure --nodes 1000 --root /tmp/NEW-v2-1k --format v2
# Repeat at --nodes 5000. Arrange copies under ROOT/{1k,5k}-trial-{1,2}.
PYTHONPATH=src:. python scripts/validate_morpion_sharded_v2_benchmark.py /tmp/ROOT
```


See the accompanying benchmark JSON for raw fresh-process stage observations.
The generated fixtures are nominal 1k/5k deterministic search graphs; actual
counts are reported rather than silently truncated. Each format uses the same
saved synthetic checkpoint in a fresh process, then exports initial G1, identical
G2, and a bounded-growth G3. Checkpoint load/search preparation and reference
validation are excluded from export wall time. Process peak RSS includes imports,
checkpoint restore and earlier stages, as a lifetime high-water mark.

The benchmark counts actual checkpoint anchor/delta decodes, checkpoint-handle
state requests and canonical dumps. Writer `state_accesses` counts only explicit
new-node reference state accesses; raw operation counts also expose any access
inside upstream value evaluation. RSS is measured by existing `StageMeasurement`
and `process_peak_rss_mb`, never estimated from Python allocations.

## Downstream behavior

Current Dataset policies use D/T, current children, node flags and canonical
state refs; they do not consume the export target/E/provenance fields. Persisting
an explicit export target does not silently redefine those nine policies. The
certainty diagnostic above identifies a shared policy question, despite exact
row parity. The built-in reevaluator decodes refs and uses terminal/backup
handling; v2 production routing remains absent. Its future integration must
retain per-value provenance/confidence and revalidate actual patch outputs.

## Bounded-reader integration and activation plan

1. Keep PR #75's existing `ShardedTrainingGenerationView`; extend its explicit
   kind/version/semantic-contract dispatch and JSONL parser. Do not create another
   reader/index. V1 reconstruction and cache semantics remain frozen.
2. Bump the index/decoder contract identity. Include v2 envelopes, descriptor and
   referenced shard hashes, generation, producer/codec contract in source identity.
   No v1 cache may be reused under v2 semantics. Preserve source checks during
   build, reuse, each deterministic pass and final output publication.
3. Load current parents/depth/values directly from updates; immutable records
   supply only state refs and creation identity. Preserve scalar/source/certainty,
   export order, source generation and metadata without v1 inference/defaults.
4. Expose per-value contracts through the bounded source interface. Keep PR #76's
   bounded disk-backed family passes and the nine existing scientific policies.
   Resolve the certainty association gate explicitly before treating node flags
   as confidence for arbitrary selected scalars.
5. Repeat exact full-reference/indexed comparisons, all-policy JSONL/manifests,
   missing/null and malformed/version cases, source changes, interrupted index/
   output, cache reuse and deterministic multi-pass tests. Measure actual peak
   RSS and throughput at 1k/5k/10k, including canonicalization and reused indices.
6. Only after those gates and a separate activation review, add opt-in Dataset/
   Reevaluation routing. Verify reevaluation decoding/model inputs and terminal
   backup handling. No activation, historical migration, gen38 run, distributed
   worker or merge is part of this change.

Remaining memory limits: the writer receives an in-memory search graph, keeps
O(N) IDs/creation-index and O(N+E) topology for validation, and upstream state
resolution may retain O(N) newly decoded states. The full reference reader is
intentionally O(N) and validates every canonical state ref. Generation descriptors
have cumulative O(N) creation indices and O(G) shard lists. No bounded-total-RSS
or production-ready reader claim is made in this PR.
