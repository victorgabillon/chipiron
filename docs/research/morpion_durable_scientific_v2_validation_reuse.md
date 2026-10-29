# Repeated durable scientific v2 publication: committed canonical validation reuse

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Local, explicitly opt-in investigation, 2026-10-21. No historical artifacts,
production routing, existing PR branches, or validated original worktrees changed.

## Source and isolated scope

The exact base is `69534167486f980afebc4a07af4e9f6c95784370`, containing the prior
publication implementation `27fa302a7768d55776a6b58d025d1cc1ae518e8b` above the
consolidated scientific/durability stack `9b08cdf77202cca07f41c1ade4aad9e737e647e6`.
The implementation commits are `bdad537a` and
`c1c737ca0d9fb76f9a9d2ad68d0f999a4c9f0be1` (final code).
The new branch is `perf/morpion-durable-scientific-v2-reuse`. Its isolated worktree
is under `/home/pompote/oldata/victor/morpion-scratch/durable-v2-reuse-20261021/`.
The original 26 worktrees were fingerprinted before starting.

The [prior protocol, crash matrix and pinned PR dependency graph](morpion_durable_scientific_v2_publication.md)
remain prerequisites. This change does not replay them. It changes only the private
publication validator and adds versioned evidence to its existing capsule seal.
Scientific exports, public reader routing, Dataset policies, Reevaluation source
identity, checkpoint/receipt contracts, retention and atomic publication remain intact.

The secondary ext4 partition had approximately 79 GiB available. The sandbox's
actual write probe returned EROFS; an authorized host probe verified file and
directory fsync before any new worktree, fixture or cache was placed there. All
new scratch data is on that partition. Existing `/tmp` worktrees/evidence were
read only. Dependency packages were reused read only in a separate CI environment.

## Evidence contract and proof

A successful publication now includes `canonical_validation` in its existing
capsule descriptor, whose exact digest is already committed in `run_state.json`.
There is no second journal, SQLite framework or transaction authority.
The evidence has kind `morpion-immutable-canonical-validation`, version 1, and binds:

* The exact immutable record descriptors: relative path, creation generation,
  row count and complete shard SHA-256.
* The complete scientific source identity already sealed by publication v2:
  schema/contract, producer/config, scientific/Growth generation, descriptor and
  catalog hashes, and checkpoint identity.
* The validator identity: explicit canonical-reference contract, Python runtime
  identity (including optimization mode, cache tag, byte order and platform size limit),
  the selected ijson backend, and hashes of the relevant codec, parser and validator
  implementations.
  This includes all source/native files in Morpion's codec/game package, Valanga,
  Anemone checkpoint serialization and ijson, plus the Chipiron validation and
  publication modules. Install paths and bytecode caches are not identities.
* If reused, the immediate committed parent's capsule ID, seal digest, generation
  and validation-evidence digest. The normal consecutive capsule ancestry persists.

Let C_I(r) mean that decoding and canonical redumping reference r with implementation
I yields exactly r. At genesis every immutable record is checked normally. For
G+1, reuse is permitted only while holding the ordinary committed-parent read pin,
after its authoritative run-state membership, inventory and source identity have
been validated. The incoming descriptor prefix must equal the parent's attested
prefix, its producer must match, and the validator identity must match exactly.
The bounded reader freshly hashes every source shard against those descriptors,
parses all records, and rehashes the source after index construction. Under the
existing immutable-file/cooperative-writer contract and SHA-256 collision assumption,
the reused references are the exact bytes for which C_I was already established.
Determinism of the unchanged codec therefore preserves C_I. New immutable records
always undergo decode/redump checks. This gives an induction from fully validated
genesis across consecutive committed generations; family membership or node
exactness plays no role in this proof.

The scope is restricted to one private generation manifest and producer, and is
reset on both success and exception. Public consumers cannot obtain reuse by
presenting a SQLite cache or an uncommitted capsule. The publication index is
always newly built, checked and deleted before sealing; an unexpectedly reused
index is rejected. Every current update/value/certainty/provenance check, creation
key cross-check, export order check and reciprocal acyclic DAG validation still runs.

Missing evidence in an older publication-v2 capsule triggers full validation.
A changed implementation/runtime/codec identity also triggers full validation and
a fresh attestation. Malformed or unsupported evidence and inconsistent scientific
ancestry fail closed. Hash changes or corrupted parent shards fail before reuse.
An older binary can ignore this additional seal member but performs its own full
canonical validation; the scientific format and publication version are unchanged.
V1 output bytes and interpretation are unchanged.

This is an integrity attestation by the already trusted local publisher, not a
cryptographic signature against an attacker who can replace authoritative run state
and all seals, nor a certificate of a game-theoretic value. Runtime/dependency code
must remain immutable during a process, as required by the existing scientific
producer contract. No live monkeypatch or hostile Python-code model is claimed.

## Transaction and recovery boundaries

The sequence stays private export → complete validation → seal and fsync → atomic
no-replace rename → durable run-state replacement/directory fsync → repairable
discovery → verified patch acknowledgement. Validation evidence is not authoritative
until its containing capsule is committed. A crash during validation/evidence creation
cannot create a reusable source. A moved but uncommitted capsule remains invisible.
Retries start from the committed parent and never seed from an abandoned attempt.

Historical inventory hashing, immutable-prefix verification, file/directory fsync,
source rechecks, stable reader pins and committed-capsule retention were not removed.
The additional parent pin/validation costs are included in measurements. Those costs
are visible after removing repeated replay work; this patch deliberately does not
memoize inventories or weaken the commit barrier to obtain further speedups.

## Measurements and validation

The final measured implementation is `c1c737ca0d9fb76f9a9d2ad68d0f999a4c9f0be1`;
control is exact `69534167486f980afebc4a07af4e9f6c95784370`. Exploratory samples from
`bdad537a` are retained separately and are not the final after column. All fixtures
are newly generated, balanced legal replay populations with 4 KiB per-node metadata
and declared ESTIMATE values. They share exact scientific source fingerprints,
checkpoint bytes and all committed scientific identities across G1/G2/G3.

The final 1k comparison passed before the final 5k comparison ran. Each timing
process is fresh; runs are serial and do not overlap CI/static/audit jobs. These
are controlled single samples on a shared host, with warm OS caches. A speed ratio
is descriptive, not a statistical confidence interval. No real gen38 estimate is made.

| Nodes | Revision | Genesis sealing s | G2 / G3 publication s | G2 / G3 CPU s | Publication peak MiB |
| ---: | --- | ---: | --- | --- | ---: |
| 1,000 | before | 3.622 | 4.013 / 3.865 | 3.402 / 3.251 | 289.73 |
| 1,000 | after | 3.392 | 1.066 / 1.125 | 0.602 / 0.668 | 290.33 |
| 5,000 | before | 26.798 | 27.418 / 26.794 | 26.247 / 25.609 | 311.42 |
| 5,000 | after | 24.788 | 3.221 / 3.447 | 2.124 / 2.334 | 312.12 |

At 5k this removes about 88% of repeated-publication wall time (7.8–8.5× faster).
Genesis still decodes and redumps all N records; its timing variation is not an
optimization claim. G2/G3 each have zero new immutable records: decode/redump counts
change from N/N to 0/0. A regression with one new DAG node requires exactly one
new decode, while the receiving generation validates all current links and values.

Genesis sealing excludes the initial source writer and input graph construction.
The initial writer measurements (before / after) are:
* 1,000: 1.508 / 1.488 s; 1,000 state accesses each.
* 5,000: 13.283 / 11.990 s; 5,000 state accesses each.

Every repeated cycle includes checkpoint restore/write, receipt handling, export,
validation, fsync, publication, discovery and acknowledgement. Writer state accesses
are zero for old records. The writer fixture owns its O(N) graph; its RSS is not the
bounded-validator-only footprint. The separate reader processes build no input graph.

| Nodes | Revision | Cold / reused reader s | Dataset rows/s | Dataset peak MiB | Index bytes |
| ---: | --- | --- | ---: | ---: | ---: |
| 1,000 | before | 3.476 / 0.0650 | 1814 | 284.31 | 5,857,280 |
| 1,000 | after | 3.245 / 0.0587 | 1909 | 284.73 | 5,857,280 |
| 5,000 | before | 25.393 / 0.2026 | 1816 | 286.21 | 29,433,856 |
| 5,000 | after | 23.927 / 0.2024 | 1942 | 286.46 | 29,433,856 |

Both reused opens report `index_reused=True`; cold opens build a fresh cache.
Complete Dataset JSONL hashes agree at both sizes. Consumer code still performs
full canonical validation on cold opens. Changes in these single-sample cold/row
rates are not attributed to this publication-only optimization. RSS includes the
same probe/helper import footprint before and after, so it must not be compared
naively with an earlier lighter 256 MiB probe.

| Nodes | Revision | G2 / G3 logical read MB | G2 / G3 logical write MB | G3 workspace allocated bytes |
| ---: | --- | --- | --- | ---: |
| 1,000 | before | 156.71 / 237.54 | 11.82 / 11.83 | 21,168,128 |
| 1,000 | after | 168.50 / 259.43 | 11.82 / 11.83 | 21,176,320 |
| 5,000 | before | 788.49 / 1192.32 | 60.15 / 60.16 | 104,718,336 |
| 5,000 | after | 843.62 / 1297.93 | 60.16 / 60.16 | 104,726,528 |

Logical I/O is `/proc` rchar/wchar, not storage traffic. Physical read/write/cancelled
byte counters are retained in the raw measurements; reads are mostly OS-cache hits.
At 5k after G2/G3, physical writes were 60,628,992 / 60,637,184 bytes. The additional
committed-parent pin/validation increases logical historical reads; it is deliberately
not hidden or optimized away. Inventory validation calls rise from 14 to 16 per cycle.

Workspace allocation counts each hard-linked inode once and includes genesis inputs
and all retained capsules. It excludes external consumer indexes and Dataset output;
index sizes are above. The private validation index is removed before sealing, so
workspace numbers are retained allocation, not transient peak disk. G3 uses only
8,192 extra allocated bytes for the added attestations in these samples.

Current-prefix scientific JSON metadata remains exactly 42,260→63,110 bytes at 1k
and 194,260→291,110 at 5k from G2→G3. Individual capsule seal, run-state and catalog
sizes are retained in the evidence. Historical metadata and dependency hashing still
grow with generation count; this patch does not make retained disk bounded.

### Phase attribution

The separate 1k cProfile runs include the initial writer, genesis sealing and complete
G2/G3 transactions. Values below are G2 inclusive seconds unless labelled otherwise.
Nested times overlap and must not be summed. Full call tables and caller attribution
are retained. Narrow creation-key/SQL and fsync counters wrap existing operations
only in the probe; no additional parser/index or profiling framework is introduced.

| Phase | Before s | After s | Scope / preserved checks |
| --- | ---: | ---: | --- |
| Graph traversal / node reconstruction | 0.097600 | 0.091321 | 1,000 build_training_node_snapshot calls |
| Writer topology | 0.001856 | 0.001731 | complete live graph |
| Historical canonical decode | 3.041479 | 0.000000 | 1,000 → zero calls |
| Historical canonical redump | 3.005552 | 0.000000 | 1,000 → zero calls |
| Value/certainty validation | 0.079540 | 0.074577 | 2,000 calls: writer and fresh bounded validator |
| Generation topology validation | 0.034744 | 0.034397 | reciprocal closed acyclic DAG |
| Bounded JSON parser | 0.132887 | 0.131054 | unchanged strict event parsing, including immutable rows |
| JSON encoding | 0.067424 | 0.062232 | generation rows, index payloads and metadata |
| JSON decoding | 0.001295 | 0.001998 | ordinary descriptor/ledger JSON; bounded parser listed separately |
| Checkpoint writing | 0.000413 | 0.000410 | scalar cold-checkpoint fixture |
| Generation export writing | 0.262555 | 0.245916 | all current updates; zero new states |
| Historical capsule validation | 0.188564 | 0.223779 | 14 → 16 full validations |
| All file hashing | 0.091459 | 0.098381 | records, updates, inventories, code and identities |
| Source identity hashing | 0.014086 | 0.013597 | three existing reader passes |
| Private reader initialization | 6.738156 | 0.658455 | fresh generation index |
| Complete scientific validation | 6.756582 | 0.718945 | canonical reuse plus all remaining checks |
| Atomic no-replace publication | 0.006492 | 0.007408 | rename and namespace fsync |
| Authoritative run-state commit | 0.014185 | 0.019069 | replacement and directory durability |
| Creation-index parsing | 0.003646 | 0.006015 | 1,000 keys; parser time only |
| Creation-index SQLite checks | 0.002927 | 0.004714 | 1,002 inserts/cross-check calls; row-loop overhead is in bounded build |
| File fsync | 0.091379 | 0.102456 | 30 → 31 calls |
| Directory fsync | 0.067400 | 0.078821 | 67 → 69 calls |
| Historical immutable-shard hashing | 0.003500 | 0.003883 | 19 → 21 hashes; exact bytes retained |
| Historical update-shard hashing | 0.051536 | 0.057342 | 16 → 18 hashes; exact bytes retained |
| Capsule-seal hashing | 0.000862 | 0.001532 | 30 → 34 hashes; exact bytes retained |
| Publish Python self time, including inventory construction | 0.000151 | 0.000201 | Excludes hashing/walk callees; not a complete standalone inventory timer |

New state-reference creation in G2/G3 is zero in both revisions. The initial 1k
writer does 1,000 canonical dumps (profiled 3.056 / 2.842 s); genesis canonical
decode+redump costs 6.098 / 5.704 s. Both retain full validation. G2 historical
decode+redump alone costs 6.047 s before and zero afterward, explaining the dominant
redundancy independently of aggregate wall time.

At 1k G2, historical record hashing is separately measured at about 0.004 s;
historical update hashing and repeated inventory verification remain measurable.
All per-category hash calls, bytes and timings are in the profile JSON. Capsule
inventory construction is not a standalone function in the existing protocol:
cProfile attributes dictionary/iteration overhead to publish self time, and payload
hashes to its file_digest callees (which also include prefix verification). These
are retained with exact callers rather than inventing a disjoint inventory timer.
No claim is made that aggregate publish time is inventory-construction time.

Creation SQL counters include inserts and the count/join consistency checks, not
Python per-key guards or transaction commit overhead. Those remain in bounded-build
self/callee times. This attribution distinguishes measured components and residual
work without pretending that overlapping profiles form an exclusive phase partition.

### Scientific and crash regression evidence

Full installed-wheel CI: **2,152 passed, two CUDA skips, zero failures**, 644.93 s
pytest wall time (645.19 s controller). All **101 original durable scientific-v2
cases** and **22 new reuse cases** ran. The normal four integration-marker exclusions
are unchanged. All 471 production Python files match the tested installed wheel.
Ruff lint/format, toolchain coherence and wheel/sdist build pass. Strict mypy,
Pyright and Pylint report zero new diagnostics (209 baseline, 206 current, three
removed), with no baseline or suppression changes.

The unchanged complete audit compares 3,780 decisions: 1,890 across all nine legacy
policies, 1,050 across five supported v2 policies and 840 exact-family policy gates.
All complete cases and the family counterexample match. **8,820 supervised JSONL
file pairs and ten source JSONL pairs are byte-identical**; all 24 flat/scientific
source fingerprints match. Two scratch model-path fields in the independently
generated fixture patch differ and are explicitly retained; that fixture patch is
not claimed byte-identical. Six managed same-source/model full-reference/indexed
worker tests compare complete patch and cursor bytes literally.

The original crash matrix retains 22 process-death boundaries, ten fsync failures,
a repeated discovery-repair crash and five genesis retry cases. It covers private
record/update/descriptor/catalog writing, hashing, validation, sealing, no-replace
publication, state replacement/directory fsync, discovery, acknowledgement and
orphan cleanup. No durable barrier or assertion was weakened. The new actual
concurrent-process test keeps an older v2 reader pinned while a writer publishes
with reuse and rejects a competing Growth owner.

New regressions also require: full genesis validation; only-new-record decoding;
full public cold validation; missing/different validator fallback; parser backend
and Python optimization-mode invalidation; malformed receipt/source/producer/hash
rejection; corrupted historical shards despite a valid cache; invalid cache identity
and payload rebuild; rejection of new invalid refs, certainty, topology and creation
entries; and successful clean retry from the committed parent.

The reachable direct ESTIMATE(42), tree/effective FORCED(35), terminal 35 example,
five enabled/nine legacy policy parity, four gates, current incoming DAG parents,
checkpoint restore/selector state and neural ESTIMATE semantics remain covered by
the retained suite. No scientific target or certainty interpretation was changed.

Development logs retain test-fixture field-path errors and a corrected expected
kernel-lock exception. One preliminary CI run was deliberately interrupted to add
explicit runtime/parser identity binding; final CI was rebuilt and rerun in full.
These logs are not counted as successful validation. The final original/new test
IDs, exact CI selection, source hashes and all reports are included in evidence.

[Resource/counter data](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_v2_reuse/evidence/resources-final.json),
[profile attribution](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_v2_reuse/evidence/profile-attribution-final.json),
[scientific parity](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_v2_reuse/evidence/scientific-parity.json),
[CI/static validation](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_v2_reuse/evidence/validation.json), and
[validator environment](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_v2_reuse/evidence/validator-environment.json).

## Remaining limits and safe review order

The optimization removes repeated canonical reconstruction of historical immutable
states, not O(N) current-generation work. All current rows, creation entries, graph
edges and indexes are still processed on disk; all dependency hashes still run.
The live writer remains O(N), while the bounded validator uses individual rows,
parser buffers, SQLite/TEMP storage and O(generation/shard metadata). No N-sized
Python record cache or second tree is introduced. Receipts and retained descriptors
add O(G) metadata per capsule, potentially O(G²) cumulatively. All committed capsules
remain retained, and a new generation still builds a fresh SQLite index.

Consumer cold initialization still fully validates canonical states. Same-source
consumer caches retain their existing source checks and digest validation. This
publication-only optimization is not a promise of faster Dataset extraction or a
real gen38 forecast.

All previous activation gates remain: selected-value confidence v2 consumers,
five enabled policies and four disabled exact-family policies, qualified fsync and
no-replace filesystem behavior, trusted fresh genesis/producer lineage, cooperative
writer fencing, downstream durability, producer/model epoch design and retention.
No new scientific proof rule, migration, epoch transition or production switch is
introduced.

Keep the new delta local above the exact validated publication base. Once the
scientific, streaming, confidence and durability sibling stacks have an approved
shared base, review the existing publication-v2 delta first and then this small
canonical-validation reuse delta. Re-run the retained checks against the actual
approved SHA. A PR against only one existing sibling would duplicate unmerged work,
so no such PR is opened.


## Reproduction and review artifacts

The branch is local; no remote PR or merge was created. The bundle retains the
local integration ancestry above the four public prerequisite heads listed in the
prior report (#76, #80, #82 and #81). The implementation patch contains only
`69534167..c1c737ca`, not any replay of those prerequisite stacks. The final evidence
archive includes that patch/bundle, complete JSONL pairs, source/capsule fixtures,
profiles, logs, source/validator fingerprints, CI reports, wheel and sdist.

Use a new directory on the verified secondary partition for every reproduction.
The [measurement driver](durable_v2_reuse/measure.py) extends the retained existing
publication probe with standard cProfile and narrow hash/creation/fsync counters.
The raw serial controllers record before/after process boundaries and exact parity
assertions. Do not execute an extracted raw run state: its paths refer to the
original scratch root, and ZIP does not preserve hard-link allocation semantics.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest --no-cov \
  tests/environments/test_morpion_canonical_validation_reuse.py \
  tests/environments/test_morpion_durable_scientific_v2.py \
  tests/environments/test_morpion_durable_scientific_v2_crashes.py \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python \
  docs/research/durable_v2_reuse/measure.py --mode durable --nodes 1000 \
  --root /home/pompote/oldata/victor/morpion-scratch/NEW-PUBLICATION-PROBE
```

The same full installed-wheel CI command and two test processes used by the
repository ran with the existing validation-safety plugin. That plugin redirects
optional leaderboard output to pytest scratch and does not weaken assertions.
All 26 original worktrees retain captured HEAD, status and tracked-diff fingerprints.
No historical evidence was deleted or overwritten, and production remains disabled.
