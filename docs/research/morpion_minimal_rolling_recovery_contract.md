# Minimal rolling recovery contract — read-only design and evidence

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

## Decision and scope

Base: **`ffa5ea7d3d9624c5efc8a562a184cf3f402d43e3`**, directly, with no
cherry-picks or changes to sibling worktrees. New local branch
`design/morpion-minimal-rolling-recovery`, secondary-disk workspace
`/home/pompote/oldata/victor/morpion-scratch/rolling-recovery-contract-20261023/worktree`.
The [validated closed-publication report](morpion_durable_closed_snapshot_publication.md)
remains the scientific/durability specification.

**The read-only analysis and proposed compact roots are implemented. Persisting a
rolling authority, executing rollback, and deleting artifacts are deliberately
blocked.** Publication protocol 3 still requires every committed capsule.
Its last two complete checkpoints/scientific snapshots are *rollback candidates*,
not complete executable rollback boundaries: the capsules do not seal the
producer cursor or an unambiguous acknowledgement/application boundary.

An intentional rollback with newer receipts can silently lose applied effects;
removing those receipts can allow the same acknowledged UUID to be applied again.
This is a protocol decision, not a retention optimization. The requested stop
condition therefore applies before changing persisted recovery semantics.
No scientific policies, worker routing, receipt formats or checkpoint formats
are changed. No historical dataset or previous worktree is modified.

## Smallest proposed authority extension

Use the **existing atomic `run_state.json` only**. A future explicit publication
metadata version 4 would replace the cumulative `generations` list with references
to the current capsule, up to two *eligible* rollback boundaries, and additional
required roots. Closed scientific capsules remain physical publication version 3
unless their recovery seal must be extended. Old readers must reject the new
authority version, rather than interpret its incomplete history as v3 ancestry.
No second ledger, catalog, database, journal or cache is needed.

The report's `proposed_active_roots` has its own **draft schema 1**, not a format
accepted by any recovery reader. It contains:

| Field | Meaning |
| --- | --- |
| `current` | One capsule identity, generation and exact seal SHA256 |
| `rollback_candidates` | Last two capsules, explicitly **not yet executable** |
| `additional` | Extra observed readers, pending work, producer cursor, claims, frozen model dependencies or explicit retention |
| `capsule_publication_version` | 3; independent of the draft projection version |

The descriptor path is derived from the capsule identity in the existing owned
namespace. Full scientific identities, producer identity, checkpoint/scientific
paths and file inventories remain in their sealed capsule; do not repeat them
for every past generation in each run state. Current top-level checkpoint/source
pointers must agree with the current sealed capsule. The report is advisory and
is never read back by Growth, Dataset, Training or Reevaluation.

Keep these kinds of state separate:

* **Active heavyweight roots:** current + two eligible rollback boundaries + live
  dependencies, retained once. Their count depends on actual work, not age.
* **Exact acknowledgements:** retain every existing receipt field and exact
  UUID/content SHA256 identity, once in the existing authoritative run-state
  ledger. Do not substitute source generation, latest patch ID or a watermark.
* **Producer context:** checkpoint selector/RNG is already in the full checkpoint;
  Reevaluation cursor, evaluator/model identity, pending publication identity and
  relevant Training state are separate mutable controls today.
* **Metrics/provenance:** keep existing lightweight history/status records with
  explicit historical provenance references. A provenance hash is not evidence
  that the referenced payload still exists or was verified now.

Future boundary eligibility requires a sealed, versioned recovery context in the
**same capsule/commit**, with the precise producer cursor and model dependencies,
and a binding to the exact receipt set at that boundary. A count/hash of an
append-only receipt prefix could identify that set without copying it, *only*
after validating the full exact ledger and proving prefix ordering. It is not
a replacement for arbitrary UUIDs, and does not solve branching/application
semantics. This alternative is documented, not implemented or approved here.

## Crash recovery, intentional rollback and exact receipt invariants

Crash recovery adopts the single complete observed run-state head. Validate its
sealed dependencies and checkpoint identity; resume pending acknowledgement only
when the receipt digest, cursor, evaluator and model identity match. A visible
orphan capsule is never a commit. A missing/corrupt committed head is an error,
not permission to fall back to a preceding checkpoint. An interrupted directory
fsync retains the existing recovery protocol's treatment of the observed state.

Exact retry means the **same bytes** and same arbitrary patch ID, with no second
application/blend. Duplicate IDs inside the ledger are rejected, even when the
duplicate is identical. A pending ID with a different byte digest is rejected.
New IDs at the same source generation remain independent requests.

Intentional rollback changes which scientific effects are present. These choices
are not interchangeable:

1. Select G105 with G106's receipts: G106-only effects disappear while their
   acknowledgements suppress replay. Scientifically incorrect.
2. Restore an old ledger with G105: already acknowledged arbitrary IDs may be
   applied again, and the live producer may already have advanced. Incorrect
   under current same-run receipt semantics.
3. Quiesce and create an **explicit new lineage** from a verified boundary,
   retaining the original run and exact acknowledgements as provenance. This is
   the recommended direction for a separately reviewed rollback operation. Its
   producer reset/restore semantics and patch namespace fence must be explicit;
   it cannot claim identical continuation without a saved producer context.

Supporting same-run rollback instead would require defining acknowledged versus
currently applied effects and how to replay/reject pending work across the
boundary. The current checkpoint plus receipt tuple does not contain the old
patch payload needed to reconstruct every discarded effect. No such machinery
is introduced here.

## Read-only reachability implementation

`scripts/report_morpion_rolling_recovery.py WORK_DIR [--retain GENERATION ...]`
prints JSON to stdout and exits 2 on incomplete validation. It does not create
an index, manifest, pin, staging directory or lock file. It does not repair
discovery, adopt orphans, acknowledge patches, quarantine or delete anything.

It reuses:

* `growth_receipt_ownership` with `create=False`: the existing exclusive,
  nonblocking Growth lock; a concurrent owner refuses analysis.
* `capsule_lease` with `create=False`: existing stable pin inodes only. Successful
  exclusive acquisition means no cooperative reader currently holds that pin;
  failing exclusive acquisition followed by shared acquisition records a reader.
  An unknown exclusive owner fails closed. All acquired leases live through the
  report. Missing pin files suppress all conditional candidates.
* `publication_records`, `validate_capsule`, `receipt_ledger`, current-checkpoint
  validation, exact patch observation and existing cursor/claim/status loaders.
  Every inspected capsule's payloads are actually hashed. A parent hash or pin
  never substitutes for checking their bytes, seal inventory or scientific and
  checkpoint identities. No scientific snapshot is reconstructed.

For each committed generation the report includes actual protocol-3 retention,
rolling reasons, observed reader protection, all transitive capsule files and
model pins. Old cursor sources, pending patch sources and explicit scientific
roots survive. **Expired claims still protect their generations.** Training
cursors and stage statuses are conservatively retained, including completed or
failed statuses; neither status nor TTL proves a worker stopped. Unknown claim,
producer/model metadata, unknown controls, malformed/missing artifacts and
uncommitted objects fail closed. A missing advisory manifest is allowed because
it is reconstructible from authority; an existing conflicting manifest is not.

Every file outside the closed-capsule namespace is retained. This includes
legacy/bootstrap inputs, original model bundles, rows, reports and recovery
evidence. The tool does not infer permission to remove them or implement a
general workspace collector. Unsupported external/derived model ownership is
refused. Fixed producer/configuration/model epoch remains the base limitation.

Control fingerprints before and after analysis detect observed changes, not ABA
or future reader acquisition. Reader/worker admission and producer publication
are not all serialized by the Growth lock. The report is a point-in-time
conditional model; its candidates cannot be reused as a deletion plan later.

Inode accounting groups `(device, inode)` across all committed capsule files.
Conditional reclaimable file blocks require every known link to be in candidate
capsules **and** observed link count to equal `st_nlink`. A link in a kept capsule
prevents counting that inode. Any unlocated external link owner suppresses the
entire conditional removal set: the tool does not guess its dependencies.
The successful report lists the precise candidate inode groups and their names,
so the aggregate allocation can be independently summed. Actual disk allocation
uses `st_blocks * 512`, not summed pathname sizes. Directory blocks, delayed
reclamation of open files, reflinks and filesystem snapshots are excluded from
the estimate. No promise is made that an unlink would immediately free that
many device bytes.

## Complexity and measurements

Let R be active roots, P exact receipts, F inventory entries and B bytes of all
retained payloads, counting each capsule's names. Let M be the largest original
model bundle size. The proposed authority has O(R) root records (generation
integers add O(log G) bits). Receipts still require O(P) metadata. The diagnostic
loads existing cumulative authority and inventories, requiring O(G + P + F)
metadata RAM and O(B + G + P × M + F) validation/I/O apart from directory sorting.
The existing receipt checks still rehash each receipt's original model dependency;
this offline report is not a new optimized publication/discovery path.
It holds O(G) existing pin descriptors during its offline scan. It is not a
history-independent production discovery path, nor suitable for an unlimited
number of open pins without redesign. It constructs no O(N) Python tree.

With bounded R, complete closed payload storage could eventually become
O(R × (N + E + canonical-state-history-bytes)) plus required models/artifacts.
**This implementation still retains O(G) physical snapshots and O(P) receipts.**
Metrics also remain cumulative. No total-history bound is claimed.
Even a future compact root set leaves whole-run-state receipt reads/writes at
O(P); one new receipt per generation implies Θ(G²) cumulative ledger bytes
rewritten over the run. This task does not change that exact deduplication
representation. Rows/models outside capsules also remain retained and may be
large. Conservatively retained stale claims and completed-stage statuses can
keep R growing until actual worker-lifetime/release evidence resolves them.
The fixed-three-root experiment is not a proof that those production lifetimes
are already bounded.

Fresh-process, serial G8/G16/G32/G64 measurements use the existing real Anemone
fixture driver with requested size 4 (actual **five nodes**), a full sharded
checkpoint and unchanged no-op-patch generations. The report instrument reuses
`StageMeasurement` and existing CPU/I/O/codec counters. No independent profiling
framework is added. Complete exact inode/content/mtime/ctime fingerprints of the
synthetic workspace must be unchanged by every analysis call.

All 84 serial fresh-process jobs succeeded. Full reports, source fingerprints and
machine/toolchain details are in
[the evidence directory](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/minimal_rolling_recovery/evidence/scaling.json).
The independent raw fixtures, logs and command manifest remain under
`/home/pompote/oldata/victor/morpion-scratch/rolling-recovery-contract-20261023/evidence/validated-history`.

| Generations | Active root records / bytes | Receipt count / bytes | Scientific allocated file bytes | History JSONL bytes | Conditional reclaimable file blocks, bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3 / 584 | 7 / 4,502 | 131,072 | 36,884 | 307,200 |
| 16 | 3 / 587 | 15 / 9,371 | 262,144 | 79,060 | 798,720 |
| 32 | 3 / 587 | 31 / 19,115 | 524,288 | 163,454 | 1,781,760 |
| 64 | 3 / 587 | 63 / 38,603 | 1,048,576 | 332,134 | 3,747,840 |

The three-byte root difference is solely the decimal width of the three
generation numbers, not a historical list. Receipt bytes count the complete
existing receipt metadata object once, using compact sorted JSON. Scientific
allocation counts unique inodes across all retained closed scientific files;
it is **not** their hypothetical allocation after deletion. Existing stage
performance records additionally occupy 35,850 / 76,989 / 159,387 / 324,099
logical bytes; metrics have not been bounded or removed.
The allocation remaining inside capsules after the *conditional* subtraction is
200,704 bytes at every measured G, including shared frozen models. The **actual**
retained capsule allocation grows from 507,904 to 3,948,544 bytes. Every measured
workspace remains intact and current-contract reclaimability is zero.

| Generations | Report wall seconds | CPU user + system seconds | Process peak RSS MiB | Hash calls / bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0434 | 0.0432 | 291.14 | 329 / 685,534 |
| 16 | 0.0859 | 0.0857 | 291.25 | 641 / 1,355,064 |
| 32 | 0.1697 | 0.1693 | 291.86 | 1,265 / 2,694,295 |
| 64 | 0.3394 | 0.3341 | 293.07 | 2,513 / 5,372,464 |

These are single warm-filesystem observations, not statistically estimated
latencies. RSS includes Python/Torch/package imports and cumulative report
metadata; raw evidence retains both current RSS readings and kernel process
peak accounting. This does not prove bounded diagnostic RAM as G grows.
Every measured analysis performed **zero state decodes, zero write syscalls and
zero fsyncs**. The pin-preparation reader ran before the measured/read-only
region, solely in the newly generated fixture. Every file/directory fingerprint
remained exact (read atime is excluded).

All four populations preserve full-reference/bounded scientific digests across
unchanged generations. Two subsequent continuation steps preserve exact selector
and RNG state. Node-shell differences are zero on these five-node fixtures;
checkpoint metadata differs across generations and is explicitly recorded in
the evidence. This small example does not remove the known larger-graph
representative-parent discrepancy.

Reproduce on a new destination only:

```bash
PYTHONPATH=src:. python -m docs.research.minimal_rolling_recovery.reproduce NEW_SECONDARY_DISK_DIRECTORY
```

## Conditions before any physical deletion

1. Review and implement the explicitly versioned compact authority; historical
   v1/v2/v3 ancestry and receipt dependencies cannot be silently reinterpreted.
   Define whether old receipt checkpoint references are retained evidence or
   provenance-only after an explicit retirement transition. Today they are roots.
2. Define intentional rollback separately and prove two complete, executable
   boundaries, including producer context and exact acknowledgement semantics.
3. Ensure **every** reader/worker holds the existing lifetime pin from admission
   through its last dependency access. Fence new admission against authoritative
   retirement using the same stable pins and Growth ownership. A stale claim,
   elapsed timeout, PID observation or this report is insufficient.
4. Resolve pending patch/cursor publication and model dependencies under that
   fence. Existing patch publication explicitly does not make cursor updates
   transactional. Prove recovery across each publication/acknowledgement fault.
5. Resolve every unknown owner/dependency, including external links and explicit
   scientific retention, before selecting an actual removal set. Preserve current,
   rollback, pending, active-reader and still-required recovery evidence.
6. Verify kept payloads with their exact sealed inventories, not ancestry hashes.
   Recheck inode/link identities and filesystem assumptions at removal time.
7. Test interrupted retirement/removal using the existing sole commit authority
   and fsync rules. Do not add a speculative second journal to solve this here.

Recommended review order: this local read-only report/lock-open option; a small
separate recovery-boundary/rollback contract review; compact authoritative roots
with lifetime admission/retirement tests; only then a separately authorized
deletion implementation. No monolithic PR, merge or production activation is
performed in this work.

## Validation and code delta

The relevant existing suites passed **881 cases in 866.20 seconds**, including
22 initial analysis cases. After the final fail-closed guards, all **24 final
analysis cases passed in 22.39 seconds**. This is 859 existing regression cases
plus 24 final dedicated cases across the two runs, not 905 independent tests.
[Exact suite counts and JUnit fingerprints](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/minimal_rolling_recovery/evidence/validation.json)
are retained. Tests ran serially against the new checkout and the existing pinned
Python 3.13.5 environment, without modifying that environment. The full installed
package CI was **not** rerun; this is the relevant regression selection, not a
claim of a new full CI result.

| Required behavior | Evidence |
| --- | --- |
| Crash before/after commit | New subprocess `before_state`/`after_state` tests; 28 closed-capsule crash/fsync cases; inherited publication crash suites |
| Exact patch retry | Real post-commit byte-identical retry applies zero times and preserves checkpoint bytes |
| Receipt conflicts | Identical duplicate ledger IDs, changed duplicate digest, changed pending bytes and conflicting producer identity fail closed |
| Old active reader | Existing shared flock retains G2; release makes it a conditional candidate again |
| Pending Reevaluation | Old source/cursor/model retained and matched to the exact committed scientific identity |
| Stale Dataset/Training claims | Both expired claims retain their generation; unknown claim dependencies block the report |
| Missing/corrupt inputs | Missing/corrupt original model, missing capsule or pin, symlink, unknown object/control, truncated v3 authority all refuse candidates |
| Intentional rollback | Older checkpoint combined with newer authoritative head/receipts is rejected; all reported rollback candidates remain non-executable |
| Growth concurrency | Independent flock descriptors refuse simultaneous ownership; 3 existing receipt and 6 publication concurrency tests also pass |
| Hard links | Unknown external owner blocks all candidates; kept/candidate shared inode remains; two candidate names count one allocation |
| Concurrent controls | Changed control fingerprint invalidates the entire report; ABA/future admission remains explicitly unproven |
| Science and compatibility | Closed/reference/bounded reader suites, nine legacy policies, five supported certainty-v2 policies, four gated policies, reachable FORCED(35)/ESTIMATE(42), exact Dataset/patch serialization and discovery suites pass |

Repository-wide Ruff checking and formatting pass (692 Python files), toolchain
versions are coherent, and the existing strict static ratchet reports **zero new
diagnostics** (209 baseline, 206 current). The inherited debt baseline is untouched.
All 33 pre-existing worktree registrations and all 16 present worktrees retain
their HEAD, status and tracked diff fingerprints, including the dirty research
checkout and the source implementation worktree.

Code-size accounting, including blank lines and docstrings:

| Area | Added | Removed | Net |
| --- | ---: | ---: | ---: |
| Installed runtime module | 16 | 5 | +11 |
| Standalone read-only diagnostic CLI | 490 | 0 | +490 |
| New regression tests | 409 | 0 | +409 |
| Research observation/reproduction drivers | 128 | 0 | +128 |

Counting the diagnostic CLI as production code gives **+506 / -5, net +501**;
the small installed-runtime delta is not the size of the whole implementation.
The runtime change only adds `create=False` to two existing lock contexts, so
read-only inspection cannot create missing ownership evidence. Defaults preserve
all existing callers. The standalone diagnostic separates control capture,
schema-specific pending-work roots, capsule/inode inspection and fail-closed
reporting; these are functions, not a generic ownership/GC framework. It reuses
existing serializers and validators and has no persistent state.

No old cumulative retention code is removed yet: deleting it before a versioned
authority/receipt dependency transition would silently change historical recovery
semantics. The prospective deletion is the *duplicated generation descriptor
history* in a future authority version, not the exact acknowledgement ledger.

## Scientific limitation retained

The known representative-parent raw-checkpoint discrepancy remains visible. The
base recorded 7/70/161 affected representatives at 1003/5013/10018 nodes. It is
not normalized away or fixed by retention metadata. Byte immutability of the
inspected artifacts and scientific/selector/RNG parity are distinct from proving
raw checkpoint roundtrip identity for all graphs. Tiny fixed-history results do
not discharge that existing issue or the four gated exact-family policies.
