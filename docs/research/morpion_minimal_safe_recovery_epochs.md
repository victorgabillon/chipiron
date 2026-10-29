# Minimal safe recovery epochs — contract and blocked implementation

Status: **design and executable boundary investigation; no epoch activation or
fork implementation**. Base `089fec3106bdbc690435e61a5b0ce236a5b4821e`, direct
parent `ffa5ea7d3d9624c5efc8a562a184cf3f402d43e3`. This report supersedes the
previous rolling report's suggestion that an intentional rollback should retain
the old producer stream. It does not change normal crash recovery semantics.

The requested stop condition applies. A metadata-only epoch extension is unsafe,
and the existing initializer cannot adopt a retained non-genesis snapshot.
There is also a pre-existing producer crash gap: a durably visible patch can lack
the corresponding cursor. Growth preserves its effects and exact receipt, but
cannot finish acknowledgement on restart. Shipping a fork before defining and
testing these boundaries would falsely imply end-to-end recovery support.

No production source has been changed. Research tests demonstrate the gaps using
the actual publication, Growth and Training paths. They are characterizations,
not passing acceptance tests for a new epoch implementation. No existing
experiment, PR branch or source worktree was changed.

## Compatibility and simplification decision

Scientific state and recoverability are mandatory. Historical internal formats
are not. Recommend **one current epoch-aware protocol**, replacing the old
publication variants after a separately verified one-time adoption. Do not add
an indefinite `if legacy / else epoch` branch to every worker. The incompatible
protocol must reject missing or unknown identities rather than infer them from
paths, timestamps, generations or permissive metadata defaults.

This investigation does not perform adoption, remove old implementations, or
relax any current checks. It identifies exactly why the existing pieces cannot
yet be advertised as that single protocol.

## Two operations, two explicit contracts

**Crash recovery:** reopen the same immutable workspace identity and its latest
valid committed `run_state`. Validate the complete current dependencies and
finish only operations justified by that commit. Preserve checkpoint, graph,
values, certainty, provenance, selector and RNG, configuration, active model,
receipts, pending work and the producer cursor. An invalid current head is an
error, not an instruction to restore G-1. Discovery/status files are derived
views, never recovery authorities.

**Intentional rollback:** explicitly select a complete retained boundary and
prepare a new, initially absent workspace with a fresh epoch ID. Restore that
boundary's scientific state, not the previous epoch's latest state. Later search
progress is deliberately absent. Begin a new producer stream with an empty
receipt ledger and explicit initial cursors only after validation and the new
durable initial commit. Leave the original workspace, receipts, pending work,
reader pins and artifacts intact. Its live workers remain confined to that old
workspace; a timeout does not revoke them.

Never change a workspace's epoch in place, reuse its path for another epoch,
retarget a symlink used by running workers, or copy its live singleton controls
as initialization. This removes the need for a distributed epoch-switch lock.
It does **not** remove the need to validate every imported result and acceptance
boundary. The user explicitly chooses the new workspace; no new global
`current_epoch` journal/authority is needed.

Trust assumptions remain explicit: local POSIX atomic rename/link, working fsync
and flock, collision-resistant content hashes, and cooperative immutable owners.
All new-workspace writers must implement the current acceptance protocol; retain
old executables only for read-only conversion/audit or their old workspaces.
Capture the expected epoch when admitting a job and carry it through computation;
reading the destination epoch at completion and attaching it to an old result
would defeat the fence. These are stale-worker and corruption protections, not
protection against an arbitrary process deliberately rewriting committed files.

## Minimal proposed current contract (not yet a supported format)

Use one explicit top-level run-state format/version. Keep the existing single
durable `run_state` replacement and Growth ownership lock. Add a fixed-size epoch
identity and one immediate fork-origin record; do not copy the ancestor's
publication or receipt arrays. Reference current and retained capsules by their
existing ID and descriptor digest, without duplicating their inventories.

Proposed dispatch is `format_kind="morpion_research_run"`, `format_version=1`;
the exact spelling is a design proposal, not a format recognized by this branch.
One `epoch_id` scopes the whole authority, including its receipt list. Every
worker result carries that same epoch ID plus the exact selected capsule ID and
descriptor digest. A different result protocol version, absent identity, or
unknown fields affecting recovery require rejection, not a legacy fallback.
Do not repeat the enclosing epoch on every historical receipt merely to enlarge
the ledger; bind it in the proposal bytes and authoritative enclosing contract.

| Component | Required content and owner |
| --- | --- |
| Epoch | Fresh unambiguous ID, protocol version; immutable for this workspace |
| Immediate origin | Source epoch/legacy namespace, source generation, capsule descriptor digest, checkpoint/scientific/config/model digests; reason `intentional_fork` |
| Current head | One committed capsule reference; paths and inventory derive from its seal |
| Recovery candidates | Up to two complete retained capsule references; not automatic fallback heads |
| Receipts | Exact active-epoch UUID and byte digest, source/model identity, application count/blend and committed checkpoint association |
| Producer state | Accepted cursor checkpoint and precisely bound outstanding proposal, if any; reconstructible from committed evidence |
| Model | Accepted model identity and exact pinned weights/configuration; Training emits proposals, Growth accepts transitions |
| Additional roots | Exact dependencies for active work and explicit scientific retention, using the existing pins/ownership machinery |
| History | Append-only lightweight metrics/provenance outside the active-root projection; not a second authority |

This is a replacement contract, not a second ledger alongside the current one.
The model and producer cursor must be coherent with the same authoritative
commit. Stage completion files may be reconstructible projections of accepted
results, but cannot independently select the model used for recovery.

The capsule must identify the selected boundary's configuration and model, not
whatever a newer mutable `active_model.json` says during adoption. A relocation
map may change physical workspace paths, with original logical identities kept
as provenance; it must not silently change evaluator configuration or reseed the
runtime. Current config hashes include workspace details, so copying a capsule
into another root and patching path strings is not an already-validated migration.

An epoch ID is **not** a model generation. Ordinary accepted model transitions
stay in the same epoch and retain the configured evaluation policy. There is no
new mathematical target, certainty or patch blending policy in this proposal.
This does not enable currently unsupported model transitions: the existing fixed
model restriction stays gated until checkpoint/value provenance and transition
recovery are independently validated. An epoch label alone proves none of that.
The seed model retains its original producer/model identity. Its old generation
number is provenance, not the new epoch's local Training lower bound. Start new
local producer cursors explicitly (no outstanding patch, start cursor, zero
completed passes), while restoring the search selector and RNG exactly. These
are deliberately different state machines. Do not fake a renumbered model or
carry an old `latest_started_generation` into the new epoch. The current inferred
source/external-model guard exceptions can disappear in the explicit protocol.

### Historical node origin

Existing closed records preserve `creation_generation`, canonical state references
and current scientific values; their generation header describes the current
publication. Keep these concepts distinct. In an epoch-aware representation,
origin needs an explicit namespace as well as the original creation generation.
Copied nodes retain that origin; newly created nodes use the new epoch namespace.
Do not rewrite every copied node as newly created at local G1.

A migration from a source without epoch IDs must declare a stable legacy origin
namespace tied to the validated source provenance; it must not claim that an old
epoch identifier was present. Preserve the original fields as evidence. A fresh
publication sequence can start locally at one only with a versioned independent
origin field. Alternatively keep publication numbering at the adopted generation;
that still needs an origin namespace for forks which independently reuse later
generation numbers. A generation counter alone is not a lineage identity.

Current blockers: `save_closed_scientific_snapshot` requires source-less genesis
at one or the previous generation plus one. `initialize_scientific_v2_publication`
requires generation one and no existing ledgers. `validate_state` requires
genesis one and a fixed producer; publication also requires a fixed model epoch.
The existing `derived` subsystem explicitly resets RNG and is not exact
continuation. None is an exact fork API. Do not weaken these checks as a shortcut.

### Why old receipts need not enter a new epoch's hot path

Conditional proof, with assumptions made explicit:

1. The adopted checkpoint already contains exactly the effects committed at the
   selected boundary, independently validated against its scientific export.
2. Every patch/result is bound to an immutable epoch **before** its work starts;
   the consumer compares that identity to the destination authority before use.
3. Missing, unknown and different epochs are rejected before receipt lookup or
   value application. Moving a file or resetting its generation does not retag it.
4. Old worker handles and paths cannot become handles for the new workspace.
5. No old pending proposal or cursor is silently imported into the new epoch.

Then an old patch can neither apply nor collide with a new epoch receipt. The
new ledger starts empty; lookup is by `(epoch, patch_id)`. Within an epoch, same
ID/same bytes is an exact retry; same ID/different bytes is an error; duplicate
entries in the ledger are malformed. No generation watermark substitutes for
UUIDs. Effects included in the seed are state, not fresh patch applications.

These assumptions are **not all enforced today**. Therefore clearing receipts in
the current protocol remains forbidden. New initial publication must be durable
before old receipts can leave the active hot path; archiving does not authorize
deleting them or dependencies still needed by old workers.

## Acceptance boundary audit

| Boundary | Existing behavior | Required change in the one current protocol |
| --- | --- | --- |
| Growth commit/receipt | Exclusive stable flock, exact bytes and head/checkpoint hashes; no epoch field validation | Validate expected epoch inside the same lock before apply/commit/ack; reject unknown schema |
| Scientific publication | Immutable v3 capsule, exact inventories and config/model pins; fixed genesis/producer/model | Bind epoch and fork origin in seal and authority; preserve node origin separately |
| Reevaluation publication | Create-only fsynced patch, then separate cursor write | Epoch-bound proposal must contain sufficient producer continuation state for deterministic recovery |
| Patch consumption | Scientific capsule ID and exact source/model checks | Keep these checks AND require epoch before dedup; do not infer identity from scientific values |
| Dataset completion | Validate manifest, then rows/manifest/status writes | Retain source pin for the work lifetime; epoch/source-bound immutable output and guarded acceptance |
| Training completion | Validate manifest; independently save status, active model, cursor | Epoch/source-bound result; accepted model transition belongs to the existing authority |
| Discovery/status | Projection/source validation, but generic metadata allows unknown keys | Require new protocol identity and rebuild only from that epoch's accepted evidence |
| Claims/readers | TTL claims and stable capsule flocks have different lifetimes | TTL is scheduling information; complete worker lifetime must hold the required pins |
| Cycle/derived alternate paths | Additional active-model/status writers outside pipeline stages | Route through the same accepted transition or remove from the current protocol |

Relevant source: `checkpoint_patch_receipts.py` (`apply`, `commit`,
`acknowledge_pending`); `reevaluation_worker.py` (patch publication followed by
`save_reevaluation_cursor`); `pipeline/stages.py` (Dataset/Training completion);
`pipeline_artifacts.py` (writers); `cycle_pipeline_manifest.py`,
`bootstrap_loop.py`, `derived/orchestrator.py` (alternate writers).

Validation followed by an unrelated replace is not a fence against an in-place
epoch switch. The proposed immutable-workspace rule avoids that switch race;
the result still needs explicit identity and coherent acceptance. Generic
serialization helpers cannot supply that identity after computation has ended.

## Demonstrated blockers and controls

The accompanying `safe_recovery_epochs/test_boundaries.py` exercises production
functions with freshly generated two-node scientific fixtures; Training itself
is stubbed, its real completion path runs. No actual fork is simulated as safe.

* An arbitrary `epoch_id` in run-state and a different one in a valid-source
  patch do not prevent application. This disproves the metadata-only shortcut;
  it does not mean current v3 promises epoch support.
* A late Training completion still publishes status/model when the run-state
  epoch label changes during training. Capsule validation remains intact in the
  fixture. Existing epoch-unaware completion is not a new-epoch fence.
* A patch from a separately initialized workspace is rejected by its different
  capsule identity. Existing integrity checks are useful and must remain.
* Erasing receipts after a cosmetic epoch retag permits a committed patch to be
  blended again: `{root: 6, leaf: 8}` becomes `{root: 8, leaf: 10}`. Keeping the
  ledger prevents that second application. The paired controls demonstrate why
  receipt erasure is conditional on a real acceptance fence, not a new label.
* An older retained G2 capsule cannot be passed to the current initializer as a
  fork. It fails before committing a target run-state; the G3 source stays intact.
* A durably published patch without its cursor can be committed exactly once.
  Three cold startup attempts leave the pending file present and cannot invent
  the missing cursor. Supplying the original cursor explicitly allows ack without
  replay. This is safe refusal/stall, not loss of already committed values.
* Existing Growth flock exclusion and an old generation's shared reader pin
  coexist with a later generation publication.

The existing Reevaluation worker fault tests separately show the producer side:
after a post-link failure, retry returns `pending_patch_exists` and keeps the old
cursor. Thus waiting for a normal producer retry does not close the cursor gap.
The missing information includes the exact next cursor and full-pass counter;
the current receipt does not contain all of it. Reconstructing a counter from a
possibly absent or newer singleton would invent producer history.

The lean candidate for closing this gap is to carry the expected prior cursor
identity and complete next cursor in the immutable patch proposal itself, then
accept its next cursor with the checkpoint/receipt in the **same run-state
commit**. A restarted producer reads that accepted cursor and any outstanding
proposal. It has no second authoritative cursor write to complete. This removes
an authority rather than adding a journal. It still needs exact tests for a
producer dying before/after the create-only visibility point, a consumer dying
before/after commit, a lost acknowledgement, concurrent proposal creation, and a
model change. It is a proposed incompatible protocol change, not a safe
reinterpretation of existing patch metadata.

### Smallest corrective sequence

1. Define the epoch-aware result envelope and bind the producer cursor transition
   to the exact patch proposal. Reuse the create-only publisher and authoritative
   receipt commit. Prove recovery at publication-before-cursor and
   commit-before-ack; remove the independent ambiguous cursor update for the new
   protocol. No additional journal.
2. Make stage results proposals and put epoch/source/model acceptance at the
   existing authority. Cover every alternate writer or remove its current-mode
   route. Reuse Growth ownership; do not hold it throughout Training. Keep old
   worker paths isolated and source pins alive for the actual read lifetime.
3. Implement explicit adoption into a new workspace using the existing capsule
   publisher after full science/checkpoint/config/model/selector/RNG validation.
   Bind node origin independently. Commit one current protocol and remove
   migrated legacy branches from its production path.
4. Only then replace cumulative active publication roots with current + two
   complete candidates + proven active roots. Extend the existing read-only
   reachability report; deletion remains a separate authorization and change.

These are separable reviews, with a single final protocol, not four new
frameworks. The present branch adds no partial production epoch helper because
an unenforced helper would suggest guarantees the real worker paths lack.

## Fork durability state machine to validate before implementation approval

Keep a source shared pin while validating/copying. Use the destination's existing
Growth ownership; a competing owner fails. Never acquire two mutable authorities
to switch the old epoch. Copy into private immutable destination dependencies,
independently verify science and continuation state, fsync files/directories,
publish the complete capsule, then atomically write and fsync its sole initial
`run_state`. The source remains unchanged throughout.

| Interruption | Permitted restart behavior |
| --- | --- |
| Before/during copying or validation | No destination head; refuse launch, original source remains valid |
| Capsule visible but no committed run-state | Still no destination head; do not infer a commit from discovery/orphans |
| Run-state replacement observed before durability confirmation | Validate all dependencies and fsync the observed complete authority, as existing publication does |
| After durable initial commit | Resume that new epoch; do not refork or reset its new receipts/cursor |
| Stale or absent discovery | Reconstruct from committed new-epoch evidence; never adopt the old epoch's statuses |
| Corrupt checkpoint/model/config/origin evidence | Fail closed; never select a different recovery candidate automatically |

Repeated forks must preserve each node's original provenance and each source
epoch's exact receipt audit, without embedding the full chain in every new head.
The short immediate-origin descriptor is audit information; it is not a
substitute for validating all copied payloads. A hash link does not prove bytes
are present, uncorrupted or scientifically coherent.

## Validation and resource evidence

Fresh serial G8/G16/G32/G64 measurements and exact commands are retained under
`safe_recovery_epochs/evidence/`. The existing driver grows a legal Morpion graph
to **five nodes** when asked for a minimum of four, then republishes the fixed
population with no-op exact receipts. This is a history-scaling experiment, not
a large-graph benchmark. Process RSS includes imported runtime/Torch overhead.

| Measure | G8 | G16 | G32 | G64 |
| --- | ---: | ---: | ---: | ---: |
| Proposed three-root projection, bytes | 584 | 587 | 587 | 587 |
| Exact active receipts, count | 7 | 15 | 31 | 63 |
| Exact receipt metadata, bytes | 4,502 | 9,371 | 19,115 | 38,603 |
| Current cumulative publication metadata, bytes | 15,596 | 31,161 | 62,297 | 124,569 |
| Illustrative epoch/origin record, bytes (not implemented) | 592 | 593 | 593 | 593 |
| Retained heavyweight capsules | 8 | 16 | 32 | 64 |
| Scientific snapshot allocated file bytes | 131,072 | 262,144 | 524,288 | 1,048,576 |
| Workspace allocated bytes, unique inodes | 1,380,352 | 2,600,960 | 5,033,984 | 9,895,936 |
| Historical metrics JSONL, bytes | 36,835 | 78,958 | 163,192 | 331,639 |
| Stage metrics, bytes | 35,934 | 77,048 | 159,303 | 323,652 |
| Conditional reclaimable file allocation | 307,200 | 798,720 | 1,781,760 | 3,747,840 |
| Actually reclaimable under current protocol | **0** | **0** | **0** | **0** |
| Read-only whole-history analysis, seconds | .0442 | .0846 | .1724 | .3541 |
| Analysis hash calls | 329 | 641 | 1,265 | 2,513 |
| Analysis bytes hashed | 685,416 | 1,354,850 | 2,693,729 | 5,371,472 |
| Analysis process peak RSS, MiB | 291.07 | 290.83 | 291.53 | 293.20 |
| Repeated publication, seconds | 1.368 | 1.404 | 1.438 | 1.371 |
| Publication process peak RSS, MiB | 294.86 | 294.50 | 294.77 | 295.63 |

The epoch-size row serializes only the illustrative fixed-field identity/origin
object in `summarize.py`; the actual protocol has **zero epoch bytes**. Its
identifiers are fixed-width examples, not invented measured epoch executions.
The three-root row is the existing analyzer's *proposal*, not a compacted actual
run-state. Differences in integer digit count explain 584→587 and 592→593.

G64 selected Dataset discovery separately takes **0.0262 s**, hashes **47,611
bytes in 29 calls**, and records 290.29 MiB process peak RSS. It reads 795,560
logical bytes in total (`/proc` rchar), despite the much smaller hashed payload.
The measured CPU is .00895 s; physical read bytes are zero with OS page cache
present. No cold-disk claim. The three fsync calls belong to existing observation
of a committed publication. This is the existing selected-generation path, not
whole-history reachability and not a new epoch performance result. G8–G32
discovery timings were not measured in this run.

The read-only analysis makes zero state decodes, writes or fsync calls and
preserves names, bytes, inodes, mtime and ctime. It still scans all history.
Repeated publication hashes 890 files/about 4.05 MB at each sampled generation;
this does not make its cumulative run-state parsing/serialization independent of
G. No old artifact was removed. The conditional allocation excludes directories,
unknown ownership, open-FD delays and filesystem sharing. Disk is measured by
unique inodes; scientific allocation is reported separately from checkpoints,
models, indexes and metrics.

All four scientific oracle digests equal
`8c7f4cceb72854fe4c3b5aa3cc03f25638bb9ca10bf721b93de84b431f6847da`.
The indexed cold/reused reader matches the full closed oracle, including values,
certainties, topology, state references and creation generations. Two additional
actual search steps compare selector/RNG after restore with different constructor
seeds. Only checkpoint metadata differs in this tiny fixture's section comparison.
This continuation probe loads both sides with `model_bundle_path=None`; it is
not evidence of epoch adoption or arbitrary neural model transition parity.
Pinned model bytes and producer identity are independently included in capsule
verification. A future fork test must also restore the explicitly selected model.

For fixed additional roots R, the proposed active descriptor uses O(R) references
(plus O(log G) numeric digits), and the immediate epoch origin is O(1) references.
Receipts remain O(P) in RAM and disk for P acknowledged patches in the active
epoch. Current run-state history remains O(G+P); repeatedly serializing it for
G generations with P proportional to G is quadratic cumulative metadata work.
Closed checkpoints/scientific snapshots are O(N+E+S) each, where S is the total
canonical state/replay payload; retaining all generations costs the sum of those
sizes. The current analyzer has O(G+P+F) metadata and O(B) validation work for F
files/B retained bytes. Even after future retirement, neither scientific state
growth nor an indefinitely long active epoch's exact receipts are bounded by a
three-generation window. No graph-size scalability conclusion comes from N=5.

The production delta is **zero lines added, zero removed, zero changed**.
Research tests/report/evidence are separate from production code. No epoch
implementation is benchmarked.

### Tested scope versus the remaining acceptance suite

Focused validation passes **174 distinct cases**: the main run has 172 passes
(166 existing cases plus six new boundary characterizations), then two additional
receipt-erasure controls pass in a targeted run. The existing cases include the
real FORCED(35)/ESTIMATE(42) endgame across
all nine policy names: five supported v2 policies and four rejected gates. Exact
Reevaluation patch/cursor comparisons and checkpoint selector/RNG tests run for
JSON-zst and sharded checkpoints. No target or certainty policy was changed.
Closed worker parity tests use the existing explicit test adapters; ordinary
worker routing has not been activated for closed snapshots or epochs.

| Requested property | Evidence or remaining blocker |
| --- | --- |
| Interrupted normal publication | Existing 19 closed-publication process-death boundaries, fsync faults, and repeated interruption pass |
| Exact retry/conflicting ID | Existing receipt tests plus cold restart counterexample; exact application remains once-only |
| Concurrent Growth and old reader | Existing contention tests plus new shared-pin/exclusive-owner test pass |
| Bad checkpoint/model and missing history | Existing fail-closed receipt/reachability tests pass |
| Stale Dataset/Training claim | Existing analysis retains even expired claims; no worker-death inference |
| Pending old patch | Current scientific source rejects a different capsule; epoch label alone does not fence application |
| Late Training result | Actual completion path counterexample shows missing epoch fence |
| Explicit fork from retained older snapshot | Existing initializer refuses; no executable adoption protocol implemented |
| Exact continuation after an epoch fork | **Blocked**; existing same-lineage continuation is not substituted for this test |
| Crash before/during/after epoch commit | **Blocked** until the new commit payload and acceptance boundaries exist |
| Restart with stale discovery | Existing same-lineage reconstruction tested; wrong-epoch discovery remains an explicit future rejection test |
| Repeated fork/recovery cycles | **Blocked**; must prove original node provenance and independent receipt domains, not just repeat same-lineage restarts |

The first installed-wheel CI attempt used an overly long secondary-disk
`TMPDIR`: multiprocessing reported `AF_UNIX path too long`, four match tests
failed, one file-descriptor regression timed out with the same socket error,
and a later DataLoader stalled. That attempt was interrupted and retained
(2,371 passes, five failures, two skips, four deselections; exit 2). The five
affected tests and both supervised-learning cases subsequently pass together
with a fresh short temporary directory on the same secondary disk: seven passes
in 41.38 s, without changing production code. Full CI is rerun with that
corrected environment; this initial attempt is not labeled successful.

The corrected complete installed-wheel run passes: **2,401 passed, two CUDA
skips, four normal marker deselections**, 129 warnings, pytest time **1,484.79 s
(24:44)**. It ran serially with the original CI marker expression and the existing
fixture-isolation plugin. This is local CI validation; its serial duration exceeds
the hosted workflow's 15-minute pytest step limit. No hosted CI result is claimed.

The sdist/wheel build, installed-package guard, pinned toolchain and repo-wide Ruff
checks pass. All **472 production Python files** are byte-identical to the
installed wheel. The strict static ratchet passes with **206 inherited findings,
zero new findings**, against a reviewed baseline of 209; the three removals are
inherited from the base, not changes made here. All 34 pre-existing worktree
registrations/HEADs/statuses/tracked binary diffs remain unchanged.

See [validation results](safe_recovery_epochs/evidence/validation.json),
[source fingerprints](safe_recovery_epochs/evidence/source-manifest.json), and
[reproduction commands](safe_recovery_epochs/README.md). The research-only Python
delta is 345 lines in two files: executable boundary counterexamples and a small
reducer of existing measurement JSON. Neither is a production protocol or cache.
No unresolved new-epoch acceptance case is reported as passing.

## Remaining retirement conditions and compatibility code to remove

Retirement requires the complete epoch admission/acceptance proof above, a
validated closed recovery boundary, exact receipt/cursor/model coherence, stable
reader pin admission (no path/lock ABA), and reachability over pending work,
explicit retention and models. Unknown ownership or incomplete validation must
retain dependencies. Count unique `(device,inode)` allocations and unknown
external hardlinks, not path sizes. A snapshot protected by an old reader remains
required even when another epoch is current. No deletion is authorized here.

Creating a new epoch does not itself close the old epoch or retire its authority.
Before any future reclamation there must also be evidence that no old producer
can start new work against the retiring dependencies; completion of observed jobs
or their expired TTLs is insufficient. Old runnable protocol-3 workspaces still
declare all their committed capsules required. An explicitly validated migration/
closure and safe admission policy are prerequisites for changing that meaning.
Merely archiving their run-state file or observing an empty pin set is not a
retirement fence. Keep these generations required until that condition is proven.

After independently validated migration, candidates for removal/consolidation:

* `generation_publication.py`: v1 flat/additive cloning and v2 cumulative catalog
  branches, format dispatch/adoption and repeated historical model search.
* `scientific_generation_publication.py`: cumulative `catalog_paths`, catalog
  cloning and historical export validation on the current closed-only path.
* `scientific_publication_initialization.py`: genesis-only special handling,
  replaced by one validated genesis/adoption publisher, not two permanent paths.
* `checkpoint_patch_receipts.py`: non-publication/flat activation branches;
  keep exact receipt validation, atomic commit and acknowledgement invariants.
* `pipeline_artifacts.py` and `pipeline/cursors.py`: legacy inferred source/
  missing identity defaults and independent authoritative singleton updates;
  keep necessary projection serialization.
* `cycle_pipeline_manifest.py`, `bootstrap_loop.py`, `derived/orchestrator.py`:
  duplicated model/status publication routes; consolidate current-mode acceptance.
* Old scientific v1/additive readers/writers and discovery adapters: remove from
  the production path once required sources have a validated explicit migration.
  A scientific compatibility oracle may remain offline, outside runtime dispatch.

Do not delete the shared bounded parser/SQLite reader, atomic I/O, stable pins,
exact hash checks, checkpoint codec, or certainty policies. Do not remove an old
model representation needed by retained weights without numerical and semantic
validation. Backward compatibility is optional; these dependencies can still be
scientifically necessary.

Known limitation remains visible: Anemone's representative-parent checkpoint
roundtrip discrepancy was previously observed in 7/70/161 representatives at
1003/5013/10018 nodes. The tiny unchanged-population continuation checks here do
not establish universal representative-parent parity or raw checkpoint byte
parity. A fork validator must compare full graph/runtime state and state this
limitation explicitly; node-value equality alone is insufficient.

## Integration strategy

Local review branch only, directly on exact `089fec31`; no existing implementation
is cherry-picked twice and no existing PR is rebased. Land this contract/evidence
as a small review on that local base once the existing stack has a clean review
base. The next independently reviewable change should close the producer
proposal/cursor recovery gap, with epoch-bound acceptance tests, before enabling
fork adoption. Its exact future base is the reviewed contract commit, not main
or a guessed PR head. No duplicate-code monolithic PR has been opened.

**Outcome:** the conditional receipt-reset proof and the simplest workspace
isolation contract are explicit. Safe executable epoch fork and eventual deletion
are still blocked; passing the existing recovery tests is not a claim that those
new operations exist or are scientifically ready.
