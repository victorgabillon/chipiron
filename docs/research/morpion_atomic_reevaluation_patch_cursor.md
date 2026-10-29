# Atomic Reevaluation patch and accepted-cursor recovery

Local implementation based on `8530ec6e12a6876c30aa41ce60be9c971a11c2e0`
(investigation), whose parent is `089fec3106bdbc690435e61a5b0ce236a5b4821e`;
validated closed publication ancestor `ffa5ea7d3d9624c5efc8a562a184cf3f402d43e3`.
Branch `fix/morpion-atomic-patch-cursor`, isolated secondary-disk worktree.
No existing PR branch, historical workspace, production configuration or data was
migrated. This is a local change; no PR, merge, deployment or epoch fork.

## Result and scope

For the explicitly initialized protocol, a committed patch has a complete
accepted cursor in the **same durable `run_state` replacement** as its exact
receipt and resulting checkpoint/scientific head. Publishing a proposal does
not advance continuation. There is no second authoritative cursor file.
An exact retry can be acknowledged after later proposals without reapplying its
values or rewinding the current cursor.

Enable only while preparing a **new isolated genesis** with the existing
`initialize_scientific_v2_publication(..., closed_scientific_snapshot=True,
atomic_reevaluation=True)` API. The new argument defaults to false; no launcher,
CLI or production default is changed. Missing or unsupported protocol fields do
not trigger a legacy fallback. Existing scientific payloads and target policies
are unchanged. Closed worker routing still requires the existing explicit test
adapter; this change does not activate it in production.

This fixes the Reevaluation acceptance boundary. It does **not** make intentional
rollback or epoch adoption safe: Dataset/Training and alternate result writers
still have independent acceptance boundaries. The fixed-model publication
restriction remains in force. No receipt history may be reset.

## Exact protocol

`run_state.metadata.atomic_reevaluation` has exactly these fields:

| Field | Meaning |
| --- | --- |
| `protocol` | `morpion_reevaluation_proposal` |
| `schema_version` | Integer 1, with booleans/unknown versions rejected |
| `workspace_id`, `epoch_id`, `producer_id` | Three fresh 128-bit identifiers generated only at new genesis; immutable for this workspace/stream |
| `accepted_cursor` | Explicit null at genesis, otherwise the complete existing `MorpionReevaluationCursor` serialization |

The existing `checkpoint_bound_reevaluation_v1` metadata key is retained as a
version fence; its **schema version becomes 2** for this protocol. Old Growth
binaries expecting schema 1 refuse it. Each v2 receipt adds
`accepted_cursor_sha256`, the SHA-256 of the complete proposed next cursor using
sorted compact JSON. The latest accepted cursor must match the latest committed
receipt and its patch ID. A nonempty receipt ledger with a null/missing cursor,
a missing transaction identity, or a malformed cursor is rejected.

A proposal remains the existing `MorpionReevaluationPatch` serialized by the
existing create-only publisher. Its metadata contains the versioned
`atomic_reevaluation` envelope with:

- The protocol/version/workspace/epoch/producer identity captured at admission.
- The selected capsule ID, descriptor SHA-256 and complete scientific identity.
- The exact active model identity: evaluator generation/name, logical path,
  bundle digest, source and source-generation/local-training provenance.
- `expected_cursor_sha256`, identifying the entire previously accepted cursor
  (including the explicit genesis null).
- `next_cursor`: all continuation fields, including next node, selected source
  generation, evaluator identity, last patch ID, timestamp, completed-pass count
  and metadata. Missing or unknown cursor fields cannot acquire defaults.

`producer_id` identifies the single logical Reevaluation stream, not an OS
process. Competing processes share that stream and compare the same accepted
predecessor; they cannot create independent continuation authorities.

The patch itself carries the exact UUID/string ID, rows, selected range and
scientific provenance. The receipt hashes **all exact published bytes**, including
this envelope; neither a generation watermark nor an inferred receipt count
replaces UUID/digest deduplication. Existing arbitrary nonempty receipt IDs stay
valid identities; there is no renumbering scheme.

The producer reads accepted state and captures source/model/epoch **before**
loading/evaluating its selected window. It rechecks source and model before
publication. It never fetches a destination epoch after computation and attaches
it to an old result. Restart reads the accepted cursor from `run_state`; a visible
outstanding proposal blocks further production until Growth accepts/rejects it.
The historical singleton is neither read nor written in this mode.

Full-pass count increases by exactly the proposal's boolean wrap indicator from
the accepted predecessor. An evaluator identity change uses the existing reset
semantics in validation, but actual model transitions remain gated by immutable
publication. Zero rows can still carry a complete cursor transition and use the
same commit. An empty source or `max_nodes_per_patch=0` currently produces no
transition and advances nothing.

## Transaction state machine

All consumer actions remain under the existing stable exclusive Growth flock.
No additional lock service, journal, database, cache or authority was introduced.

| State/boundary | Durable authority | Recovery |
| --- | --- | --- |
| Computing or private proposal temporary | Previous head/receipts/cursor | Ignore private temporary; recompute from accepted cursor |
| Create-only hard link visible, including producer death before directory fsync | Still previous authority | Consumer validates and fsyncs the exact visible proposal before provisional application |
| Applied in a private runtime; checkpoint/capsule being prepared | Still previous authority | Discard uncommitted runtime; retry from previous checkpoint/cursor |
| Complete immutable dependencies published, before run-state replace | Still previous authority | Orphan capsule is not a commit; retry through existing recovery |
| Atomic run-state replacement observed | New checkpoint + scientific head + receipt + accepted cursor together | Validate all dependencies and fsync the complete observed authority; never combine old cursor with new values |
| Committed but acknowledgement lost | Same new authority, exact pending proposal may remain | Verify exact receipt bytes and cursor evidence, then unlink/fsync pending entry; never apply again |
| Exact old proposal retried after newer accepted progress | Newest authority remains | Verify old receipt/cursor digest, acknowledge without assigning the old cursor |
| Different bytes for existing ID, stale predecessor, foreign/unknown protocol, corrupt source/model/cursor | No new accepted state | Fail before application; retain mailbox for explicit diagnosis |

New proposal acceptance verifies committed source and model, current predecessor,
complete next cursor and exact bytes; applies once with unchanged blending;
prepares immutable checkpoint/science dependencies; rechecks pending bytes/model
and authority; then stores receipt and cursor in one replacement. Publication,
discovery and metrics remain projections/dependencies, not commit authorities.
The accepted cursor is explicitly omitted from generation discovery metadata to
avoid copying a second apparent continuation authority. The dashboard displays
accepted state for this protocol instead of an obsolete singleton.

Acknowledgement is a consequence of the commit, not a prerequisite for it. A
failed unlink or directory sync cannot authorize another application. A producer
that loses a create-only race cannot replace the winner. If it later republishes
its stale result after the winner is consumed, predecessor validation rejects it.

## Integrity and trust limits

Preserved: exact file/source/model hashing, capsule inventory validation,
immutable committed dependencies, shared source pins, fsync barriers, atomic
replacement and exclusive Growth ownership. No parent hash is substituted for
verification. The new envelope does not sign proposals or defend against a
malicious process rewriting the authoritative run-state and all its evidence.
The trust model remains cooperative immutable writers, local POSIX link/rename,
working fsync/flock and collision-resistant hashes.

A copied workspace retaining the same identities is the same logical epoch,
not a supported fork. No tool here changes an epoch in place. Foreign epoch
rejection tests deliberately mutate only generated fixtures to demonstrate that
old computation is not retagged or accepted; they do not validate epoch adoption.

## Validation evidence

The baseline run reproduced both gaps before edits:

1. A closed patch committed without a singleton cursor survives three cold
   restarts with effects/receipt intact but acknowledgement still blocked.
2. A real producer suffering a post-link failure retries as
   `pending_patch_exists` and retains its old cursor.

Sixteen baseline characterization/producer tests passed. These document legacy
behavior; they are not acceptance tests for the new protocol. The first legacy
regression run after the implementation passed 59 cases.

New tests are in `tests/environments/test_morpion_atomic_reevaluation*.py`:

| Coverage | Evidence |
| --- | --- |
| New/duplicate/stale/conflicting proposals | Values applied once; raw authority unchanged on refusal; old exact retry does not regress a newer cursor |
| Missing singleton and repeated restart | Complete accepted cursor recovered from the commit; no singleton is manufactured |
| Empty rows and wrap | Cursor-only proposal creates one checkpoint/receipt acceptance boundary; pass counts checked |
| Corrupt envelope/accepted cursor, missing fields and identity/model/source changes | Fail before application or fail closed at startup |
| Producer source/epoch capture | A mid-computation synthetic retag leaves the old captured identity on the result; Growth refuses it |
| Concurrent producers and Growth ownership | Two real local producer processes race at create-only publication; one wins; existing Growth flock excludes another consumer |
| Consumer process termination | All 19 existing closed-publication kill points reused, including partial checkpoint/science, seal, fsync, before/after replace, discovery and acknowledgement |
| Producer process termination | Actual worker dies at proposal file fsync, before link and immediately after link |
| Filesystem errors | Six capsule/checkpoint/model/state fsync faults and three producer fsync/temporary-cleanup faults |
| Complete patch equality | Full scientific reference and bounded worker produce byte-identical proposals for fixed identities; committed continuation matches the proposal |
| Actual scientific endgame | `FORCED(35)`/direct `ESTIMATE(42)` preserved; complete row/metadata/JSONL comparison for five supported policies; four exact-family policies explicitly rejected; direct/tree preference and exact-only filtering exercised |
| Actual checkpoint continuation | JSON-zst and sharded checkpoints retain scientific values, selector and RNG continuation through explicit atomic genesis |

The new tests reuse existing independent scientific assertions and crash hooks;
they do not weaken expected values or normalize output differences. Existing
nine-policy legacy tests remain part of the full suite. During development,
fixture adapter/signature mistakes were corrected; their failed logs remain in
the scratch evidence. Final validation: **2,502 passed, 2 CUDA-unavailable
skips, 4 normal marker deselections, 129 warnings**, exit 0, 1,742.60 seconds
(pytest; 1,744.48 seconds in the CI metrics wrapper). All **101 new protocol
cases** are included. Six supplementary installed-wheel research cases also
pass, reusing the older before/after-application and before/after-unlink kill
hooks and injecting failure of the acknowledgement directory fsync; exact
proposal bytes are deliberately resurrected to model an unpersisted unlink.

The wheel/sdist build, installed-distribution guard, Ruff and toolchain checks
pass. All 473 installed Python sources match the working sources byte-for-byte.
Strict static ratchet: baseline 209 diagnostics, current 206, **zero new**; the
three inherited removals are not attributed to this protocol. No baseline was
rewritten. This is complete **local serial CI**, not a claim that hosted GitHub
CI ran; its roughly 29-minute duration must not be confused with a hosted step
time limit. Full JUnit XML, summary, environment and source fingerprints are in
`atomic_reevaluation/evidence/`.

The known representative-parent discrepancy in larger Anemone checkpoint
round trips remains unresolved and visible in the prior closed-snapshot report.
This change does not rewrite graph parents or claim universal raw checkpoint
byte equality. Current DAG/value/certainty/canonical-state semantics and the
existing independent continuation checks remain intact.

## Resource comparison

Fresh serial processes, three repetitions each, generated identical two-node
Morpion states; baseline implementation **8530ec6e**, new implementation working
sources identified in `atomic_reevaluation/evidence/source-manifest.json`.
The reused `durable_scientific_v2.measure` instruments record actual process peak
RSS, CPU, I/O, fsync and state-decode counters. No new profiling framework.
Exact `records.jsonl` and `updates.jsonl` hashes match across every paired phase.

| Phase | Baseline median (range), seconds | Atomic median (range), seconds | Median peak RSS, baseline → atomic MiB |
| --- | --- | --- | --- |
| Construct/publish fixture proposal | .0405 (.0401–.0421) | .0393 (.0391–.0406) | 291.38 → 291.78 |
| Accept, deliberately lose ack | 1.0071 (1.0012–1.0333) | 1.0197 (1.0124–1.0204) | 291.59 → 291.94 |
| Fresh restart and finish ack | .0905 (.0787–.0933) | .0820 (.0801–.0820) | 291.01 → 291.19 |

A separate six-process measurement runs the **actual bounded producer**, cold
index included, with the same preloaded scalar evaluator and logical model name:
legacy median .2733 s (range .2683–.2811), atomic .2857 s (.2809–.3578);
median process peak RSS 290.12 → 290.81 MiB. Both perform three state decodes,
six file fsyncs and twelve directory fsyncs. File hashes increase from
68/148,662 bytes to 70/148,712 bytes: two complete model identity checks around
computation, each hashing the fixture's 25 bytes. For a real bundle of M bytes
these extra verification passes read 2M bytes; no claim about neural inference
throughput follows from a preloaded scalar evaluator. Whole row fingerprints
match exactly (`47e51c1b0dd962c67616cb7fc2053e5b0c13b0559fee74c45cb89fe9799e788c`).
The initial worker probe used different dummy-model paths in row provenance;
that mismatch remains in scratch evidence. Fresh fixtures were regenerated with
the same logical identity. No output fields were normalized away.

Acceptance includes cold restore, closed publication, existing validation and
run-state commitment; acknowledgement is deliberately deferred to measure crash
recovery independently. These tiny fixtures measure protocol overhead, not
large-N throughput or bounded total history. Sub-millisecond/RSS differences are
not evidence of a general speed improvement.

| Phase | File/directory fsync, baseline → atomic | File hash calls/bytes, baseline → atomic |
| --- | --- | --- |
| Fixture proposal | 2/4 → 2/4 | 11/15,107 → 12/15,132 |
| Acceptance | 32/75 → 32/75 | 447/3,193,338 → identical |
| Recovery | 7/19 → 6/16 | 86/340,438 → identical |

All measured phases perform zero canonical state decodes in this tiny unchanged
publication fixture. The extra proposal hash binds the model's 25 fixture bytes;
real producer computation also rehashes its captured model before publication.
The four removed recovery fsync calls were solely the singleton cursor barrier.
The existing create-only publisher still fsyncs its temporary and both directory
transitions. Full validation is retained. These are file-digest counts from the existing
instrumentation; small in-memory cursor/inventory hashes are not counted as file
reads. Cursor hashing is linear in the fixed-size continuation serialization.

After acceptance the formatted run-state is 9,762 versus 8,934 bytes (+828);
the compact current transaction is 551 bytes and receipt metadata 1,000 versus
903 bytes. After acknowledgement the measured workspace has 299,008 versus
303,104 unique allocated bytes (one fewer singleton file/block), and 65,550 versus
65,017 unique logical file bytes. Cumulative metrics and UUID-dependent path
lengths affect total bytes; raw records are retained rather than claiming a
universal per-generation disk reduction.

Let W be patch-window rows, R exact receipts and G retained publication roots.
The proposal adds O(1) identity/continuation metadata to O(W) existing rows;
accepted cursor state is O(1); each receipt adds one fixed-size digest. Existing
run-state parsing, receipt dictionary/serialization and capsule history remain
O(R + G) metadata memory/I/O, plus existing capsule/source validation costs.
Growth still uses its existing live search graph and node lookup; this change
does not make Growth O(W) memory. The new protocol does not construct a complete
scientific snapshot or a second in-memory tree. It does not solve cumulative receipt/publication growth or
historical discovery overhead. At one receipt per generation, cumulative
rewriting still costs O(G²) over the run; this task does not optimize receipts.

## Production code review

Six production files: **487 lines added, 74 removed, net +413** (including
comments/docstrings; per-file counts in `atomic_reevaluation/evidence/code-size.json`).
The one new 227-line module defines and validates the shared versioned envelope,
complete cursor and committed source reference. Genesis, producer and consumer
need exactly the same contract; separating these pure checks avoids duplicated
validators or an import cycle. It contains no I/O loop, storage engine or cache.
Existing model identity calculation is shared rather than copied. The publisher,
checkpoint/scientific serialization, receipt ledger, flock and atomic writer
remain the existing implementations.

The active-mode singleton authority is removed; the file API and old barrier
remain only for explicitly legacy workspaces. This is not a net code-deletion
claim. Source/receipt metadata is still read in full: in particular producer
admission temporarily materializes O(R + G) existing run-state metadata. Future
work on compact roots/receipt storage must preserve this acceptance contract.

## Compatibility and removal plan

Backward compatibility is not a design objective for the eventual current
protocol. In this focused change, existing unactivated workspaces keep their
old behavior; no historical receipt or missing continuation is reinterpreted.
The new mode has one envelope and one accepted cursor, using existing classes,
serialization, publication and commit functions. The old code remains only to
avoid silently adopting experiments or activating an unreviewed migration.

A separately authorized, validated one-time adoption can later remove:

- Legacy worker singleton load/save and legacy cursor-gated acknowledgement.
- Schema-1 dispatch and optional cursor digest handling in receipts.
- Obsolete standalone cursor dashboard/file-retention references.
- The temporary explicit genesis opt-in once a single current pipeline protocol
  is intentionally activated and old executables can no longer write it.

An old pending patch lacking complete next-cursor information must not be upgraded
by guessing from a singleton or receipt count. Keep its original valid source;
resolve its state explicitly before any validated adoption. Removing these
branches today would either break unadopted workspaces or disguise the existing
unsupported migration. No new converter was added here.

## Remaining requirements before safe epoch adoption

1. Bind Dataset/Training and every alternate result writer to identities captured
   at admission, and enforce guarded acceptance. Training's model/cursor writes
   are still independent; this task establishes no global stale-worker fence.
   Reuse/consolidate the committed workspace/epoch identity at that authority;
   do not mint independent stage epochs or another registry.
2. Validate a retained non-genesis boundary in a new immutable workspace: graph,
   canonical states, direct/tree/effective values and certainty/provenance,
   checkpoint, selector/RNG, exact model/configuration and origin namespace.
3. Define and validate model transition acceptance; current fixed-model rejection
   remains. Do not reinterpret a generation counter as an epoch.
4. Durably commit the new workspace before touching any old source. Preserve old
   receipt evidence and active/pending dependencies; no in-place epoch switch,
   implicit rollback or clearing of current receipts.
5. Independently validate reader/worker lifetime pins and reachability before any
   future retention/deletion feature. This change authorizes no deletion.

The scientific exact-family proof gate and known representative-parent issue
remain separate prerequisites. No new mathematical proof rule, target policy or
FORCED/ESTIMATE interpretation is introduced.
