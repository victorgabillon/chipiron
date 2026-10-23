# Minimal fenced Dataset publication

Base: `58176ff3b7278f893c9cc4deb282d57f163c30a8`, the validated atomic
Reevaluation implementation. Implementation commit: `fb81af3b9b2b7a6adf83cf04d4ef95c6de2c8eee`.
Worktree: secondary ext4 disk,
`morpion-scratch/fenced-dataset-20261023/worktree`; branch
`fix/morpion-fenced-dataset`. Existing branches and experiments are not adopted.
Implementation fingerprints and the exact production diff are in
[`fenced_dataset/evidence/`](fenced_dataset/evidence/).

## Scope and explicit architectural decision

Dataset computation and publication use the existing bounded scientific reader,
row serializer, immutable-directory publisher, source pins, Growth ownership and
durable `run_state` replacement. No new parser, SQLite framework, epoch, journal,
cache, database or lock service is introduced. Training receives a verified
Dataset input; its model/cursor publication protocol is unchanged.

The existing scheduler can finish G2 and subsequently select pending G1. A lone
"last accepted Dataset" pointer cannot remember both decisions without changing
that policy or adding another acceptance history. Following the user's explicit
choice, **one optional `accepted_dataset` reference is attached to each existing
committed publication entry**. This adds cumulative metadata to the existing
protocol; it does not make its history bounded. There is no separate result
catalog. This tradeoff is explicit, not disguised as a constant-size solution.

Enable only at fresh genesis through
`initialize_scientific_v2_publication(..., closed_scientific_snapshot=True,
atomic_reevaluation=True, fenced_dataset=True)`. All flags retain their false
defaults. No CLI, launcher, production workspace or historical artifact is
activated or migrated. Fenced extraction requires the existing explicit streaming
and scientific-v2 opt-ins and uses the existing closed-snapshot reader option.
Legacy policy behavior remains available as reference behavior; scientific use
still requires the selected-value-v2 confidence contract and its five supported
policies. Publication correctness does not turn legacy confidence into valid v2
scientific evidence.

## Admission, artifact and acceptance contract

`run_state.metadata.fenced_dataset = {"schema_version": 1}` is a dispatch marker.
It introduces no identity of its own. The authoritative workspace and epoch come
from `atomic_reevaluation`. Unsupported/malformed markers fail closed; accepted
references cannot silently downgrade into an unmarked legacy mode.

Before row processing, under the existing source lifetime pin, admission captures:

- Workspace and epoch, selected generation and complete scientific identity.
- Committed capsule ID and exact descriptor digest.
- Eight Dataset options: depth/visit/row filters, backed-up-value preference,
  exactness filter, family policy, blending and confidence contract selection.
- Source model pins and source configuration digest, from the validated capsule.
- A new unique attempt ID and the existing scheduling claim's identity.

The source identity includes the scientific contract, generation and producer.
Source model provenance belongs to that source; Dataset does not re-label old
values using whichever active model happens to exist later. Arguments used by
extraction are copied before admission. There is no destination epoch/source
lookup that re-tags computed rows.

Rows are written by the existing streaming serializer into
`dataset_publications_v1/staging/<attempt>/rows.jsonl`. Every attempt has a
private directory, including the serializer's temporary filename. The completed
rows are parsed with the existing streaming row reader, checking count and size;
no complete Dataset or scientific snapshot is loaded. A small `proposal.json`
contains captured admission, exact row SHA-256/size and reconstructible
manifest/status evidence, including record/frontier diagnostics. Row JSONL bytes
are not changed by publication.

The existing fsync and `renameat2(RENAME_NOREPLACE)` primitive durably publishes
this directory at `dataset_publications_v1/objects/<attempt>`. Its existence is
not acceptance. Neither another attempt nor a cooperative row serializer can
replace files under this immutable-output namespace. An unaccepted object is
still an orphan, regardless of how complete it looks.

The existing exclusive Growth flock then guards acceptance. The consumer reloads
current authority, validates exact receipt/cursor coherence, rechecks captured
epoch/source/model/configuration bindings, holds the scientific pin, validates
the completion envelope and hashes the exact output bytes. It verifies that the
claim still belongs to this attempt and the scientific generation is still
committed. A newer Growth head alone does not disqualify an older source.

Acceptance adds exactly this reference to the existing generation entry:

```json
"accepted_dataset": {
  "attempt_id": "<32 lowercase hex characters>",
  "descriptor_sha256": "<SHA-256 of proposal.json>"
}
```

The existing durable `run_state` replacement is the sole commit point. Checkpoint,
scientific head, exact patch receipts and accepted Reevaluation cursor retain
their values. The existing serializer still rewrites the complete run-state file;
there is no claim that receipt metadata avoids that I/O.

A different accepted attempt cannot replace the first, even if its output happens
to have identical row bytes. The same accepted reference can be retried exactly.
A normal repeated stage invocation returns the accepted result only for identical
Dataset options, after integrity verification. Changing filtering/policy on a
completed generation is an explicit conflict, not an implicit overwrite.

## Recovery, claims and consumers

| Boundary | Authoritative result | Recovery |
| --- | --- | --- |
| Private rows or partial temporary | None | Recompute after normal claim recovery; no shared output was replaced |
| Complete immutable object, before acceptance | None | Object remains uncommitted; visibility cannot feed Training |
| Run-state replace observed | Exact accepted descriptor and rows | Validate dependencies and fsync observed authority; retain this result |
| Commit succeeded, projection write failed | Same accepted identity | Reconstruct manifest/status from accepted evidence |
| Exact accepted retry | Same identity | Verify and repair projections; do not alter acceptance |
| Foreign epoch, different source, changed output, replaced claim or conflicting attempt | No new acceptance | Fail closed; keep evidence for diagnosis |

TTL expiration only permits scheduling a replacement. If replacement has acquired
a different claim, the old worker cannot accept, even while it retains its source
pin. Its cleanup cannot unlink the replacement claim. If an expired claim has not
been replaced and no result has been accepted, its owner can still finish safely.
The stable kernel source pin, rather than TTL, protects reader lifetime.

Dataset holds exclusive Growth ownership during admission, acceptance and claim
release, **not during extraction**. The inherited Growth API owns that lock for
its invocation (default one cycle). Busy ownership produces an explicit retryable
lock failure, never an unlocked commit. Long multi-cycle Growth invocations can
delay Dataset admission/acceptance; no fairness or automatic mailbox retry is
introduced. Source closure/output verification also costs time inside acceptance.

Discovery resolves Dataset completion from the accepted reference and its small
hashed descriptor. It does not hash every historical JSONL. Selected Training
admission verifies the exact accepted rows while retaining the source pin. A
merely visible output or forged `done` projection cannot become a Training input.
Missing/corrupt accepted rows stop selected consumption; they are not silently
recomputed under the same identity. Missing or stale completion projections can
be repaired. Source-identity conflicts remain errors.

Scientific discovery projections explicitly omit the mutable Dataset reference,
just as they omit continuation authority. This preserves their committed
scientific-source interpretation. Ordinary Training status fields remain
advisory; their model acceptance is not made safe by this change.

## Validation

Four baseline characterizations reproduce shared-path replacement, eligibility
based on a visible completion after file loss, latest-pending backfill, and a
TTL takeover overwritten by a still-running old invocation. An early fixture
forgot to force G2 publication; the corrected fixture commits a real synthetic
patch before checking backfill. Logs of that harness failure are retained.

The new normal suite contains **85 cases**:

- Normal publication, restart, exact retry, configuration and attempt conflicts.
- Unaccepted visible rows refused by selected Training; missing/corrupt rows and
  descriptors; missing and stale projection repair.
- Foreign workspace/epoch, source/model/configuration mismatches, malformed or
  incomplete JSONL, changed source bytes during computation, and a synthetic
  epoch change proving the result is not re-tagged.
- G1 pinned through real Growth advancement to G2, followed by G1 acceptance;
  later Growth preserves earlier accepted references and receipt/cursor progress.
- Real shared source pins and exclusive Growth lock exclusion.
- Ten real local process-death boundaries: before/partial/after rows,
  before/after immutable publication, before acceptance, before/after authoritative
  commit, and before/after projection repair.
- Two local processes with explicit barriers: an expired old worker resumes after
  a replacement has committed and is rejected without changing its result.
- Four injected fsync/projection failures, including error after run-state replace.
- **45 scientific cases** using the actual reachable Morpion endgame. Nine legacy
  policies compare entire reference/indexed JSONL bytes. For selected-value v2,
  all nine names are checked with both target preferences and both exactness
  filters: five emit byte-identical reference rows, four remain explicitly gated.
  The 29 emitting cases contain **53 paired rows**; all full-file hashes and byte
  counts are retained in `scientific-parity.json`. The other 16 cases are refusals.
  `FORCED(35)`, neural `ESTIMATE(42)`, terminal protection, counters and provenance
  are retained. Full-snapshot loading is forbidden in the actual bounded stage.

Seven supplementary installed-wheel cases cover takeover before the replacement
finishes, missing/unsupported protocol markers, and both cooperative row
serializers refusing to overwrite accepted paths, and byte-identical actual
Reevaluation proposals before/after Dataset acceptance.

Development-only harness mistakes were corrected without relaxing scientific
assertions: a blocking test pin needed the explicit nonblocking option; the
partial-row kill fixture initially selected no rows because its direct-only
values were queried with backed-up preference; and the legacy research adapter
needed to override the new explicit closed-reader argument. Raw failed logs
remain in scratch. The corrected focused protocol/scientific cases pass; these
are not hidden as production defects or counted as green failed runs.

Full installed-package local CI: **2,587 passed, 2 skipped (CUDA unavailable),
4 normal marker deselections, 141 warnings**, exit 0; pytest **1,858.24 s**
(30:58). All 85 new normal cases are included; the seven supplementary
installed-wheel cases and four corrected baseline characterizations also pass.
Wheel/sdist build, installed-distribution guard, pinned toolchain, Ruff (709 Python
files) and strict static ratchet pass. Static baseline 209/current 206, zero new;
the three inherited removals are not attributed to this change. No baseline was
rewritten. All 474 installed Python sources match the working tree, and the 473
baseline installed sources match exact commit `58176ff3`. All 36 pre-existing
worktree registrations, HEADs, statuses and tracked diff hashes remain unchanged.
Full JUnit, summaries, source fingerprints and resource observations are retained
in evidence. No hosted GitHub CI run is claimed; this serial local run exceeds
the hosted test-step timeout and is not represented as hosted green.
The known larger-checkpoint representative-parent discrepancy remains an
independent limitation. Dataset acceptance does not rewrite checkpoints or claim
to fix that continuation issue.

## Resource evidence and complexity

Six fresh serial processes, three per mode, use identical newly generated
2-node Morpion sources and direct-target preference, producing two rows each.
Baseline uses the installed exact `58176ff3` implementation; fenced runs use the
new installed wheel whose source fingerprints are recorded. Existing
`StageMeasurement`, RSS/CPU/I/O, decode/fsync and disk instruments are reused.
Cold reader indexing is included; source files are already in the OS page cache
after fixture publication (kernel physical read bytes are zero). No benchmark
overlaps full CI.

| Observation | Baseline | Fenced |
| --- | ---: | ---: |
| Median Dataset wall time | 0.2782 s | 0.3656 s |
| Wall-time range | 0.2751–0.2992 s | 0.3648–0.4232 s |
| Median CPU user + system | 0.02621 s | 0.04443 s |
| Median process peak RSS | 292.426 MiB | 292.523 MiB |
| Rows/s on this tiny fixture | 7.190 | 5.470 |
| Canonical state decodes | 2 | 4 |
| File/directory fsync calls | 5 / 8 | 13 / 28 |
| Process logical bytes read (`rchar`) | 300,532 | 499,525 |
| Kernel bytes written | 225,280 | 237,568 |
| Workspace allocation, unique inodes | 204,800 B | 237,568 B |
| Authoritative run-state size | 4,019 B | 4,283 B |

All six exact row hashes are
`4d8cf7e0ab20aca5f359e0bbbf47230e15ab1e6c526e6fd0ecc0ab1c6d300acd`
(2,291 bytes). Exact scientific-source fingerprints are recorded separately.
The extra decode pass validates the serialized output through the existing row
reader. These are protocol-overhead measurements, not large-N throughput or
memory-scaling evidence; earlier header-only probes remain in scratch and are
not used for throughput comparisons.

For B output bytes, R emitted rows, G committed scientific publications and P
exact patch receipts, the additional row validation and hashing cost O(B + R)
time and one row/state plus parser buffers in RAM. The existing bounded Dataset
family/index paths remain unchanged. Existing run-state parsing/serialization
still costs O(G + P) metadata memory and I/O, and scientific closure validation
still reads the selected capsule. Discovery reuses the already parsed publication
records so this extension does not introduce a new pairwise historical scan.

Each accepted generation adds one fixed-size reference to its existing entry,
plus its immutable output and descriptor on disk. At one acceptance per generation,
cumulative run-state rewriting costs the sum of O(G_t + P_t) over acceptances t,
bounded by O(G² + G·P_final) for at most one acceptance per generation. This is
O(G²) when receipt count grows linearly with generation. Descriptors do not duplicate the
publication/receipt ledgers. Unaccepted attempts also retain disk space; there is
no deletion or retention implementation here.

## Production delta and remaining work

Eight production files: **682 lines added, 47 removed, net +635**, including
comments/docstrings and the 491-line `dataset_publication.py`. This is a correctness
change, not a claimed net deletion or speed optimization. The one new module owns
the explicit Dataset contract shared by admission, acceptance and consumers;
its single attempt record carries captured identity and a private path. Existing
format serializers, pins, hashing, atomic writes and directory publication stay
shared. Per-file counts and source hashes are retained in evidence.

A future verified adoption of one current protocol can remove legacy shared-row
replacement, direct `done`/`failed`/`extracting_rows` publication branches and
visible-file-based recovery. The reference scientific oracle remains useful for
independent tests; permanent dual production protocols are not the goal. This
change does not perform migration or erase the only valid source.

Before production activation, Training still needs captured Dataset/model/epoch
admission and guarded atomic model/cursor acceptance; Dataset input verification
alone does not fence a Training worker's eventual output. Fixed-model publication
restrictions, intentional rollback/epoch adoption and alternate writers require
their own validated transition. The four exact-family policies still need an
explicit mathematical proof contract. No new proof rule is introduced.

Before any future cleanup, reachability must include accepted Dataset descriptors
and row files, in-flight attempts/claims, active source/Training readers, and all
model/patch/rollback dependencies. Source pins must protect this transitive
closure; TTL or age cannot justify deletion. Unknown ownership must fail closed.
No artifacts are deleted or quarantined by this implementation.

Integrity uses the existing cooperative immutable-writer, POSIX fsync/rename/flock
and collision-resistant-hash assumptions. It does not authenticate a malicious
worker capable of rewriting authority and all evidence, nor provide scheduling
fairness. No gen38 operation, distributed service worker, production activation,
historical-data mutation, remote PR, merge or migration was performed.
