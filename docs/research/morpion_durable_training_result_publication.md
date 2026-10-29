# Lean durable Training result publication

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Base: `1dec3f79af0989d52921d16ffb1785d2fa801a25`, containing Dataset implementation
`fb81af3b9b2b7a6adf83cf04d4ef95c6de2c8eee` and atomic Reevaluation
`58176ff3b7278f893c9cc4deb282d57f163c30a8`. No implementation was reapplied.
New secondary-ext4 worktree:
`/home/pompote/oldata/victor/morpion-scratch/durable-training-20261023/worktree`;
branch `fix/morpion-durable-training`.

Implementation: `81100b883e6582b9156d3bbe4c88665e7003f8f0`.
Compact RNG encoding: `9d2d5fd7eb6648c95eedc54aef79304dde4f6d30`.
Private input/cache cleanup: `403b3d407947fb881494bceb3204ecba0b7b0605`.
Continuation test head: `e63f2b01f564db8e711233150015bf573a6c612a`.
The exact 476 production source hashes, installed-wheel identity, dependency
versions, ancestry and measurements are in
[`durable_training/evidence/`](https://github.com/victorgabillon/chipiron/tree/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/durable_training/evidence).

## Scope and ownership

Training computation produces candidates. Acceptance records exactly one result
for that Dataset generation. **Neither step activates a model in Growth.**
The active-model pointer, checkpoint, search evaluator identity, node provenance,
exact patch receipts and accepted Reevaluation cursor retain their existing values.
The fixed-model closed-publication restriction remains in force.

Only fresh explicit genesis can enable
`initialize_scientific_v2_publication(..., closed_scientific_snapshot=True,
atomic_reevaluation=True, fenced_dataset=True, durable_training=True)`.
Every flag still defaults to false. No CLI, launcher, historical workspace or
production route is activated. This task performs no migration or epoch fork.

`run_state.metadata.durable_training = {"schema_version": 1}` dispatches the
current protocol. Missing markers with accepted references, unsupported versions,
boolean versions and missing Dataset/epoch dependencies fail closed. An older
binary is not an authorized writer for this new protocol. Permanent historical
multi-format support is not a goal.

The existing Growth flock protects admission, acceptance and claim release. It is
released during computation. The existing scientific-generation shared pin spans
the public Training operation and acceptance also acquires its own pin. Future
retention must treat the accepted Dataset and Training references as dependencies
of that pinned generation. No cleanup of committed generations is implemented.
TTL claims schedule attempts; expiration does not prove that a reader has stopped.

## Captured admission

Before Training starts, the private attempt records:

- Schema version, a unique 32-hex attempt ID, the existing claim ID and generation.
- The authoritative workspace/epoch, without manufacturing a Training epoch.
- The exact accepted Dataset attempt and descriptor digest, scientific capsule
  ID/descriptor/scientific identity, row SHA-256 and row byte count.
- Ordered candidate names; complete effective `MorpionTrainingArgs` including
  defaults; resolved model arguments; maximum rows/chunk size and diagnostic options.
- Captured forced-candidate control and the existing selection-policy contract.
- Python/PyTorch versions, the existing Torch initial seed and complete current
  CPU/CUDA RNG bytes in reversible hexadecimal, determinism/backend settings,
  thread count and relevant Morpion/CUDA/BLAS environment settings.

The trainer previously used process Torch RNG for model initialization and the
configured validation seed for split/shuffle behavior. Admission **observes** those
states; it does not reseed them. The hex encoding round-trips exact RNG bytes.
Candidate order is an explicit list: JSON key sorting cannot change tie breaking.
The source manifest separately pins the implementation/dependency build used in
validation. The versioned contract does not promise identical results across
unvalidated PyTorch versions, GPU hardware or nondeterministic kernels.

The existing accepted-Dataset validator checks the descriptor, source binding,
completion and exact row bytes. A visible row file or mutable `done` status cannot
supply the input. A private `admission.json` records capture before computation;
publication and acceptance require it to equal the completion's captured admission.
It is evidence, not another transaction authority.

## Computation and private artifacts

The existing `train_and_select_evaluators_streaming` performs Training. Its
architectures, optimizer, losses, rows, row limit, split, shuffle, seeds, diagnostics
and candidate selection are unchanged. Normal model serializers remain in use.
The private paths keep every candidate, metric and diagnostic separate from other
attempts and from shared model output paths.

Existing trainers create tensor caches beside their input. Training therefore makes
one bounded-copy, exact-hash private copy of the accepted JSONL. The existing cache
implementation operates only there. After the synchronous trainer returns, the
copy is rehashed and **only that attempt's private input/cache directory is removed**
before publication. No cache is added to the accepted result. This is cleanup of
new, private temporary files; it never deletes accepted Dataset rows or any
historical/committed artifact. An interrupted attempt can still leave private files;
no abandoned-attempt collector is introduced.

Candidate bundles retain the existing weight, model-argument, manifest and readable
weight files. Publication reloads one model at a time with the existing loader to
check architecture/weight compatibility, compares resolved architecture,
hyperparameters and final metrics to admission/completion, and preserves caller RNG
around model reconstruction. Acceptance checks the exact bundle digest. This is the
existing sorted `(relative path, size, file SHA-256)` inventory digest, so weights,
configuration and model metadata are all bound, not just a visible directory name.

Model manifests retain the original computational input/output path metadata;
those paths describe the private attempt before rename. The accepted proposal's
Dataset reference and model paths identify the durable dependencies. No post-hoc
rewrite of model metadata or custom model serialization is introduced.

The family Training API used here has no exact mid-training-resume contract. A
partial attempt is not an accepted result; a fresh attempt may recompute. The
separate value-training API's existing optimizer/RNG checkpoint/resume behavior is
unchanged and remains covered by existing tests. A transient checkpoint or complete
visible model is never imported as accepted Training merely because it exists.

## Immutable result and sole commit point

`training_publications_v1/staging/<attempt>/` contains the private outputs.
The small `proposal.json` binds admission, every candidate's complete bundle digest
and durable path, and the existing Training status serialization: selected candidate,
selection policy, final train/validation metrics, per-candidate performance and
completion metadata. Both candidates, including the nonselected one, are retained.

Existing `sync_artifact` and `publish_directory_if_absent` fsync the files and
namespace and publish at `training_publications_v1/objects/<attempt>/` using
`renameat2(RENAME_NOREPLACE)`. Serializers reject writes into accepted namespaces.
The object remains unaccepted until the run-state transaction.

Under Growth ownership, acceptance:

1. Reloads `run_state` and validates its exact receipt/cursor coherence.
2. Rechecks version, workspace/epoch, generation, accepted Dataset reference and
   scientific identity; retains the selected scientific pin.
3. Checks immutable admission evidence, complete candidate paths, selection policy
   and selected candidate using the original policy in the captured candidate order.
4. Rehashes every model bundle and the original accepted Dataset rows, and fsyncs
   the output dependencies before the commit.
5. Requires the current claim's stage, generation and ID to match for a new result.
   A competing accepted identity is rejected. The identical already accepted
   reference can be acknowledged again, even after the claim is gone.
6. Adds one field to the **existing** scientific publication entry:

```json
"accepted_training": {
  "attempt_id": "<32 lowercase hex characters>",
  "descriptor_sha256": "<SHA-256 of proposal.json>"
}
```

The existing durable `save_bootstrap_run_state(..., durable=True)` replacement is
the only acceptance point. No model registry, journal, cache, lock service, Training
ledger or authoritative singleton cursor is added. The commit does not alter
`latest_model_bundle_paths`, active model, checkpoint or producer continuation.
Discovery projections omit this mutable result reference from scientific identity,
as they already do for accepted Dataset references.

After the commit, advisory manifest/status projections are repaired from accepted
evidence. The normal accepted-stage retry requires the same Training configuration.
A changed request is a conflict, not permission to replace the result. Selected
consumers hash actual model/row dependencies; ordinary discovery reads descriptors
without scanning every historical model's weights.

The existing latest-pending/monotonic Training scheduling rule remains. Its completed
frontier is derived from accepted entries, with live claims providing the started
frontier, rather than trusting the legacy cursor singleton. An already admitted
older generation may finish after a newer result; it does not activate the older
model or rewind any frontier. A newer Growth head alone does not revoke that
admission. Legacy stale-state recovery cannot unlink replacement claims in this mode. The old
status-backup/reset procedure and its extra grace heuristic are not used to establish
authority: claim TTL controls replacement eligibility, while kernel pins protect
actual readers. Failed/unaccepted work can be retried without retaining a historical
started-cursor blocker. These are explicit recovery-protocol differences, not changes
to Dataset ordering, model selection or the accepted-generation scheduling frontier.

## Recovery and integrity assumptions

| Failure boundary | Official result after restart |
| --- | --- |
| Before/during Training, partial model serialization | None; fresh attempt can recompute |
| Output fsync/rename, visible object before acceptance | None; visibility is insufficient |
| Before run-state replacement | Previous authority |
| Replacement observed, including error before directory fsync returns | Validate and fsync the complete observed accepted authority |
| After acceptance, during projection repair or acknowledgement | Same accepted descriptor and exact model bytes; repair and acknowledge again |
| Replaced claim, foreign epoch/source, corrupt/missing input/model, competing result | Reject without committing a replacement |

Claims are advisory: an expired owner can finish if nobody replaced its claim;
a replacement identity fences it even if it is still alive. Late cleanup checks
identity and cannot release the replacement's claim. Busy Growth ownership is an
explicit retryable failure; no unlocked commit or fairness/mailbox service is added.
The existing Growth API holds ownership for its invocation. Long Growth invocations
can delay admission/acceptance. An unaccepted attempt may require recomputation;
there is no new automatic adoption/resume protocol for orphaned results.

Integrity assumes cooperative immutable writers, trusted computation/serializers,
SHA-256 collision resistance and local POSIX flock/rename/fsync semantics. Output
validation proves binding and artifact integrity; it is not an independent proof
that an arbitrary hostile worker executed an optimizer correctly. No code path
silently trusts corrupt or unsupported dependencies. Reader pins protect lifetime;
they do not replace content verification.

## Scientific and model evidence

The new normal suite contains **45 cases**, including:

- Two actual tiny CPU candidates, automatic and forced selection, exact weight-file
  bytes/tensors, resolved architecture and numerical metric parity against the
  unchanged trainer on the same rows and RNG. Paths/timings are intentionally not
  claimed byte-identical across independent computations. Restart preserves the
  complete accepted artifact bytes exactly.
- Explicit tie-order parity and preservation of the existing rejection of nonfinite
  persisted candidate metrics; no new selection/proof rule.
- Input/reference/configuration/model/descriptor corruption, missing weights/config,
  stale/foreign identities, configuration conflicts and protocol downgrade refusal.
- Growth advancing during Training, newer Training finishing first, older admitted
  Training finishing later, and subsequent Growth preserving both accepted results
  while keeping the original active model.
- Missing projection repair, exact accepted retry, visible-unaccepted refusal,
  immutable serializer guards and real source pin/exclusive-owner exclusion.
- Thirteen actual local process-death points: before/during Training, partial model
  serialization, before/after publication, publication namespace sync, before
  acceptance, before/after replacement, after durable acceptance, before/after
  projection, and acknowledgement.
- Two real local processes with barriers: the expired old worker resumes only after
  its replacement commits, then fails without changing the accepted result.
- File-fsync, post-replace directory-fsync and projection failures.
- Byte-identical actual Reevaluation proposals before/after Training acceptance.

The existing actual Morpion endgame tests retain `FORCED(35)` versus neural
`ESTIMATE(42)`, selected-source confidence and terminal protection. Nine legacy
policies and all nine v2 names remain checked: five supported policies, four gated
exact-family policies. The Dataset reference/bounded comparisons preserve complete
JSONL rows/metadata: the complete final CI fixtures yield **29 byte-identical file
pairs containing 53 paired rows**, with all fingerprints retained in
`scientific-parity.json`; the other 16 v2 combinations are gated refusals. No
scientific payload or target policy is edited here.

The known Anemone representative-parent checkpoint discrepancy remains independent:
older large fixtures showed 7/70/161 raw node-shell differences at
1003/5013/10018 nodes. This change neither normalizes those differences nor claims
that Training acceptance resolves general checkpoint scientific-readiness questions.

## Validation status

Final complete local CI on `e63f2b01`, using the rebuilt installed wheel:
**2,632 passed, 2 skipped, 4 deselected, 141 warnings**, exit 0, in **1,945.27 s**
(32 min 25 s). This is one complete invocation, not a sum of targeted reruns.
Both skips require unavailable CUDA; the four deselections are the repository's
normal integration/external-data marker exclusions. All 45 new Training cases
passed within that invocation. The measured whole-command wall time was 1,950.58 s;
peak RSS was 1,060,624 KiB (1,035.77 MiB) for the entire suite, distinct from the
tiny protocol benchmarks below. JUnit, the full log, command resource counters and
the exact selected/deselected summary are retained with the evidence.

The installed-package guard, pinned toolchain check, sdist/wheel build and install,
Ruff lint and formatting all passed. The existing strict static-analysis ratchet
reports 209 baseline diagnostics, 206 current, **zero new**, three removed; its
baseline was not edited. The final wheel's 476 production Python files match both
the checkout and recorded SHA-256 manifest byte-for-byte. Nonfatal warnings include
PyTorch deprecations/transformer settings, existing fork-based tests and SQLite
connection resource warnings in reader/cache tests. They are preserved in the log,
not suppressed or presented as resolved by this Training change.

After the complete run, all 37 previously registered worktrees still had the same
registration/HEAD, existence, status and tracked-diff SHA-256 as before this task.
This check does not claim an exhaustive content audit of every historical artifact.

Before the final run, 42 installed-package Training cases, three independent installed
producer/serializer cases, 68 focused Training/scientific cases and 55 shared
Dataset/Training/supplementary cases passed in their respective runs. Those overlap
and must not be summed as a total. After private-cache cleanup, 28 focused cases and
the subsequent-Growth check passed. The complete result above supersedes these
targeted checks as the final CI evidence.

One earlier full-suite invocation was deliberately interrupted after 171 passes,
4 normal deselections, at 180.60 seconds, to remove attempt-private caches before
publication. It is **not** a complete passing CI run. Development-only harness
mistakes (missing temporary parent, relative benchmark root, moved test imports)
and an overly strict temporary JSON numeric comparison were corrected; their logs
remain in scratch. A speculative nonfinite-loser fixture established that the existing
status type already rejects nonfinite values; the final regression preserves that
behavior. No scientific expected values or production static baseline were weakened.

## Resource measurements and limits

Three fresh serial CPU Training processes and three independent fresh restart
processes use newly generated identical two-node/two-row sources, two candidates,
one epoch each, batch/chunk size one. Existing `StageMeasurement`, process RSS/CPU/I/O,
fsync counters, disk accounting and cProfile are reused. No benchmark overlaps a test
suite or another benchmark. Kernel physical read bytes are zero after fixture creation
and writes; these are OS-cache-warm protocol measurements, not cold-device throughput.
CUDA is unavailable; no GPU resource or cross-GPU parity claim is made.

| Phase | Median wall time (range) | Median CPU | Actual process peak RSS |
| --- | ---: | ---: | ---: |
| Output publication, including one-at-a-time model reload | 88.82 ms (79.20–89.90) | 11.32 ms | 381.59 MiB |
| Authoritative acceptance and projections | 74.65 ms (73.76–74.95) | 20.54 ms | 381.59 MiB |
| Fresh-process accepted recovery | 56.56 ms (54.92–60.27) | 16.41 ms | 291.23 MiB |
| Complete tiny Training stage | 2.694 s (2.643–2.790) | See per-phase evidence | 381.59 MiB |

Publication uses 13 file/17 directory fsyncs; acceptance 16/17; accepted recovery 4/6.
Median file/directory fsync times are respectively 65.47/12.13 ms, 23.92/30.71 ms,
and 27.38/12.26 ms. Existing file hashing performs 9/24/24 calls, with cumulative
0.402/0.477/0.512 ms on these tiny fixtures. Zero canonical state decodes occur in
these publication/acceptance/recovery phases. These counts measure actual calls,
not an inference from nominal file count.

Accepted output: 12 files, about 101.5 KB apparent bytes, **163,840 B allocated**
including unique directory inodes. Descriptor: 28,249–28,264 B. Accepted input/cache
copies: **zero retained**. Run state grows from 4,342 to 4,550 bytes: **208 B** for
this generation's accepted Training reference. Scientific source fingerprints and
both weight hashes match across all final runs; exact values are in `resources.json`.
The fixed RNG bytes were first represented as indented integer arrays; replacing
that draft encoding with reversible hex reduced the descriptor from approximately
77.6 KB to 28.3 KB. Earlier prototype measurements remain separate in scratch and
are not mixed into the final medians.

For a fixed configured candidate family, Dataset byte size B, candidate model/output
bytes M, selected scientific closure S, G committed generations and P receipts: the added copying/validation/hashing
work is O(B + M + S) plus inherited O(G + P) authority parsing/rewriting. Working
memory added by the protocol is bounded I/O buffers, one model reload, candidate
metadata and inherited O(G + P) run-state metadata. It creates no O(N) Python search
snapshot or second tree. The existing trainer's tensor caches/index schedules can
still have O(N) resident memory in some modes; this work does not claim that every
pre-existing trainer is bounded. Its existing modes and numerical behavior remain.

Private scratch includes B and existing trainer caches while computation is active.
Accepted results retain model/diagnostic/provenance outputs, not the copied rows or
caches. Abandoned attempts can still retain scratch. One result reference is added
to each existing publication entry; exact receipts and publication history remain
cumulative. Run-state rewriting remains O(G²) over a linear accepted history (and
also includes cumulative receipt costs). Deriving the completed Training frontier scans the existing G entries; deriving
the started frontier checks their existing claim paths, adding O(G) filesystem
metadata work. No constant-history scheduling cost is claimed. Accepted model
artifacts accumulate until a separately validated reachability/retention protocol exists. Neither total disk
history nor metadata RAM is claimed constant.

## Production delta and review boundaries

Production: **963 lines added, 92 removed, net +871 across 13 files**, including
comments/docstrings and type/formatting lines. This is not a claimed net deletion
or a performance optimization of Training. Per-file counts are retained in evidence.

`worker_result_io.py` extracts the existing Dataset generation/epoch lookup,
attempt-directory and descriptor identity checks, version-fence validation and
pure attachment to an existing publication entry. Dataset and Training use those
same small operations. It does not dispatch stages or own a transaction.
`training_publication.py` owns only the Training admission, candidate/provenance
checks, private computation, accepted projection and short commit boundary.
Existing filesystem primitives, lock, pins, run-state serializer, model loaders,
row readers, metrics and selector remain authoritative implementations. No third
parser, serializer, storage framework or cache is introduced. The repeated overall
admit/compute/accept shape is explicit domain control flow, not a new workflow engine.

After a separately verified one-time adoption of this protocol, opportunities for
removal include the old shared-path Training completion/activation branch in
`pipeline/stages.py`, legacy Training cursor writes/reads, visible-file recovery,
and the old stale-status backup/reset path. They are retained only for the existing
unactivated entrypoints in this change. A broad migration or deletion of historical
sources is outside this task; no only-valid source is overwritten. The current legacy
Training body spans 360 lines; its two cursor-write helpers span 32 lines and the
legacy recovery mutation function 62 lines. These are removal opportunities, not
independent guaranteed savings: retained callers and shared helpers must first be
reviewed. Exact locations are in `obsolete-code-opportunities.json`.

## Future model-activation transaction and integration order

The future interface must consume an exact accepted Training reference, selected
candidate and complete bundle digest together with the expected current workspace,
epoch, checkpoint/head, active-model identity and producer continuation. It must
explicitly define model-epoch transition, node-value provenance/reevaluation,
selector/RNG continuity, pending-patch fencing, reader/worker dependencies, rollback
and exact receipt behavior. Checkpoint, scientific values and active evaluator must
be committed coherently through the existing authority. An accepted Training result
alone is insufficient evidence for that transition. Acceptance also does not upgrade
a legacy Dataset confidence contract: scientific activation still requires the
selected-value-v2 contract and its five supported policies; the four exact-family
policies remain gated. No such interface is activated
or implemented here.

The local dependency chain is the validated Dataset head followed by the four
small local implementation/refinement/test commits listed above. A clean merged
upstream base equivalent to these local scientific/streaming/durability dependencies
has not been established in this task.
Review this delta against `1dec3f79`; do not open a PR containing duplicate copies
of those unmerged stacks. Once an exact equivalent upstream base is established,
port only this delta and rerun the same evidence before any draft PR or migration.
No remote PR, merge, production activation, gen38 operation, distributed service,
epoch fork or historical artifact modification was performed.
