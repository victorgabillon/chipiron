# Exact receipt model evidence without unrelated historical trees

Exact M1/M2 receipt acknowledgement at M3 now authenticates committed descriptors
and verifies the actual selected frozen model bundle without validating unrelated
historical checkpoint/scientific payloads. Current-head recovery and new-patch
admission retain their complete validation. No format, authority, scientific
policy, activation setting or deletion permission changes.

## Source and scope

- Base: `60184a7c0618ecbd6042fff1b8791b9742292506`.
- Base FULL-tested executable: `11b82d5cb08dfdb0678c4ae0f3558a0e56f193a4`.
- This executable: `fe0fe7a576f25dfdd6464060495cbd1356b03590`.
- Anemone: `90c7ef4454edf52459d7f7051b6c904ddf7f7a35`.
- Anemone wheel SHA-256:
  `5f3015c1a07d7a838a729896a388fad2b31f666f9abea7e82eb6f3381cb52890`.
- Branch: `fix/morpion-exact-receipt-evidence`.
- Isolated ext4 worktree:
  `/home/pompote/oldata/victor/morpion-scratch/receipt-evidence-20261025/worktree`.

The exact [source manifest](receipt_model_evidence/evidence/sources.json) verifies
all 477 installed Chipiron Python files against this checkout and all 289 Anemone
files against both the exact pinned Git tree and actual wheel. Package version `0.2.26` alone is not used
as dependency evidence. The new environment copies the small exact dependency
packages and reads other dependencies from the previous environment without
modifying it. No existing worktree, historical experiment or Anemone source is
changed. Commits remain local; no PR, merge or production activation is performed.
The final [preservation audit](receipt_model_evidence/evidence/protected.json)
matches all 46 pre-existing Chipiron and 13 Anemone worktrees' HEAD, status and
tracked-diff fingerprints; original E2E bytes and installed dependencies also match.

## Root cause and choice

The concrete class is `DurableGenerationPublication`, not the task's shorthand
`GenerationPublicationSession`. Its `model_pin(state, origin, digest)` has one
production caller: exact receipt acknowledgement. Previously its backwards scan
called `validate_capsule` before inspecting each descriptor's `model_pins`.
Consequently an exact M1 retry at G8 validated whole G8, G7, G6, G5 and G4 capsules
before finding M1 in G4. This was in addition to the existing current-head checks.

Option A would remove historical model authentication from acknowledgement.
An authoritative exact receipt can justify not repeating an inference, but that
would change the existing model-integrity guarantee and require a separate trust
contract. Option B preserves that guarantee with a call-site substitution:
`read_committed_descriptor` replaces `validate_capsule` **only in model lookup**.
The existing `artifact_digest` still verifies every actual file of the chosen
model. This is the smaller change to the established contract.

The selection remains deterministic and newest-first. Missing/corrupt selected
models do not fall back to older copies. Descriptor scanning remains; this is not
an O(1) model index or a claim of history-independent recovery. No registry, cache,
ledger, schema adapter or duplicate validation function was added.

## Three distinct contracts

| Operation | Required evidence | Permitted effect |
| --- | --- | --- |
| NEW proposal admission | Actual committed scientific source; current frozen model and runtime model identities; workspace/epoch, producer, cursor, numeric policy and exact proposal; existing Growth ownership/fence | Apply compatible values, then atomically commit checkpoint, science, receipt and accepted cursor |
| EXACT committed acknowledgement | Current full head; authoritative receipt UUID/content hash; original frozen model bytes and authenticated binding; committed cursor digest and exact atomic proposal; unchanged pending publication | Unlink the matching pending entry and fsync its directory |
| Historical model resolution | Authoritative generation records, authenticated descriptor chain scanned to newest matching `(origin, sha256)`, confined physical path, actual bundle digest | Return that existing physical bundle |

Acknowledgement never invokes inference or patch application, advances no cursor,
creates no generation, changes no epoch/model and cannot promote M1 over M3.
Absence of a receipt returns `False`, preserving pending work for the independently
validated NEW-admission path. Stale new M1 proposals still fail against M3.

`read_committed_descriptor` checks the committed descriptor hash, capsule path,
protocol/version/id/generation, parent, scientific producer identity and evaluator
binding when present. It authenticates metadata, **not the scientific payload**.
The actual bundle digest then checks its recursive sorted file inventory, including
weights/configuration/manifests, rejects symlinks and missing/empty bundles, and
compares the result with the receipt's exact model digest. The caller's existing
second bundle digest check is retained.

Logical origin, physical location, model bytes and current evaluator remain
different identities. M1's logical origin is `seed`, but this experiment resolves
its chosen frozen bytes in G4; M2 resolves in G7. Their exact original digests,
sealed descriptor digests and physical paths appear in
[receipt model evidence](receipt_model_evidence/evidence/reachability-after.json).
Identical weights elsewhere do not authorize discarding these bindings.

The trust root remains the existing committed `run_state`, accessed under Growth
ownership. This does not add signatures or claim to detect a coherently forged
authority. Independent descriptor/model corruption, malformed/duplicate receipts,
changed proposal bytes and cursor-evidence conflicts are refused. Existing exact
receipt UUIDs/digests, not generation watermarks, determine deduplication.

## Genuine E2E and lifetime separation

The fresh [40-process experiment](receipt_model_evidence/evidence/e2e-processes.json)
ran the original configuration with actual accepted Dataset/Training results and
two explicit activations: G2 Dataset/Training; G3 M1 receipt; G4 M2 activation;
G5 Dataset/Training; G6 M2 receipt; G7 M3 activation; G8 Growth/pending M3 proposal.
It ends with 277 nodes and 281 DAG edges. No fabricated acceptance record or
historical input was used. The independently restored scientific files, selector,
search RNG and rollout RNG were compared throughout.

Before changing production code, traces reproduced the full G8→G4 dependency;
the new focused regression failed because the old graph retained G3 in its smaller
fixture. Both [baseline traces](receipt_model_evidence/evidence/before-m1.json) and
[failing-before output](receipt_model_evidence/evidence/failing-before.log) survive.

The full original E2E graph was validated before inspecting conditional candidates.
On a `cp -a` disposable copy, the corrected analyzer identifies **only G4** as
conditionally unpinned. Its checkpoint and scientific directory were renamed into
a preserved sibling directory, with every moved file hash checked. Descriptors,
all model bundles and the current head remained in place. The new pending M3
proposal was preserved separately while exact older proposals occupied its slot.

The [lifetime result](receipt_model_evidence/evidence/lifetime.json) establishes:

- Three exact M1 and three exact M2 retries leave authority bytes, accepted cursor,
  epoch, model, complete cold checkpoint payload and scientific state unchanged.
- Historical M2 activation acknowledgement still succeeds without reactivation.
- Every required historical frozen model digest is identical before/after.
- The complete predicted next checkpoint and scientific files are equal before
  and after the old payload becomes unavailable.
- Actual normal G9 Growth matches that untouched oracle, including selector and
  both RNG states; the saved pending M3 bytes are restored afterward.
- Original experiment file hashes remain unchanged. The general full-history
  retention auditor correctly refuses the incomplete disposable copy.

This experiment's opening graph has no terminal/FORCED certificates. Their
preservation is established non-vacuously by the separate real-endgame regressions,
not inferred from the opening run. The inherited representative-parent discrepancy
remains visible; no field is normalized away or omitted to obtain parity.

## Matched resource evidence

Fresh processes used copies of the **same** fresh workspace, same source bytes and
same exact raw proposals. Each row below is five uninstrumented repetitions, with
a separate cProfile/file-hash trace. These are warm local filesystem measurements;
they are not cold-storage or large-N scalability claims.

| Operation | Before median ms | After median ms | Hash calls before→after | Bytes hashed before→after | Historical tree bytes before→after |
| --- | ---: | ---: | --- | --- | --- |
| Exact M1 ACK | 51.741 | 32.412 | 238→82 | 2,952,479→1,079,719 | 1,174,086→0 |
| Exact M2 ACK | 37.959 | 31.246 | 143→79 | 1,919,434→1,080,688 | 298,828→0 |
| Current-head evaluator validation | 6.998 | 4.221 | 29→29 | 455,267→455,267 | 0→0 |

M1's full-capsule trace changes from `[8,8,8,7,6,5,4]` to `[8,8]`;
M2 from `[8,8,8,7]` to `[8,8]`. The two current-head validations were already
present and remain. Descriptor lookup still reads G8…G4 / G8…G7. Model bundle
validator calls change from 11→4 for M1 and 7→4 for M2, counting both existing
validator entry points. The four surviving calls verify current M3 twice and the
required historical model twice. Per-file traces identify every read.

Process peak RSS: M1 259.95→258.70 MiB; M2 260.01→258.71 MiB. Both use the same
31,527-byte authority, whose hash remains identical. Current-head validation's
work is unchanged despite the timing variation. No scientific snapshot/tree is
loaded by the lookup: it retains existing publication metadata and one descriptor
at a time. Existing O(G) publication/receipt metadata and descriptor scans remain.
Hash-byte counters count repeated reads, not unique bytes or physical disk I/O;
the trace includes CPU and `/proc` I/O counters separately.

Across five complete calls including proposal publication, M1 CPU time falls
0.1742→0.0802 seconds and process `rchar` 16,334,046→6,790,946 bytes; M2 CPU time
0.1076→0.0765 seconds and `rchar` 10,929,806→6,664,356 bytes. Physical `read_bytes`
is zero in these cache-warm samples; `write_bytes` is 61,440 in each five-call set.
Historical tree bytes in the table are specifically **bytes hashed**; aggregate
process read counters also include manifests, authority, proposals and other I/O.

For D inspected descriptors and M bytes in the required model, historical lookup
now costs descriptor authentication plus O(M) actual model verification, instead
of reading every payload in those D capsules. It retains O(G) existing publication
metadata plus one descriptor and the model's file inventory, without an O(N)
scientific Python tree. Full current-head validation remains proportional to its
actual checkpoint/scientific/model size. Persistent format and authority size are
unchanged; cumulative receipt/publication metadata remain separate limitations.

## Complete roots and allocated-byte accounting

The [before](receipt_model_evidence/evidence/reachability-before.json) and
[after](receipt_model_evidence/evidence/reachability-after.json) graphs preserve:

| Generation | Remaining independent conditional root |
| --- | --- |
| G1 | Explicitly retained original scientific evidence |
| G2 | M1 receipt source; accepted Dataset/Training and status dependencies |
| G3 | M1 receipt checkpoint evidence, conservatively retained |
| G4 | No remaining full-tree root; selected M1 bundle and authenticated descriptor remain |
| G5 | M2 receipt source; accepted cursor; accepted Dataset/Training/status dependencies |
| G6 | Rollback candidate and M2 receipt checkpoint evidence |
| G7 | Rollback candidate; selected M2 model/activation evidence |
| G8 | Complete current M3 recovery and pending M3 proposal |

Active readers, claims regardless of TTL, explicit retained artifacts, pending
work and unknown ownership still fail closed or retain their complete dependencies.
Receipt model evidence now retains **exact physical bundle files** and scanned
descriptors independently of full historical trees. A per-generation mapping
reuses the existing in-memory report objects; it avoids rescanning the generation
list for every receipt. It is not a persistent index or new authority.
The auditor still verifies the entire historical inventory; its model-holder
maximum scans can take O(RG) for R receipts sharing a long model history. This
research audit is not a history-independent production discovery/cleanup path.

Current publication protocol reclaimability is **zero**. Under a future explicit
retirement contract the audited conditional G4 file allocation is:

| Files | Unique inodes | Logical bytes | Allocated bytes |
| --- | ---: | ---: | ---: |
| Checkpoint | 7 | 32,551 | 53,248 |
| Scientific snapshot | 3 | 213,213 | 221,184 |
| Nonretained capsule metadata | 2 | 5,744 | 12,288 |
| Total conditional | 12 | 251,508 | 286,720 |

Counts use `(st_dev, st_ino)` and `st_blocks*512`; hard-linked model bytes are not
counted twice or mistaken for reclaimable blocks. The original selected physical
model paths remain protected even when another hardlink exists. Unknown external
links suppress the conditional set. Directory overhead, reflinks, open descriptors
and future readers are not represented as reclaimable space. The retention probe
deletes/reclaims no bytes. See the [complete allocation](receipt_model_evidence/evidence/allocation.json).

## Validation and code size

FAST: 148 passed. Focused new receipt, activation-completion and retention tests:
65 passed, including all 22 new cases. PR: **2,651 passed, two CUDA-unavailable
skips**, 935.30 seconds. All **47 affected exhaustive fault cases passed** in
164.78 seconds. The single frozen-executable **FULL run passed 2,798 tests**, with
two CUDA-unavailable skips, **80% coverage**, in 1,517.47 seconds. Its 196 complete
certainty/Dataset/atomic-science matrix cases all passed, including the real
FORCED(35)/ESTIMATE(42) endgame. See [validation](receipt_model_evidence/evidence/validation.json)
and [FULL test identities/results](receipt_model_evidence/evidence/full.xml).
FAST⊆PR⊆FULL and affected-exhaustive⊆FULL selections were checked, with no missing
or duplicate test execution phases.

The focused cases cover M1/M2 after M3, repeated cold restarts, exact cursor
stability, conflicting raw bytes under the same UUID, missing/duplicate/altered
receipts, corrupt/missing selected historical model or descriptor, corrupt current
model/checkpoint/science, new valid M3 admission, stale new M1 refusal, reader and
explicit retention roots, historical activation retry and actual concurrent NEW
activation excluded by the existing Growth lease. Process death (`os._exit(73)`)
is exercised after model validation, before unlink, after unlink, and after directory
fsync; restarting recognizes either the same pending operation or its completed
acknowledgement without authority mutation.

The existing exhaustive matrices additionally cover late M1 producers, proposal
link/fsync/cursor/commit crashes, activation interruption, model identity, recovery
and scientific certainty, including FORCED(35) versus direct ESTIMATE(42), all nine
legacy policies, five enabled v2 policies and four gated v2 exact-family policies.

Ruff and all 750 formatting checks pass. Build, installed distribution checks and
toolchain verification pass. Strict static ratchet: 206 inherited diagnostics,
baseline 209, **zero new**. These are local equivalents of the lean CI workflow,
not a claim that a remote GitHub workflow ran. Final CI uses the built wheel and
the exact verified Anemone package; the safety plugin redirects default test
leaderboard output into synthetic temporary fixtures.

The nearly full secondary filesystem cannot retain another complete copy of all
passing CI fixtures. Exhaustive/FULL use pytest's existing
`tmp_path_retention_policy=failed` for their **new disposable test directories**;
assertions, test selection and application retention behavior are unchanged. All
original experiments, baseline/lifetime copies, failure evidence, test reports and
source fingerprints are preserved. This is test-fixture teardown, not a pipeline
retirement operation.

[Code delta](receipt_model_evidence/evidence/code-delta.json), excluding this report
and generated evidence: production **+8/−3** (net +5, including docstrings), analyzer
**+40/−10**, tests **+412/−5**, research probes **+349/−0**. The only executable
production change is importing/calling the existing descriptor authenticator in
place of full capsule validation. The analyzer's additional output is necessary
to expose the exact physical model/descriptor lifetime; the research probes are
reproducers and counters, not runtime infrastructure.

## Remaining work and reproduction

This removes one demonstrated retention edge. It does not compact receipts,
publication history or descriptors; eliminate repeated current-head/model hashing;
release accepted Training/Dataset source dependencies; provide a deletion fence;
or make rollback executable. G6/G7 are complete rollback **candidates**, not
authorized old research epochs that can overwrite `run_state`.

The smallest next design step is to bind a chosen boundary's model, checkpoint,
science, exact receipts, producer cursor and epoch in the existing authority and
define explicit fork/rollback admission under existing ownership. Current crash
recovery must continue to select the committed head only. Before any retirement,
specify retained metadata/model bindings, pending/worker dependency fencing and
reader lifetime protection; independently validate recovery and hard-link
accounting. This task implements none of that protocol or deletion machinery.

Reproduce with the pinned wheel, `PYTHONPATH=.`, `CHIPIRON_TEST_INSTALLED=1`,
`PYTHONDONTWRITEBYTECODE=1`, short secondary `TMPDIR`, and the source manifest:

```bash
python -m docs.research.e2e_bootstrap.reproduce NEW_SECONDARY_DIRECTORY
# Make independent cp -a copies of its workspace for before/after/lifetime probes.
python -m docs.research.receipt_model_evidence.measure receipt COPY RESULT.json \
  --proposal NEW_SECONDARY_DIRECTORY/evidence/06-check/pending.json
python -m docs.research.receipt_model_evidence.separate \
  DISPOSABLE_COPY NEW_SECONDARY_DIRECTORY LIFETIME.json
make test-fast
make test-pr
make test-full
```

The baseline probe uses the same script with the protected base's `src` on
`PYTHONPATH`; it never edits that checkout. Model M2's exact archived proposal is
`evidence/22-check/pending.json`. Raw subprocess/profile/test/build logs and all
original generated workspaces remain in this task's parent scratch directory.
The final evidence commit is a documentation-only descendant of `fe0fe7a5`;
FULL is not repeated for documentation changes.

For a graphical VS Code diff, open this worktree and run:

```bash
git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt 60184a7c fe0fe7a5 -- \
  src/chipiron/environments/morpion/bootstrap/generation_publication.py \
  src/chipiron/environments/morpion/bootstrap/generation_publication_io.py \
  scripts/report_morpion_rolling_recovery.py
```
