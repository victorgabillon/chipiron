# Atomic model activation: checkpoint prerequisite remains open

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

## Decision and scope

**No durable M1 → M2 activation was implemented.** This task stops at the user's
explicit checkpoint prerequisite. The pinned dependency does not preserve the
literal checkpoint/runtime ordering contract, and this investigation does not
establish a sufficient general continuation-equivalence proof to waive it.

This is **not a demonstrated wrong target or divergent search decision**. New
1k probes preserve scientific Values, tested decisions and RNG through 30 further
steps, including after the corrected complete neural refresh and with rollouts.
Both checkpoint formats preserve complete closed scientific files in the new
round-trip comparison. These finite observations do not prove that every admitted
configuration, future expansion, refresh, patch and restart is order-independent.
No discrepancy was normalized away in the literal round-trip test.

The unimplemented transaction must not be described as validated, crash-safe or
ready for activation. The existing fixed-model publication restriction remains.
No production source, normal test, configuration, dependency or CI command changes.

## Exact sources and isolation

- Chipiron CI implementation: `b67bd94572a1398bd82cb6187e70c0e20240a06c`.
- New branch base: `f1a4a6e61bcfd5925e5151c7caa1b7e2daced72c`, its direct
  documentation-only descendant. Its parent is exactly `b67bd945`.
- Their production ancestry includes `f7230d13`, `ecccef06` (numeric contract),
  `1b83dea4` (committed evaluator restore), `7e29c9f5` (producer fence),
  `60fdf85b`, and accepted Training head `dc142573`.
- Anemone implementation: `c015194533857a52d07bafdd603b762314692cfb`; its registered
  worktree is at documentation-only descendant `28d4e77f`. Neither is modified.
- Anemone wheel SHA-256:
  `16aaa5b3a3ac339b8aef34ef75a93134136ed07010e9c2da0816d412dd109c47`.
- Installed Chipiron wheel SHA-256:
  `0fd52731093f27ddc7266f7133bda3d7c8859f8fd8688b1b5cb8f7f002388a68`.

All 476 installed Chipiron and 289 Anemone Python modules match these Git sources
byte-for-byte. Version `0.2.26` alone is not a reproducible dependency identity.
The independent copied environment and complete hashes are recorded in
[`atomic_model_activation/evidence/`](https://github.com/victorgabillon/chipiron/tree/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_model_activation/evidence).

New secondary-ext4 worktree:
`/home/pompote/oldata/victor/morpion-scratch/atomic-model-activation-20261024/worktree`.
Branch: `research/morpion-atomic-activation-preflight`.
Research diagnostic commit: `c605d9d22823c28fecbfc356fc077a928f744076`.
The final report/evidence commit changes no executable diagnostics. All passing
production/scientific tests use unchanged `f1a4a6e6` production and test sources;
the final explicit checkpoint witness is rerun on committed `c605d9d2`.
All fixtures, temporary outputs and logs use that scratch root or the short sibling
`/home/pompote/oldata/victor/am24`. No historical source was used as a fixture.

## New checkpoint evidence

The explicit research-only test
[`test_checkpoint_gate.py`](atomic_model_activation/test_checkpoint_gate.py)
**fails for both `sharded` and `json-zst`**. It is outside normal pytest selection
and is intentionally left failing as an unresolved prerequisite witness, without
xfail, rounding, field removal or a weakened expectation. It tests the literal
representation route; a separately established equivalence contract could instead
justify changing that requirement. Its failure alone does not prove semantic harm.

Generate the legal default graph with seed 4, branch limit 100000 and at least
1000 nodes. The resulting graph has 1003 nodes. In both formats:

- Seven checkpoint node records change `parent_node_id` and/or
  `branch_from_parent`; all other fields in those records remain equal.
- 36 nodes change incoming-parent iteration; 62 change outgoing-child iteration;
  93 nodes change one or both. Counts overlap.
- Full evaluator version, selector state, search RNG, rollout RNG and latest
  expansion payload compare exactly.
- All three closed scientific files compare byte-for-byte. This includes canonical
  references, creation generation, full Values/certainties, topology and provenance.

Concrete node 82: representative parent **66 → 50**, branch
`(3, -5, -1, 2) → (0, 1, 1, 4)`. Both edges exist in the original graph.
This is not missing-parent corruption. The complete per-node differences, actual
unsorted runtime lists and file SHA-256 values are retained in the evidence.

Source attribution in pinned Anemone:

1. `checkpoints/build_node_payloads.py::_serialize_linked_children` sorts outgoing
   edges by serialized branch representation and child ID; original insertion
   order is absent from the saved list.
2. `checkpoints/node_restore.py::_link_nodes` inserts children in saved order and
   incoming parents in node-payload traversal order. It does not restore the live
   incoming-parent order from a stored ordered list.
3. `checkpoints/build_state_payloads.py::_representative_parent_link` chooses the
   first live incoming edge, but prefers a reusable delta's codec parent on a
   restored handle. Graph representative metadata and codec parent selection can
   consequently differ between two writes of the same scientific graph.
4. `AlgorithmNodeCheckpointPayload` explicitly describes representative fields as
   topology/debug metadata. They must not be mislabeled as a proof of lost values
   or node-creation generation. The scientific creation-generation field is separate.

The default opening helper regenerates legal branches from state and tests opened
membership; it does not simply consume the changed child order. Linoo candidate
heap keys include stable node IDs; saved selector state is preserved. These explain
why the observations can pass despite order loss, but are not a complete audit of
all consumers or a general induction proof. Sorted graph-content fingerprints are
reported separately from unsorted-order fingerprints.

## Fresh continuation probes and resource limits

The retained `probe.py` generates fresh legal states and invokes the public
`refresh_with_evaluator(scope="all", batch_size=64)` where requested. Its neural
model is the existing deterministic test evaluator with bias 42. This is a
component probe, not an accepted Training model or an activation transaction.

| Probe | Starting nodes | Further steps | Final nodes | Wall seconds | Process peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same evaluator, sharded restore | 1003 | 30 | 1474 | 8.18 | 312.92 |
| Complete neural refresh on each counterpart | 1003 | 30 | 1416 | 34.61 | 319.84 |
| Rollout enabled, one extra rollout step | 1028 | 30 | 1762 | 46.70 | 333.51 |

Every tested step has equal Value/PV/topology-content fingerprints, selector reports
and RNG states. Unsorted ordered-link hashes remain unequal. No 5k extension is
needed to reproduce the unresolved representation prerequisite.

These are serial fresh-process, **two-counterpart test-oracle** measurements. Their
RSS includes both runtimes and O(N) diagnostic dictionaries. They are not a bound
on a future one-runtime production activation, nor isolated timings of inference,
backup, publication or restart. No inference/feature optimization is attempted.
The existing 256-state resolver remains unchanged. Full activation phase timings,
temporary-disk peak and M1/M2 transaction throughput are **not measured**, because
that transaction was not built. Source checkpoint fingerprints and GNU time logs
for the refreshed/rollout probes are retained. The first probe records `ru_maxrss`
and wall time internally, without an external GNU time log.

The measurement source before import/annotation/format-only cleanup is retained
as `evidence/probe-measured.py.txt`. There is no algorithm change between that
source and the checked-in probe. Measurements are not attributed to a later commit
that did not yet exist.

## Minimal activation preconditions, after resolving the gate

Use one explicit request and the existing Growth ownership lock. The request is
not a persistent journal or a new authority. It must name:

| Binding | Evidence to validate under ownership |
| --- | --- |
| Workspace, epoch | Current atomic Reevaluation authority identities |
| Expected head | Generation, capsule ID/descriptor digest, checkpoint and scientific identity |
| Expected active M1 | Explicit active binding and complete sealed bundle digest |
| Accepted M2 | Existing publication entry's `accepted_training` attempt/descriptor reference |
| Candidate | Exact accepted selected candidate, bundle files and full inventory digest |
| Scientific policy | Frozen config digest, objective, feature/schema/model arguments, score transformation |
| Producer continuation | Complete accepted cursor digest and digest of the complete existing exact receipt ledger |

Validate the selected Training result with the existing Training validator; a
visible directory, `done` status or mutable active-model projection is insufficient.
Receipt identities remain arbitrary exact UUID/content-digest pairs, never a
generation watermark. The request need not copy the ledger into another artifact.
Stale requests must fail before a state mutation. The minimal initial policy should
admit only the demonstrably supported default max proof/selector configuration,
selected-value-v2 Dataset contract and Growth numeric schema 2. Saved-value
Reevaluation fallback remains inadmissible because its node-level certainty can
certify the wrong selected scalar. This restriction is not implemented here.

Preflight certificates using actual states and the existing proof rule. An unknown
direct FORCED certificate must fail; the enum and exporter identity do not prove
its origin. For terminals, independently verify the terminal state and score;
`is_game_over` alone does not certify an arbitrary attached scalar. Reconstructible
closed-subtree proofs retain their exact Value and objective. No invented proof
transfer or family policy is needed. Direct ESTIMATE(42) remains estimated even
when the same state's effective/tree Value is FORCED(35).

## Proposed authority state machine — not implemented or crash-validated

1. **M1 committed.** Acquire Growth ownership, verify all request bindings, pin
   dependencies, validate source/model/configuration and admissible proof origins.
2. **Drain M1.** If a pending admissible M1 proposal exists, finish it through the
   existing patch/checkpoint/cursor transaction. Refuse corrupted or conflicting
   proposals without deletion or fabricated acknowledgement. An already receipted
   exact retry is acknowledged without replay or cursor rewind.
3. **Post-drain M1 boundary.** Keep the resulting exact cursor/receipts. Record the
   requested pre-drain head and actual predecessor in eventual transition evidence.
   A crash here leaves a valid newer M1 head; retrying the old expected-head request
   is stale and requires a new explicit request. No automatic activation retry.
4. **Private refresh.** Restore one verified runtime with M1 supplied directly to
   the loader; load M2 under preserved unrelated Torch RNG. Call the corrected
   complete refresh once in bounded windows. No worker observes the intermediate
   runtime. Failed partial refresh is discarded and M1 is restored.
5. **Private candidate.** Reuse checkpoint serialization, closed writer, model/config
   freezing, inventory validation, fsync and no-replace capsule publication.
6. **Single authority replacement.** Existing durable run-state replacement commits
   the new head and explicit active-evaluator binding together. Before it, M1 is
   authority, even if an M2 capsule is visible. After it, only the complete M2
   checkpoint/source/model may be restored.
7. **Repair projections.** Discovery and active-model projection are rebuilt from
   authority. Recovery supplies M2 directly to the checkpoint loader and does not
   call `set_evaluator` or increment evaluator version again.

The existing producer fence holds Growth ownership for final identity/cursor/source
recheck plus no-replace publication. Computation stays outside. Keep that fence.
The cursor is not reset at activation: the first M2 traversal starts according to
the existing model-mismatch rule, while its proposal binds the retained accepted
M1 cursor digest; acceptance replaces cursor and exact receipt atomically.

One required refinement: `current_model_identity` still reads the mutable active
projection, whereas committed Growth restore already reads the capsule. An
activation-capable version must resolve both from the same explicit active binding
in the existing descriptor/authority. Current zero/one-pin inference cannot choose
M2 if the capsule also retains M1 receipt dependencies. Never pick the first or
last model in a list.

The current publisher rejects changed model pins against genesis. A narrowly
versioned transition must replace that restriction with verified predecessor and
accepted-result bindings. The closed writer currently requires unchanged producer
identity to reuse old state records. Changing evaluator identity must preserve
canonical states and creation generations under an explicit transition contract;
pretending to create generation 1 would lose provenance. No bypass is added here.

## Required failure matrix and minimal next changes

The new M1/M2 crash matrix is **not executed**: there is no transaction to kill.
It remains required at load, inference, backup, selector refresh, checkpoint write,
scientific write, validation/fsync, pre-authority replacement, post-replacement,
projection repair and producer restart. Before commit, recover latest committed
M1 (possibly the drained boundary); after commit, recover complete M2. A power
loss around directory fsync requires validating the complete observed authority;
process death alone does not emulate device-cache loss.

Also require competing requests, late M1 producer, pending proposal drain, duplicate
receipt, corrupted model/checkpoint/science, unsupported proof origin, repeated
restart and M1 → M2 → M3. Existing M1 kill/fence tests are prerequisites, not proof
of that unimplemented transition.

Recommended review sequence:

1. Resolve the pinned Anemone checkpoint prerequisite separately. Either establish
   an explicit admitted configuration and an order-independence induction covering
   all affected consumers, or preserve the required order/representative metadata
   in the existing checkpoint mechanism. Keep codec parent distinct from graph
   representative. A format change must be explicit; do not add a second parser,
   cache or caller-side reconstruction of a second graph. Test anchors, deltas,
   reordered/multiple DAG parents, typed/sharded/repeated round trips, partial/full
   opening, rollouts and long uninterrupted-versus-restored sequences.
2. Integrate that exact dependency in a small Chipiron change and rerun the independent
   seven-node oracle, real endgame, all five v2 policies/four gates, numeric policy,
   canonical source/checkpoint and selector/RNG checks. Do not bless mere enum labels.
3. Add the activation transition to existing publication and authority helpers,
   with explicit active-model binding and drain-before-switch. Run the complete
   new crash/concurrency matrix and phased one-runtime resource measurements.
4. FAST during development, all affected exhaustive matrices plus PR, then one
   FULL on the stable implementation. No publication or production activation follows
   merely from passing these checks.

No dependency change or speculative checkpoint fix is implemented in this task.
Review the current local delta only against `f1a4a6e6`; the upstream consolidated
stack is not established as a clean merged base, so no duplicate-code PR is opened.

## Validation, code size and review

Fresh FAST: **145 passed**, 83.69 seconds. Focused independent scientific suite:
**101 passed**, 68.29 seconds. Complete committed-evaluator, numeric-policy and
producer-fence files: **57 passed**, 150.97 seconds, including their exhaustive
process-death parameters. These counts overlap FAST and are not summed as unique
coverage. Scientific/component results and exact commands are recorded in
`evidence/validation.json`. The explicit literal checkpoint
witness fails twice as described above; it must not be counted as passing CI.

PR and FULL are **not rerun for this stopped investigation**. There is no stable
activation implementation to submit to those final gates. The inherited report's
2707-pass FULL belongs to `b67bd945`; it is not a new result or activation evidence.
Ruff lint/format, toolchain coherence, source/wheel build and rebuilt installed-wheel
smoke pass. Static checks have **0 new diagnostics**, 206 current against the
unchanged 209 baseline. Fresh quality/installed-source checks are recorded separately.
The new files are research diagnostics and evidence only: **269 research Python
lines added; 0 production lines added / 0 removed**, and no normal tests changed.
All 42 pre-existing Chipiron and 12 Anemone worktree registrations, HEADs, statuses
and tracked-diff hashes are unchanged. Installed production-source hashes also
remain identical after validation.

Future deletion opportunities remain the unmanaged restore-time model replacement
route, legacy Training completion/activation branch, stale active-model fallback
authority and duplicate status/cursor recovery paths. Their removal requires
verified current-protocol adoption; this task performs no migration or deletion.

Open the isolated worktree in VS Code:

```bash
code --new-window /home/pompote/oldata/victor/morpion-scratch/atomic-model-activation-20261024/worktree
```

For exact graphical per-file diffs from the specified base, from that worktree:

```bash
git -c diff.tool=activation-vscode \
    -c 'difftool.activation-vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
    difftool --no-prompt f1a4a6e61bcfd5925e5151c7caa1b7e2daced72c HEAD -- docs/research
```

The comparison is only the local research delta. All existing worktrees, commits
and evidence remain intact. No gen38, historical modification, distributed worker,
production activation, merge or release was performed.
