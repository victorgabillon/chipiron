# Lean scientific model activation: contract and blockers

Base: `dc142573a1163178ae8fd5fc9feb90ff8ffee21e` (accepted Training, no activation).
New isolated secondary-ext4 worktree:
`/home/pompote/oldata/victor/morpion-scratch/model-activation-20261024/worktree`.
Branch: `research/morpion-model-activation`.

## Decision

Prefer **A: one explicit complete refresh**. A bounded direct-evaluation traversal,
one deepest-first backup pass, and one selector refresh are much simpler to verify
than stale-value read barriers throughout search. The existing runtime can perform
these numerical operations without building another Python search tree.

**Do not activate yet.** Stock `refresh_with_evaluator(scope="all")` is not a safe
activation primitive: executable witnesses show erased certificates and a stale
ancestor even after a nominal full refresh. The research-only replacement kernel
matches an independent default-max oracle on the tested small graphs, but it is
not a production transition. Unknown proof origins, decoded-state retention,
atomic model authority and the Reevaluation publication race still need focused
work. No production code or storage format is changed here.

## Exact scientific contract

The objective is Morpion's existing single-agent maximum of the final total move
score, not a new remaining-moves or reward-difference objective. Preserve the
configured rules, objective, tensor conversion, score transformation, neural
architecture and evaluation policy. The existing Anemone direct evaluator applies
`(1 / 0.99999999) ** tree_depth * prediction` to nonterminal predictions. Ordinary
Morpion Reevaluation patch rows currently carry the raw neural score instead.
These are demonstrably different numeric policies at depth > 0. This study retains
the Growth policy exactly; neither rounding nor silently adopting patch semantics
is an admissible correction. A transition must declare its policy, and future
patches must be compatible with it before they may modify refreshed values.

| Existing field | Required transition behavior |
| --- | --- |
| Terminal direct Value | Preserve exact score, TERMINAL certainty and outcome; no neural inference |
| Nonterminal direct FORCED Value | Preserve the value and its certificate origin; never clear it for neural inference |
| Nonterminal direct ESTIMATE/missing | Reevaluate under M2, including solved nodes whose **tree** value is exact; stamp actual new direct origin |
| Derived tree estimate | Recompute from current child **effective** Values, after children settle |
| Default closed-subtree FORCED | Re-derive under the same proof rule and require the original certificate/value unchanged |
| External/non-reconstructible tree certificate | Fail before mutation until its proof contract is established; do not downgrade or guess |
| Effective Value/source | Recompute using the unchanged partial/full-opening rule; do not copy an effective exact flag to direct/tree/target |
| Selected raw target | Preserve the declared target-source policy; certainty comes from that selected Value |
| Family-derived training target | Keep selected-value-v2 confidence, five supported policies and four gated exact-family policies |
| PV / child ordering | Recompute from settled child Values using existing score, exactness, tactical and stable-ID tie ordering |
| Selector | Rebuild derived priorities/depth caches; preserve selection-step parity, RNG and configured policy |
| Topology / canonical states | Same node IDs, actual state identities, all incoming/outgoing edges, opening completeness and unopened branches |
| Provenance | Bind old/new bundle digests and accepted Training reference; distinguish the exporting producer from the origin of a preserved certificate |

For the default max backup, a tree value is FORCED only when all legal branches
have been generated and every linked child's effective Value is exact. An exact
terminal child makes its nonterminal parent FORCED, not TERMINAL. With unchanged
topology, unchanged terminal/direct certificates, and children processed first,
induction on depth preserves reconstructible default tree proofs. This argument
does not establish correctness for custom backup policies or unverified external
certificates. The experiment refuses partial tree certificates rather than
inventing proof-transfer rules.

For these Morpion graphs each move increases the played-move count by one, so
reverse depth is a topological order even when a state has several incoming parents.
The refresh does not transfer a certificate between unrelated PV-family states.

Partial nodes compare `(score, is_exact)` and prefer the tree on remaining ties.
Fully opened nodes use an available tree Value even if a direct Value is exact.
This pre-existing rule is characterized, not silently changed. A direct proof is
preserved as a direct proof; it does not automatically certify a chosen estimated
tree target. Contradictory certificates for the same state/objective must fail.
The activation preflight must validate proof provenance; the certainty enum alone
cannot distinguish a genuine external proof from a historically miscertified value.
Characterizing the full-node preference does not authorize downgrading a proven
state: if an input's valid same-state certificate and the existing effective-value
rule cannot be reconciled, activation must refuse it pending a specific proof-rule
decision. Preserving a direct field alone is not a proof of scientific readiness.

An estimate at M1 must either be replaced under M2 or explicitly remain identified
as M1; a complete-refresh commit promises no active model-derived M1 estimate.
Certificates are independent of the active neural model, and keep their original
provenance. Missing and explicit-null Values remain distinct in exported contracts.

## Demonstrated Anemone limitations

Installed dependency: `algorhino-anemone==0.2.26`; exact module hashes are in the
source manifest. The code, not a moving upstream branch, is the investigation source.

1. **Direct proof erased.** A public-API-valid synthetic direct `FORCED(9)` becomes
   `ESTIMATE(42)` after stock full refresh. `NodeDirectEvaluator.evaluate_nodes`
   clears every nonterminal direct Value. This is a proof-contract counterexample,
   not a claim that the initial Morpion board actually has optimum 9.
2. **Partial tree proof erased.** An externally supplied partial-node tree
   `FORCED(35)` is recomputed as ESTIMATE by stock refresh. Its derivation is not
   represented by the default partial max backup. The experimental kernel rejects
   this input before calling `set_evaluator`.
3. **Real partial-state propagation failure.** Grow the actual game for two steps
   with branch limit 6: seven nodes, internal node 2 partially opened. A controlled
   evaluator changes only its direct estimate to raw 100, leaving every other
   prediction exactly unchanged. Full refresh leaves root tree/effective value
   `2.2600000452000004`; the independent reference requires `100.00000100000001`.
   The child's local backup/PV did not change, so `ValuePropagator` fails to mark
   its parent dirty although the child's effective value changed. The backup
   result compares backed-up Values, not the old direct/effective candidate.
   One unconditional deepest-first pass corrects this witness.
4. **Lazy attach has no freshness barrier.** `set_evaluator` increments a runtime
   version and changes the evaluator; all stored Values remain untouched. The
   selector still makes a choice with old per-node versions. No version-sensitive
   backup or selector read barrier proves decisions equivalent to M2 search.
5. **Resume is not a switch.** Calling `set_evaluator` after restoring the same
   model increments the runtime version again while nodes retain their saved
   version. The existing Anemone checkpoint loader already accepts an evaluator;
   the minimal runner integration should supply the committed evaluator there,
   avoiding another switch on every restart.
6. **Inference windows do not bound decoded states.** The checkpoint resolver's
   `_resolved_states` retains decoded states for all visited nodes. The actual
   small checkpoint test demonstrates growth to the whole graph even with window
   size one. Existing compact-state eviction has a bounded LRU, but it is a
   different experimental live resolver, not automatically the restore resolver.
7. **Scientific origin is incomplete for mixed models.** Checkpoints retain
   `direct_evaluation_version`; the current scientific value contract exports
   Values/certainties and a global producer, but not a per-value model identity.
   A lazy mixed-model snapshot cannot be correctly described by changing that
   global producer alone. Full refresh avoids mixed model estimates, but still
   needs explicit certificate versus estimator provenance and a resume binding.

The genuine endgame (move 34, one legal action, terminal score 35) retains direct
ESTIMATE(42), tree/effective FORCED(35), and terminal TERMINAL(35) in the experimental
refresh. The Dataset comparisons exercise all nine legacy policies and all nine v2
names, both target preferences and exact-only filtering. Legacy scientific flaws
remain visible under legacy_v1; the five supported v2 policies protect terminal 35,
and all four exact-family names continue to reject.

The runner also defaults `reevaluation_scope` to `leaves`: its existing
`reevaluate_tree=True` (including the `reevaluate_all` policy boolean) does not
guarantee that internal direct estimates are refreshed. A dedicated regression
retains the unchanged old root direct Value/version after this wrapper call. An
activation integration must explicitly demand the complete scientific pass.

## Why A, and why not B now

A operates on one quiescent live runtime. Iterate its existing depth registry,
filter actual direct certainties, evaluate windows, settle **every** opened parent
in descending depth, rebuild selection caches once, validate against the committed
source, and serialize one candidate checkpoint/snapshot. No second graph, global
Python value dictionary, evaluator-version database or persistent cache is needed.
The independent O(N) dictionary in `probe.reference` is a small-fixture oracle only;
it is never used by the measured refresh path.

B would require freshness rules at every direct/effective/backup/selector read,
ancestor invalidation for each lazy update, certificate ownership and checkpoint
serialization of mixed provenance. A mere version counter is not that contract.
Stale estimates are not certified mathematical bounds; arbitrary neural M2 can
reverse two M1 candidate scores. No existing theorem or algorithm makes a decision
using those old rankings equivalent to the declared M2 policy. B is therefore not
implemented or benchmarked as a scientifically admissible shortcut.

## Minimal durable transition proposal (not implemented)

Use one operation with an explicit requested accepted Training descriptor, expected
current head and expected previous model. Do not let a Training completion trigger
it. Keep workspace and research epoch stable; a model transition is not an epoch
fork and must not invalidate existing accepted Dataset/Training identities.

1. Acquire the existing Growth exclusive ownership for the entire transition.
   Capture/validate current run-state, current capsule/checkpoint/source hashes,
   exact receipts and accepted cursor, config/evaluation policy, and the accepted
   Training descriptor/selected candidate/full bundle digest. Retain current-source
   and accepted-input pins. A changed expected head/model/config is a conflict.
2. Establish the pending-proposal boundary described below. Restore the current
   committed graph exactly once using its committed evaluator and configuration.
   Preflight supported objectives, proof origins and bounded decode retention.
3. Load verified M2 bytes using the existing model loader in eval mode, preserving
   unrelated Torch RNG during reconstruction. Change the evaluator once. Run the
   complete refresh; no Growth selection or publication observes intermediate
   values. Any exception invalidates this live attempt: recover M1 from disk.
4. Verify every preserved certificate and all expected direct versions using the
   existing indexed committed scientific source in selected windows. Recompute
   targets/source certainty without changing policy. Preserve current RNG, selector
   step count, graph and branch-generation state; reject mismatches.
5. Reuse the capsule's private staging, model freezing, checkpoint serializer,
   closed writer, hash validation, fsync and no-replace publication. The capsule
   descriptor binds `{previous_head, previous_model, accepted_training,
   selected_candidate, new_model_digest, evaluation_policy, configuration_digest}`
   and the complete new checkpoint/science. This is immutable evidence attached to
   the existing capsule, not a new journal or authoritative registry.
6. One durable run-state replacement makes the entire new head/model current.
   Active-model/status/discovery files become reconstructible projections of that
   same head. Missing/corrupt committed model bytes must fail closed, not fall back
   to a default evaluator. Old M1 dependencies remain immutable and valid until
   this commit; rollback and active-reader retention continue protecting them.

Current publication v3 explicitly rejects changed model pins against its initial
capsule. The closed writer also requires identical producer identity for copying
state records, and no source means generation 1. Therefore calling it with a new
producer or disguising a transition as another genesis is **not** integration.
A small versioned current publication contract must explicitly allow a verified
same-workspace transition with the bound previous head. Immutable state-byte reuse
may ignore evaluator identity only after validating stable codec/rules, exact source
hashes, state IDs and canonical state identity. It must not weaken validation or
replace actual byte verification with a parent hash. Initially a complete state
rewrite is simpler; measured component costs below inform that choice.

No permanent pair of old/new production mechanisms is required. After a separately
authorized validated migration, the current protocol can replace obsolete internal
branches. This task neither migrates nor deletes historical sources.

## Pending Reevaluation: smallest deterministic policy

Prefer **drain-before-switch**, not a new rejection ledger. While holding Growth
ownership, an admissible visible M1 proposal is completed by the existing M1
checkpoint/receipt/cursor transaction first. This creates a valid M1 boundary; if
activation later fails, that latest committed M1 boundary remains authoritative.
An invalid/conflicting visible proposal blocks activation without deletion.

Checking that the mailbox is empty is insufficient. The producer currently checks
`current_model_identity` and subsequently does create-only publication outside
Growth ownership. An old live producer can publish between the check and rename,
including after activation has committed. Its new M1 patch is correctly rejected
by the consumer, but can occupy the singleton mailbox and strand progress.

The minimal required change is to put the producer's final identity/cursor/source
recheck **and** existing no-replace publication inside the same short Growth
ownership boundary. Computation stays outside it. During activation that gate is
closed; afterward old computations fail the captured expected-model check and must
restart from authority. TTL or a missing claim is not proof of producer termination.
All compatible producer code must use this gate before activation is enabled.

Keep the last accepted cursor unchanged at activation. The existing ledger requires
it to match the latest exact receipt; resetting it to null while receipts exist
is invalid. Existing `cursor_matches_active_model` already starts traversal at the
beginning and pass count zero for a different model while the next proposal still
binds the old accepted-cursor digest. The first M2 patch atomically replaces that
cursor alongside its receipt/checkpoint. Do not invent a fake receipt for activation.
Retain exact old UUID/content digests and next-cursor evidence; duplicate M1 retries
may be acknowledged from their exact receipt, never reapplied or allowed to rewind.
Workspace/epoch/producer identities need not rotate merely because a model changed.

This proposal needs an executable publish-versus-activation race test. It is not
claimed validated just because existing atomic M1 patch tests pass.

## Crash/restart matrix required before activation

| Kill boundary | Required observation on restart |
| --- | --- |
| Before M2 load; during inference; during backup/selector | Complete latest committed M1 state; discard private live attempt |
| Checkpoint writing; scientific writing; validation | M1; incomplete/unaccepted outputs cannot be discovered as current |
| After immutable capsule rename, before run-state replacement | M1; visible M2 capsule remains an orphan |
| After replacement, including failure before fsync returns | Validate the complete observed authority; a power loss may expose old or new durable head, never mixed dependencies |
| Immediately after durable commit; discovery/active-model projection repair | Complete M2; repair all projections from the head, never load a stale singleton as authority |
| Producer restart / old computation resumes | Exact committed cursor/receipts retained; new M2 traversal starts under explicit model identity; old unaccepted computation cannot publish |

No new activation transaction was built, so these M2 commit death tests are a hard
prerequisite, **not a completed test claim**. Existing M1 capsule/atomic patch crash
regressions are rerun separately. Process-kill tests do not emulate loss of a drive's
volatile cache; POSIX durability still relies on the existing fsync contract.

## Measurements

All three serial fresh-process runs completed successfully. The requested sizes
are lower bounds because real expansion batches add multiple children. Raw phase
records, CPU/I/O counters, fingerprints and profiler attribution are in
[`model_activation/evidence/`](model_activation/evidence/). These are component
measurements, **not timings of a committed M2 activation**.

| Requested | Nodes / links | Multi-parent nodes | Maximum depth | Model evaluations | Peak RSS, MiB |
| --- | --- | --- | --- | --- | --- |
| 1k | 1003 / 1114 | 106 | 20 | 1001 | 310.41 |
| 5k | 5013 / 5931 | 759 | 62 | 5005 | 410.38 |
| 10k | 10018 / 11846 | 1391 | 62 | 10009 | 543.82 |

| Phase, seconds | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| Verify source checkpoint | 0.0012 | 0.0014 | 0.0014 |
| Restore current checkpoint | 0.248 | 0.947 | 1.842 |
| Verify accepted Training / load selected model | 0.015 | 0.024 | 0.025 |
| Direct evaluation | 23.378 | 142.476 | 281.516 |
| Complete backup pass | 0.308 | 1.593 | 3.248 |
| Selector refresh | 0.019 | 0.093 | 0.192 |
| Save and fsync refreshed checkpoint | 0.320 | 1.143 | 1.999 |
| Write, validate and fsync private closed snapshot | 3.621 | 44.799 | 106.449 |
| Independent cold indexed validation | 2.359 | 29.985 | 71.237 |
| Sum of measured refresh phases | 30.270 | 221.061 | 466.510 |
| CPU user + system across those phases | 29.642 | 218.933 | 462.552 |
| Fresh-process reload of refreshed checkpoint | 0.237 | 0.890 | 1.736 |

The full 5k/10k refresh processes take 224.65/470.31 s including interpreter startup,
final fingerprints and reporting. `/usr/bin/time -v` confirms their process peak RSS
of 420228/556868 KiB. The 1k refresh uses the existing `ru_maxrss` instrumentation;
no external refresh time log was captured for that run. Fresh restart verifies all
recorded Values/certainties, effective sources, direct versions, PV, edge sets and
search RNG; it is not a production restore with newly committed model authority.

| Allocated bytes, unique inodes | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| Refreshed checkpoint | 131072 | 491520 | 950272 |
| Closed scientific output | 1044480 | 6094848 | 12554240 |
| Whole scratch work directory after validation | 2666496 | 14819328 | 30326784 |

The last row includes the retained source checkpoint, refreshed checkpoint,
scientific output and validation index. It is final allocation, not an observed
instantaneous temporary-disk high-water mark. At 10k, snapshot creation and its
independent validation report 50040832 and 37466112 physical write bytes respectively;
their temporary SQLite files include later-cancelled writes. No output was accepted
into a run-state, and no committed artifact was removed. The selected tiny model's
complete bundle digest is
`d88d21ba545da5bf868bccee46cbba4a606cf23391fa832d834ea6e9754910da`.

At 10k, neural adapter evaluation takes 234.08 s cumulatively, of which input tensor
and feature preparation takes 213.07 s (75.7% of the entire direct phase). Candidate
segment generation accounts for 188.29 s cumulatively within that work. These are
nested profiler totals, not additive costs. The small network arithmetic is not
the dominant bottleneck. Decoded states retained by the restore resolver explain
the direct phase's RSS rise from 336.32 to 511.13 MiB. A larger network or different
state history can change this balance; no gen38 estimate is derived.

| State-codec calls | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| Direct phase: anchors / deltas | 268 / 735 | 1372 / 3641 | 2772 / 7246 |
| Snapshot: canonical decodes / dumps | 1003 / 2006 | 5013 / 10026 | 10018 / 20036 |
| Additional indexed validation: decodes / dumps | 1003 / 1003 | 5013 / 5013 | 10018 / 10018 |

For each size, the instrumented source verification hashes seven checkpoint files;
accepted-Training verification hashes eleven files; snapshot creation invokes
`hashlib.file_digest` 19 times plus the scientific reader's `_sha` ten times;
independent indexed validation invokes them 13 and four times. These are **64
actual per-file hash calls across those phases**, not 64 independent model-bundle
passes. Wrapper calls are excluded to avoid double-counting. Recording the selected
bundle and final checkpoint fingerprints additionally hashes four and seven files
outside those phase profiles. Raw profiles distinguish in-memory identity hashing
from source-file reads. Source bytes are actually checked; no parent digest is
substituted for verification. Hash work is small in these fixtures; canonical
encoding/verification is substantial. Reusing a validated writer index may avoid
some repetition later, but validation is not weakened here.

The measurements use newly generated legal search graphs, one live runtime per
process, separate serial prepare/refresh/restart processes, fixed inference window
64, and an actual tiny CPU model selected by a freshly accepted Training result.
The accepted model belongs to the benchmark's new Training fixture; the graph
kernel is offline, not a same-workspace production activation transaction.
The source evaluator for these size probes is the existing heuristic. Small tests
also exercise repeated neural 42→7→42 transitions on identical graph populations.
No historical tree, gen38 or distributed worker is used.

`StageMeasurement`, cProfile, existing state-codec/fsync counters and unique-inode
disk accounting are reused. Timings are profiled CPU measurements on OS-cache-warm
files, not predictions for a particular production graph or GPU. The current neural
adapter's `evaluate_batch_items` loops over scalar evaluations; window 64 bounds
query storage, it does not claim 64-state vectorized GPU inference. CUDA is absent.
Both plain canonical-reference decodes and incremental anchor/delta decodes are
reported; claiming zero state decoding from only `load_state_ref` would be false.

Memory: the traversal holds at most B query nodes, one parent's branch set/order,
and transient PV/feature data. The bound depends on those states' feature sizes,
maximum outgoing degree and depth, not just B. This is in addition to the existing
resident graph, model, selector and resolver; stored PV sequences can themselves
occupy O(sum of PV lengths). The existing checkpoint serializer still builds
O(N+E) typed records plus their encoded state/PV payload. The decoded-state cache
can retain O(sum of state sizes); complete memory independence from N is not claimed.
A production transition needs a validated bounded lifetime for decoded states using
existing resolver/eviction primitives, without a second state catalog. Restored graph
RAM and proof/receipt metadata are accounted separately from the query window.

Time: N nodes, E links, depth D, model/feature cost F(s), checkpoint/science bytes S,
model bytes M, history G and receipts P. Direct work is sum F(s) over eligible nodes;
backup traverses E and existing branch-ordering sorts add sum d(v) log d(v); selector
rebuild scans N/E and maintains its existing heaps. State encoding/validation depends
on actual replay length, not merely N. Hash and serialization work is O(S+M), plus
inherited O(G+P) authority work. No history-independent commit claim is made.

## Executable validation and its limits

The new research suite passes **62 tests**: 17 scientific/continuation witnesses and
45 Dataset cases. Of the latter, 29 full-reference/indexed file pairs contain 53
paired rows with identical entire JSONL bytes, including metadata; 16 cases verify
that the four unsupported v2 exact-family policies reject both target preferences
and both exact-filter settings. All nine legacy policies are compared unchanged;
the five supported v2 policies retain selected-source confidence. The tests preserve
the terminal target 35 and leave the direct target 42 estimated.

Small-tree checks cover actual partially expanded nodes, seven DAG transpositions,
exact/estimated ties, terminal and nonterminal certificates, unknown partial-proof
refusal, windows 1/7/64 and repeated neural 42→7→42 transitions. A separate oracle
implements the pinned default max/effective/PV rules without using Anemone's backup,
ordering or effective-value helpers. Three subsequent Growth selections match after
checkpoint restore, both with and without rollout; search and rollout RNG states
are preserved. The test attaches the identical restored evaluator without a version
increment to demonstrate the needed resume interface; this private test assignment
does not repair production restart.

Existing-protocol regressions: **325 passed in 361.12 s** across value certainty,
checkpoint parity, atomic Reevaluation, durable closed snapshots and their existing
scientific/crash suites. A first invocation failed at fixture setup because its
chosen temporary-directory parent was not writable; it was rerun in a writable
short path on the same secondary disk. No assertions or expected scientific results
were weakened. Ruff checks and formatting pass for all four research Python files.
The complete CI from the validated base is not rerun for this docs-only investigation;
all 476 production source hashes still match that base exactly. The new M2 commit
death/race tests remain unimplemented for the reasons above.

The large-graph refresh is a resource and checkpoint-fingerprint probe, not an O(N)
independent reference evaluation. It never builds the full scientific reference
snapshot. Stock-defect tests deliberately assert the observed wrong behavior;
their passing result does not mean Anemone has been fixed. Earlier exploratory
harness failures (a mismatched restore branch limit, and ordered-link identity
mistakenly included in the edge-set comparison) were diagnosed and corrected;
separate scratch evidence is retained. The ordered-link difference remains explicit.

## Minimal sequence and code removal opportunities

1. In Anemone, independently fix the demonstrated effective-value propagation bug
   and certificate-erasing refresh behavior. Add the seven-node witness and proof
   tests upstream. Keep the default max proof semantics unchanged.
2. Validate one full-refresh operation using bounded windows, deepest-first backup,
   selector/RNG continuation, explicit evaluator resume and bounded decode retention.
   Adopt the direct-value scale/provenance contract explicitly; retain the existing
   neural ESTIMATE correction and selected-target confidence gates.
3. Add the short final producer publication gate. Test live stale producers and
   retain the accepted cursor/receipt invariant without resetting or adding a ledger.
4. Add only the model-transition extension to the existing capsule/run-state protocol,
   including the authoritative model projection and every death/race test above.
   Revalidate actual accepted Training identity, source integrity and recovery under
   corrupted/missing artifacts before an explicitly authorized production trial.

After verified adoption, remove Training's legacy active-model write branch in
`pipeline/stages.py`, independent active-model fallback authority in
`pipeline/active_model.py`, legacy status/cursor-driven Training recovery, and the
ambiguous attach-without-refresh routes used as model switches. Keep a plain attach
only for loading an already committed evaluator at restart. The full-refresh entry
should replace, not duplicate, existing model-replacement control flow. Do not remove
or redefine generic external-proof APIs before their callers and proof contracts
are audited.

Keep this investigation local on the exact accepted-Training base above. The first
fixes belong in the pinned Anemone dependency, followed by the focused Chipiron
changes in this order. A future integration PR needs an upstream base equivalent
to the existing consolidated local stack; copying all unmerged predecessor code
into a new activation PR would obscure these small changes.

The known representative-parent checkpoint discrepancy remains visible: previous
validated fixtures had 7/70/161 differing raw node-shell representatives at
1003/5013/10018 nodes. This work does not fix or normalize those fields. Ordered child
iteration also changes on restore (62 nodes in the new 1003-node diagnostic), while
edge sets and all tested values/PV are equal; ordered fingerprints are retained
separately from graph-content fingerprints. Exact selector/RNG continuation is tested
on the small graph, not assumed from a normalized topology hash.

The activation gates are therefore concrete: correct effective-value propagation;
preserve and validate same-state certificates; choose and record one compatible
Growth/Reevaluation numeric policy; bound decoded-state lifetime; attach the
committed evaluator without incrementing its version on resume; close the producer
publication race; and commit the accepted-model binding with checkpoint/science in
one authority update. Representative-parent and ordered-link differences require
either an independently established continuation-equivalence contract or a focused
checkpoint correction before a broader exact-continuation claim. Unknown custom
backup/proof policies must fail closed. The complete M2 crash/concurrency matrix is
required after these prerequisites; no passing test here bypasses those gates.

The source manifest pins all 476 production Python files and 344 Anemone/Atomheart/
Valanga Python modules. Critical entry points for the proposed small changes are
`TreeExploration.refresh_with_evaluator`, `NodeDirectEvaluator.evaluate_nodes`,
`ValuePropagator._is_parent_relevant_after_local_value_change`,
`CheckpointStateResolver.resolve`, the runner's
`_set_runtime_evaluator_from_bundle`, `GenerationPublication`, and the final
check/publication sequence in `run_morpion_reevaluation_worker_once`. Existing
`current_model_identity` currently reads the active-model projection: it must read
the committed head's model identity before that projection can safely be repaired
after an M2 commit. The report is based on pinned local code, not moving PR heads.

Production delta: **zero lines**. All new code is executable research evidence under
`docs/research/model_activation/`. No model activation, epoch fork, production route,
new journal, persistent cache, migration, historical deletion, PR or merge occurs.
The four research Python files total 984 lines (independent oracle, two test modules
and resource driver); their purpose and hashes are recorded in the evidence. All
38 pre-existing worktree registrations/heads, statuses and tracked diffs are
unchanged, including the dirty research checkout and missing registered worktrees.
