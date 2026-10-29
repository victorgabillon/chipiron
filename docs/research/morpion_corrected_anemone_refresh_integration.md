# Corrected Anemone refresh: Morpion integration and remaining activation gates

## Result and scope

The corrected public Anemone refresh is usable directly by the existing Morpion
runtime. The integration tests request `scope="all"` and explicit bounded windows;
they do not execute a second Chipiron backup/refresh algorithm. An independent
Morpion oracle verifies Values, certainty, effective source, PV and decision order.
The actual seven-node propagation witness is corrected, and the real endgame keeps
direct ESTIMATE(42), tree/effective FORCED(35), terminal TERMINAL(35).

One small production correction closes the final Reevaluation check/publication
race under the **existing Growth flock**. It adds no lock, authority, registry,
cache, receipt format or scientific format. The full activation transaction is
still absent. Production model switching is not enabled, and no automatic Training
completion-to-activation route was introduced.

The remaining work is precise: bound the existing restore resolver; share the
numeric scoring operation; supply the committed evaluator at restore without a
version increment; establish genuine external-proof provenance; and bind the
accepted Training model/checkpoint/science in the existing durable authority.

## Exact dependencies and ancestry

| Input | Exact identity |
| --- | --- |
| Chipiron durable Training foundation | `dc142573a1163178ae8fd5fc9feb90ff8ffee21e` |
| Chipiron reviewed research base | `60fdf85bb26e34f7d502122f34f3208ae2fc19d9`, direct child of that foundation |
| Corrected Anemone | `74bbd1250f31b72497166e624659d547324a0e5a` |
| Locally built Anemone wheel SHA-256 | `225f78efa26f9937ad078a1fd4129b1e7e1ed309ae50c0de0a3cecb0b176c742` |
| Python | 3.13.5 |
| Packages | Anemone distribution 0.2.26; Atomheart 0.2.11; Valanga 0.2.5; Coral 0.1.15; Torch 2.14.0+cpu |

New ext4 worktree:
`/home/pompote/oldata/victor/morpion-scratch/anemone-chipiron-integration-20261024/worktree`.
Branch: `research/morpion-corrected-anemone-refresh`.
Dedicated interpreter: sibling `../venv/bin/python`.

All 288 installed Anemone production Python modules match the corrected commit
byte-for-byte. The original environment and worktrees were not modified. Build
inputs came from `git archive` of the exact commit into new scratch. The wheel still
declares 0.2.26, so a version-only requirement cannot distinguish it from the old
published implementation. The dependency manifest records hashes and import paths;
no package release or dependency download is substituted for that verification.
Anemone PR #77/#78 changes are not assumed present. No Anemone source was copied
into Chipiron production.

The existing Chipiron production dependency requirement is not changed to pretend
the unpublished fix is available as a released version. A future deployment needs
the reviewed release or exact artifact binding. This environment is an explicit
integration validation, not production activation.

The measurement control's old environment was also checked against the inherited
research manifest: all 344 recorded dependency modules and 476 Chipiron production
modules are unchanged. `corrected_anemone_refresh/evidence/control-dependencies.json`
records that check and
the prior manifest digest. Its dependency identity is defined by those source
hashes, not the ambiguous shared 0.2.26 version string.

## Scientific parity and proof boundaries

The previous research implementation remains historical evidence and a measurement
control. New scientific tests call only:

```python
runtime.refresh_with_evaluator(
    evaluator,
    scope="all",
    batch_size=64,
    validate_direct_certificate=validated_original_proof,
)
```

The callback is omitted where there are no external direct FORCED certificates.
Missing proof validity is not synthesized from the certainty enum or evaluator
version. The real direct-certificate test exhausts every legal action of the actual
move-34 one-action Morpion state, verifies its successor is terminal with total
score 35, and binds that proof to the same state, objective and original Value.
Refresh retains the original certificate object and evaluator version. This tiny
independent proof fixture is **not** a general production certificate validator.

The independent oracle is the reviewed `model_activation/probe.reference`; it calls
no Anemone backup/effective-value helper. Additional order assertions derive the
score/exactness/PV-length/stable-ID ordering independently. Tested cases cover:

- The actual seven-node graph: root becomes exactly `100.00000100000001`, replacing
  the old incorrect `2.2600000452000004`.
- Real DAG transpositions with at least seven multiple-parent nodes; windows 1/7/64;
  repeated neural 42→7→42 transitions; all estimated direct versions advance once.
- Actual terminal and nonterminal proofs, independent certainties, partial and full
  expansion, exact/estimated equal-score comparisons and refusal before mutation
  of unverified direct or unreconstructible partial tree certificates.
- Unchanged graph links, search RNG and selector step count; exact small-graph
  checkpoint continuation with and without rollout through both codecs.
- All nine legacy Dataset policies; five supported selected-value-v2 policies;
  explicit refusal of four exact-family policies, for both target preferences and
  both exact-only-filter settings.

Dataset comparison: 29 complete reference/indexed JSONL pairs, 53 paired supervised
rows, byte-identical metadata and rows; 16 v2 gated cases. There is no target
normalization or weakening of selected-source certainty. The terminal target 35 is
protected while the direct target 42 remains an ordinary estimated target.

The API's known-proof preflight can refuse before mutation. Inference errors or a
new contradiction may occur after mutation; a failed refresh runtime must be
discarded and restored from the committed checkpoint. This remains a caller
transaction obligation. General historical direct FORCED provenance is unavailable
in current Morpion artifacts; activation with such inputs must refuse until that
proof authority is established. Neither a source file hash nor a global producer
identity proves the mathematical validity of an arbitrary certificate.

## Numerical policy: exact existing difference

Anemone's `NodeDirectEvaluator.process_evalution_not_over` applies
`(1 / DISCOUNT) ** node.tree_depth * evaluation`, with `DISCOUNT=0.99999999`.
The built-in Reevaluation adapter evaluates the canonical state, extracts the raw
neural scalar and writes it unchanged. A new executable test observes both paths
on the same actual graph/model: raw patch 42 versus the exactly transformed Growth
value at depth > 0. Corrected refresh preserves the Growth expression and order of
operations bit-for-bit.

The smallest future correction is an Anemone-level pure scalar operation taking
the raw score and **the same checkpoint-relative tree depth**, called by its direct
evaluator and Chipiron's model-patch adapter. Do not use the absolute played-move
count: an endgame root at move 34 has tree depth zero. Keep terminal Values outside
that estimate transformation. Bind the scoring-policy identity to the future
activation and proposals, and reject incompatible pending proposals; do not silently
reinterpret already saved patch rows or training targets. No different optimization
objective, discount choice, rounding or historical target rewrite is introduced here.

## Restore semantics and remaining checkpoint discrepancies

The existing Anemone loaders already accept `master_state_value_evaluator`. Supplying
the known committed evaluator there preserves the saved runtime version, all direct
versions, Values, selector state and RNG. The four integration cases exercise both
checkpoint codecs and rollout on/off, followed by three matching Growth selections.
They inject the known evaluator at the loader boundary; they do not assign a private
post-restore evaluator or call `set_evaluator`.

Chipiron currently supplies its default evaluator to both loaders and subsequently
calls `_set_runtime_evaluator_from_bundle`. With `reevaluate_tree=False`, that calls
`set_evaluator`, incrementing the version even when the same model is restored.
A regression observes this remaining defect. The minimal caller correction is to
pass the already verified committed model through `_load_runtime_from_checkpoint`
and its sharded counterpart, and omit the second attach for normal resume. Loading
a checkpoint with an arbitrary newly supplied bundle is not evidence that the bundle
is its committed evaluator: the authority/model binding must be checked first.

The older runner helper also still defaults to a partial `leaves` refresh through
its configuration. Future activation must explicitly call the complete API, not
reinterpret the existing boolean as a promise to refresh all nodes. This task does
not repurpose that legacy behavior into an implicit model switch.

The known representative-parent checkpoint discrepancy remains visible and is not
repaired by Anemone's refresh change. Ordered child-link insertion can also differ
after restore; tests record it separately from edge-set/value equality. Decision
ordering and the tested subsequent selections match, but this small test is not
proof that every representative-parent difference is harmless for continuation.
No historical payload field is normalized away.

## Memory diagnosis and minimal upstream correction

`CheckpointStateResolver._resolved_states` is unchanged in the corrected commit.
Every resolved anchor/delta state is retained in that dictionary. Thin checkpoint
handles share it; bounded inference windows do not evict its entries. The small
tests observe the entire restored graph retained even at batch size one.

Chipiron's existing `_LiveCompactStateResolver` has a bounded LRU, but also owns a
different live-eviction payload catalog and per-node diagnostic maps. Switching
restored handles to it would create another O(N) catalog and duplicate generic
resolver responsibilities. No such workaround, cache clearing or second graph is
implemented.

The smallest upstream change is to bound retention **inside the existing**
CheckpointStateResolver, threaded through the current restore factories. Replace
its retention policy, not its payload authority; reuse the standard bounded-LRU
pattern, original immutable payload store and existing cycle detection. A decode
must keep its active ancestor chain alive locally while it resolves a child.
Validate capacities 0/1/B, shared ancestors, long chains, non-dense IDs, malformed
references, checkpoint reuse and exact continuation. Merely limiting cache entries
does not bound the size of a long-history state; account for cache capacity plus
active chain/window and maximum state size. Existing graph, PVs, selector and
checkpoint serialization remain separate O(N+E/payload) costs.

Repeated decoding can increase CPU; measure that tradeoff after the upstream change.
The present task measures the existing behavior and makes no memory-bound claim
for the complete restored runtime.

## Matched-input resource comparison

All paired output checkpoints **and every scientific snapshot file** are identical
byte-for-byte. Model, full scientific fingerprint, ordered-link digest and RNG also
match. Raw source identities, phase CPU/I/O/fsync/codec counters, profiles selected
by function, disk allocations and process measurements are in
[`resources.json`](corrected_anemone_refresh/evidence/resources.json) and the adjacent
per-run JSON files. Both runs retain the same existing ordered-link discrepancy
between the newly generated graph and its restored checkpoint.

| Actual graph | Engine | Complete refresh s | Nodes/s | All measured phases s | Peak RSS MiB | Retained decoded states |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,003 nodes / 1,114 edges, depth 20 | Previous research kernel | 24.374 | 41.15 | 31.241 | 310.426 | 1,003 |
| Same source and model | Corrected Anemone | 24.295 | 41.28 | 31.083 | 310.809 | 1,003 |
| 5,013 nodes / 5,931 edges, depth 62 | Previous research kernel | 136.815 | 36.64 | 214.407 | 410.957 | 5,013 |
| Same source and model | Corrected Anemone | 135.835 | 36.91 | 213.889 | 411.453 | 5,013 |

The control's complete-refresh time sums its direct/backup/selector phases; the
candidate measures one API call including preflight. These are single profiled
trials, not statistical evidence of a speedup. No substantial speed or memory
improvement is demonstrated. All-phase CPU user+system totals are respectively
30.409/30.252 s at 1k and 211.698/211.161 s at 5k. The complete runtime retains
O(N) decoded states despite windows of at most 64; their bytes also depend on state
history length. The core graph, its E links, payload catalog and selector remain
additional live memory. No second Python tree or full scientific snapshot is loaded
by this measured path.

Whole-process `/usr/bin/time` wall times, including imports and final evidence
hashing, are 34.89/34.90 s at 1k and 217.98/217.50 s at 5k (control/candidate).
Its peak RSS agrees with the table; phase totals deliberately exclude that overhead.

Corrected API nested cProfile attribution:

| Operation | 1,003 nodes: calls / cumulative s | 5,013 nodes: calls / cumulative s |
| --- | ---: | ---: |
| State-to-model feature preparation | 1,001 / 18.207 | 5,005 / 101.873 |
| Candidate-segment generation, inside features | 1,001 / 15.985 | 5,005 / 90.310 |
| Actual Morpion model `forward`, excluding features | 1,001 / 0.074 | 5,005 / 0.370 |
| Resolver calls, including cache hits | 53,041 / 2.097 | 267,581 / 13.125 |
| Actual anchor decodes | 268 / 0.713 | 1,372 / 5.759 |
| Actual delta decodes | 735 / 1.370 | 3,641 / 7.329 |
| Node backup entry point | 62 / 0.330 | 366 / 1.609 |
| Selector state refresh | 1 / 0.021 | 1 / 0.094 |
| Certificate preflight + postflight | 2,006 / 0.034 | 10,026 / 0.171 |

The decoded-state cache starts empty and retains every node. Features consume
approximately 75% of refresh time; candidate-segment enumeration remains the major
cost, not neural arithmetic. The inference interface is window-bounded, but the
adapter still evaluates individual states; this is not a claim of vectorized GPU
inference. The accepted model here is the tiny `small_mlp` fixture; these timings do
not predict the arithmetic cost of a larger production network. Canonical
decoding/serialization remains substantial outside refresh:
the 5k snapshot performs 5,013 canonical loads and 10,026 dumps; its bounded index
validation performs another 5,013 loads and 5,013 dumps.

| Corrected run phase / final allocated disk | 1,003 nodes | 5,013 nodes |
| --- | ---: | ---: |
| Checkpoint restore | 0.255 s | 0.963 s |
| Checkpoint write + synchronization | 0.343 s | 1.152 s |
| Closed snapshot write + synchronization | 3.700 s | 45.273 s |
| Bounded snapshot validation | 2.463 s | 30.642 s |
| Checkpoint allocation, unique inodes | 131,072 B | 491,520 B |
| Scientific snapshot allocation, unique inodes | 1,044,480 B | 6,094,848 B |
| Entire measured workspace, including index/source | 2,666,496 B | 14,819,328 B |
| Measured phase logical read bytes (`rchar`) | 13,574,629 B | 122,753,635 B |
| Measured phase kernel write bytes | 4,956,160 B | 37,728,256 B |

Kernel physical read bytes are zero in these warm-page-cache runs; fresh processes
do not imply a cold disk cache. Identical paired artifacts also have identical final
allocated sizes. Greater history depth increases payload/codec cost; these samples
are not a proof of asymptotic scalability. No new 10k run was performed: 1k/5k and
the source-level retention invariant already establish the unresolved cache issue.
The previous study's roughly 544 MiB / 466 s at 10k remains previous evidence only,
not an extrapolated or remeasured result for this corrected API.

Fresh legal graph generation is done once per size. Two copies of that exact source
checkpoint and one freshly accepted tiny model feed separate serial processes.
The control uses the original dependency and previous research refresh kernel; the
candidate uses the corrected wheel's public API. This is a comparison against the
previous scientifically checked kernel, not a timing comparison with the broken
old public full-refresh operation. No full scientific reference reader is used in
the measured path. Large outputs are compared as serialized scientific artifacts.
The reused research driver's synthetic producer revision label is held constant
across the pair (`dc142573-model-activation-research`); it is a fixture label, not a
claim about the executable Git head. Actual source/dependency identities are bound
in the separate evidence manifest. Neither output is committed as a real generation.

The existing driver, cProfile, StageMeasurement, codec counters and RSS/disk tools
are reused. In candidate runs the phase named `complete_refresh` includes preflight,
inference, backup and selector; `inference.prof` records that whole call. Separate
component times come from its nested profile and must not be summed as independent
work. CPU-only inference is used; no GPU-memory claim is made. Scratch disk totals
are final unique-inode allocation, not a measured instantaneous temporary-disk peak.

## Publication fence, recovery and intentional behavior change

The original producer captures workspace/epoch/producer/source/model/cursor before
inference, then checks the model and performs create-only publication outside Growth
ownership. A stale producer can cross a concurrent identity or cursor transition.
The deterministic baseline witnesses use the actual worker and flock, not sleeps:
the model identity case simulates visibility under the lock; the cursor case commits
a genuine synthetic M1 patch; the publication case shows Growth entering the gap.

The fix reuses `growth_receipt_ownership` only after expensive computation and source
byte validation. Under that lease it reloads the existing authority, calls the
existing `validate_proposal` for source/epoch/producer/cursor, rechecks full model
identity, then completes the existing no-replace/fsync publication before release.
It does not require the whole run-state JSON to equal its earlier bytes: unrelated
Growth status updates are allowed. Pins still protect the selected immutable source.

The lease retains its existing nonblocking behavior. Contention raises
`BlockingIOError` without publishing or advancing a cursor; an explicit retry after
ownership release succeeds. No waiting service, TTL, lock or persistent retry state
is introduced. Current opt-in callers must retry this existing ownership error.
Growth holds ownership for its invocation, so eventual progress also requires that
the owner release it; no fairness or starvation guarantee is invented here.

Intentional change: a synthetic mid-computation epoch mismatch is now rejected by
the producer before visibility. The existing test is strengthened to require no
patch/cursor publication, while retaining the consumer rejection of an old proposal.
Exact receipt UUID/digest semantics and duplicate acknowledgement remain unchanged.
Current `current_model_identity` still reads the active-model projection. Future
activation must derive it from the committed model binding before relying on
post-commit projection repair. The fence closes the demonstrated race among callers
using the existing lease; it does not implement a durable M2 transition or claim
safety against old binaries that ignore ownership.

For a future switch, the simplest pending-work rule is to drain an admissible M1
proposal under the same authority before activating M2. Preserve its exact accepted
receipt and cursor across the transition; the first M2 proposal must bind its
expected-cursor digest to that exact accepted cursor. Model-specific selection and
pass counters must follow the declared new-model policy, rather than rewriting the
old acknowledgement. Refuse activation while a proposal cannot be validated or safely
completed. This requires no fabricated acknowledgement, cursor reset or epoch fork.
A superseding/rejection transition would need its own explicit durable semantics;
it is not implemented or assumed here.

The publication lease is short relative to inference, but not constant-cost: it
still reads cumulative run-state metadata and hashes the model bundle through the
existing identity helper. Those O(history + receipts) and O(model bytes) costs are
not disguised as a history-independent protocol. No extra full receipt-ledger scan
or persistent identity cache was added.

## Validation, code size and next steps

Full installed-wheel local CI equivalents pass: **2,638 passed, 2 skipped,
4 deselected, 0 failures/errors**, with 141 warnings, in **2,131.11 s** (35m31s;
metrics-driver wall time 2,133.34 s). The normal marker expression is unchanged:
`not integration and not external_data`; execution is deliberately serial (`-n 0`).
The two skips require unavailable CUDA. The four excluded tests are the existing
chess checkmate integration cases. No Morpion scientific case was removed. Overall
line/branch coverage is 80.24%, with 83.25% statement coverage. The full JUnit file,
exact skipped/excluded identities, raw artifact hashes and local paths are recorded
in `corrected_anemone_refresh/evidence/validation-summary.json` and `full-ci.xml`.
These are local CI gates against the pinned dependency, not a remote CI run or a
published package release.

Targeted final checks:

- Corrected-API scientific/restore/retention suite: **63 passed in 56.836 s**.
- Deterministic race baseline using verified original installed Chipiron:
  **3 failed in 6.442 s**, demonstrating stale model, stale cursor and publication
  ownership gaps. The same three cases on the correction: **3 passed in 6.420 s**.
- All six new fence scenarios: **6 passed in 9.114 s**. Updated epoch regression
  plus all six fence cases: **7 passed in 10.784 s**.
- Existing focused run initially had **172 passed / 1 failed in 210.134 s**. Its
  sole failure expected the old producer to publish a stale-epoch proposal. The
  final test now checks earlier rejection and still explicitly tests consumer
  rejection of old-producer evidence; this is an intentional protocol-boundary
  change, not a weakened scientific assertion.
- Source/wheel build, installed-distribution smoke, pinned toolchain checks, Ruff
  lint and formatting pass. Static ratchet: **0 new diagnostics**, 206 current
  against 209 inherited baseline; no baseline was edited (mypy 89, pyright 73,
  pylint 44). Packaging release checks are outside this task; an extra attempted
  `twine check` could not run because Twine is absent from the dedicated environment.

Development reruns are not hidden: a combined research/repository pytest invocation
failed collection because repository path manipulation shadowed `scripts`; the
documented separate invocations pass. The initial race pause was before admission,
so it was moved to the post-capture proposal boundary. An attempted baseline omitted
`CHIPIRON_TEST_INSTALLED=1` and inadvertently tested the checkout; only the verified
installed baseline above is used as red evidence. One exploratory unrelated-Growth
test incorrectly expected a new generation from a fixture that performs only a
status update; the final assertion checks that actual independent status update.
Raw logs remain in the isolated scratch directory; targeted JUnit identities and
hashes are retained in `corrected_anemone_refresh/evidence/targeted-validation.json`.

Production delta: one file, **17 added / 8 removed, net +9 lines**. The only new
production behavior is the fenced publication block, reusing current validation and
I/O utilities. No new production module/interface is added. New research code is
tests and an adapter to the existing profiler/measurement driver: 447 lines across
three research Python files. The repository adds 190 lines of deterministic race
tests and changes the existing epoch test by +9/-1 lines. Dependency/evidence JSON
is measurement data, not an additional runtime implementation.

Final preservation checks confirm all **39 pre-existing Chipiron** and **11
pre-existing Anemone** worktree registrations, existence states, statuses and tracked
diff fingerprints are unchanged. Missing old temporary worktrees were not pruned.
All 288 corrected Anemone and 476 installed Chipiron production module hashes were
rechecked after CI. The only Git worktree addition is this isolated Chipiron checkout.

Recommended order:

1. Review the exact corrected Anemone artifact and this integration evidence;
   obtain a distinct reviewed release/artifact binding before deployment.
2. Bound the existing Anemone restore resolver and extract its existing numeric
   score operation without changing the mathematical policy.
3. Bind and supply the committed evaluator at Chipiron restore, eliminating the
   version-incrementing resume attach; call the complete Anemone API explicitly for
   a separately authorized transition. Refuse unavailable certificate proofs.
4. Extend only the current capsule/run-state transition to commit accepted model,
   checkpoint, scientific state and numeric policy together; retain exact pending
   patch/receipt/cursor handling through this ownership gate.
5. Execute the complete M2 death/restart matrix, source/model corruption and live
   producer races before any explicit activation. Existing M1 crash tests do not
   establish that matrix. Resolve or prove continuation-equivalence of the known
   checkpoint representative/order discrepancies first.

The demonstrated performance follow-up is the existing Morpion feature extractor's
candidate-segment work. It belongs in that domain implementation, with exact feature
tensor and model-output comparisons on the same states before any optimization is
accepted. More refresh orchestration, another cache, or nominally larger batches
would not address the measured dominant cost. No speculative feature rewrite is
included here.

After that verified current protocol exists, remove the redundant same-model
post-restore attach, ambiguous partial refresh-as-switch route, independent
active-model fallback authority and obsolete Training active-model write branches.
Do not maintain permanent dual formats merely to preserve internal historical
layouts. No migration or deletion is performed here.

## Local review

Implementation/test commit:
`7e29c9f58f2d9559e28565afc9371e963e570311`.
Exact base: `60fdf85bb26e34f7d502122f34f3208ae2fc19d9`.
The subsequent report/evidence commit changes no production behavior.

Open the isolated checkout in VS Code:

```bash
code --new-window /home/pompote/oldata/victor/morpion-scratch/anemone-chipiron-integration-20261024/worktree
```

From a terminal in that checkout, open the actual base-to-implementation production
Git diff in VS Code's graphical two-pane diff. These command-local settings do not
change Git configuration or any old checkout:

```bash
TMPDIR=/home/pompote/oldata/victor/morpion-scratch/anemone-chipiron-integration-20261024/tmp \
git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt \
  60fdf85bb26e34f7d502122f34f3208ae2fc19d9 \
  7e29c9f58f2d9559e28565afc9371e963e570311 -- \
  src/chipiron/environments/morpion/bootstrap/reevaluation_worker.py
```

Omit the final path to review all implementation/test files sequentially. Use
`git diff --stat 60fdf85b HEAD` for the complete implementation plus evidence scope.
This local branch is not a new monolithic PR duplicating the unmerged durability
stack. Review the small fence separately after its exact foundation is accepted;
Anemone remains an independently pinned dependency, never a Chipiron cherry-pick.

No production activation, gen38 operation, distributed worker deployment, historical
artifact mutation, package release, PR merge or epoch fork was performed.
