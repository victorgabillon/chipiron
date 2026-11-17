# Morpion bootstrap: current implementation and ownership map

> **Source-reviewed snapshot:** 2026-11-17, local C6 integration on shared
> `308e04f524369733c2acdbc38876b86ac1013be5` (reviewed main `4c882db5`, C2–C5).
> This candidate is not deployed or fully qualified. Exact source and validation
> identities belong to the C6 delivery receipt; remote/FULL qualification requires
> the owner's integration gate.
> This is an index of **implemented contracts**, not a new design, a readiness
> certification, or permission to enable experimental policies. Read the source
> at the pinned commit for exact behavior. Later changes must update this map.

## Why this page exists

Chipiron has several overlapping-looking Morpion bootstrap entrypoints:
historical single-process and artifact-pipeline modes, the opt-in closed
scientific publication/atomic activation path, and newer continual coordination
policies. The names *Reevaluation*, *refresh*, *promotion*, *Training* and
*activation* do not denote interchangeable operations. Older design and
implementation reports are retained as historical evidence; some describe
features that were incomplete *at their original commit*. This page records the
current source-level boundaries to prevent inventing a second implementation.

Review coverage includes the entrypoints, contracts and cross-boundary files named
below. The original architecture review did not exhaustively verify every branch.
C6 adds focused integration, native-state, mixed-source, promotion and recovery
tests; it does not replace the separately authorized final FULL qualification.

## One-page process and authority map

```text
chipiron-bootstrap start / lifecycle
                  |
         monitored supervisor
                  |
   +--------------+---------------+------------------+
   |              |               |                  |
 Growth        Dataset          Training         Reevaluation
 process       worker           worker           worker
   |              |               |                  |
   |     immutable published   accepted results   pinned tree
   |          source           + own Adam/weights  snapshot
   |              |               |                  |
   |              +---------------+                  |
   |                separate acceptance              |
   |                          |                  neural estimates
   |                optional accepted promotion      |
   |                          |                  immutable singleton
   |                   Growth-owned activation  reevaluation patch
   |                          |                  |
   +--------------------------+------------------+
   |            one authoritative tree/commit owner
   |      apply eligible patch -> Anemone backup
   |      grow -> closed export/checkpoint -> durable run_state
   v
 immutable generation publications, accepted receipts/cursor, model binding
```

The four roles may run in different processes. **Computing an evaluation
asynchronously is already implemented.** Concurrent mutation of a single
authoritative live tree is **not** implemented or desirable: Growth applies
reevaluation results at its owned boundary. A persistent Python worker can
retain interpreter imports between jobs, but does not retain a live Growth tree,
Dataset reader, training optimizer or model across jobs as shared authority.

## Which entrypoint is which?

| Entry point / owner | Implemented responsibility | Explicit non-responsibility |
| --- | --- | --- |
| [`operator_cli.py`](../../src/chipiron/environments/morpion/bootstrap/operator_cli.py), [`lifecycle.py`](../../src/chipiron/environments/morpion/bootstrap/lifecycle.py) | Operator `start`, current-run selection, preview and controlled native resume | Does not recreate scientific configuration, train or maintain another coordinator |
| [`launcher.py`](../../src/chipiron/environments/morpion/bootstrap/launcher.py) | Routes `--scientific` stages and explicit prepare/run/activation operations | Does not authorize a stage by the CLI flag alone |
| [`continual_operator.py`](../../src/chipiron/environments/morpion/bootstrap/continual_operator.py) | Owns monitored four-worker supervisor/resources | Scientific work scheduling is elsewhere |
| [`persistent_worker.py`](../../src/chipiron/environments/morpion/bootstrap/persistent_worker.py) | Optionally reuses the Python process for bounded jobs | Does not keep one mutable live tree or neural model between jobs |
| [`pipeline/stages.py`](../../src/chipiron/environments/morpion/bootstrap/pipeline/stages.py), [`pipeline_orchestrator.py`](../../src/chipiron/environments/morpion/bootstrap/pipeline_orchestrator.py) | Actual stage entrypoints and per-generation artifact dispatch | The local sequential `loop` path is not a replacement for four scientific workers |
| [`runtime/runner.py`](../../src/chipiron/environments/morpion/bootstrap/runtime/runner.py) | Adapts the Morpion game to one Anemone search runtime; Growth, checkpoint, export and patch APIs | Does not own a second generic search-value/backup implementation |
| [`generation_publication.py`](../../src/chipiron/environments/morpion/bootstrap/generation_publication.py), [`generation_publication_io.py`](../../src/chipiron/environments/morpion/bootstrap/generation_publication_io.py) | Closed capsule, model binding, read pins, Growth ownership and immutable publication | Discovery files do not supersede committed `run_state.json` |
| [`checkpoint_patch_receipts.py`](../../src/chipiron/environments/morpion/bootstrap/checkpoint_patch_receipts.py) | Exactly-one-committed-effect patch receipt and accepted cursor in durable Growth state | Producer publishing a patch does not commit its application |
| [`dataset_publication.py`](../../src/chipiron/environments/morpion/bootstrap/dataset_publication.py) | Dataset proposal, source identity, admission and acceptance | Does not promote a model |
| [`training_publication.py`](../../src/chipiron/environments/morpion/bootstrap/training_publication.py) | Immutable Training results, candidate continuation/selection and acceptance | Acceptance is not model activation |
| [`model_activation.py`](../../src/chipiron/environments/morpion/bootstrap/model_activation.py) | Growth-owned cold activation: full refresh by default, or fresh opt-in `owned_cold_model_only_v1` | No second scheduler, mutation owner or prediction authority |
| [`reevaluation_worker.py`](../../src/chipiron/environments/morpion/bootstrap/reevaluation_worker.py) | Independent active-model inference over bounded saved-snapshot windows | Does not mutate Growth's live tree or authoritative checkpoint |
| [`reevaluation_patch_consumer.py`](../../src/chipiron/environments/morpion/bootstrap/reevaluation_patch_consumer.py), [`runtime/reevaluation_patching.py`](../../src/chipiron/environments/morpion/bootstrap/runtime/reevaluation_patching.py) | Growth applies accepted direct-value patches and uses native Anemone propagation | Does not implement its own minimax/maximum backup |

## Distinguish the three execution families

1. **Historical inline/single-process bootstrap** is implemented in
   [`bootstrap_loop.py`](../../src/chipiron/environments/morpion/bootstrap/bootstrap_loop.py)
   and its `cycle_*` helpers. It remains a compatibility path.
2. **Artifact-pipeline/four-worker execution** uses the same stage entrypoints,
   worker claims, immutable files and a separate Reevaluation producer. Its
   older `active_model.json`-driven examples remain in the operator README,
   but are **not** the scientific model-authority contract.
3. **Opt-in scientific continual execution** requires fresh prepared genesis,
   checked installed source/runtime, closed scientific publication, accepted
   Dataset and Training, and Growth-owned model activation. The
   [`learning_schedule.py`](../../src/chipiron/environments/morpion/bootstrap/learning_schedule.py)
   policy dispatch can select no window, bounded lookahead, ready-result
   lookahead, or `bounded_continuous_v1`. For the latter,
   [`continuous_contract.py`](../../src/chipiron/environments/morpion/bootstrap/continuous_contract.py)
   defines six wire slots, [`continuous_transitions.py`](../../src/chipiron/environments/morpion/bootstrap/continuous_transitions.py)
   proposes transitions, and the `continuous_dataset`, `continuous_training`,
   `continuous_publication`, `continuous_lifecycle`, `continuous_resources`
   and [`promotion.py`](../../src/chipiron/environments/morpion/bootstrap/promotion.py)
   modules bind them to real admission, resources, common validation and durable
   publication. It does **not** replace Growth's tree owner or add a new optimizer.

The `bounded_continuous_v1` policy remains explicit/default-off. Its activation
default mode is `owned_cold_full`. Fresh genesis may explicitly select
`owned_cold_model_only_v1`, preserving the same six slots and promotion comparison. Do not describe successful
isolated pilot results as default-on or production-certified behavior.

## Current Reevaluation: exact producer → consumer sequence

The separate producer is real; it is **not** a proposed B4 component.

1. The scientific launcher dispatches
   `run_morpion_reevaluation_worker_once(...,
   use_streaming_sharded_reader=True, allow_scientific_v2=True)`.
   [`reevaluation_worker.py`](../../src/chipiron/environments/morpion/bootstrap/reevaluation_worker.py)
   requires managed execution for the scientific worker, finds an active model
   through committed sealed authority, and selects the latest published
   suitable tree snapshot. Non-scientific legacy mode has different discovery.
2. A source manifest is pinned; the bounded sharded
   [reader](../../src/chipiron/environments/morpion/bootstrap/sharded_training_reader.py)
   selects sorted node IDs through the historical circular cursor, or the fresh
   schema-3 finite rolling cursor. The built-in evaluator
   loads the pinned bundle, reconstructs selected Morpion states, predicts
   nonterminal direct values and preserves stored terminal scores.
3. **Historical schemas retain a model-bound cursor.** `cursor_matches_active_model` compares
   evaluator generation, name and bundle path. A mismatch resets the next
   selection to the beginning and resets the completed-pass count. Tree
   generation alone does not reset it. A full-pass count measures cursor wraps,
   not freshness against every future/new model. Schema 3 instead freezes one
   source inventory and accepts exact finite prefixes without wrapping a packet.
   Model switches preserve its source, continuation and pass count; future Growth
   IDs enter a later pass. Incomplete sources remain retention roots. Completion
   records traversal, not a claim that all direct predictions use the newest model.
4. Inference runs outside the Growth mutation ownership boundary.
   [`reevaluation_publication.py`](../../src/chipiron/environments/morpion/bootstrap/reevaluation_publication.py)
   creates **one** immutable `pipeline/reevaluation_patch.json` via
   create-only publication. A pending patch blocks additional publishing;
   the producer does not replace it. In the atomic protocol it carries the
   **proposed** next cursor, but does not advance the accepted cursor.
5. A verified model activation that supersedes a worker's captured model
   currently **discards the unpublished batch**. It does not allow a patch for
   the previous evaluator to overwrite the new committed model. See
   [`reevaluation_activation.py`](../../src/chipiron/environments/morpion/bootstrap/reevaluation_activation.py).
6. At a Growth-owned boundary, `GrowthPatchReceiptSession.apply()`
   validates patch ID/bytes, source generation, producer/cursor evidence and
   the **exact current model**. It applies the patch through the live runner.
   Native [Anemone value propagation](https://github.com/victorgabillon/anemone)
   updates affected ancestors/selector. A patch-driven save commits the
   checkpoint, receipt **and accepted cursor** together through durable
   `run_state.json` replacement. Publication and acknowledgement are
   distinct; crash recovery deduplicates accepted patch IDs.

Consequences: Reevaluation computation and Growth can overlap, but the consumer
only incorporates a patch at a Growth boundary, under the single owner. Patch
application may force checkpoint/publication work; very small frequent patches
can hurt total throughput. Workers do not directly edit saved checkpoints.

### Model authority is different in scientific mode

For the current closed scientific path, the authoritative evaluator is the
sealed committed generation's `active_evaluator` pinned to the model bundle.
`pipeline/active_model.json` is a repairable projection/discovery aid, **not**
a second activation authority. [`model_activation.py`](../../src/chipiron/environments/morpion/bootstrap/model_activation.py)
requires a verified accepted Training model, an explicit request or accepted
automatic promotion path, Growth ownership, complete certificate preflight,
the genesis-selected operation, and an atomic checkpoint/scientific commit.
An accepted Training cohort can legitimately retain the incumbent; each
candidate's accepted weight/Adam lineage continues separately.

Default `owned_cold_full` activation retains the synchronous native full refresh
and its original request/completion policy. Fresh model-only genesis requires
tracked actual origins, rolling schema 3 and explicit Dataset targets. Its sealed
genesis descriptor authenticates a distinct versioned policy; requests use schema
3 (explicit) or 4 (reserved continuous promotion), and completion schema 2 records
an evaluator switch with zero rewritten nodes, without a full-refresh assertion.
Historical workspaces cannot adopt this mode in place.

Both operations cold-restore a private runtime, verify certificates, drain one
valid pending old-model patch through the existing receipt/cursor commit, pin the
exact accepted candidate and use the same checkpoint/capsule/run-state commit.
Model-only uses the supported runner `switch_model_evaluator()` capability and
Anemone `set_evaluator()`. Existing direct/tree/effective values, proofs, edges,
selector and RNG remain unchanged; only future native direct predictions use the
new evaluator. Later accepted bounded patches update earlier nodes through native
propagation. A novel old-model packet refuses; an exact accepted retry only repairs
projections. Failure discards private RAM; a committed drain remains authoritative
if activation fails afterward. Existing automatic renewal validates that precise
adjacent drain before issuing another bound request.

C3 actual-write evidence and immutable creation IDs are sealed inside the same
capsule as the checkpoint. In the combined rolling/tracked path, C5 binds the
mixed header *after* origin evidence preparation and *before* export validation,
inventory, seal and run-state replacement. Header, descriptor and generation
identity share the same reference. Both readers call the real committed-origin
reader and authenticate every source node, including rows excluded from Dataset.
Derived tree/effective targets do not acquire a single neural model stamp.
C3-only homogeneous and fully untracked historical paths retain their contracts.
Model pins and origin sidecars survive checkpoint/science retirement; incomplete
rolling sources additionally retain their payloads.

Cold restore, certificate validation, checkpoint and export remain on the activation
path. Model-only does not establish homogeneous newest-model predictions, a
wall-clock staleness bound, learning convergence or a measured performance gain.

## Integrated J5 reliability: existing owners, stronger boundaries

The [continuous coordination contract](../morpion_continuous_coordination.md)
describes the operational sequence. J5 changes that sequence's failure handling,
not the target policy, Training selection or `owned_cold_full` activation:

- **J5.1 — Dataset backpressure and shutdown (#128).**
  [`continuous_resources.py`](../../src/chipiron/environments/morpion/bootstrap/continuous_resources.py)
  distinguishes unavailable completion capacity from invalid accounting.
  `dataset_publication.admit_dataset` releases only the unadmitted attempt's
  advisory claim; the paced worker defers that admission with pending Source,
  debt, existing consumers and their pins unchanged. No new job clock is
  created or renewed. Corrupt authority still fails. On stop,
  `continual_operator` durably records the decisive reason/clock before
  `continuous_lifecycle.close_admission` tries Growth ownership without waiting.
  If busy, supervisor acknowledgement and native reap precede the bounded retry
  under the original shutdown deadline. Finalization requires proven admission
  closure and native shutdown; uncertainty retains measured usage and pins.
- **J5.2 — authenticated terminal idle (#129).** Growth seals scientific
  node-budget termination into its publication.
  [`managed_sessions.py`](../../src/chipiron/environments/morpion/bootstrap/managed_sessions.py)
  authenticates that seal and the worker's actual live managed launch before
  raising `ManagedScientificTerminationError`. Existing persistent workers
  catch this typed refusal, including a terminal commit racing dispatch, and
  heartbeat in idle until native supervised shutdown. The finite fallback
  prevents an endless terminal wait. This authorizes waiting, never more
  scientific work; invalid/missing/stale worker evidence and unexplained exits
  remain failures. Historical unsealed runs are not migrated or refunded.
  The shutdown test fixture now hands off the actual post-`setsid` worker PID
  and verifies its published status before monitor sampling; production
  identity checks remain strict.
- **J5.3 — bounded prepared-Dataset contention (#130).**
  `prepared_dataset.borrow_continuous` computes one deadline from the original
  admitted job, capped by the original managed launch's remaining allowance
  and supervisor limit. It acquires stable Dataset then native preparation
  ownership without holding Growth while waiting. It revalidates authority
  after the stable wait and again under Growth ownership before the consumption
  commit in
  [`continuous_inputs.py`](../../src/chipiron/environments/morpion/bootstrap/continuous_inputs.py).
  Ready binding, admitted slot, claim, launch and Dataset state must still
  match. The same deadline bounds disposable recovery. Expiry refuses
  consumption/comparison completion; retries cannot renew the clock, and
  signals unwind ownership normally. Damaged inputs may rebuild only before
  consumption, for the same admitted Dataset and recipe; post-consumption
  corruption remains fatal evidence.

These guarantees are part of integrated main, while continuous scheduling
remains fresh opt-in/default-off. C6 adds the separately versioned model-only policy and actual origins above;
these J5 guarantees continue to apply to the existing owners.

## Values, targets and provenance: three different owners

- **Valanga** defines `Value` (score, `ESTIMATE`/`FORCED`/`TERMINAL`,
  outcome). It does not model model generation, wall-clock age or workflow.
- **Anemone** owns the generic node direct/tree/effective selection, proof,
  backup, DAG propagation, PV/ordering and local
  `direct_evaluation_version`. Its normal `reevaluate_nodes` updates direct
  predictions/version; `apply_node_value_updates` updates scalar values and
  propagates but **does not currently stamp a producer model identity or
  current `direct_evaluation_version`**.
- **Chipiron** owns Morpion feature conversion, model bundles, scientific
  producer/model SHA-256 identities and training-target choices. For **new**
  scientific preparation, [target policy](../../src/chipiron/environments/morpion/learning/target_policy.py)
  uses canonical exported *effective* targets by default (PR #127); explicit
  tree/direct policies and historical backed-up recipes retain their own
  compatibility semantics. Canonical effective value means Anemone's chosen
  direct-or-tree value, not necessarily the backed-up tree value.

A checkpoint/row containing mixed-old/new predictions cannot silently claim
that every direct value came from the latest model. C3 tracks actual direct-write origins separately; `direct_evaluation_version` alone is local to a
runtime and not durable model-digest proof. Do not add model timestamps to
Valanga's scalar `Value`.

## Caches and expensive work already implemented

- [`evaluator_input_cache.py`](../../src/chipiron/environments/morpion/players/evaluators/evaluator_input_cache.py)
  implements **default-off** `CHIPIRON_EVALUATOR_INPUT_CACHE=1`, scoped to
  owned candidate activation's full refresh. It stores canonical *input
  features*, not predictions or certificates. Only qualified strict CPU
  FP32 scalar flat linear/MLP recipes are supported; the normal linear-41
  admission ceiling is 20,000 total nodes. It is **not automatically attached
  to the independent Reevaluation worker** and is not a generic cache.
- [`prepared_dataset.py`](../../src/chipiron/environments/morpion/bootstrap/prepared_dataset.py)
  with [storage](../../src/chipiron/environments/morpion/bootstrap/prepared_dataset_storage.py)
  provides separately bounded, disposable accepted-Dataset preparation; it
  is not model-prediction caching or a second scientific Dataset authority.
- [`runtime/checkpoint_io.py`](../../src/chipiron/environments/morpion/bootstrap/runtime/checkpoint_io.py)
  and Anemone's native checkpoint/streaming restore supply lazy state
  resolution. The [closed snapshot](../../src/chipiron/environments/morpion/bootstrap/closed_scientific_snapshot.py)
  and [scientific stream](../../src/chipiron/environments/morpion/bootstrap/scientific_stream.py)
  are distinct from runtime checkpoint and reevaluation patch formats.
- [`managed_sessions.py`](../../src/chipiron/environments/morpion/bootstrap/managed_sessions.py),
  [`continuous_resources.py`](../../src/chipiron/environments/morpion/bootstrap/continuous_resources.py)
  and [`persistent_worker_state.py`](../../src/chipiron/environments/morpion/bootstrap/persistent_worker_state.py)
  already own original time/resource/session limits and quiescence evidence.
  Do not add competing monitors, leases, claims or process owners.

## B4: only the *remaining* experimental work

As of this source snapshot, **asynchronous inference is implemented; asynchronous
model-only activation with rolling cross-model coverage is not**. A narrowly scoped
B4 study should first verify these integration seams:

1. **Activation:** introduce an explicitly identified, opt-in scientific
   model-only commit without the current all-node synchronous refresh, preserving
   existing accepted model pins, Growth ownership, exact receipts, recovery and
   compatibility. The old complete refresh remains the reference/default.
2. **Traversal:** make the accepted Reevaluation scan cursor independent of
   evaluator identity **without** weakening batch/patch model identity.
   Continuous node coverage across growing snapshots needs a tested fairness
   invariant; a naive sorted cursor reset or wrap count is insufficient.
3. **In-flight model changes:** current model-bound proposals are rejected or
   unpublished work discarded. Decide whether to discard/retry only the affected
   batch (simple and safe) or prove a monotonic per-node provenance rule for
   accepting old-model patches. Neither may silently overwrite a newer value.
4. **Provenance:** worker scalar patch application lacks the direct-version stamp
   that native direct reevaluation emits. A durable mapping to actual model
   digests is still needed if consumers interpret mixed-model values.
5. **Science and latency:** current closed Dataset/export and activation receipts
   assume canonical source semantics. Mixed-model scientific targets need explicit
   coverage/freshness declarations or a safe refusal. Benchmark checkpoint-on-
   patch cost, application pauses and CPU contention; a separate worker is not
   zero-cost. Preserve every terminal/exact certificate and native propagation.

The C6 implementation above addresses origins, finite traversal and mixed
admission. Scientific quality, throughput and long-lived storage remain research
questions requiring separately bounded measurements.
Do not add another generic worker, event bus, tree database, backup, or model
authority before proving the existing protocol cannot supply the contract.

## Tests and reading order

Start with the entrypoint/source pair for the behavior under investigation,
then read its existing contract tests. In particular:

- `tests/environments/test_morpion_scientific_launcher.py`,
  `test_morpion_continual_operator.py`, `test_morpion_continuous_runtime.py`,
  `test_morpion_continuous_races.py`, `test_morpion_continuous_lifecycle.py`.
- `test_morpion_continuous_backpressure.py`,
  `test_morpion_continuous_shutdown_contention.py`,
  `test_morpion_scientific_terminal_workers.py`,
  `test_morpion_persistent_shutdown.py`, `test_morpion_continuous_lease.py`.
- `test_morpion_bootstrap_reevaluation_worker.py`,
  `test_morpion_streaming_reevaluation.py`,
  `test_morpion_atomic_reevaluation.py`,
  `test_morpion_atomic_reevaluation_crashes.py`,
  `test_morpion_reevaluation_activation_race.py`,
  `test_morpion_reevaluation_contention.py`.
- `test_morpion_checkpoint_patch_receipts.py`,
  `test_morpion_model_activation.py`,
  `test_morpion_model_activation_crashes.py`,
  `test_morpion_generation_publication_crashes.py`.
- `test_morpion_fenced_dataset.py`, `test_morpion_durable_training.py`,
  `test_morpion_prepared_dataset.py`, `test_morpion_target_policy.py`,
  `test_morpion_target_compatibility.py`,
  `test_morpion_evaluator_input_cache.py`.

See [operator README](../../src/chipiron/environments/morpion/bootstrap/README.md),
[continuous coordination](../morpion_continuous_coordination.md),
[normal lifecycle](../morpion_run_lifecycle.md),
[atomic Reevaluation cursor](../research/morpion_atomic_reevaluation_patch_cursor.md),
[activation](../research/morpion_atomic_model_activation.md), and
[effective target semantics](../research/morpion_effective_target_semantics_2026_11_15.md).
Their older historical implementation notes are not automatically today's
feature status.

## Required pre-change review discipline

Before proposing another Morpion bootstrap worker, coordinator, evaluator refresh,
cache, checkpoint writer or scientific schema:

1. Identify **which run family** is active (legacy loop, four-worker artifact
   pipeline, or opt-in closed scientific/continuous mode).
2. Search the **current pinned source and installed dependency** for the role,
   not only its documentation, then trace the actual launcher dispatch to the
   owning entrypoint and tests. Old research worktrees are not current main.
3. Write an **existing-versus-missing** table: which part is already implemented,
   which actual contract must change, and which other consumers depend on it.
4. Preserve **one authoritative owner** for each scientific artifact, and keep
   operational caches, proposals, projections and receipts in their documented
   non-authoritative roles.
5. State whether the proposal is **current default**, **fresh opt-in**,
   **historical compatibility**, or **research only**. Never claim production
   equivalence or speedup from a synthetic proof/prototype alone.
6. Update this map and the nearest operator document in the same PR when
   interfaces/ownership change. Do not silently rewrite historical evidence.
7. Require focused contract/crash tests and a new source/HEAD check before
   calling the change integrated. Do not merge or bypass protections based only
   on a previous green run.

This procedure limits architectural rediscovery; it cannot honestly certify
that every code path in the whole repository has been exhaustively verified.

## Version/status fence

J5.1/J5.2/J5.3 merged in order through #128 → #129 → #130 at
`2a5925be3e70b3eb736bb919c050c1aaa32a30f1`, before the B6 integration.
That J5 baseline's [post-merge CI](https://github.com/victorgabillon/chipiron/actions/runs/43878496866)
passed FULL, FAST/PR, integration, quality, strict static and aggregate gates.
This documentation refresh did not rerun those suites or a scientific pilot.
Earlier stacked-branch pilot results retain their original source identities;
they are not new experiments on this documentation candidate.

**B6 Anemone 0.4.1 is integrated through PR #132.** The current source baseline
above has tree `79038ddfc2c286c22c8db54889269b11017885c7` and pins
`algorhino-anemone==0.4.1` in [`pyproject.toml`](../../pyproject.toml).
The current scientific producer requires Anemone **0.4.1** in
[`sharded_training_v2.py`](../../src/chipiron/environments/morpion/bootstrap/sharded_training_v2.py).
Exact producer, evaluator and configuration checks still apply; this is not a
checkpoint migration. Historical 0.4.0 evidence, frozen contracts and experiment
environments retain their original identities. See the
[scientific dependency contract](../testing.md).
C6 integrates the reviewed local C3/C4/C5 contracts and the new fresh opt-in
model-only path. It remains a local candidate pending owner-authorized final
installed-package FULL qualification, review and remote integration.

Historical research reports and the September canonicalization audit retain
their dated/source-pinned claims. Their old branch, launch and unimplemented-work
instructions are evidence of those revisions, not today's operational checklist.
Use this map and the current operator/coordination contracts first. C6 changes
local runtime contracts as described above; dependency versions and historical
experiment bytes remain unchanged.
