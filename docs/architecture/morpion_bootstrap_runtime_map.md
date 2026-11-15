# Morpion bootstrap: current implementation and ownership map

> **Source-reviewed snapshot:** 2026-11-15, Chipiron protected `main`
> `fcf0fcc74128362f0aef99f5f0d7141adce6c1f8` (includes PR #127).
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

Review coverage: the current source tree contains 185 Python modules under
`src/chipiron/environments/morpion/bootstrap/` and 190 Morpion-named test/helper
modules under `tests/environments/`. The entrypoints, contracts and
cross-boundary files named below were inspected. This is **not** a line-by-line
verification of every branch in those modules or a fresh execution of all
tests. Claims about crash safety are descriptions of the current code's intended
and tested contracts, not an independent filesystem proof from this document.

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
| [`model_activation.py`](../../src/chipiron/environments/morpion/bootstrap/model_activation.py) | Explicit model activation within Growth authority, current **full** scientific refresh | Does not currently offer a model-only/partial-refresh activation policy |
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
mode in the current contract is `owned_cold_full`. Do not describe successful
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
   selects sorted node IDs through a circular cursor. The built-in evaluator
   loads the pinned bundle, reconstructs selected Morpion states, predicts
   nonterminal direct values and preserves stored terminal scores.
3. **Current cursor is model-bound.** `cursor_matches_active_model` compares
   evaluator generation, name and bundle path. A mismatch resets the next
   selection to the beginning and resets the completed-pass count. Tree
   generation alone does not reset it. A full-pass count measures cursor wraps,
   not freshness against every future/new model.
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
full `refresh_model_estimates`, and an atomic checkpoint/scientific commit.
An accepted Training cohort can legitimately retain the incumbent; each
candidate's accepted weight/Adam lineage continues separately.

This explains an important limit: **the current scientific activation already
refreshes the tree synchronously.** The separate Reevaluation worker is real,
but does not yet replace that stop-the-world activation work.

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
that every direct value came from the latest model. Freshness metadata is a
separate future contract; `direct_evaluation_version` alone is local to a
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

These items are **proposed research**, not functionality claimed to be present.
Do not add another generic worker, event bus, tree database, backup, or model
authority before proving the existing protocol cannot supply the contract.

## Tests and reading order

Start with the entrypoint/source pair for the behavior under investigation,
then read its existing contract tests. In particular:

- `tests/environments/test_morpion_scientific_launcher.py`,
  `test_morpion_continual_operator.py`, `test_morpion_continuous_runtime.py`,
  `test_morpion_continuous_races.py`, `test_morpion_continuous_lifecycle.py`.
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

## Version/status fence

At this document's source commit, the reliability stack was **not yet merged**:
PRs #128 (Growth backpressure/shutdown), #129 (terminal-idle worker), and #130
(bounded prepared-Dataset lease), in that dependency order, were open on stacked
branches with their respective successful CI runs. Their changes are **not**
silently described as present on this `main` snapshot. A running B4 Codex
experiment is also not part of this snapshot. Revisit the map when any of those
changes lands. No production behavior or historical experiment was modified by
this documentation review.
