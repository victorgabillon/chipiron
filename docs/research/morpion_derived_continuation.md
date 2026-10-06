# Explicit derived continuation from historical Morpion checkpoints

This experiment derives a new workspace from an actual runtime checkpoint. It is
**not an exact historical continuation**. Preparation and dry-run never construct
a search runtime or start Growth, Dataset, Training or Reevaluation. Historical
files are read-only. Real execution requires a separate explicit confirmation.

## Representation compatibility

The gen38 bundle used `graph_tokens_v1` with 26 ordered features. The historical
`type_value` column and explicit VALUE input row are retained, as is Coral's
additional learned value token. These two tokens are intentionally both present.
The validity feature remains column 25; Coral removes it before the 25-wide input
projection. No modern entity representation or extra geometry feature is substituted.

The tokenizer is preserved from Chipiron's parent of commit
`9a94332a0afab79d00321dd12cf9a248ee94579c`, with only an inference-protocol adapter.
Compatibility lives under `neural_networks/legacy_graph/`. Config preparation wraps
all nine original fields in an explicit `legacy_graph_tokens` block containing
`representation: graph_tokens_v1`. Values are unchanged:

| Historical field | Explicit compatibility field | Source value |
|---|---|---|
| graph_max_tokens | legacy_graph_tokens.graph_max_tokens | 1536 |
| graph_input_feature_dim | legacy_graph_tokens.graph_input_feature_dim | 26 |
| graph_d_model | legacy_graph_tokens.graph_d_model | 64 |
| graph_n_head | legacy_graph_tokens.graph_n_head | 4 |
| graph_n_layer | legacy_graph_tokens.graph_n_layer | 2 |
| graph_dim_feedforward | legacy_graph_tokens.graph_dim_feedforward | 256 |
| graph_dropout_ratio | legacy_graph_tokens.graph_dropout_ratio | 0.0 |
| graph_pooling | legacy_graph_tokens.graph_pooling | value_token |
| graph_output_tanh | legacy_graph_tokens.graph_output_tanh | false |

Unknown graph fields, incomplete blocks and incompatible dimensions are rejected.
Original bundles are recognized by the complete historical argument schema and
validated against their `graph_tokens_v1` manifest. Missing/incompatible weights
fail; they cannot cause fresh random model initialization. Newly saved legacy
bundles retain the explicit block and representation. Legacy tensor caches use a
separate filename and schema, so modern cached tokens cannot be reused accidentally.

Modern configuration defaults, tokenizer, models and selection remain unchanged.
The golden fixture `tests/data/morpion_legacy_graph_v1.json` was generated using
original Chipiron tokenization and original Coral code at `791168a`; it records
source hashes, fixed states, token hashes and independent forward outputs. Nine
cases cover three states and limits of 2, 32 and 1536 tokens. An external read-only
audit also compared original and compatibility inference using the actual gen38
weights on three fixed states: maximum absolute CPU difference was 0.0.

## Preparation and provenance

Use the dedicated module (the normal operator launcher is unchanged):

```bash
python -m chipiron.environments.morpion.bootstrap.derived.cli derive \
  --source-work-dir /path/to/historical \
  --target-work-dir /path/to/new-derived \
  --code-root /path/to/validated-checkout \
  --checkpoint-generation 38 --expected-node-count 242680 \
  --reset-rng --search-seed 0 --rollout-seed 0 --training-seed 0 \
  --enable-legacy-graph-tokens-v1 \
  --max-growth-steps-per-cycle 2 --max-generations 10 \
  --dry-run
```

Replace `--dry-run` with `--prepare-only` to create the workspace without executing
any scientific stage. Existing targets, overlapping source/target paths, escaped
paths and symlinked inputs are refused. Interrupted preparation remains visibly
incomplete and cannot launch. Files are copied, never hard-linked. Copy hashes
and the unchanged source inventory are verified.

The source config is authoritative. Only the growth-step limit changes from 10 to
2, plus the audited representational wrapping of legacy fields. All configured
models, epochs, rollout settings, branch limit, dataset target rules, validation
split, selection criterion and evaluator-update policy are retained. Canonical
serialization makes previously implicit current defaults explicit where needed.
The complete additive training export is copied so earlier node shards remain
available, but runtime restoration **always** uses the runtime checkpoint.

`derived_experiment_provenance.json` includes source paths, generation, cycle,
node count, external active-model identity, all copied-source SHA-256 hashes,
checkpoint manifest identity, timestamp, code SHA, package fingerprint, dependency
versions, exact effective configuration, migration map, RNG policy, scheduler and
drain policies, intended generation bound and prepared restore-file checksums.
The experiment identity excludes timestamps and destination paths, making it
repeatable from the same scientific inputs and reset seeds.

## RNG and model generations

The copied checkpoint metadata receives `random.Random(search_seed).getstate()`
and a **separate** `random.Random(rollout_seed).getstate()`. No node, selector,
expansion or state-payload shard changes. Rollout-disabled checkpoints retain no
rollout RNG. Subsequent checkpoint saves/restores use the released Anemone RNG
contract. Identical seeds mean identical initial states, not a shared RNG object.

Before each complete training stage, Python, NumPy and Torch use
`training_seed + local_generation`; CUDA is seeded when available. Retraining an
interrupted generation starts from the same initialization and evaluator order.
The persisted validation seed and split policy remain unchanged. This fixes
initialization; cross-device/version GPU bitwise reproducibility is not promised.

The copied active model retains `generation=430`, `source=external_seed`,
`source_generation=430` and no local trained generation. Its metadata is bound to
the derived experiment identity. Only this explicitly matched provenance makes
the publication guard use the starting local tree generation (38). Ordinary
external seeds keep the existing guard. After training39, the existing evaluator
selection result can publish local39 normally. Original external provenance and
weights remain auditable; no fake renumbering or forced evaluator is used.

## Sequential generations and final drain

One foreground process, protected by an exclusive workspace lock, executes:

1. One bounded Growth cycle, requiring checkpoint restore and forcing a save at
   the generation barrier while preserving the configured normal save thresholds.
2. Dataset for that exact generation.
3. Training of the complete configured evaluator family and normal selection.
4. Reevaluation of the complete generation in one patch, avoiding singleton
   patch contention. This changes orchestration/window size, not evaluation math.
5. Patch application without Growth; save and flush a new checkpoint, then
   atomically commit its run-state pointer and receipt, then acknowledge the patch.

The next generation cannot begin before all five stages finish. There is no
latest-first generation selection. The final drain saves generation48's runtime
without creating generation49. The final training export remains the pre-drain
training snapshot; the committed runtime checkpoint is the post-drain search state.

The journal `derived_orchestration.json` records stage starts, finishes, artifacts,
failures and completed generations. A failed stage remains incomplete. Already
committed Growth/training is detected; incomplete training is retried under the
exclusive derived claim. A crash before drain commit restores the earlier runtime;
a crash after commit acknowledges the matching patch without applying it again.
Uncommitted drain checkpoints remain available for diagnosis. No orphan stage
workers are created. Memory guards stop with recoverable artifacts. A twelve-hour
budget is checked between stages; a running stage may exceed that boundary, so
this is not a hard wall-clock kill timer. Actual completion time is not promised.

The first generations record actual steps and nodes/step through the existing
observability implementation; steps are never automatically adjusted. Ten cycles
is a maximum, not a guarantee if the search, memory guard or another stage stops.

## Launch gate and observability

The generated `run_derived_bootstrap.sh --dry-run` prints provenance, source
checkpoint, pinned code and dependency identities, configuration and plan.
`--launch --confirm-derived-experiment` is the deliberately separate execution
interface. Preparation does not execute it. A changed configuration, mismatching
code/environment, missing checkpoint/model, incomplete preparation or missing
PR #70 instrumentation prevents launch.

The draft derived-continuation PR is temporarily based on PR #70
(`perf/bootstrap-pipeline-observability`). Its existing commits are retained
without rewriting its branch. PR #70 provides the stage/RSS/CUDA observations;
this implementation does not reimplement them. The sequential journal additionally gives generation barrier
and critical-path timing evidence. Keep the derived-continuation PR in draft;
launch requires Victor's explicit later approval even when validation is green.
