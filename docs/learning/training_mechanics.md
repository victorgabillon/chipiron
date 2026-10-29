# Shared tensor mechanics and training contracts

`chipiron.learning.optimization` contains two small operations:
`forward_tensors(model, inputs)` forwards positional tensors, and
`backward_and_step(loss, optimizer)` backpropagates once and steps once.
Neither operation changes model mode, transfers devices, clears gradients,
extracts scalar metrics, checks finite loss, advances schedules or saves state.
The update accepts optional context managers for the existing regression timing
phases; synchronization happens only when those caller-provided scopes request it.
Without scopes the primitive introduces no device synchronization.

`chipiron.learning.supervised.regression` retains the public batch wrapper and
its returned statistics. Its phase names and order remain `batch_transfer`,
`zero_grad`, `forward`, `loss`, `backward`, `optimizer_step`, and
`metric_accumulation`. Evaluation retains its transfer, forward and metric phases.
`scalar_loss_from_regression_metrics` reconstructs L1/MSE `mean` using the number
of target elements, or `sum` from the corresponding accumulated sum. Nonempty
unreduced/unsupported criteria raise `TypeError`; empty aggregates retain zero.
Both chess scripts use this function through their existing private helper names.

## Training-path inventory

| Path | Optimizer and objective | Update ordering and measurements |
| --- | --- | --- |
| Generic regression batch | Caller-provided optimizer and criterion | Default `zero_grad()` before forward; one backward and step; detached pre-update predictions provide L1/MSE sums, scalar loss and counts. |
| Chess supervised script | SGD with configured momentum; L1 | Generic regression batches; StepLR events remain under the script's rolling-loss policy. Evaluation reconstructs the configured scalar loss. |
| Morpion eager, flat-cache, entity-cache, relational-cache and full-dataset training | Existing recipe's Adam and MSE | Existing orchestration calls the regression wrapper. Splits, tensor/cache representations, batch ordering, target transforms, selection metrics and bundle metadata remain with each recipe. |
| Morpion accepted-round continuation | Validated parent model and optional saved Adam state | Reuses the Morpion training path. Round acceptance, configuration of the new round and publication remain domain responsibilities. |
| Standalone `train_entity_value` | AdamW; elementwise MSE mean | Assign warmup/cosine LR, `zero_grad(set_to_none=True)`, forward, reject nonfinite loss, backward, step, increment progress. Existing logging and checkpoint boundaries retain their synchronization. |
| Toy `train_weighted_regressor` | Fresh Adam per invocation; `sum(weight * error²) / clamp(sum(weight), min=1e-12)` | Forward and construct loss **before** default `zero_grad()`. Every batch steps, including all-zero weights. Epoch metrics accumulate weighted errors from each batch's pre-update predictions and divide by total epoch weight (floor `1e-12`). |

Model architecture, data validation, targets, splits, augmentation, shuffling,
device requests, model mode, schedules, finite-loss policy, checkpoint schemas,
progress counters, bundle activation and scientific acceptance are domain-specific.
The common operations introduce no trainer lifecycle or callback registry.
Device resolution and synchronization use the shared
[Torch runtime policy](torch_devices.md).

## Continuation contracts

**Chess optimizer/scheduler restoration** constructs SGD against the supplied live
model and StepLR against that optimizer, then restores supported legacy state.
This preserves momentum, LR and scheduler progress for a compatible single
ordered parameter group. Given the same batches and explicit scheduler events,
the deterministic CPU update sequence continues. Batch position, RNG,
rolling-loss history and scheduler-decision history are not all persisted, so
this is not a whole-script replay guarantee. See the
[chess migration inventory](chess_supervised_learning_migration_inventory.md).

**Morpion accepted-round continuation** loads a validated parent bundle and may
reuse its Adam moments according to the existing continuation policy. New-round
configuration, LR and acceptance rules still apply. This is continuation between
accepted training rounds, not restoration of an arbitrary minibatch position.

**Standalone exact minibatch resume** uses the existing `checkpoint.pt`: model,
AdamW state, completed step, training time, identity and Torch/CUDA RNG states.
The step reconstructs deterministic epoch/sample order, hash-based augmentation
and LR schedule. With the same configuration and provenance, an interrupted run
continues dropout and warmup/cosine updates. CPU regression tests compare final
model, optimizer state, RNG and metrics across interruptions before, at and after
a nonzero warmup boundary. They do not assert hardware-independent bitwise GPU
reproducibility or equality of elapsed time.

**Toy training** starts a fresh Adam each invocation and has no optimizer-resume
checkpoint contract. Its epoch shuffling and caller-selected model mode remain
unchanged. Zero-weight batches still matter once Adam has accumulated momentum.

## Validation

Independent pre-refactor computations compare several L1/SGD(momentum), MSE/Adam,
relational MSE/AdamW and nonuniform weighted MSE/Adam updates. The tests compare
parameters, optimizer state, pre-update metrics and RNG consumption exactly on
CPU, including dropout, unequal batches and a short final batch. Additional tests
cover timing phase order, caller-owned gradient clearing/finite checks, scalar
mean/sum reconstruction, and absence of hidden scalar extraction or device sync.
Existing cache, full-dataset and accepted-round tests retain their own contracts.
Final integration follows the installed-wheel and strict-static gates in
[testing.md](../testing.md).
