# Chess supervised learning migration inventory

## Current entrypoints

### `src/chipiron/scripts/learn_nn_supervised/learn_nn_from_supervised_datasets.py`

- Purpose: supervised chess value-network training from persisted board/value
  datasets.
- Invocation: script-factory entrypoint for `ScriptType.LEARN_NN`; default
  options live next to the module in `exp_options.yaml`.
- Current tests:
  `tests/scripts/learn_nn_supervised/test_learn_nn_from_supervised_datasets.py`.
- Current trainer config/checkpoint dependency: imports `NNTrainerArgs` from
  `chipiron.learningprocesses.nn_trainer.factory` and checkpoint helpers from
  `chipiron.learningprocesses.nn_trainer.checkpoint_helpers`.
- `chipiron.learning` usage: directly uses
  `chipiron.learning.supervised.train_regression_batch` and
  `evaluate_regression_batch`; `FenAndValueData` implements the common
  `SupervisedBatch` protocol.

### `src/chipiron/scripts/learn_from_scratch_value_and_fixed_boards/learn_from_scratch_value_and_fixed_boards.py`

- Purpose: chess bootstrap-style script that refreshes fixed-board values with
  an evaluating player and is intended to train a neural evaluator from those
  values.
- Invocation: script-factory entrypoint for `ScriptType.LEARN_NN_FROM_SCRATCH`;
  default options live next to the module in `exp_options.yaml`.
- Current tests:
  `tests/scripts/learn_from_scratch_value_and_fixed_boards/test_learn_nn_from_scratch_and_fixed_boards.py`.
- Old trainer dependency: imports `NNTrainerArgs` from
  `chipiron.learningprocesses.nn_trainer.factory`.
- `chipiron.learning` usage: indirect through `FenAndValueDataSet`. The method
  `learn_model_some_steps` is currently empty, so there is no active train loop
  to migrate in this script yet.

### `src/chipiron/scripts/evaluate_models/evaluate_models.py`

- Purpose: evaluates saved chess neural-network model bundles on a stockfish
  evaluation dataset and writes an evaluation report.
- Invocation: script module and `__main__` block; default dataset path points at
  external data.
- Current tests: `tests/scripts/evaluate_models/test_evaluate_models.py`.
- Old trainer dependency: removed in PR19.
- `chipiron.learning` usage: directly uses
  `chipiron.learning.supervised.evaluate_regression_batch`.

### Chess neural evaluator runtime modules

- `src/chipiron/environments/chess/players/evaluators/boardevaluators/neural_networks/chess_model_bundle_evaluator.py`
  builds chess neural evaluators from resolved model bundles by adding the chess
  input converter before delegating to generic model-bundle runtime helpers.
- `src/chipiron/environments/chess/players/evaluators/boardevaluators/wirings/chess_eval_wiring.py`
  wires chess evaluator configuration into runtime evaluators.
- `src/chipiron/environments/chess/players/evaluators/boardevaluators/master_board_evaluator.py`
  participates in board-evaluator construction.
- These are evaluation/runtime entrypoints, not training loops. They should
  continue to work while training internals move.

## Current data abstractions

### `DataSetArgs`

- Path:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/datasets/datasets.py`.
- Purpose: dataset file names and preprocessing toggle for chess learning
  scripts.
- Classification: chess-learning configuration.
- Target/input: describes files only.

### `MyDataSet`

- Path:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/datasets/datasets.py`.
- Purpose: pandas-pickle dataset base class with optional preprocessing.
- Classification: generic-ish, but currently coupled to the chess dataset
  package and pandas pickle loading.
- Target/input: defined by subclasses.

### `FenAndValueData`

- Path:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/datasets/datasets.py`.
- Purpose: tensor sample or batch containing chess model input and target value.
- Classification: chess-specific concrete batch object that already satisfies
  `chipiron.learning.supervised.SupervisedBatch`.
- Target type/shape: `torch.Tensor`; value evaluator targets are scalar-like
  model outputs, usually transformed from a white-perspective value in
  `[-1.0, 1.0]`.
- Input tensor/model input type: `torch.Tensor` produced by the configured chess
  content-to-input converter.

### `custom_collate_fn_fen_and_value`

- Path:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/datasets/datasets.py`.
- Purpose: stacks `FenAndValueData` samples into a batched `FenAndValueData`.
- Classification: chess-specific adapter, but compatible with the common
  supervised batch protocol.
- Target/input: stacked target tensors and input tensors with batch dimension
  first.

### `FenAndValueDataSet`

- Path:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/datasets/datasets.py`.
- Purpose: converts rows containing FEN and value columns into
  `FenAndValueData`.
- Classification: chess-specific dataset.
- Target type/shape: transforms raw dataset value to a white value, then through
  the evaluator output converter into a model-output tensor.
- Input tensor/model input type: chess `ChessState` converted to model input by
  `ContentToInputFunction[ChessState]`.

### Input representation helpers

- Paths:
  `src/chipiron/environments/chess/players/evaluators/boardevaluators/neural_networks/input_converters/`
  and
  `src/chipiron/players/boardevaluators/neural_networks/input_converters/`.
- Purpose: convert chess positions into neural-network tensors and name the
  configured representation.
- Classification: chess-specific model input adapters.
- Current tests:
  `tests/players/boardevaluators/neural_networks/input_converters/test_representation.py`.

## Current trainer abstractions

### `NNPytorchTrainer`

- Path: `src/chipiron/learningprocesses/nn_trainer/nn_trainer.py`.
- Classification: removed in PR19.
- Notes: no active source callers remained after the supervised learning script
  and model evaluation script moved to common learning primitives.

### `compute_loss`

- Path: `src/chipiron/learningprocesses/nn_trainer/nn_trainer.py`.
- Classification: removed in PR14.
- Notes: this duplicated forward-plus-criterion helper became dead after PR13
  routed normal train/eval paths through common learning kernels.

### `compute_test_error_on_dataset`

- Path: `src/chipiron/learningprocesses/nn_trainer/nn_trainer.py`.
- Classification: removed in PR19.
- Notes: `evaluate_models.py` now aggregates
  `chipiron.learning.supervised.evaluate_regression_batch` metrics directly.

### `check_model_device`

- Path: `src/chipiron/learningprocesses/nn_trainer/nn_trainer.py`.
- Classification: removed in PR14.
- Notes: all remaining internal device lookup uses
  `chipiron.learning.module_device` directly.

### `NNTrainerArgs`

- Path: `src/chipiron/learningprocesses/nn_trainer/factory.py`.
- Classification: keep as a public supervised chess script configuration
  dataclass.
- Notes: contains optimizer, scheduler, batch-size, saving, and model input
  configuration.

### `create_nn_trainer`

- Path: `src/chipiron/learningprocesses/nn_trainer/factory.py`.
- Classification: removed in PR19.
- Notes: the active supervised chess script constructs optimizer and scheduler
  state directly while preserving `NNTrainerArgs` semantics.

### `safe_nn_architecture_save`, `safe_nn_param_save`, `safe_nn_trainer_save`

- Path: `src/chipiron/learningprocesses/nn_trainer/checkpoint_helpers.py`,
  re-exported from `src/chipiron/learningprocesses/nn_trainer/factory.py`.
- Classification: live compatibility APIs used by the supervised chess learning
  script.
- Notes: `safe_nn_param_save` now writes CPU-normalized state dicts through
  `chipiron.learning.state_dict_on_cpu`; architecture YAML and trainer optimizer
  or scheduler pickle saving remain unchanged.

## Current model/bundle/checkpoint paths

- `src/chipiron/players/boardevaluators/neural_networks/model_bundle_runtime.py`
  loads architecture args and builds generic NN evaluators from resolved model
  bundles and caller-provided input converters.
- `src/chipiron/environments/chess/players/evaluators/boardevaluators/neural_networks/chess_model_bundle_evaluator.py`
  adds the chess input converter and delegates to generic model-bundle runtime.
- `src/chipiron/environments/chess/players/evaluators/boardevaluators/neural_networks/chipiron_nn_args.py`
  reads and writes `chipiron_nn.yaml`, including version, `game_kind`, and input
  representation.
- `src/chipiron/players/boardevaluators/neural_networks/neural_net_board_eval_args.py`
  defines bundle-backed neural evaluator config.
- `src/chipiron/learningprocesses/nn_trainer/checkpoint_helpers.py` writes
  legacy checkpoint artifacts: architecture YAML, weights `.pt`, readable
  weights YAML, optimizer pickle, scheduler pickle, and training copies.
- Checked-in chess model data exists under
  `src/chipiron/data/players/board_evaluators/nn_pytorch/`.
- Current tests include `tests/models/test_chess_model_bundle_evaluator.py`,
  `tests/models/test_model_bundle_runtime.py`, and
  `tests/models/test_neural_net_board_eval_args.py`.

## Existing tests

- `tests/scripts/learn_nn_supervised/test_learn_nn_from_supervised_datasets.py`
  runs the supervised chess learning script on a tiny local dataset.
- `tests/scripts/learn_from_scratch_value_and_fixed_boards/test_learn_nn_from_scratch_and_fixed_boards.py`
  runs the scratch/fixed-board learning script path on a tiny local dataset.
- `tests/scripts/evaluate_models/test_evaluate_models.py` covers the model
  evaluation script.
- `tests/players/boardevaluators/neural_networks/input_converters/test_representation.py`
  covers chess board input representation behavior.
- `tests/models/test_chess_model_bundle_evaluator.py`,
  `tests/models/test_model_bundle_runtime.py`, and
  `tests/models/test_neural_net_board_eval_args.py` cover model-bundle and
  neural evaluator configuration paths.
- `tests/learning/test_chess_learning_inventory_imports.py` now pins import
  behavior for the main chess learning modules and verifies `FenAndValueData`
  can pass through the common supervised batch helpers.
- `tests/learning/test_chess_supervised_batch_common_kernel.py` verifies
  chess `FenAndValueData` batches work with common supervised train/eval
  kernels.
- `tests/learning/supervised/test_regression_quality_chess_like.py` now checks
  that common regression quality diagnostics handle chess-like value targets.

Obvious missing tests:

- A direct chess evaluation-loop test using the common
  `evaluate_regression_batch` primitive.
- Broader script-level checkpoint assertions once optional chess dependencies
  are available locally.
- A script-level assertion that PR13 preserves the tiny supervised chess
  learning outputs and metrics while changing the trainer internals.

## Migration plan

PR13:

- Keep public chess scripts and configuration stable.
- Route the active supervised train/eval paths in
  `learn_nn_from_supervised_datasets.py`, `NNPytorchTrainer`, and
  `evaluate_models.py` through `chipiron.learning.supervised` primitives.
- Preserve `NNTrainerArgs` as a compatibility shim unless replacing it is tiny
  and low risk.
- Add tests comparing legacy-compatible behavior against the common supervised
  regression kernel on a small chess-like batch.

PR14:

- Remove duplicated legacy trainer/shim code once chess and Morpion tests pass.
- Delete or replace `NNPytorchTrainer`, duplicated loss/evaluation helpers, and
  redundant device helpers.
- Decide whether legacy checkpoint helpers stay as chess-specific bundle
  writers or are replaced by common bundle/checkpoint utilities.

## PR13 progress

- `NNPytorchTrainer.train` now delegates normal supervised regression updates
  to `chipiron.learning.supervised.train_regression_batch`.
- `NNPytorchTrainer.test` and `compute_test_error_on_dataset` now delegate
  forward/metric collection to
  `chipiron.learning.supervised.evaluate_regression_batch`.
- At PR13, `check_model_device` remained as a compatibility helper delegating
  to `chipiron.learning.module_device`.
- `NNTrainerArgs`, `create_nn_trainer`, and legacy checkpoint helpers remain
  compatibility shims.
- `train_next_boards` remains a legacy special case because its targets are
  generated from the next board rather than supplied by a supervised batch.

## PR14 progress

- Removed dead duplicated helpers: `compute_loss` and `check_model_device`.
- Kept compatibility wrappers: `NNPytorchTrainer`, `compute_test_error_on_dataset`,
  `NNTrainerArgs`, `create_nn_trainer`, and legacy checkpoint helpers.
- `_loss_value_from_regression_sums` now explicitly supports only scalar
  `torch.nn.MSELoss` and `torch.nn.L1Loss` aggregate reconstruction.
- `_loss_value_from_regression_sums` now raises a clear `TypeError` for
  unreduced losses and unsupported criteria instead of falling through to a
  non-existent metric field.
- At the end of PR14, `train_next_boards` was still unreferenced by current
  source/test searches and remained pending PR15 removal review.
- Remaining PR15/PR16 work at that point: decide whether to remove
  `NNPytorchTrainer` itself and whether legacy checkpoint helpers should become
  a chess-specific bundle writer.

## PR15 progress

- `train_next_boards`: removed from `NNPytorchTrainer` because current
  source/test/doc searches found no live callers outside its own definition and
  migration notes.
- `safe_nn_param_save`: kept as a live compatibility API and updated to save
  CPU-normalized state dicts via `chipiron.learning.state_dict_on_cpu`.
- `safe_nn_architecture_save`: kept unchanged because the supervised chess
  learning script still calls it to write legacy architecture YAML.
- `safe_nn_trainer_save`: kept unchanged because the supervised chess learning
  script still calls it to write optimizer and scheduler artifacts.
- Remaining PR16 work: decide whether `NNPytorchTrainer` can be removed once
  script-level tests run with optional dependencies, and decide whether legacy
  checkpoint helpers should move into a chess-specific bundle writer.

## PR16 progress

- `NNPytorchTrainer` no longer imports Coral at runtime; the legacy trainer
  wrapper tests now run with plain `torch.nn.Linear` and
  `TensorSupervisedBatch`.
- Legacy checkpoint save helpers moved into the dependency-light
  `chipiron.learningprocesses.nn_trainer.checkpoint_helpers` module and remain
  re-exported from `factory.py` for existing script imports.
- `chipiron.players` and `chipiron.scripts` now lazy-load their public package
  exports so narrow learning/model imports do not pull move-selector factories,
  GUI code, or the parser stack.
- The tiny supervised chess learning script test now constructs
  `LearnNNScript` directly with prebuilt dataclass args and local observability
  stubs; it runs locally without `parsley`, `PySide6`, `mlflow`, or
  `torchinfo`.
- The chess script/model validation selection improved locally from all skipped
  to `6 passed, 2 skipped`. The remaining skips are the scratch/fixed-board
  script test and the YAML board-evaluator config parsing test, both because
  `parsley` is not installed in this environment.
- Remaining PR17 blocker: decide whether the still-public
  `NNPytorchTrainer`, `NNTrainerArgs`, `create_nn_trainer`, and checkpoint
  helper APIs can be removed or moved once parser-backed script coverage is
  available in an environment with `parsley`.

## PR17 progress

- Parsley canonical import path: `parsley`. The PyPI distribution declared in
  `pyproject.toml` is `parsley-coco>=0.1.34`, which exposes the `parsley`
  import package.
- Dependency declaration status: Parsley is already a runtime dependency via
  `parsley-coco`; local lightweight environments may still miss it if Chipiron
  was installed without project dependencies.
- `PlayerConfigTag` no longer imports Parsley just to expose enum values;
  Parsley is imported only by the parser-backed `get_players_args()` method.
- The scratch/fixed-board learning script no longer imports the parser-backed
  `Script` class at module import time; it follows the supervised script path
  and derives its default output folder directly from `output_root_path_str()`.
- Removed unnecessary parser and GUI skips from the scratch/fixed-board script
  test by constructing `LearnNNFromScratchScriptArgs` directly and using a tiny
  base-script adapter.
- Split `test_neural_net_board_eval_args.py` into direct dataclass/model-bundle
  behavior, which runs without Parsley, and YAML parser behavior, which skips
  precisely when `parsley` is absent.
- Added `test_parsley_dependency_contract.py` to document that parser-backed
  Chipiron config uses `import parsley` even though the dependency distribution
  is named `parsley-coco`.
- Parser-backed tests pass in the anemone/dev environment where `parsley-coco`
  is installed and exposes `import parsley`. Remaining skips only occur in
  lightweight environments that do not install Chipiron runtime dependencies.

## PR18 progress

- The active supervised chess learning script no longer constructs
  `NNPytorchTrainer` through `create_nn_trainer`.
- The script now uses `chipiron.learning.supervised.train_regression_batch` and
  `evaluate_regression_batch` directly, while preserving `NNTrainerArgs` as the
  public script configuration dataclass.
- Optimizer and scheduler construction in the script preserves the legacy
  `NNTrainerArgs` SGD and `StepLR` behavior, including reloading existing
  optimizer/scheduler checkpoints.
- Checkpoint helpers remain live compatibility APIs.
- `safe_nn_trainer_save` now accepts any optimizer/scheduler holder, so direct
  training code can save legacy optimizer and scheduler checkpoints without
  constructing `NNPytorchTrainer`.

## PR19 progress

- Removed `NNPytorchTrainer` and the old dependency-light
  `nn_trainer.py` wrapper module because no active source callers remained.
- Removed `create_nn_trainer` from the legacy factory module because the active
  supervised chess script now uses common learning directly.
- Migrated `evaluate_models.py` away from the legacy dataset evaluator; it now
  aggregates `evaluate_regression_batch` metrics directly.
- Kept `NNTrainerArgs` as the public supervised chess script config dataclass.
- Kept checkpoint helpers and factory re-exports as compatibility APIs.
- Remaining PR20 work: decide whether to rename or move `NNTrainerArgs` out of
  `learningprocesses.nn_trainer.factory`; decide whether checkpoint helpers
  should move to a chess-specific training/checkpoint module; remove stale
  compatibility re-exports if desired.

## Risk notes

- Several chess learning modules are reached through the dynamic script factory
  and YAML/dataclass parsing, so migration tests should include script-factory
  coverage and not only direct function calls.
- Importing some chess learning modules pulls in optional GUI/script/runtime
  dependencies such as `parsley`, `PySide6`, `coral`, `atomheart`, and
  `anemone`; tests should skip only when those dependencies are genuinely
  absent.
- `learn_from_scratch_value_and_fixed_boards.py` constructs search/player and
  Syzygy-related runtime objects even though `learn_model_some_steps` is empty.
  This path should be treated carefully in PR13.
- `evaluate_models.py` defaults to external dataset and model-bundle paths, so
  tests should continue to patch or isolate filesystem/model access.
