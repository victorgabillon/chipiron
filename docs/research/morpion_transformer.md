# Morpion Solitaire Transformer research

## Objective

The current neural evaluator predicts **V(state)** from the current Morpion board.
`MorpionRegressor` wraps a Coral value network; `build_morpion_regressor` constructs
it from `MorpionRegressorArgs`. This is a state-value estimate, not a policy or an
optimal action-value oracle.

## Retained architecture and compatibility

`MorpionEntityTokenConverter` produces GLOBAL, DOT, EDGE and legal MOVE tokens.
The GLOBAL token represents board-level information; it is distinct from any
internal value/readout mechanism in the network.
`MorpionRelationalEntityTokenConverter` adds typed relation triples describing
move paths, edge endpoints and moves sharing a new dot.

The useful research configuration is
`model_kind="relation_biased_entity_token_transformer_value_net"`,
`entity_n_layer=2` and `relation_bias_scale=0.25`, backed by Coral's
`RelationBiasedEntityTokenTransformerValueNet`. The model retains a scalar
state-value readout. The existing conservative representation is
`global_geometry_features="none"`, `edge_token_mode="drawn_only"` and
`latent_window_move_features="none"`.

**Research preference is not a default change.** Existing public argument and
bootstrap-preset defaults remain unchanged, including `relation_bias_scale=1.0`.
The ordinary entity-token Transformer and handcrafted-feature evaluators also
remain supported.

The optional `latent_window_move_features="promoted_only"` representation appends
`promoted_latent_window_count` to MOVE tokens. Its optimized implementation builds
one state-level window index and computes action counts in a batch. It does not
construct successor states to obtain this feature. A small scalar reference and
parity tests protect its semantics; normal conversion uses the batched path.

Baseline token features retain their ordering and width (25), and promoted-only
features retain width 26. Token ordering, validity position, relation IDs,
`MORPION_ENTITY_RELATION_SCHEMA` and persisted representation names remain
unchanged. Bundle construction/loading, target transforms, dataset adapters,
padding/batching and training caches remain part of the reusable implementation.
Historical opt-in geometry/edge/blocked-feature settings remain readable where
they are part of existing converter or bundle contracts; their retention is not
an endorsement of the rejected experiments.

## Completed representation experiments

| Experiment | Conclusion |
| --- | --- |
| Relation-biased attention | Useful; retain. |
| Relation-bias scale around 0.25 | Useful research setting; retain configurable scaling. |
| Depth 2 versus tested depths 3/4 | Retain depth 2. |
| Global normalization-extent feature | No robust benefit; reject for the chosen architecture. |
| Prospective/free edge tokens | Regression; reject for the chosen architecture. |
| Promoted latent-window count on MOVE tokens | Predictive signal and improved supervised value MSE; retain as an optional feature. |
| Extra blocked-window features | Not justified. |
| `majority_top1` ensemble aggregation | Did not improve gameplay; reject as deployed default. |

These conclusions describe the tested settings, not universal architecture
claims. Lower supervised MSE did not always translate into better gameplay.
Future architecture decisions should consider both prediction metrics and
separately authorized gameplay evaluation.

## Future policy work: postponed

The intended direction remains one current-state representation, one
relation-biased Transformer and **one score per legal MOVE token**. This could
support direct action ranking without separately evaluating every successor.
No MOVE head is introduced by this cleanup.

The obstacle is supervision. The existing dataset has sparse labels for
alternative actions. Fresh independent validation/test source groups were
successfully generated, but the cheap search-derived teacher was repeatable
for a fixed seed and sensitive to changes of seed. Confidence filtering selected
more reproducible comparisons without establishing a sufficiently broad and
consistent protocol for large-scale label generation.

The completed PR 5F5 audit analyzed nine complete historical training states.
For independent four-seed means, retaining the largest nominal 50% of selector
margins retained 49.6% of non-tied comparisons, with 91.2% pair-weighted and 88.8%
state-weighted sign agreement. SNR ≥ 2 retained 43.6%, with 93.8% and 88.1%
agreement respectively. No high-confidence top-action reversal was observed at
four or eight seeds. The recommendation remained `need_stronger_teacher`.
These descriptive results do not establish action-value accuracy or held-out
policy performance; the historical action pairs did not share seed IDs.

Teacher targets had the semantics `teacher_target_value(T(state, action))`,
with no added `+1`: search-derived successor total-score estimates. They were
not true Q values or ground truth. The interrupted dense generation and
historical stability study should not be mistaken for completed datasets.
The bounded teacher-frontier implementation did not produce a real experiment
result during its implementation here.

**Direct MOVE-head work is postponed until trustworthy action-level supervision
is available.** Existing external artifacts, model bundles and sealed test groups
are preserved. This cleanup launches no new label generation.

## Evaluator-v1 status

Evaluator v1 is selected, trained and integrated on the dedicated goal branch.
It retains the simple depth-2, width-64 shared-projection baseline and uses the
predeclared seed-0 bundle from 100k-row training. Normal configuration and
verification are documented below. No further architecture experiment is
justified by this roadmap; policy/action work remains postponed.

## Source-tree policy

Keep reusable conversion, relation, model, serialization and training code, plus
small diagnostics used by those components. Completed study orchestration,
report builders, manifests, recommendation logic and experiment-only tests do
not belong permanently in the active package. Public schemas and retained
model numerical behavior take precedence over removing a few compatibility
branches. Historical implementations belong in version history; untracked
research work must be preserved before removal because it is not in Git history.

## Evaluator-v1 goal — checkpoint 1

Work began from cleanup commit `f545304c` on the isolated
`feat/morpion-transformer-evaluator-v1` branch. The user's dirty Chipiron and
Coral workspaces remain untouched. Some historical reusable features were never
committed, so the isolated baseline needed explicit relation scaling and the
Coral correction that preserves additive attention-mask magnitudes during
inference. Three historical scale-0.25 baseline bundles now give bit-identical
predictions on the checked states in both workspaces. The default scale remains
1.0; the controlled training recipe explicitly selects 0.25.

The isolated branch currently implements the original 25-feature representation.
Promoted-only and standardized-target support present in the user's uncommitted
workspace must be reconciled before final evaluator-v1 integration; this is an
inherited compatibility gap, not a claim that those features have been removed
from the user's workspace.

D4 augmentation transforms complete states around the fixed start center
(-1/2, -1/2) using Atomheart point/action conventions, then rebuilds tokens and
relations. This handles direction permutations, canonical action reversal and
missing-dot/relation slots. Each training row chooses one deterministic D4
element per epoch from an independent hash of seed, epoch and source row index.
Targets, training shuffle and the validation split remain unchanged. Validation
MSE uses canonical states; an additional 128-state diagnostic measures all eight
views without averaging them into the primary predictions.

`EntityValueTrainingConfig` / `train_entity_value` provide the reusable training
path. A resumable disk cache stores each original state and target once, avoiding
repeated checkpoint decoding without materializing eight copies of the dataset.
The external runner supplies progress/ETA, source snapshots, atomic results and
a cumulative 12-hour active-time limit that persists across resumes.

| Question | Variants | Dataset / split | Seeds | Result / decision |
| --- | --- | --- | --- | --- |
| Does D4 training augmentation improve V(state)? | None versus random D4; all other settings identical | First 50,000 rows of `generation_000038.jsonl`; validation indices 4, 9, …, 49,999 | 0, 1, 2 | Complete: D4 mean MSE +20.54%, all three seeds worse. Retain no augmentation. |

Both arms use depth 2, width 64, four heads, FFN 256, zero dropout, a VALUE-token
readout, relation scale 0.25, raw targets/MSE, batch size 8 and 20 epochs. AdamW
uses weight decay 0.01, learning rate 0.001, 5% linear warmup and cosine decay to
0.01 of the initial learning rate. Both controls are trained through the same
new input pipeline; historical scores are not reused as paired controls.

Predeclared decisions favor no augmentation for neutral/inconsistent results.
A relative mean MSE improvement of at least 2% needs at least two improving
seeds and no seed regression greater than 10%. A 1–2% gain can retain this
training-only change with the same seed checks and non-worse mean MAE. Individual
seeds, MAE, R², Pearson, prediction/target spread and the three-model arithmetic
ensemble are reported for review. These are internal, repeatedly inspected
holdouts from one historical provenance group, not independent generalization
tests. Sealed policy/action test groups are outside this experiment.

Artifacts and the exact launch script live outside Git under
`/home/pompote/oldata/victor/morpion_runs/generic_linoo_fresh_with_bigrun_models_v1/evaluator_v1/`.
The planned result directory is `d4_50k_20epochs_3seeds_20260912_v1`.
Bounded pilots were used only for runtime estimation. The representative estimate
is approximately 7.5 hours, or 11.2 hours with a 50% margin; it is not scientific
evidence for either architecture. The complete comparison used 5.31 charged active hours, including preparation
and recovery, within the 12-hour cap. Token encoders use no augmentation.

The first user run completed baseline seed 0's 100,000 optimizer steps in
50.1 minutes, then exhausted file descriptors during canonical validation.
Retaining worker-backed target tensors kept a shared-memory descriptor open
for every validation batch. Validation now clones those targets into local
storage; training, targets and metric calculations are unchanged. A regression
test with two spawned workers and a 128-descriptor limit fails on the original
code and passes with the fix. The saved checkpoint subsequently completed
all 10,000 validation rows and the D4 diagnostic, with MSE 31.372818.

The recovery performs no training updates. Original checkpoint/config/source
artifacts are SHA-verified in the run's `repairs/validation_fd_20260912/`
directory, which also records the exact source patch and provenance migration.
The resumed user run completed all five remaining models while retaining the
original cumulative runtime budget.

The completed September 13 audit verified source/dataset hashes and snapshots,
all six configurations, 100,000 optimizer steps per model, every canonical target
against the original JSONL, recomputed metrics and ensembles, checkpoint/bundle
weight equality, and 24 saved prediction rows per model through normal loading.
The audit is recorded externally as `evaluator_v1/d4_scientific_audit.json`.

| Training | Seed | MSE | MAE | R² | Pearson | Prediction/target std | Training min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No augmentation | 0 | 31.372818 | 3.638631 | 0.433512 | 0.658439 | 0.663487 | 50.12 |
| No augmentation | 1 | 32.586948 | 3.762805 | 0.411589 | 0.643438 | 0.655530 | 53.82 |
| No augmentation | 2 | 28.574371 | 3.467807 | 0.484043 | 0.696515 | 0.729337 | 46.83 |
| Random D4 | 0 | 39.396843 | 4.117097 | 0.288626 | 0.537427 | 0.523313 | 53.74 |
| Random D4 | 1 | 37.066063 | 3.971586 | 0.330712 | 0.576035 | 0.545646 | 51.77 |
| Random D4 | 2 | 35.077145 | 3.906462 | 0.366625 | 0.605522 | 0.603393 | 46.60 |
| No augmentation mean | — | 30.844713 | 3.623081 | 0.443048 | 0.666131 | 0.682785 | 50.26 |
| Random D4 mean | — | 37.180017 | 3.998382 | 0.328654 | 0.572995 | 0.557451 | 50.70 |

Both variants contain 106,049 parameters. Paired D4-minus-baseline MSE differences
are +8.024025 (+25.58%), +4.479115 (+13.75%), and +6.502773 (+22.76%). The mean
difference is +6.335304 (+20.54%): zero seeds improve and three worsen. Arithmetic
three-seed ensemble MSE is 28.325850 without augmentation versus 34.925835 with D4.
Mean within-state eight-view population standard deviation drops from 1.587817
to 1.012231 (36.25% lower). Better symmetry consistency therefore does not translate
into better fit to the canonical search-derived targets in this controlled run.

**Decision: reject random D4 augmentation for evaluator v1; retain canonical
training.** This follows the predeclared regression rule. It does not establish
that geometric invariance is wrong or explain the cause of the regression;
orientation effects in finite-search targets and optimization remain possible
explanations, not findings of this experiment. No further augmentation sweep is
planned. The next comparison changes only the shared versus type-specific input
projection, retaining the baseline data, targets, optimizer and depth-2 model.

## Evaluator-v1 goal — checkpoint 2

The next candidate replaces the shared 24-to-64 input projection with four
linear projections, selected by the existing GLOBAL/DOT/EDGE/MOVE one-hot bits.
Each projection receives the same features. The separate learned VALUE token,
token ordering, relation triples, width, depth, relation scale and readout remain
unchanged. Common Transformer/readout weights retain the baseline initialization
under each seed; only the replacement projections have new parameters.

The candidate is serialized as `entity_input_encoder=type_specific_linear_v1`.
Absence of that field means `shared`, and shared bundle serialization retains
the previous format. The token/relation schemas remain v1. Unknown encoder
versions are rejected. Parameters increase from 106,049 to 110,849 (+4,800;
4.53%). Padding remains inert and real tokens require exactly one type bit.

The surviving configuration has no augmentation. The completed checkpoint-1
controls can be reused: all three seeds match the archived source exactly for
input batches, initial weights, RNG state, predictions, gradients, the first
AdamW update and checked historical bundle predictions. The shared execution
path is unchanged. This source-bound verification and hashes of each reused
artifact are recorded in the new experiment provenance.

Only the three candidate seeds require training: first 50,000 original rows,
20 epochs, seeds 0/1/2, the same modulo-5 canonical validation split, raw targets,
AdamW schedule and batch size. A bounded 512-row representative pilot measured
approximately 2.9 hours for all three runs, or 4.4 hours with 50% margin. Its
scores are runtime diagnostics only and do not support an architecture decision.

The predeclared adoption rule requires at least 2% mean paired MSE improvement,
at least two improving seeds, and no seed more than 10% worse. Neutral results
and gains below 2% retain the simpler shared projection. The report includes
individual/mean metrics, paired differences, parameter/training costs and the
three-seed arithmetic ensembles.

| Question | Control | Candidate | Status |
| --- | --- | --- | --- |
| Do type-specific linear input encoders improve V(state)? | Audited checkpoint-1 shared projection, no augmentation | Four linear projections, no augmentation | Complete: mean MSE +5.06%; one slightly improved seed, two worse. Retain shared projection. |

External command: `evaluator_v1/run_encoder_50k.sh`. Output directory:
`evaluator_v1/encoder_50k_20epochs_3seeds_20260913_v1` under the existing run root.
Training progress and ETA are logged every 15 seconds, checkpoints every minute,
and the cumulative active-time maximum is 12 hours across resumes. The user-run candidate comparison completed in 2.35 charged active hours.
Width screening retains the shared projection and no augmentation.

The checkpoint-2 audit verifies the dataset/source/cache hashes and source
snapshot, every reused-control hash, all 10,000 canonical targets against the
original JSONL, configuration and 100,000 optimizer steps per model, recomputed
metrics/ensembles, checkpoint/bundle weights and 24 checked predictions per model.
The audit is external: `evaluator_v1/encoder_scientific_audit.json`.

| Input encoder | Seed | MSE | MAE | R² | Pearson | Prediction/target std | Training min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared | 0 | 31.372818 | 3.638631 | 0.433512 | 0.658439 | 0.663487 | 50.12 |
| Shared | 1 | 32.586948 | 3.762805 | 0.411589 | 0.643438 | 0.655530 | 53.82 |
| Shared | 2 | 28.574371 | 3.467807 | 0.484043 | 0.696515 | 0.729337 | 46.83 |
| Type-specific | 0 | 32.990208 | 3.715034 | 0.404308 | 0.636147 | 0.655237 | 43.87 |
| Type-specific | 1 | 32.444607 | 3.705412 | 0.414160 | 0.644402 | 0.665246 | 48.07 |
| Type-specific | 2 | 31.783007 | 3.665557 | 0.426106 | 0.652814 | 0.645167 | 46.35 |
| Shared mean | — | 30.844713 | 3.623081 | 0.443048 | 0.666131 | 0.682785 | 50.26 |
| Type-specific mean | — | 32.405940 | 3.695334 | 0.414858 | 0.644454 | 0.655217 | 46.10 |

Paired candidate-minus-shared MSE differences are +1.617390 (+5.16%), -0.142342
(-0.44%), and +3.208635 (+11.23%). Mean difference is +1.561228 (+5.06%): one seed
improves slightly and two worsen. Three-seed ensemble MSE worsens from 28.325850
to 30.347506. The candidate adds 4,800 parameters (+4.53%). Training durations
come from separate invocations and do not establish a controlled speed advantage.

**Decision: reject type-specific linear encoders for evaluator v1; retain the
shared projection.** The candidate fails the predeclared accuracy/consistency
rule and adds inference complexity. This conclusion concerns this simple tested
encoder and recipe, not every possible type-specific architecture. No additional
encoder sweep is planned. Checkpoint 3 screens the prescribed depth-2 widths
32/64/96 with no augmentation, shared projection and relation scale 0.25.

## Evaluator-v1 goal — checkpoint 3 (complete)

Screen exactly the prescribed widths using seed 0, the first 50,000 original
rows and 20 epochs. All candidates retain depth 2, four heads, FFN width four
times model width, relation scale 0.25, shared projection, no augmentation,
VALUE-token readout and the established optimizer/split/targets. The existing
audited width-64 seed-0 run is reused. Source bytes are unchanged since the
checkpoint-2 parity proof; only the checkpoint commit IDs advanced.

| Width | FFN | Parameters | Screening work |
| ---: | ---: | ---: | --- |
| 32 | 128 | 27,457 | New seed-0 run |
| 64 | 256 | 106,049 | Reuse completed seed-0 control |
| 96 | 384 | 235,841 | New seed-0 run |

The existing implementation supports these widths without production-code
changes. Cheap checks verify parameter counts, finite forward/backward execution,
unchanged input/relation tensors and exact bundle round trips at all three sizes.

Predeclare a meaningful screening improvement as at least 2% lower MSE than
width 64. If neither challenger qualifies, retain 64 and stop the width study.
Otherwise confirm only the strongest qualifying challenger with seeds 0/1/2,
preferring the smaller qualifying width when within 1% of the best screening
MSE. Screening alone never adopts a challenger. The final confirmation applies
the requested preference for the smallest model within about 1% of the best
robust validation MSE; a larger model needs a clear reproducible gain.

The screening report contains MSE, MAE, R², Pearson, prediction/target standard
deviation, paired differences, parameters and training duration. A three-model
ensemble is unavailable at this one-seed stage and is only relevant if
confirmation proceeds. Runtime pilots do not select a width or shorten the
scientific epoch budget. No extra widths or depth/readout changes are planned.

The user completed both new runs: 100,000 optimizer steps each, with 10,000
canonical validation predictions. Total charged active time was 1.658 hours,
below the 12-hour cumulative cap. Artifacts are under
`evaluator_v1/width_screen_50k_20epochs_seed0_20260913_v1`; the independent
`evaluator_v1/width_scientific_audit.json` verifies source/cache/data hashes,
the source archive, control reuse, raw targets, split, checkpoint/bundle weights,
all saved metrics and 24 predictions per model through the normal bundle loader.

| Width / FFN | MSE | MAE | R² | Pearson | Pred. std / target std | Parameters | Training min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 128 | 31.184700 | 3.639690 | 0.436909 | 0.661410 | 0.684134 | 27,457 | 50.36 |
| 64 / 256 (reused) | 31.372818 | 3.638631 | 0.433512 | 0.658439 | 0.663487 | 106,049 | 50.12 |
| 96 / 384 | 33.227608 | 3.773004 | 0.400021 | 0.632909 | 0.656232 | 235,841 | 47.09 |

Paired seed-0 MSE differences against width 64 are -0.188118 (-0.60%) for
width 32 and +1.854790 (+5.91%) for width 96. Neither meets the predeclared 2%
screening threshold. **Retain width 64; no confirmation runs.** The smaller
model's neutral single-seed result does not establish robust equivalence, and
the staged protocol explicitly stops here. This is not evidence that width 32
cannot match the baseline across seeds. Timing comes from separate invocations
and does not establish an inference or training speed ranking.

## Evaluator-v1 goal — checkpoints 4–5 (architecture selected)

**Skip type-aware readout.** These experiments provide no concrete evidence that
the VALUE-token readout is a bottleneck: the wider encoder regressed, and no
token-information or residual diagnostic demonstrates a readout limitation.
This is a reason to stop the planned ablation sequence, not a claim that all
alternative readouts would fail.

| Component / question | Tested candidate | Evidence on first 50k, historical 80/20 split | Decision |
| --- | --- | --- | --- |
| D4 training augmentation | Random rooted D4, unchanged targets | 20 epochs, seeds 0/1/2; mean MSE +20.54%, all worse | No augmentation |
| Token input encoding | Four type-specific linear projections | 20 epochs, seeds 0/1/2; mean MSE +5.06%, 2/3 worse | Shared linear projection |
| Capacity at depth 2 | Widths 32 and 96, FFN = 4×width | 20 epochs, seed 0; MSE -0.60% / +5.91%; neither qualifies | Width 64 / FFN 256 |
| Readout | No candidate justified | No concrete evidence of a readout bottleneck | Existing VALUE-token readout |

Evaluator v1 is the existing simple baseline: `morpion_entity_tokens_v1`,
GLOBAL/DOT/EDGE/legal-MOVE tokens with the existing ordered 25 features and
validity in column 24; at most 1,536 input tokens. No optional geometry,
prospective-edge, or promoted features enter this selection. The shared linear
projection consumes the 24 non-validity features. A distinct learned VALUE
token supplies the scalar readout. Model width is 64, with four attention heads,
two Transformer layers, FFN 256, zero dropout and no output tanh. Relations use
the existing 16-type `morpion_entity_relations_v1` schema and additive learned
bias scaled by 0.25. Total parameters: **106,049**.

Training uses canonical states without D4 augmentation; raw existing search
values, MSE, AdamW with learning rate 0.001 and weight decay 0.01, 5% linear
warmup then cosine decay to minimum LR ratio 0.01, batch 8 and 20 epochs.
Validation uses original rows 4, 9, 14, ...; no target standardization or
symmetry averaging. Public legacy model defaults remain unchanged.

The surviving baseline's three-seed 50k mean MSE is **30.844713**, with individual
MSEs 31.372818, 32.586948 and 28.574371; arithmetic ensemble MSE is **28.325850**.
The ensemble is a reporting diagnostic, not an adopted deployment default.
Architecture selection completed at this checkpoint. The larger-data training
and normal integration described below subsequently completed the goal. All rows share one historical provenance group, so the
internal holdout does not establish independent generalization or gameplay gains.

## Evaluator-v1 goal — checkpoint 6 (complete)

Train the selected baseline on the first **100,000 existing rows**, with 20
epochs and seeds 0/1/2. Each seed uses 80,000 training rows and 20,000 canonical
validation rows (indices 4, 9, ..., 99,999), for 200,000 optimizer steps. All
architecture and optimizer settings above remain fixed. No new search or action
labels are generated. Seed 0 is the predeclared single-model candidate; seeds
1/2 check consistency. Do not choose the lowest-MSE seed after seeing results.

Complete 50k timings already put three-seed training on all 242,680 rows above
12 hours. A 126.6-second pilot sampled 1,024 states uniformly across the first
125,000 rows and measured 40.87 ms/step after startup, slower than the completed
50k runs. That projected 14.29 hours with the 50% planning margin for a 125k
plan, so the final protocol uses 100k. Reusing the measured larger-prefix rate
and retaining all three seeds and 20 epochs projects **7.63 hours**, or **11.44
hours with margin**, including cache preparation, validation and diagnostics.
The pilot is only throughput evidence and does not influence model selection.
Its original source archive, data mapping and rejected 125k estimate are retained.

Run `evaluator_v1/run_final_100k.sh`. The unique output directory is
`evaluator_v1/final_100k_20epochs_3seeds_20260913_v1`, with a sibling
`.console.log`, per-seed `bundle`, `checkpoint.pt`, `validation.pt` and
`result.json`, and a final `summary.json`. All artifacts remain outside Git.
The command logs progress/ETA every 15 seconds, saves checkpoints every minute,
and combines a shell timeout with the existing 12-hour cumulative runtime
budget across resumes. The identical command resumes interrupted work. Source
identity and configuration must remain unchanged while it runs.

The final report includes per-seed and arithmetic-ensemble metrics over the
expanded 20k holdout. It separately compares the 10k validation rows shared
with the 50k experiment using paired seeds and ensemble metrics; expanded and
original holdout MSEs are not directly comparable. All rows still belong to
one historical provenance group, and the common holdout already informed
architecture selection. Neither report is an independent generalization test.

All three user-run seeds completed 200,000 steps. Charged active time was
**5.6875 hours** (including 28 minutes of cache preparation), below the cap.
The independent `evaluator_v1/final_scientific_audit.json` verifies the source
archive, dataset/cache hashes, all 20,000 original validation targets, configs,
checkpoint/bundle weights, 32 predictions per seed and all summary/ensemble metrics.

| Seed | Expanded-holdout MSE | MAE | R² | Pearson | Pred. std / target std | Training min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 34.923115 | 3.506265 | 0.526345 | 0.725713 | 0.737813 | 104.67 |
| 1 | 34.877495 | 3.456996 | 0.526963 | 0.726329 | 0.733865 | 104.82 |
| 2 | 33.486862 | 3.407729 | 0.545824 | 0.739143 | 0.758819 | 98.60 |

Expanded-holdout mean MSE is **34.429157**, with arithmetic ensemble MSE
**32.446564**. On the original 10k common holdout, mean MSE regresses from
30.844713 to 34.322110 (+11.27%), with all three seeds worse; ensemble MSE
regresses from 28.325850 to 32.266407. This adverse result is retained explicitly.

A subsequent **evaluation-only diagnostic**, prompted by that regression,
compared the frozen 50k models with the new models on the same full 20k holdout.
It took 95 seconds including the audit, performed no training/search, and verified
that recomputed old-model predictions match the saved original 10k predictions.
Artifacts: `evaluator_v1/final_expanded_holdout_comparison_20260914_v1`.

| Seed | 50k-trained MSE on expanded holdout | 100k-trained MSE on same holdout | Relative difference |
| ---: | ---: | ---: | ---: |
| 0 | 40.204330 | 34.923115 | -13.14% |
| 1 | 45.669987 | 34.877495 | -23.63% |
| 2 | 38.969414 | 33.486862 | -14.07% |

Mean MSE improves from **41.614577** to **34.429157**
(-17.27%, all three seeds improve). Old/new expanded-holdout ensemble MSE is
36.050030 / 32.446564. These models trade some fit
on the original subset for better coverage of the added states. The expanded
comparison was a post-result diagnostic, not a predeclared independent test.

**Retain the 100k-trained seed-0 bundle as evaluator v1**, honoring the seed
chosen before training. Its expanded-holdout MSE is 34.923115; seed 2 has lower
MSE but is not substituted after inspecting validation. The architecture and
parameter count remain unchanged, and no ensemble becomes the deployment default.
No further architecture or training sweep is justified by this bounded roadmap.
Normal integration, inherited promoted/standardized-target compatibility and
final consolidation/verification are recorded in checkpoint 7 below.

## Evaluator-v1 goal — checkpoint 7 (integrated and verified)

The selected normal bundle is outside Git:

```text
/home/pompote/oldata/victor/morpion_runs/generic_linoo_fresh_with_bigrun_models_v1/evaluator_v1/final_100k_20epochs_3seeds_20260913_v1/seed_0/bundle
```

Seeds 1/2, checkpoints, validation predictions, configurations, source archives
and summary tables remain beside it. `evaluator_v1/selected_bundle.json` records
the chosen artifact and its SHA-256 checksums. The named
`morpion_evaluator_v1_model_args()` preset fixes the selected architecture while
legacy model defaults remain unchanged.

`load_morpion_evaluator_from_model_bundle` now lives in the normal Morpion
`players.evaluators` package. Existing bootstrap imports re-export the same
loader. It selects the converter from persisted bundle arguments, applies any
legacy target transform once, returns raw value scores, and preserves the exact
terminal-value override. CPU is the portable default; `device="cuda"` is opt-in.
No experiment runner is required for inference.

```python
from chipiron.environments.morpion.players.evaluators import (
    load_morpion_evaluator_from_model_bundle,
)

evaluator = load_morpion_evaluator_from_model_bundle("/path/to/bundle")
value = evaluator.evaluate(state)
```

The ordinary tree-player evaluator configuration accepts:

```yaml
evaluator_args:
  master_board_evaluator:
    board_evaluator:
      type: morpion_neural
      model_bundle: /path/to/bundle
      device: cpu
```

`MorpionNeuralEvaluatorArgs` is serializable through the existing evaluator union;
normal Morpion player construction consumes it. Programmatic game-state scoring
also accepts `MorpionEvalWiring(neural=...)`. A missing configured bundle fails
instead of silently selecting the heuristic. Existing heuristic settings remain
the default. Use the matching Chipiron/Coral goal branches when running these
APIs; the original dirty workspaces and their installed editable packages were
preserved, and no merge or installation was performed automatically.

The committed cleanup baseline omitted compatibility code still present in the
original dirty workspace. The goal branch now restores only the retained token
representation/target-transform modules needed to load those models; unrelated
training/bootstrap drafts were not copied. Baseline/promoted bundle schemas,
feature order, validity position, relation IDs and raw prediction semantics are
preserved. The six legacy baseline/promoted bundles and three final bundles
produce bit-identical tensors and predictions on eight checked source states
against a reference from the untouched original workspace. Standardized-target
round trips and terminal handling are also tested.

The rejected type-specific input encoder and its dedicated tests were removed
from active Chipiron/Coral code. Its commit and external source snapshot preserve
reproducibility. Exact D4 geometry, canonical caches, dataset conversion and the
resumable value trainer remain reusable infrastructure; symmetry transforms also
serve the retained consistency diagnostic. All one-off orchestration, audits,
weights, caches and logs remain outside Git. Historical experiment commands
require their recorded source snapshot; they are not normal inference commands.

Verification: **221 retained Chipiron tests and 66 Coral tests pass**, including
normal configuration/player construction, target scaling, padding/conversion,
symmetry, restart and bundle tests. Ruff and formatting pass; Pyright reports no
new errors (19 inherited training diagnostics plus one inherited wiring
annotation, reproduced against committed source). Import and diff-whitespace
checks pass. Both original dirty workspaces retain their original HEAD, status
and modified/untracked-file checksums. The durable 183-file cleanup archive was
preserved.

A broader, non-evaluator player-tag parser check still fails because Parsley
tries to call Anemone's `LinooDepthSelectionPolicy` type alias. The same failure
was reproduced at committed baseline `87086567`; it is not introduced here.
The new serializable configuration and ordinary programmatic player construction
pass independently. This inherited CLI parsing limitation is not claimed fixed.

Normal selected-bundle CPU/CUDA inference was checked against saved predictions
on 16 nonterminal validation states (maximum absolute difference <0.00002).
With one CPU thread and batch size one, 48 measurements per device gave median
16.79 ms on CPU and 15.36 ms on the RTX 3060 Laptop GPU; p95 was 20.43/17.78 ms.
These include terminal checking, conversion, transfer and scalar evaluation;
they exclude bundle loading/cache reads and are a small latency sample, not a
throughput or gameplay benchmark. Parameters remain 106,049.

Final limitations: targets are search-derived and all rows share one historical
provenance group. Architecture selection and internal holdouts do not establish
independent generalization or gameplay strength. The selected larger-data model
trades worse original-subset fit for better expanded-holdout fit; the latter
comparison was performed after seeing the regression. No new labels, search
data, policy head, automatic merge, or additional sweep was introduced.
