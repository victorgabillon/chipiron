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

## Next evaluator work

The near-term objective is to settle a clean first V(state) Transformer
architecture using the existing dataset. Possible later studies include symmetry
augmentation, type-specific token encoders, a small width/capacity comparison
and possibly a readout that distinguishes token types. These are research notes,
not jobs launched or scheduled by this PR.

## Source-tree policy

Keep reusable conversion, relation, model, serialization and training code, plus
small diagnostics used by those components. Completed study orchestration,
report builders, manifests, recommendation logic and experiment-only tests do
not belong permanently in the active package. Public schemas and retained
model numerical behavior take precedence over removing a few compatibility
branches. Historical implementations belong in version history; untracked
research work must be preserved before removal because it is not in Git history.
