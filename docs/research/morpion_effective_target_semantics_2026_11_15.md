# Morpion explicit training-target semantics — 2026-11-15

This change makes canonical Anemone effective values the default recipe for
**new** scientific bootstrap preparation. It changes learning semantics, not the
search evaluator or a claim about playing strength. The November 15 audit's
historical findings and inconclusive strength pilot remain unchanged.

## Review status and integration boundary

The independent Dataset/Training implementation and focused tests are complete.
Continuous integration is **blocked** by two existing compatibility gates:
`bootstrap/promotion.py::_evaluate` and
`bootstrap/training_publication.py::_compare_incumbent` require model
`target_kind == "backup_value"`. New effective bundles truthfully declare
`effective_value`, so those gates reject them. The initial external seed's
historical target provenance also needs an explicit comparison policy; it must
not simply be relabeled as effective-trained.

These gates have deliberately not been altered: the request explicitly protects
J4.2's validated implementation during integration. Approval of the narrow
semantic integration change is pending. Do not treat the fresh continuous CLI
default as validated end-to-end promotion until that integration is resolved.
No coordinator, promotion, activation, Anemone or Valanga implementation changed.

The working branch is `feat/morpion-effective-target-semantics`, based on clean
commit `7cd5121b838ca2243830472fef73a6f42b87ca95` in the existing
`/tmp/chipiron-runtime-boundary-20261029` checkout. It includes the preceding
uncommitted provenance correction described in
[morpion_effective_provenance_fix_2026_11_15.md](morpion_effective_provenance_fix_2026_11_15.md).
The original research checkout's 44 pre-existing modified/untracked files and
the J4.2 branch pointer remain unchanged. No worktree or clone was created.

## Exact target policy

Chipiron selects an exported slot; it never implements Anemone's candidate
comparison or infers opening completeness. The three explicit choices are:

| Policy | Anemone value used | Missing slot |
| --- | --- | --- |
| `effective` | Canonical `effective_value`, with `direct_self` or `tree_child` provenance | Skip Anemone's explicit `none`; refuse unavailable legacy provenance |
| `tree` | `tree_value`, the child-derived candidate | Skip; no direct fallback |
| `direct` | `direct_value`, the node's own evaluation | Skip; no tree fallback |

Every selected scalar must match its full scientific Value and its own
certainty. Effective source, Value and certainty must agree with the declared
origin slot. Node-level exactness is not borrowed by another selected value.
FORCED and TERMINAL selected values remain exact; node terminality remains
separate. Contradictory certificates and missing certainty fail explicitly.

Python callers create a new recipe with:

```python
args = MorpionBootstrapArgs.new_scientific(work_dir)
# Alternative: new_scientific(work_dir, target_source="tree" or "direct")
```

Fresh CLI `--scientific --scientific-operation prepare-continuous` selects
`effective`; `--dataset-target-source tree` and `direct` are explicit alternatives.
The compatibility boundary above still blocks the continuous promotion stage.
A persisted explicit recipe contains:

```json
{"dataset": {"target_source": "effective", "use_backed_up_value": null}}
```

This fragment supplements the existing Dataset settings. No major schema
redesign or checkpoint format was added. Existing generic argument construction,
legacy configuration files, and resume without a new policy retain their frozen
recipe. Supplying an explicit policy alongside a historical boolean is refused.
Changing policy inside an existing scientific workspace is refused; create a
fresh experiment instead. Explicit legacy CLI boolean flags remain available.

## Formats, family adjustment and provenance

Flat Anemone snapshots already preserve canonical effective scalar/source.
Scientific-v2 and closed scientific exports already preserve the full Value
contracts; their calculation and serialization are unchanged. Flat extraction
with an explicit target additionally requires the matching scientific Value and
source-provenance sidecar. A bare legacy flat snapshot cannot supply that proof.

The preceding legacy-shard fix copies Anemone's observed effective scalar/source
instead of reconstructing them tree-first. On the fresh two-node partial tree,
the old reconstructed effective value was `99.00000099000002`; canonical and
corrected values are `100.0`, `direct_self`, ESTIMATE. The full-opening endgame
has direct `66.0` ESTIMATE but effective/tree `35.0` FORCED, and its terminal
child has value `35.0` TERMINAL. Legacy shards missing either observation expose
`unavailable_legacy`, never a guessed canonical value. Even newly corrected v1
shards lack a full per-value certainty contract and cannot by themselves satisfy
an explicit scientific target policy.

Family adjustment takes its raw inputs from the same requested slot in eager
and indexed/streaming extraction. It cannot reselect a backup behind `effective`.
Smoothing is an explicit subsequent transformation; supported prediction-based
family transformations retain ESTIMATE certainty and protect certified raw
anchors. The historical metadata field `effective_target` refers to this
post-family training target and is distinct from Anemone `effective_value`.
Existing cross-state proof policies remain gated by the confidence contract.

New Dataset metadata records `dataset_target_source` and
`dataset_target_contract="anemone-exported-target-selection-1"`. The existing
extraction identity includes these fields. Rows retain selected source,
certainty and full Value provenance; JSON and JSONL validation reject disagreement
with the Dataset header. Dataset admission and accepted manifests carry the
policy. Training metadata and bundles carry the policy, smoothing settings,
Dataset extraction identity and scientific source provenance. Raw bundles name
`effective_value`, `tree_value` or `direct_value`; smoothed bundles name
`family_adjusted_value`. Historical bundles retain their historical label.

Optimizer continuation checks the complete policy and smoothing settings against
the accepted parent bundle. Matching continuation succeeds; switching policy or
crossing from a legacy lineage is refused. Existing checkpoint/bundle manifests
bind the provenance: tensor/optimizer serialization formats are unchanged.
Runtime checkpoints remain bound to the scientific config identity, without
embedding a new learning rule in Anemone.

## Compatibility and validation

Historical `use_backed_up_value` is not an alias for an explicit slot policy.
Its row fallback, family-input behavior, flags and serialized identities remain
unchanged. No historical Dataset, checkpoint or experiment was migrated.
Eight small baseline/current comparisons cover both historical boolean values,
legacy and selected-value confidence, and unsmoothed/blended family targets;
config hashes and complete Dataset JSONL bytes match.

The combined focused suite passes **150 tests**, with 37 existing broader cases
excluded. It includes 20 new target-policy cases plus the real-tree provenance
regressions, historical configuration/extraction, scientific launcher,
selected-value confidence and continuation tests. Coverage includes legal partial
direct winners, child winners, full expansion, FORCED and TERMINAL values, DAG
transpositions, exact alternative policies, flat/sharded/eager/streaming equality,
serialized-policy refusals, one-epoch eager/streaming training and optimizer
continuation, and publication through accepted Dataset/Training with checkpoint
identity. It does not exercise or claim complete continuous promotion.

Affected Ruff formatting/lint, strict mypy (16 files), Pyright (15 production
files) and Pylint checks pass. Wheel and source distribution builds and strict
Twine checks also pass; logs and package artifacts accompany the runtime evidence. Chipiron has no `make check` target. No large matrix or playing-strength
benchmark was repeated.

Evidence and disposable runtime outputs live in
`/home/pompote/oldata/victor/morpion-effective-target-semantics-20261115`.
The runtime uses Python 3.14.10 as required by Chipiron, Anemone 0.4.0,
Atomheart 0.3.1 and Valanga 0.3.0. No merge or publication was performed.

## Changed files

- `bootstrap/bootstrap_args.py`, `bootstrap/config.py`, `bootstrap/launcher.py`:
  explicit policy, fresh scientific defaults, frozen legacy/resume behavior.
- `learning/target_policy.py`, `learning/target_confidence.py`,
  `learning/tree_to_dataset.py`: exact slot selection, certainty association,
  Dataset provenance and serialization checks.
- `bootstrap/cycle_dataset.py`, `bootstrap/dataset_family_targets.py`,
  `bootstrap/streaming_dataset.py`: policy propagation and family consistency.
- `bootstrap/dataset_publication.py`, `bootstrap/pipeline/stages.py`: existing
  Dataset admission/manifests include explicit target identity.
- `players/evaluators/neural_networks/bundle.py`, `training/service.py`,
  `training/continuation.py`: truthful model semantics and continuation provenance.
- `bootstrap/sharded_training_export.py`: preceding canonical observation fix.
- `tests/environments/test_morpion_target_policy.py`,
  `test_morpion_effective_provenance.py`, `test_morpion_sharded_training_export.py`,
  `test_morpion_sharded_training_v2.py`, `morpion_activation_helpers.py`: focused
  regressions and optional explicit-policy fixture support.
- This report and the preceding provenance-correction report.

Production paths above are relative to `src/chipiron/environments/morpion/`.
