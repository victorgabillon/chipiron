# Morpion target compatibility follow-up — 2026-11-15

Status: `MORPION_EFFECTIVE_TARGET_SEMANTICS_READY`.

The user authorized a narrow correction to the backup-only compatibility gates
in J4.2. This follow-up removes the integration blocker recorded in the
[semantic implementation report](morpion_effective_target_semantics_2026_11_15.md).
It changes compatibility checks, not evaluation, training, coordination,
scheduling or activation algorithms.

## Exact behavior

Comparison validates each model's own declared target policy independently.
A historical `backup_value` incumbent and a scientifically declared
`effective_value` candidate both run inference on the **same current held-out
Dataset**, with the existing split, population digest, execution recipe and
finite-loss checks. Neither model's original training policy is changed. The
incumbent's historical cached validation loss is not used in place of evaluation
on the current Dataset. Candidate cached validation metrics in the older
comparison path must still match the current population; its explicit target
policy and scientific Dataset provenance must also match that Dataset.

Continuation remains stricter than comparison: the accepted parent must match
the candidate's complete target policy, including smoothing, as well as existing
architecture, bundle digest, weight-file digest, candidate name and tensor
identity checks. New explicit parents additionally require scientific source
provenance and consistent nested continuation policy. A legacy parent cannot
silently become an effective-trained candidate's parent. Matching effective
parents continue their own weights and Adam state normally.

Declared historical `backup_value` bundles remain valid without inventing a new
per-value contract for them. New effective/tree/direct policies require the
known selection contract, supported family transformation and finite blend in
[0, 1]. Trained explicit models require a Dataset extraction identity, scientific
source provenance and the selected-value confidence contract. Unknown target
kinds, missing/null explicit policies,
inconsistent kind/policy, unsupported family policies, invalid blends, missing
or invalid scientific provenance and contradictory continuation metadata are
refused. Final integration preserves the historically supported omitted-kind
format: the loader retains the old `backup_value` interpretation while exposing
`target_kind_declared=False`. Serialization preserves the omission. Frozen
legacy Dataset comparison and legacy candidate continuation retain their old
behavior, but explicit-policy comparison/continuation refuses an undeclared
bundle. A bundle with an explicit target policy but no declared kind is invalid.
This distinction preserves replay without inventing scientific declarations.

No historical model, Dataset, checkpoint or experiment was rewritten. No Anemone
or Valanga changes were made, and no playing-strength claim follows from this
semantic correction.

## Changed files in this follow-up

Production paths below are relative to `src/chipiron/environments/morpion/`:

- `learning/target_policy.py`: shared target-kind, explicit-policy and trained-model
  provenance checks, reusing the existing scientific confidence contract.
- `bootstrap/promotion.py`: replace the backup-only gate with validation of each
  model's declared policy and the current Dataset's explicit provenance.
- `bootstrap/training_publication.py`: validate the incumbent independently;
  additionally bind candidate target provenance to the current Dataset before
  using its already computed held-out metrics.
- `players/evaluators/neural_networks/bundle.py`: reject unknown, undeclared or
  inconsistent target-kind/policy declarations when loading a bundle.
- `players/evaluators/neural_networks/training/continuation.py`: validate the
  accepted parent's scientific target provenance before the existing strict
  policy, identity and lineage checks.

Tests and documentation:

- `tests/environments/test_morpion_target_compatibility.py`: 14 focused positive
  and negative cases with real trained bundles and immutable proof references.
- `tests/environments/test_morpion_continuous_runtime.py`: run the existing real
  two-round continuous case for both legacy and effective policies, asserting
  candidate policy provenance and own-parent/Adam continuity.
- `tests/environments/test_morpion_continual_publication.py`: optional explicit
  target policy in the existing fixture; its historical default is unchanged.
- This report and the updated semantic implementation report.

## Focused results

**65 distinct focused tests passed** across the following runs:

- 14 target compatibility cases. Cross-policy comparison observes the same
  Dataset file, held-out indices and validation split in both promotion and
  incumbent comparison. Both model bundles remain byte-identical. Negative
  cases cover policy/provenance corruption, model digest changes, population
  mismatch, wrong candidate Dataset identity and cross-policy/candidate parents.
- 20 target-semantic cases, including the earlier value-selection, format parity,
  certainty and Dataset/Training provenance regressions.
- 2 real two-round continuous cases: historical and effective recipes, each with
  two candidate types, reserved promotion and independent Adam continuation.
- 18 existing model-bundle and 11 existing continuation tests.

The first effective continuous run failed because the fixture's second argument
replacement reinstated the historical boolean. That fixture was corrected and
the effective case passed on rerun. No production algorithm was changed to make
the test pass.

Strict mypy passes on the five production files plus the new regression file;
Pyright and Pylint pass on all five production files. Ruff formatting passes on
all eight Python files. Ruff lint reports only the same two pre-existing F811
fixture-import findings in the two older test modules, confirmed against the
saved pre-follow-up files; production and the new regression file are clean.
Ordinary wheel/sdist builds and strict Twine checks are recorded in the runtime
evidence. No broad matrix, CI polling or benchmark was run.

## Review and preservation

The work remains unmerged on `feat/morpion-effective-target-semantics` in the
existing `/tmp/chipiron-runtime-boundary-20261029` checkout. The source baseline
is `7cd5121b838ca2243830472fef73a6f42b87ca95` plus the preceding semantic change.
A standalone follow-up diff is generated against the exact files at the start
of this authorization, so reviewers can inspect these compatibility changes
separately from the earlier implementation.

Evidence, logs, package artifacts and `follow-up.patch` are under
`/home/pompote/oldata/victor/morpion-effective-target-compatibility-20261115`.
The original research checkout's 44 pre-existing changed/untracked files and
Anemone/Valanga dependency identities are preserved. No new worktree, merge,
publication or historical migration is part of this follow-up.

## Final integration against main

Integration base: `5e25250d17ce4dfa4d1960a7efa3c2fbe4c00a87` (merged J4.2).
Its source tree is identical to the earlier `7cd5121b` implementation base.
The clean integration checkout is
`/tmp/chipiron-effective-target-integration-20261115`, on
`integrate/morpion-effective-targets-20261115`. The semantic implementation and
its dependent compatibility correction remain two logical commits. J5.1 is
excluded, as are changes to scheduling, resource reservations, shutdown and
activation algorithms.

Review of the combined diff confirmed exact-slot selection in eager and streaming
extraction, identical family raw inputs, separate smoothing, preservation of
legacy configuration serialization, explicit policy in accepted Dataset/model
provenance, cross-policy comparison and same-policy/candidate continuation.
The shared contract constant now lives with the existing confidence constants,
removing the import cycle detected by checking the entire combined change.

The implicit bundle fallback was introduced in commit
`c4657d9a5fbefe7c5fb8473f5612404f04cb4b8a` on 2026-04-11. A fresh copy of a real
trained legacy bundle with that optional field omitted loads with identical
weights and predictions under the original and integrated loaders. This checks
a historically supported format; it does not claim an exhaustive inventory of
historical experiment files. The new regression also proves byte-preserving
replay, omission-preserving manifest serialization, preserved legacy comparison
and continuation, and refusal in explicit-policy workflows. No artifact is
rewritten or migrated.

The two Ruff findings above are resolved in the integration commit by requesting
the existing fixtures through pytest's request object, without shadowing their
module imports. Their affected integration cases pass. Integration evidence is
under `/home/pompote/oldata/victor/morpion-effective-target-integration-20261115`;
earlier focused evidence remains reusable and unchanged. No CI monitoring or
automatic merge is part of this integration.

Final integration validation:

- 63 tests passed across target-policy, target-compatibility, effective-provenance
  and bundle tests plus the affected continuous/publication integration cases.
- 82 tests passed across target compatibility, historical bootstrap configuration
  and scientific launcher tests after the integration adjustments. Together these
  runs cover 130 distinct tests (the 15 compatibility cases overlap).
- Ruff lint and formatting pass on all 25 affected Python files; strict mypy
  passes on 17 production files plus the three new regression modules; Pyright
  and Pylint pass on all 17 affected production files.
- Wheel and sdist build successfully from a separate source snapshot, and both
  pass strict Twine checks. No source checkout contains generated build outputs.
- The earlier 150-test semantic and 65-test compatibility evidence is retained;
  no broad matrix or playing-strength benchmark was repeated.
