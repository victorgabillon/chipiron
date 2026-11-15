# Morpion effective-value provenance correction — 2026-11-15

The November 15 training-target audit identified a lossy reconstruction in
`sharded_training_export.py`. This correction preserves canonical Anemone
effective-value observations in new legacy-format exports. It does not change
the default training target or establish a playing-strength advantage.

## Baseline and isolation

The audit's original counterexample was node 11 at depth 3: direct/effective
`5.860297855710058` (`direct_self`, ESTIMATE), versus backup and reconstructed
effective `5.762219659504898`. That checkout contained unrelated research edits.

Reproduction for this correction uses clean Chipiron commit
`7cd5121b838ca2243830472fef73a6f42b87ca95` in the already existing
`/tmp/chipiron-runtime-boundary-20261029` checkout. Development uses the new
`fix/morpion-effective-provenance` branch there. The previous J4.2 branch pointer
is unchanged. No new worktree was created. The original research checkout,
including the audit and all pre-existing uncommitted files, is preserved.

The two-node fixtures use legal Morpion states, the production runner, node
factory, canonical selection, backup and proof policy. A deterministic test
evaluator supplies `100 - moves` for nonterminal states. The partial fixture
opens one initial action. The full fixture legally follows first actions to a
one-action endgame and expands its terminal child. Values and expansion flags
are not injected after search. This is a provenance reproduction, not a new
playing-strength experiment.

## Demonstrated before and after

| Root | Canonical effective | Old sharded reconstruction | Corrected export/read |
| --- | --- | --- | --- |
| Partially expanded | `100.0`, `direct_self`, ESTIMATE | `99.00000099000002`, `tree_value` | `100.0`, `direct_self` |
| Fully expanded endgame | `35.0`, `tree_child`, FORCED | `35.0`, `tree_value` | `35.0`, `tree_child` |

The full root's direct value is `66.0`, ESTIMATE. Its child-derived effective
value is FORCED; the root remains nonterminal and exact. The partial root remains
nonterminal and inexact. Both fixtures' default Dataset rows are identical before
and after, including metadata. Their training targets remain the backups shown
above, not the partial root's effective value.

## Change and compatibility

The sharded writer already calls Anemone's `build_training_node_snapshot`. It now
retains that snapshot's `effective_value_scalar` and `effective_value_source` in
each mutable update instead of discarding them. Loading copies those observations
without reimplementing canonical selection. These are two optional additive
fields in the existing version-1 training export; no runtime checkpoint format,
format-version bump, migration or historical rewrite is introduced.

If either field is missing, the reader returns `effective_value_scalar=None`
and `effective_value_source="unavailable_legacy"`. This means unavailable
provenance, distinct from Anemone's `none` observation. Scalar comparisons and
node-level exactness cannot recover candidate certainty or opening completeness.
The reader does not attempt that inference, even where two scalars happen to
agree. All other legacy snapshot fields, Dataset extraction and target fallback
semantics remain unchanged. Appending a corrected generation leaves earlier
shards byte-identical. Both the materialized and indexed readers use this merge.

Older software can still read the additive JSON fields but ignores them and
retains its defective reconstruction. Consumers requiring canonical provenance
must use the corrected reader. An index built by older software that omitted
the fields can only expose unavailable provenance until rebuilt; it must not be
treated as a complete canonical observation.

This repair does not make legacy v1 a full scientific Value format. Its existing
node-level `is_exact`/`is_terminal` flags remain unchanged and must not be
reinterpreted as certainty for every scalar. It retains its other historical
limitations, including frozen ancestry and direct target fallback.

| Format | Finding |
| --- | --- |
| Flat Anemone training snapshot | Already preserves canonical scalar and source through serialization; no full per-value certainty contract. |
| Legacy sharded v1 | Defective scalar/source reconstruction corrected here; historical observations may remain unavailable. |
| Scientific-v2 JSONL | Already preserves canonical scalar/source, complete Values, per-value certainties and opening completeness. Unchanged. |
| Closed scientific snapshot | Already uses the same canonical builder and Value contract. Unchanged. |

Scientific-v2 and closed reference/indexed readers agree on the fresh legal
fixtures. Their contracts, validation, publication and activation code are not
modified. Anemone, Valanga, J4.2 coordination and all training defaults remain
unchanged. One existing v2 test that deliberately froze the v1 bug is updated;
its assertions for historical targets, ancestry and v2 separation remain.

## Validation and evidence

- 92 focused tests passed across effective provenance, sharded export, Dataset
  extraction and selected scientific-v2 contracts. Four broader tests were
  excluded, including the 648-case Dataset matrix and interruption sweep.
- Regressions cover partial direct winners, fully opened child proofs, terminal
  children, source/certainty consistency, missing either/both optional fields,
  immutable earlier generations, and indexed-reader construction/reuse.
- Ruff lint and formatting checks pass for all four changed Python files.
- Focused mypy passes for production code and the new regression file. Checking
  the two older test files reports the same 27 pre-existing errors on the clean
  baseline and corrected source. Those unrelated typing issues are not changed.
- `make check` is unavailable in Chipiron. The focused commands above replace
  that nonexistent target; no large historical matrix was repeated.
- Wheel/sdist build and strict Twine checks are run from an isolated packaging
  staging directory containing the clean baseline plus only this correction.

Runtime evidence is in
`/home/pompote/oldata/victor/morpion-effective-provenance-20261115`: source
identities, preservation hashes, `reproduce.py`, `before-after.json`, fresh
exports, test/static logs, and package artifacts. The runtime environment uses
Python 3.14.10 with Anemone 0.4.0 and Atomheart 0.3.1 built from clean Git archives;
Valanga is 0.3.0. The shared audit environment is not modified.

No merge, publication, model activation, historical checkpoint/Dataset migration
or historical experiment mutation is part of this correction.
