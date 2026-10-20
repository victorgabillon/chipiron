# Morpion value certainty and reevaluation correctness

## Findings and scope

The built-in model reevaluator falsely certifies neural predictions. A legally
reached two-node endgame reproduces this with the real Morpion runner,
`NodeMaxEvaluation`, default `MaxProofPolicy`, neural evaluator, patch serialization,
runner adapter and checkpoint loader. The parent has direct ESTIMATE(34.01),
tree FORCED(35), effective FORCED(35). Before this correction a neural prediction
42 becomes direct FORCED(42). Its effective value remains tree FORCED(35).
This distinction matters: the reproduction proves false direct certainty, not
automatic corruption of the effective search score.

Dataset also associates node flags with independently selected scalar targets.
This is **production-reachable with direct target preference**, even without
reevaluation: the same proven parent supplies estimated direct target 34.01 to
an exact-only Dataset. After an estimated prediction 42, the four exact-family
policies can even replace its terminal child's training target 35 with 42.
The live terminal value remains unchanged. Dataset v1 is deliberately unchanged;
the versioned confidence proposal below requires scientific review.

Source baselines:

* Chipiron main `50fb67a42577e195bc565f4d03e38d919640d08a`.
* Anemone main `a07fb2fdd75e2020d9795d34ec1cf896eb86f3a0`, extracted read-only
  into `/tmp/anemone-certainty-baseline`, not the moving local main checkout.
* Scientific v2 PR #80 `bfd8f2ab05f017ffda5e14af5dbf6e1f1afa3fde`, read only;
  its `morpion_sharded_training_v2.md` certainty gate informed this investigation.
* Streaming Reevaluation `79693ce2dc59c6d6ccb951951c9ba75fa69057f3`, tested in
  a separate disposable integration, with the bounded window loop retained.

No Anemone semantics/API, Dataset row format/policy, export format, production
activation, historical artifact or existing PR branch is changed.

## Exact semantics and reachability

Anemone's FORCED means an exact value for a nonterminal state. TERMINAL means
an exact value for the node's own terminal state. ESTIMATE has no proof.
FORCED may carry an exact descendant's outcome metadata without making the
parent terminal. `has_exact_value()` tests the **effective candidate**.
It does not certify the direct, tree and training target values independently.

Partial nodes compare direct and tree values using `SingleAgentMaxObjective`:
higher score wins; at equal score exact beats estimated; remaining ties prefer
tree. Fully opened nodes use an available tree value. Without a tree value,
direct remains the fallback, including terminal leaves.

For a child-derived max backup, the default proof policy requires **all branches
generated and every child effective value exact**. Otherwise the backup is
estimated. A parent's terminal child produces parent FORCED, not TERMINAL.
Tests vary both proof preconditions separately and retain the valid proof.

| Producer/path | Can introduce nonterminal direct FORCED? | D=FORCED(9), T=ESTIMATE(3), E=D |
| --- | --- | --- |
| Normal heuristic/master evaluator | No; nonterminal output is ESTIMATE | Not from normal evaluation |
| Normal neural evaluator | No; nonterminal output is ESTIMATE | Not from normal evaluation |
| Default max backup | Produces tree proofs, not direct proofs | Not from backup alone |
| Partial/full opening with normal evaluators | Partial child backups stay estimated; full proofs select tree | Not from opening alone |
| Old built-in model patch | **Yes**, copying effective exactness onto its prediction | In the normal closed proof subtree the node stays fully opened with exact tree; this alone does not establish the partial counterexample |
| Checkpoint restoration | Preserves separately serialized Value payloads | Preserves an already supplied state; does not create this association by itself |
| Normal explicit model switch | Installs evaluator; optional reevaluation returns estimated nonterminal direct values | Not with the standard evaluators |
| Explicit externally certified update/custom certified evaluator | Yes, by the caller's authority contract | API-valid; caller must supply a genuine proof |
| Dataset processing | Changes supervised outputs, not live search Values | Cannot create a search state; can misassociate selected target confidence |

Under the normal producer invariant, nonterminal direct values remain estimates
and effective exactness comes from a fully opened exact tree. Therefore the
specific partial D=FORCED/T=ESTIMATE witness is a **synthetic API-valid state**,
not evidence of normal Morpion reachability. The normal-state direct-preference
Dataset witness is independent of that synthetic configuration. Neither the
small search tests nor this argument certifies arbitrary external updates,
custom proof policies, historical inputs, or previously miscertified values.

## Real reevaluation reproduction and correction

The fixture starts at the usual initial board and repeatedly plays the first
legal Atomheart action. It legally reaches move 34 with one action remaining;
the child at move 35 has no legal action. Only this last two-node graph is
searched. No terminal/forced Value is injected in this reproduction.

The actual neural adapter uses handcrafted feature conversion and a Torch linear
regressor with zero weights and a fixed bias. It returns the normal evaluator's
ESTIMATE on the parent and TERMINAL on the child. Bundle loading alone is injected;
the neural evaluation, codec, worker row generation and runtime update are real.

| Observation | Before | Corrected |
| --- | --- | --- |
| Parent prediction 42 | direct FORCED(42) | direct ESTIMATE(42) |
| Parent prediction equal to tree score 35 | direct FORCED(35) | direct ESTIMATE(35) |
| Exact tree / effective score | FORCED(35), `tree_child` | Identical |
| Actual terminal direct value | TERMINAL(35) | Identical; reused without inference |
| Explicit direct proof followed by estimated model prediction | Proof replaced by FORCED(42) | Original direct proof retained; row reported skipped |
| Ordinary partial estimated nodes | Existing weighted direct update and backup propagation | Identical |

Both alpha=1 and alpha=.25 are covered. Existing live effective exactness already
bypasses blending, so the solved parent's direct prediction remains unblended
42 at alpha=.25. This behavior is preserved explicitly. Ordinary estimated
nodes still compute `(1-alpha)*old + alpha*prediction`; no new certainty is
created by the blend. Estimated child updates recompute the parent's estimated
backup through the real propagator. Selector classification and its public
no-frontier behavior remain consistent with proven/opened/terminal nodes.

The small correction has two parts:

1. The built-in model producer certifies only its preserved terminal score.
   Nonterminal model rows explicitly carry `is_exact=False`. Terminal nodes use
   their existing backup, or their existing direct score when no backup exists.
   A terminal row with neither score fails instead of inventing a score.
2. Before either runner application path, unproved built-in model rows are
   skipped if they would replace an existing exact direct value, or an exact
   effective value at a partial node. Fully opened nodes with an estimated
   direct value still accept model updates while retaining their tree proof.
   Applied counts exclude skipped rows; diagnostics retain skipped IDs. Patch
   IDs, evaluator/model identities and persisted patch contents are preserved.

Explicitly authoritative external rows retain the public `NodeValueUpdate`
contract. The guard recognizes the existing built-in model source and an explicit
estimated row; it does not reinterpret arbitrary external updates or silently
rewrite old pending patches. Already persisted false FORCED values cannot be
distinguished from genuine proofs by this fix and are not automatically repaired.
The guard adds a live-node lookup when model estimate rows are present; this is
a correctness change, not a bounded-memory or performance claim.

The opt-in **v1 snapshot-value fallback remains ambiguous**: replaying the solved
parent's D=34.01/T=35 with the single node flag certifies both scalars in Anemone.
Its legacy explicit-authority behavior is characterized, not silently changed.
A per-value update contract is needed to replay mixed certainties faithfully.
Do not use this fallback as evidence that a model prediction is certified.

## Dataset consequences across all nine policies

`tree_to_dataset._choose_target_value_and_source` selects the scalar, while
`_passes_filters` and the output `row.is_exact` use node flags. Family context
independently selects raw targets and constructs `exact_or_terminal_node_ids`
from those flags. It is built from the full snapshot, before emitted-row
filtering/truncation. Filtering rows does not remove their family evidence.

For the synthetic partial node D=FORCED(9), T=ESTIMATE(3), E=D, backup preference
selects **estimated 3**. Exact-only extraction admits it and sets `row.is_exact`.
For a two-member synthetic family with peer raw target/prediction 2, predictions
9 and 2, and blend .25, actual Dataset outputs are:

| Policy | Root target | Peer target | Root/family treatment |
| --- | ---: | ---: | --- |
| `none` | 3 | 2 | `hard_exact_anchor` |
| `pv_mean_prediction` | 3 | 5.5 | `hard_exact_anchor` |
| `pv_min_prediction` | 3 | 2 | `hard_exact_anchor` |
| `pv_blend_mean_prediction` | 3 | 2.875 | `hard_exact_anchor` |
| `pv_blend_min_prediction` | 3 | 2 | `hard_exact_anchor` |
| `pv_exact_then_mean_prediction` | 3 | 3 | `pv_exact_family` |
| `pv_exact_then_min_prediction` | 3 | 3 | `pv_exact_family` |
| `pv_exact_then_blend_mean_prediction` | 3 | 3 | `pv_exact_family` |
| `pv_exact_then_blend_min_prediction` | 3 | 3 | `pv_exact_family` |

All nine protect the selected estimated root scalar as an exact anchor or exact
family evidence. All four exact-then policies propagate 3 numerically to the
peer as family exact evidence. Its `row.is_exact` remains false, while
`family_has_exact_or_terminal=True` and `family_exact_target=3` remain in metadata.
With exact-only extraction, the peer is omitted but the incorrect root is still
admitted. With direct preference in this synthetic fixture, root target 9
actually has the supplied direct proof; the nine policies retain their existing
outputs rather than being globally demoted.

The stronger **normal-reachable** case has D=ESTIMATE(34.01), T=FORCED(35).
Direct preference makes 34.01 a hard exact anchor under the first five policies.
The four exact-then policies choose family max 35, masking the bad root evidence
because the terminal child supplies valid 35. After a normal estimated model
prediction 42, these policies choose family max **42**, and change the terminal
training row from **35 to 42**, retaining both `is_terminal=True` and `is_exact=True`.
This occurs after the reevaluation fix too; changing the direct certainty alone
cannot fix a Dataset that sees only node flags and scalars. Backup preference
uses certified tree 35 and terminal direct 35 in this normal fixture; all nine
then retain 35. This local safety result is not a general guarantee for custom
certified direct producers or mixed-certainty external inputs.

No Dataset source code or v1 row interpretation is changed in this PR.

## Versioned confidence proposal requiring review

Use scientific sharded v2's separate direct/tree/effective/target Value payloads,
certainties, explicit source and `backup_exactness_compatible` diagnostic. The
diagnostic catches the synthetic backup mismatch; it does not certify a direct
target or resolve every policy/source choice. Do not infer missing v1 certainty
from a node flag or migrate historical scientific artifacts in place.

A separately reviewed Dataset/row v2 could retain `node_is_exact` independently
of `raw_target_certainty`, `effective_target_certainty` and target provenance.
Version the confidence policy and include it in dataset/cache identity. There
are at least three explicit alternatives:

* **Selected-value confidence:** exact-only filtering and hard anchors use the
  actual selected raw Value's certainty. Estimated direct 42 cannot be an exact
  anchor despite an exact effective tree 35. Define separately whether estimated
  members may receive exact family targets and how row confidence changes.
* **Certified-source preference:** when confidence is required, explicitly select
  a certified tree/effective/direct source, with recorded priority. This changes
  target selection (e.g. 42 becomes 35), rather than merely changing a flag.
* **Strict compatibility gate:** refuse mixed-confidence/unknown cases until a
  supported producer invariant is established, without silently coercing targets.

All alternatives must define conflicting family proofs and protect a certified
terminal target from a different family scalar. Do not use `max(exact_targets)`
as a proof-resolution rule without a scientific contract. Preserve original
sources, producer/checkpoint identity and unknown confidence for legacy data.
The nine policy names and v1 supervised row format remain frozen here.

## Retained evidence and validation

[Regression tests](../../tests/environments/test_morpion_value_certainty.py)
exercise the real APIs and intentionally preserve v1 Dataset/fallback behavior
for review. [Failing-before evidence](value_certainty/failing_before.txt) records
six failures on untouched Chipiron main: four false neural certainty cases
(equal/different scores, both alphas), and two proof overwrite cases. The corrected
runtime passes those assertions without changing Anemone's proof semantics.

[Audit driver](value_certainty/audit.py), [baseline results](value_certainty/baseline.json.gz),
[corrected results](value_certainty/corrected.json.gz), and
[exact comparison](value_certainty/comparison.json) retain 12 cases × nine policies
× two scalar preferences × three family blends (0/.25/1) × two exact filters:
**1,296 outputs per revision**. Hashes cover every serialized supervised row field,
state reference, extraction counter and metadata field, without dropping scientific
fields or applying tolerances. Compact readable target decisions accompany hashes.

**1,080 outputs are byte-equivalent under deterministic serialization.** The only
216 changed outputs belong to the two synthetic direct-proof/model-update cases:
the correction retains the original proof rather than overwriting it and triggering
a new local backup. These fixtures explicitly declare cached values; they are not
claims of naturally solved initial boards. All ordinary and normally proven
Dataset outputs, including the confidence-policy counterexamples, are unchanged.

Local validation includes installed-wheel tests for the affected runner/worker/
consumer/Dataset paths; the existing bootstrap/family sanity suite; sdist and wheel
build; toolchain coherence; repository-wide Ruff lint/format; and all three static
analyzers with **zero new diagnostics** (209 inherited, 206 current; debt baseline
unchanged). Tests redirect the global leaderboard to fresh pytest scratch paths
without changing HOME or record assertions. Python 3.13.5, Torch 2.14.0+cpu,
Anemone 0.2.26, Atomheart 0.2.11 and Valanga 0.2.5 are used, with the Anemone source
explicitly pinned above. Streaming integration additionally uses ijson 3.4.0.post0.
The installed affected suite passes **218 tests, three streaming-only skips**;
the focused installed-wheel suite passes **89, three skips**; the existing
bootstrap/family suite passes **88**; the separate streaming integration passes
**122**, including those three streaming-only cases. The installed-distribution
contract script verifies the actual wheel imports and evaluator/checkpoint APIs.
[Machine-readable validation](value_certainty/validation.json) and
[protected worktree status](value_certainty/protected_status.json) retain details.

The separate streaming integration tests the same correction in
`evaluate_node_window`, plus the pre-existing legacy/indexed differential tests.
The new real-endgame test compares complete persisted patches, cursors and applied
Value payloads across flat, full sharded and indexed sharded worker paths. Model
identities, patch IDs and terminal behavior agree. That integration is evidence,
not a modification or activation of the protected streaming PR.

Checkpoint tests compare complete direct/tree/effective serialized Value payloads,
sources and RNG state after save/restore, then bounded continuation. They cover
both preserved direct proofs and normal child-derived tree proofs. Full remote
installed-package CI and final local counts are linked in the draft PR and final
delivery; CI runs against the branch tip, not a protected worktree.

Reproduce the matrix from this checkout with a fresh output path:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=SELECTED_CHIPIRON_SRC:PINNED_ANEMONE_SRC:. \
  python docs/research/value_certainty/audit.py --output /tmp/NEW-certainty.json
# Select untouched 50fb67a4 source for baseline, this branch's src for correction.
# Retained JSON is gzip-compressed with deterministic headers; no data is omitted.
```

The dirty checkout, PR #74–#80, Anemone PR #77/#78, historical checkpoints and
derived experiments are untouched. No gen38 operation, bootstrap continuation,
distributed worker, production activation or merge was performed.
