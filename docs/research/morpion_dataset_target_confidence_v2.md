# Scientific Dataset v2: selected-target confidence

## Decision and status

Dataset v2 implements the scientific decision made during this investigation:
certify the actual selected same-state Value; retain ordinary estimates for
training; keep every cross-state heuristic result estimated; protect certified
same-state targets; gate all four `pv_exact_then_*` policies. Different states
may legitimately have different exact values. A family is not a proof domain.

Implementation is **opt-in and local**, on
`research/morpion-dataset-v2-confidence-contract`, in
`/tmp/chipiron-dataset-v2-confidence`. No production route was activated and no
historical input was edited. No PR was opened because its required upstream
stacks are still sibling drafts. The dependency plan below identifies the
appropriate review boundary instead of publishing a duplicate-code PR.

Implementation commit: `eea417271a822311e08dd8b88ee938ccb32ef2f4`.

## Exact ancestry and scope

| Source | Exact commit |
| --- | --- |
| PR #75, bounded reader | `79693ce2dc59c6d6ccb951951c9ba75fa69057f3` |
| PR #76, bounded Dataset | `5c8906ba0d520c63d821e14a2d52530f8bed2307` |
| PR #80, scientific export v2 | `bfd8f2ab05f017ffda5e14af5dbf6e1f1afa3fde` |
| PR #82, model certainty fix and actual endgame | `cae9ec523edaaed277c8f953fafae5d9ac766701` |
| Local scientific reader integration | `02212bc6130205b7d2ef898c7b1eb0f7d384f7aa` |
| Local matched-input performance investigation | `7d808c37113b5ea127bd473bc8c5b8e774492b8a` |
| This investigation's isolated integration base | `77267b64a675602f9567920be74555999792cc68` |

`77267b64` is a local merge with exactly parents `7d808c37` and `cae9ec52`.
Its one conflict preserves the reader's node-window loop and PR #82's terminal
direct fallback. No source PR was rewritten or rebased. An old integration
test expecting model inference for a terminal direct-only node was updated to
PR #82's corrected `terminal_existing_value` provenance.

The earlier integration merge `f09a038b2589e124ffbb96a0d9936081692b3195`
has parents PR #76 and PR #80. Their common ancestor, also PR #82's base, is
`50fb67a42577e195bc565f4d03e38d919640d08a`. PR #80 is stacked on PR #77;
PR #76 is stacked on PR #75. PR #82 independently targets main. All four
queried source PRs remained open drafts at their pinned heads on 2026-10-21.
[Retained dependency metadata](dataset_confidence_v2/dependencies.json).

The existing reader, its parser, SQLite source index and immutable-state
validation are reused unchanged. New confidence consumes one indexed value
contract at a time; there is no new parser, full snapshot, source index or
population-sized Python confidence dictionary on the bounded path.

## Selected-target confidence contract

The explicit bootstrap option is
`dataset_confidence_policy="selected_value_v2"`. Its default remains
`"legacy_v1"`. The confidence contract is
`morpion-selected-value-confidence-1`; `dataset_version` is 2.

Scalar selection retains the existing direct/backup preference. This change
does not select a tree scalar merely because it has a better certificate.
The original selection rule is retained as `target_selection_rule`; v2's
`target_source`, `raw_target_source` and source counters name the actual
`direct_value` or `tree_value`.

Each v2 row's `metadata.target_confidence` stores:

| Field | Meaning |
| --- | --- |
| `node_is_exact` | Original effective-node exactness flag; never authority for another Value |
| `node_is_terminal` | Original node terminal flag, independently retained as `row.is_terminal` |
| `node_effective_certainty` | Scientific source's effective Value certainty |
| `raw_target_value`, `raw_target_source`, `raw_target_certainty` | The actual selected scalar and its own certificate association |
| `training_target_value`, `training_target_source`, `training_target_certainty` | The final target and the computation that produced it |
| `source_value_contract` | Complete original D/T/E/export-target Value payloads, certainties, declared target source and opening diagnostics |
| `contract` | The explicit confidence contract identifier |

Anemone's existing semantics are retained: `FORCED` is exact for the receiving
nonterminal state; `TERMINAL` is exact terminal evidence; `ESTIMATE` has no
certificate. Missing confidence remains unknown (`null`), never inferred from
`node_is_exact` or terminality. Unknown enum names and inconsistent payloads
are rejected. The strict scientific source readers already reject malformed
v2 shards with missing mandatory certainty; the policy API additionally tests
an unavailable selected certificate without inventing one.

In v2, `row.is_exact` describes the **final training target**. Exact-only
filtering uses the **selected raw target's certainty**. These agree for emitted
proofs under the five supported policies: certified raw targets are preserved,
and heuristic policies never promote an estimate. Other depth/visit/order and
truncation behavior remains the existing extraction behavior. Family context
still sees the whole source before emitted-row filtering and `max_rows`.

The raw selection, raw certainty and export-declared target source remain
independent: choosing direct 42 does not consume the export's tree-target
certificate for 35. Matching numbers alone do not promote an estimated direct
35 to an exact target, either.

## Family semantics and nine-policy result

The supported policies preserve certified raw targets as hard same-state
anchors. Mean, minimum and blend arithmetic on other rows is unchanged, but
every heuristic result is `ESTIMATE`, including zero-weight blends and results
that happen to equal a certified number elsewhere. This implementation does
not add promotion based on an unselected same-state certificate.

For v2, `family_has_certified_member` and `family_certified_member_count`
describe membership only. The retained legacy-named membership counters have
that same explicitly versioned interpretation. `family_exact_target` and
`family_proof_contract` are null. No maximum of certified members is presented
as a family certificate or used to overwrite another state's certified target.
Contradictory independent certificates in one node's same-state/objective
contract are rejected. Different nodes' certified scalars are not compared as
if PV membership established equality; distinct-state 9/4 declarations have
regression coverage and retain both anchors.

The actual PR #82 endgame has direct `ESTIMATE(42)`, tree/effective
`FORCED(35)`, and a terminal child `TERMINAL(35)`. With direct preference,
blend 0.25 and no filtering, exact outputs are:

| Policy | Frozen v1 parent / child | Dataset v2 parent / child | Parent v2 certainty |
| --- | --- | --- | --- |
| `none` | 42 / 35 | 42 / 35 | ESTIMATE |
| `pv_mean_prediction` | 42 / 35 | 38.5 / 35 | ESTIMATE |
| `pv_min_prediction` | 42 / 35 | 35 / 35 | ESTIMATE |
| `pv_blend_mean_prediction` | 42 / 35 | 41.125 / 35 | ESTIMATE |
| `pv_blend_min_prediction` | 42 / 35 | 40.25 / 35 | ESTIMATE |
| `pv_exact_then_mean_prediction` | 42 / 42 | Gated, no output | No cross-state proof contract |
| `pv_exact_then_min_prediction` | 42 / 42 | Gated, no output | No cross-state proof contract |
| `pv_exact_then_blend_mean_prediction` | 42 / 42 | Gated, no output | No cross-state proof contract |
| `pv_exact_then_blend_min_prediction` | 42 / 42 | Gated, no output | No cross-state proof contract |

V1 incorrectly marks both rows exact in this matrix. V2 always preserves the
child's certified 35. Exact-only direct-preference v2 outputs contain only that
child. Backup preference retains parent `FORCED(35)` and child
`TERMINAL(35)` under all five supported policies. Four exact-family policies
fail before extraction regardless of filtering, blend or row limit; they do
not silently fall back to a prediction policy.

## A cross-state equality counterexample using legal Morpion moves

Starting at the usual initial board, repeatedly choose the first legal action.
The known PR #82 path ends at 35. Its last parent at move 34 has one legal
action; the real runner proves `FORCED(35)` using its default proof policy.

Take the earlier state at move 31. Its first-legal continuation ends at 35.
Instead choose its second legal action, then repeatedly choose the first legal
action: this legally reaches terminal **71**, certified `TERMINAL(71)` by the
actual evaluator. The evidence retains every action and canonical endpoint
reference. This is a lower-bound witness, not a claim that 71 is optimal.

A five-node partial exported path at moves 31–35 places the move-31 state and
terminal-35 state in the same existing PV family. Its first three values are
explicitly declared estimates; its last two use the real PR #82 proof. Since
the objective is the maximal reachable total move count,
`V(state31) >= 71 > 35 = V(terminal35)`.
Thus PV-family membership does not imply equal state values. Transferring the
terminal's certificate to that ancestor would be unsound. Existing v1's false
`row.is_exact` association is a separate problem: its estimated ancestors are
not all marked exact; this counterexample disproves an additional blanket
family-proof interpretation, rather than attributing a new flag bug to v1.

For future exact transfer, scientific review must specify score convention,
the objective, which child is optimal, branch completeness, all relevant
certainties, and an explicit proof relationship for each receiving state.
`max(exact_family_targets)` is not such a proof rule. None is invented here.

## Exact evidence and intentional differences

[Audit driver](dataset_confidence_v2/audit.py) generates five fresh cases:
actual endgame before/after the estimated neural update, ordinary estimates,
the explicitly API-declared D=FORCED(9)/T=ESTIMATE(3) case, and the legal
unequal-state family counterexample. The 9/3 declaration is a contract fixture,
not a claim that the initial board is solved at 9.

Five cases × nine policies × two preferences × three blends (0/.25/1) × seven
filter/limit choices × two confidence contracts gives **3,780 decisions**:

* **1,890 v1 outputs** match untouched isolated base `77267b64` exactly.
* **1,050 supported v2 outputs** match saved/reloaded flat snapshots supplied
  with the same scientific sidecar, the full v2 reference, and the indexed path.
* **840 v2 decisions** reject the four gated policies identically on all three
  paths, producing no dataset.
* All **2,940 successful triples** match complete rows, order, references,
  targets, flags, provenance, counters, summaries and literal JSONL bytes.
  All **8,820 saved JSONL files** reload with identical fields, including v2 confidence/provenance.

No fields or numerical differences were normalized away. All 24 flat/export
source-file fingerprints agree between the separately generated baseline and
current cases. The two retained patch files differ only in two explicit
scratch model-path fields; both differing file hashes and field paths are
retained. These paths do not enter Dataset rows.

Among shared rows of the 1,050 accepted v2 comparisons, the deliberate change
alters 85 target scalars and 165 exactness flags; selected-confidence filtering
removes 60 rows. These are aggregate observations across repeated filter/blend
configurations, not counts of unique game states. The full reference
is used only by the oracle; a production-path test forbids full readers and
snapshot constructors and asserts one-node contract windows. Repeated pipeline
outputs reuse the existing verified source index.

[Complete before outputs](dataset_confidence_v2/before.json.gz),
[complete current outputs](dataset_confidence_v2/after.json.gz), and
[comparison including failing-before/passing-after assertions](dataset_confidence_v2/comparison.json)
retain source fingerprints, independent Value contracts, full scientific
provenance and every serialized supervised field. Differences between v1 and
v2 are intentional policy/contract changes, never claims of parity across
different policies. The explicit scientific-requirement probe produces
[nine failures before](dataset_confidence_v2/failing-before.txt) and
[nine passes after](dataset_confidence_v2/passing-after.txt), with four after
cases passing by explicit policy rejection. It evaluates the actual retained
before/new outputs; it does not assert that frozen v1 changed. Its hashed
reproducer accompanies the archive, while the 62 normal regressions exercise
the live APIs against freshly generated inputs.

Additional regressions cover equal-number estimated 35, genuine same-state
proof retention, ordinary usable estimates, missing/unknown confidence,
contradictory same-state certificates, legitimate different-state certificates,
source-generation identity, malformed serialization, unsupported contracts,
pipeline manifests, source mutation during output, atomic cleanup and refusal
to overwrite a v1 artifact with a v2 artifact.

## Compatibility and artifact identity

V1 selection, family policy, filtering, row flags, row schema and saved bytes
remain frozen. Absent new options use v1. No historical confidence is inferred,
relabelled or migrated. Flat snapshots alone cannot supply v2 confidence: the
full API explicitly requires the scientific v2 value-contract/provenance sidecar.

V2 JSON uses `format_kind="morpion_scientific_supervised_rows"`,
`format_version=2`, and `scientific_rows` instead of `rows`. V2 JSONL uses
`kind="morpion_scientific_supervised_rows_metadata"` and format version 2.
The new loader validates envelope/metadata agreement and every row's
scalar/source/certainty association. It rejects fabricated promotions and
heuristic overwrites of certificates. The previous loader **refuses both new
namespaces**, avoiding silent historical flag interpretation.

Headers and pipeline manifests record the confidence contract, Dataset version,
full scientific producer/generation provenance and `dataset_extraction_identity`.
That identity hashes the complete extraction header, including confidence,
source identity, family policy/blend, filtering and counters. The pipeline
rejects an existing v1 output destination before claims/status changes for a
v2 request. The source SQLite index is intentionally shared: it stores verified
source facts and has no Dataset-policy-dependent interpretation. No second
index is needed. Its v1/v2 identity separation and source/cache checks remain
the existing integration's contract.

Bounded worker extraction requires all existing reader/scientific opt-ins plus
the new confidence opt-in. The full-reference API is explicit and offline;
the bounded production path never invokes it as a fallback.

## Validation and bounded resources

The complete installed-wheel CI suite passes **1,872 tests, with two CUDA
skips and four standard integration-marker exclusions**, in 462.89 seconds
(pytest; controller 463.15 seconds). All **62 new confidence regressions** run
in that suite. The wheel matches all 463 production Python source files. The
affected source suite additionally passes 247 tests.

The repository-wide Ruff lint/format checks, toolchain coherence, sdist/wheel
build and installed-distribution contract guard pass. All three static analyzers
report **zero new diagnostics** (209 baseline, 206 current, three removed);
the debt baseline was not changed. The same established safety plugin redirects
optional leaderboard persistence into pytest scratch paths without altering
HOME or scientific assertions. This is the standard two-process pytest CI
execution, not a bootstrap or distributed search worker run.

[Machine-readable validation](dataset_confidence_v2/validation.json) records
commands, selection, exclusions, skips, package hashes, checks and elapsed time.
The complete CI logs, test reports, source files, JSONL pairs and resource
reproducer are retained in the accompanying evidence archive.

Fresh-process, serial 1k/5k checks use the existing `StageMeasurement` and
benchmark CPU/I/O collectors. Both confidence modes consume the **same
scientific v2 source and same verified reused index**, using direct preference
and `pv_blend_mean_prediction`. The balanced synthetic graph has legal replay
states and 4 KiB node metadata; all declared values are estimates. No game
optimality is claimed for these resource-fixture scores. Target/reference/flag
hashes and all source fingerprints agree between confidence modes.

| Nodes | Cold source index (s) | Reused init v2 (s) | Legacy-confidence Dataset (s) | V2 Dataset (s) | V2 rows/s | Legacy / v2 peak RSS (MiB) | TEMP logical bytes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1,000 | 3.1478 | 0.0278 | 0.3286 | 0.5381 | 1,858.5 | 254.00 / 253.93 | 266,240 |
| 5,000 | 23.6107 | 0.1321 | 1.6633 | 2.6532 | 1,884.5 | 255.53 / 255.41 | 1,200,128 |

V2's extra validation/provenance has measurable CPU, serialization and output
cost: output grows from 4,980,611 to 5,987,892 bytes at 1k, and from 25,140,629
to 30,172,215 at 5k. Those costs are retained rather than discarding scientific
fields. RSS values are actual process high-water observations, include imports,
and remain about 254–256 MiB at these sizes; small between-run decreases are
noise, not a claimed memory improvement. TEMP storage scales on disk. The
source index is shared unchanged, and neither confidence mode rebuilds it.

These are single controlled samples on a shared host. “Cold” means a new source
index; files have just been generated and the OS page cache is warm. They are
not comparisons with the older v1 export benchmarks or with long endgame replay
histories. They establish bounded-confidence working behavior at these sizes,
not a universal throughput or scientific-readiness guarantee. No new 10k or
production-scale result is inferred from them.

[Raw RSS, CPU, I/O, disk, hashes and timings](dataset_confidence_v2/resources.json)
include both initializations and Dataset passes. The hashed reproducer is
retained in the accompanying evidence archive.

The bounded confidence context holds one loaded contract and one row at a time;
global family context remains in the existing SQLite TEMP tables. Reader and
TEMP page caches retain their explicit bounds. Source validation and atomic
publication guards remain in place, including after the last emitted row.
Full reference/flat extraction remains O(N) by design. Writer input graphs,
the full reference loader, per-record Value/proof-line size, and the underlying
source-manifest limits remain the previously documented memory limitations.
There is no new bounded-memory claim for the exporter or full oracle.

## Integration and activation order

1. Review PR #82's producer correction and independently review the #77 → #80
   scientific format stack and #75 → #76 reader stack. Their present heads are
   pinned above; this branch does not rewrite them.
2. Establish one maintained integration base containing those approved stacks.
   Review `02212bc6` as a delta over its explicit two-parent source integration,
   and the demonstrated performance delta `ae35addb85ce75bd2438214fbadfe22ad677c307`
   independently of benchmark/report commits.
3. Apply this confidence implementation as its own delta over that base; the
   local merge is a reproducibility vehicle, not a proposed duplicate of the
   other PRs. Retain all pinned full/indexed scientific comparisons and CI.
4. Keep v1 historical artifacts and scientific v2 sources distinct. Require
   trustworthy producer/checkpoint lineage: previously persisted false FORCED
   values cannot generally be distinguished from genuine proofs by this policy.
   The new reader does not retroactively validate mathematical truth of a
   producer's certificate.
5. Before separately approved production activation, validate supported producer
   invariants and training consumers of Dataset v2; measure representative bounded
   extraction, inspect exactness/filter/counter changes, and keep routing opt-in.
   Passing export/reader parity alone is not scientific activation approval.
6. Keep the four `pv_exact_then_*` policies disabled. A future change requires
   a separate mathematical proof contract and review; family membership, scalar
   equality or a maximum over different states is insufficient.

No gen38 restore/continuation, historical experiment, distributed worker,
production activation, automatic merge, remote push or duplicate-code PR is
part of this investigation. The dirty research checkout and all source PR
worktrees remain untouched.

## Reproduction

Use Python 3.13 with the repository's pinned Anemone 0.2.26, Atomheart 0.2.11,
Valanga 0.2.5 and test dependencies. Always choose new scratch roots:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python \
  docs/research/dataset_confidence_v2/audit.py --root /tmp/NEW-confidence-v2

# Extract exactly 77267b64's src into an independent scratch directory first.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/BASELINE/src:. python \
  docs/research/dataset_confidence_v2/audit.py --legacy-only \
  --root /tmp/NEW-confidence-before

PYTHONPATH=src:. python -m pytest --no-cov \
  tests/environments/test_morpion_dataset_target_confidence.py \
  -p docs.research.benchmarks.morpion_training_export_20261020.validation_safety
```

The audit refuses to replace its root. It compares all saved fields and bytes,
keeps every scientific distinction, and uses only newly generated small inputs.
