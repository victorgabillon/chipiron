# Lean versioned Morpion discovery metadata

## Scope and dependency base

This is only the discovery-projection step from the
[architecture review](morpion_lean_durable_pipeline_architecture.md).
Implementation: `a0666f45978a0daf44fddadc8eb9ba9e8b99d5c5`.
Exact base: `3623bb0f60d784476294f93fa533f7ec2d0e35db`, whose sole parent is
`1f335acfa362469dae3636329fa1054365caeb0b`. The reviewed implementation is
`c1c737ca0d9fb76f9a9d2ad68d0f999a4c9f0be1`. The review changed no production
Python files above that implementation. The new isolated branch is
`refactor/morpion-versioned-discovery`, under
`/home/pompote/oldata/victor/morpion-scratch/lean-discovery-v2-20261021/worktree`.
All scratch, generated histories, caches, logs and CI outputs are on the secondary disk.

The complete source stack is already contained in the base; no implementation was
cherry-picked or applied twice. See [ancestry evidence](lean_discovery/evidence/ancestry.json)
and the [existing consolidation graph](morpion_scientific_durable_consolidation.md).
This remains a local implementation because the reviewed base is a local union of
sibling scientific and durability stacks. A PR against one of those siblings would
include unrelated, already proposed changes. No remote PR, merge or activation was
performed. All **28 pre-existing worktrees** retain their HEAD, status and tracked-diff
fingerprints; see [protection check](lean_discovery/evidence/protected-worktrees.json).
Existing worktrees and historical evidence are preserved.

## Exact old/new contract

Top-level generation, timestamps, checkpoint/source/rows/model paths, Dataset and
Training states, evaluator results, record/frontier status and all existing
nonledger metadata retain their meanings and serialization. No scientific export,
checkpoint, receipt, claim, cursor or Reevaluation patch wire format changes.

| Field | Historical discovery (no new marker) | New managed discovery |
| --- | --- | --- |
| `metadata.durable_generation_publication_v1` | Opaque cumulative publication ledger, unchanged when read/round-tripped | Absent |
| `metadata.checkpoint_bound_reevaluation_v1` | Opaque cumulative patch-receipt ledger, unchanged when read/round-tripped | Absent |
| `metadata.pipeline_discovery` | Absent; no version or certainty inferred | Explicit envelope below |
| Other `metadata` keys | Preserved | Preserved exactly, including diagnostics and stage results |
| Authoritative ledgers | Committed `run_state.json` | Same committed `run_state.json`; no second ledger |

Discovery version 2 is independent of the scientific-export and capsule versions.
The envelope has exactly four keys:

```python
{
    "schema_version": 2,
    "generation": state.generation,
    "publication": state.metadata["durable_generation_publication_v1"]["generations"][
        -1
    ],
    "checkpoint": None,
}
```

This Python projection serializes `None` as JSON `null`. `publication` is the
existing record object, not a new record schema. It contains capsule ID, parent ID, generation, descriptor path and
SHA-256, checkpoint/source paths, and, for scientific publication v2, the existing
publication version and scientific identity. That identity retains source format,
scientific contract/generation, producer/config/API/package identities, catalog and
generation descriptor hashes and checkpoint hash. The existing sealed capsule
inventory binds model and source dependencies. **A parent hash does not replace
inventory verification.** Only the current root record is copied; no ancestor list,
receipt list, inventory or canonical-validation history is copied.

Receipt-only mode uses `publication: null` and
`checkpoint: {"path": <committed checkpoint path>, "sha256": <receipt-head hash>}`.
Exactly one reference is present. A receipt-only baseline may later be adopted
into a v1 capsule at the same generation. Its existing projection is retained;
recovery still compares its checkpoint identity with committed run state. In
capsule discovery this checkpoint-only form is allowed for the adopted baseline,
with the existing authoritative path and full capsule checks. Its advisory hash
is not a substitute for those checks. Later managed generations use capsule records.

The manifest and Dataset/Training status types share one envelope-shape check.
Unknown/noninteger versions, generation mismatches, malformed checkpoint hashes,
missing references and mixed old-ledger/new-envelope metadata are rejected. The
current root record is compared against authoritative run state at discovery;
existing publication validation checks the actual dependencies. There is no
parallel capsule validator in the envelope decoder.

Historical files are not migrated or compacted. An existing cumulative historical
prefix therefore retains its old space cost. Pending historical generations also
keep their legacy metadata on stage updates; the linear-growth claim applies to
newly published managed generations, not a silent migration of old artifacts. If recovery finds an existing historical
manifest, it keeps the original bytes and semantics. Stage updates to historical
artifacts continue to round-trip their old metadata; they are not relabeled v2.
Missing discovery for a committed head is reconstructed in the new format. This
compatibility promise concerns historical artifacts read by the new implementation;
it is not authorization to downgrade a running durable pipeline to old writers.

## Consumer audit and ownership

The audit covered source and script references to both ledger keys, manifest/status
metadata access, stage selection/recovery, and dashboard inputs. No downstream
worker or dashboard uses historical ledger entries from discovery/status metadata.
The one behavioral presence check was `save_pipeline_manifest`: the old ledger key
selected the durable JSON writer. The new marker now also selects that same writer.

| Consumer | Consumed fields, all retained |
| --- | --- |
| Dataset discovery and worker | Generation, source/checkpoint pointers, Dataset state, claims; scientific source supplies Values, confidence and provenance |
| Training selection and worker | Rows path, Dataset/Training states, generation bound, claims, evaluator/model fields; `dataset_rows`, `num_rows`, `dataset_completed_at_utc`, `auto_recovery` |
| Reevaluation | Selected generation/source and model identities; scientific provenance comes from the verified source |
| `pipeline/manifests.py` | Metadata forwarded to stage status; record/frontier fallback and prior Dataset status |
| Training recovery | Existing metadata plus `auto_recovery`; reset eligible stage state and preserve partial results/ownership checks |
| Dashboard history/performance | `dataset_rows`, `num_rows`, `tree`, `memory.rss_mb`, checkpoint/export timing diagnostics |
| Dashboard training/model views | `training_config`, validation metrics, parameter counts, model source-row metadata |
| Dashboard runtime/control/tree views | Authoritative run-state runtime/control/checkpoint fields; these remain untouched |

See [consumer references](lean_discovery/evidence/consumer-references.json).
We retain all nonglobal metadata rather than introduce a fragile allowlist or
remove extension diagnostics. Arbitrarily large user diagnostics can still be
large; this change specifically removes the known cumulative ledgers.

## Durability and scientific invariants

`run_state.json` remains the sole Growth commit point. Capsule creation, sealing,
inventory hashing, fsync, immutable dependencies, ownership locks, exact receipt
application/acknowledgment, producer cursor handling and orphan cleanup are unchanged.
Missing discovery is repairable from committed state; a plausible projection cannot
commit an orphan or select a different capsule. Existing historical manifests are
not rewritten on startup. Stage manifest rewrites with the new marker retain
`derived.provenance.atomic_json` (flush/file fsync, replace, directory fsync).
Receipt-only new stage manifest rewrites also use that durable writer. The existing
atomic Dataset/Training status writer is unchanged; this PR does not expand the
Growth transaction to downstream outputs.

No writer/reader, target policy, neural evaluator or scientific source serializer
changes. As with any change to a module covered by existing canonical-validation
implementation fingerprints, old validation evidence can require fresh validation
at the next publication. The existing fallback and integrity rules remain unchanged;
this does not alter source semantics or introduce a cache migration. All nine legacy
Dataset policies retain v1 meanings. The five enabled
confidence-v2 policies retain selected-source certainty and certified terminal
protection; all four `pv_exact_then_*` policies remain gated. No family proof is
invented. Complete scientific and recovery regression evidence is listed below.

## Validation

Local quality checks, sdist/wheel build, installed-package smoke checks and the
strict mypy/pyright/Pylint ratchet pass. The ratchet reports 206 inherited diagnostics
against a 209-diagnostic reviewed baseline, **zero new diagnostics**. All 32 new
projection regressions pass. The complete serial CI collection ran:
**2,173 passed, 11 failed, 2 skipped**,
with four normal CI marker exclusions. The 11 failures were the fixture-coherence
cases explained below. After the test-only correction, all **22 cases** in that
module passed. Final known outcomes across the full run and focused rerun are
**2,184 passed and 2 CUDA-unavailable skips**, covering all 2,186 selected test
identities. This is not represented as a single all-green full-suite invocation.
See [validation summary](lean_discovery/evidence/validation.json) and
[per-test initial outcomes and corrected rerun](lean_discovery/evidence/test-outcomes.json).
The full serial run took 1,305.53 seconds of pytest time; no benchmarks overlapped it.

The new regression fails on the exact review base because it still serializes the
publication ledger. Targeted tests cover scientific v1/v2 managed discovery,
receipt-only mode, same-generation baseline adoption, historical manifest/status
round trips, malformed versions, mixed envelopes, missing/corrupt capsule
artifacts, forged projection references, missing-manifest repair and durable stage
rewrites. Old/new eligibility and ordering are compared across the Dataset/Training
status matrix, including active claims and lower-generation bounds. Real status
writers and dashboard output are compared; stale Training recovery preserves
metadata and committed run-state bytes. The same immutable scientific source is
used for the five enabled v2 worker runs: complete JSONL bytes and resulting
manifest fields agree, except the explicit discovery representation and wall-clock
completion timestamp. The four gated policies reject both representations.

The existing CI scientific tests cover nine v1 policies, five v2 policies/four
gates, complete Values/certainties/links/references, source provenance, JSONL,
Reevaluation patches/cursors, the real FORCED(35)/ESTIMATE(42) regression, checkpoint
restore and selector state, interrupted publication, exact receipts and restarts.
Relevant passing coverage includes 134 streaming Dataset cases,
62 selected-target confidence cases,
96 Value-certainty cases, 71 scientific-reader cases,
68 durable scientific-v2 cases, 33 scientific publication crash cases,
51 checkpoint/receipt cases and 8 checkpoint/selector parity cases.
Production scientific files are also checked unchanged against the reviewed base.

The first full CI invocation exposed 11 existing canonical-validation fixture
failures: those tests deliberately resealed a synthetic predecessor and updated
run state while leaving discovery bound to the previous descriptor hash. The new
stale-projection guard correctly rejected that inconsistent fixture before its
intended scientific check. Both fixture constructors now remove that temporary
discovery file so normal recovery reconstructs it from the changed committed test
head. No production guard or test assertion was relaxed. All 22 tests in that
module then pass, including the six full-validation fallbacks and five malformed-
evidence rejections. The dedicated forged-projection test separately retains the
new rejection behavior. These edits only affect newly generated test artifacts;
there is no historical-data repair operation.

## Structural history experiment

[The driver](lean_discovery/measure.py) reuses the review's four-node history probe,
which reuses the existing scientific publication fixture and resource tools.
It adds one representative Dataset and one Training status file per generation
through the existing status writers. N=4 and K=0 after genesis; same legal states,
producer/model/config and patch sequence in both runs. Every production hash,
validator, transaction and fsync executes normally. The fixed graph is a control
for metadata growth; mutable scientific Values can change through the same patches.

Before and after histories are newly generated, serial runs. Each G8/G16/G32/G64
complete discovery measurement uses a fresh process and cProfile/hash counters;
tracemalloc runs separately. A second allocation measurement only loads manifests,
so metadata retention is distinguished from temporary authoritative-state/capsule
validation allocation. Allocations count retained Python allocations, not allocator
calls over time. RSS is actual process peak, including imports. Logical I/O is
`/proc/self/io`; it is distinct from hash bytes and physical disk reads. OS caches
are warm and these are single samples, not statistically established speedups.

[Raw measurements and assertions](lean_discovery/evidence/scaling.json),
[scientific fingerprints](lean_discovery/evidence/scientific-fingerprints.json).
Before/after in each cell:

| G | Retained manifest bytes | Dataset + Training status bytes | Historical publication / receipt entries in manifests |
| ---: | ---: | ---: | ---: |
| 8 | 137,562 → 37,751 | 269,604 → 69,982 | 36 / 28 → 0 / 0 |
| 16 | 496,882 → 77,180 | 982,724 → 143,320 | 136 / 120 → 0 / 0 |
| 32 | 1,876,592 → 156,050 | 3,731,104 → 290,020 | 528 / 496 → 0 / 0 |
| 64 | 7,280,231 → 313,798 | 14,516,302 → 583,436 | 2,080 / 2,016 → 0 / 0 |

Status artifacts duplicate twice the historical entries in the final column before
the change and zero afterward. At G64 the three artifact types together contained
**12,288 cumulative ledger entries**; new artifacts contain only 192 current
publication references (one per artifact). Manifest bytes fall **95.7%**. Doubling
G32 to G64 multiplies old manifest bytes by 3.88 and new bytes by 2.01.

| G | Manifest-only retained allocations | Manifest-only retained traced bytes | Full-discovery retained / peak traced bytes |
| ---: | ---: | ---: | ---: |
| 8 | 2,658 → 1,178 | 239,499 → 92,357 | 260,251 / 600,666 → 113,109 / 453,524 |
| 16 | 8,384 → 2,397 | 796,389 → 188,246 | 818,071 / 1,212,461 → 210,282 / 604,672 |
| 32 | 28,865 → 4,873 | 2,848,384 → 382,266 | 2,870,336 / 3,384,806 → 404,218 / 918,688 |
| 64 | 105,913 → 9,784 | 10,704,232 → 767,887 | 18,418,733 / 19,148,819 → 8,481,562 / 9,211,648 |

At G64, manifest-only retained memory falls **92.8%**. Full-discovery retained
allocations still reach 8.48 MB, well above the 0.768 MB of manifests. The remaining
validation/runtime allocations are outside this projection change; the full path
is not bounded independently of history.

| G | Hash calls / hash bytes, identical both runs | Discovery logical bytes read | Profiled wall s | CPU s | Process peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 180 / 949,743 | 1,401,560 → 1,301,765 | 0.139 → 0.142 | 0.138 → 0.142 | 284.66 → 284.64 |
| 16 | 552 / 3,445,143 | 5,060,029 → 4,640,295 | 0.415 → 0.422 | 0.415 → 0.421 | 284.91 → 284.64 |
| 32 | 1,872 / 13,376,151 | 19,446,083 → 17,725,509 | 1.389 → 1.394 | 1.389 → 1.393 | 292.41 → 292.39 |
| 64 | 6,816 / 55,070,410 | 78,565,772 → 71,598,955 | 5.164 → 5.034 | 5.163 → 5.033 | 298.75 → 298.73 |

Physical read bytes were zero in these warm-cache discovery samples. Logical reads
decrease mainly by the removed manifest bytes; hashing remains unchanged. Wall time and
process peak RSS show no substantial improvement: the G64 poll is still about
**five seconds and 299 MiB**. This is a structural metadata fix, not a claim of a
history-independent or faster end-to-end pipeline.

All 193 final scientific source files match exactly. All 64 committed scientific
identities, including checkpoint hashes, match; all G2–G64 complete-export and
checkpoint hash pairs match. Every new current-record projection equals its
authoritative committed root. Producer identity and scientific writer counters
match. Resource counters and timestamps are compared as measurements, not required
to be equal. UUID capsule names are independently generated; no scientific bytes
are normalized to obtain parity.

## Complexity and remaining bottlenecks

For generation g let R_g be cumulative receipt count, P_g the one current root
record and D_g retained nonglobal diagnostics/stage metadata. Before, each
manifest/status copied Θ(g+R_g) ledger entries; retaining G generations required
Θ(Σ(g+R_g)), quadratic with one receipt per generation. New projections require
Θ(P_g+D_g) per artifact and Θ(Σ(P_g+D_g)) across generations. For fixed-size
identities/diagnostics this is linear in G, apart from encoded integer digit widths.
It is independent of N/E except where existing diagnostics themselves store data.
Constructing the projection shallow-copies a fixed set of metadata keys and selects
one existing root; it does not traverse or serialize the removed ledgers.

This is **not** a history-independent discovery algorithm. Candidate selection still
loads every manifest. Every capsule still undergoes full historical verification.
The current run state still holds Θ(G+R_G) authoritative metadata and is reread for
each candidate. Capsule inventories, historical descriptors and linked export
names still grow. For fixed N/E, the review's Θ((N+E)G²+G³) hash-byte model for one
all-history discovery remains. Retained manifests lose the ledger-quadratic term;
transient parsing of current run state and a capsule inventory remains. No cache,
ancestry compaction, retention deletion or integrity shortcut is introduced.

## Code delta and review sequence

Only three production modules change. Two private functions are justified:
`_discovery_metadata` expresses the single projection at the existing post-commit
publisher; `_discovery_metadata_dict` applies the explicit envelope fence to the
three existing artifact types. One shared key names that wire contract. Current
record comparison reuses the existing publication checks; no new model hierarchy,
ledger, database, cache, serializer, journal or transaction interface is added.
The duplicated cumulative serialization is removed at its source, so existing
status writers naturally inherit lean metadata. Production Git numstat is
**126 added / 5 removed, net +121 lines** (including comments and blanks), with 5 replaced lines counted on both sides: 42/1 in the
post-commit publisher, 13/0 in publication I/O, 71/4 in artifact types. There are
no new production modules or public classes. Tests add 466/remove 4 lines; the
measurement/assertion drivers are separate research code. See
[exact counting method and per-file delta](lean_discovery/evidence/code-delta.json).
The new lines predominantly enforce explicit version/reference validity and
retain existing durability behavior; no performance cache is hidden in the delta.

Recommended independent review order:

1. Establish an upstream union containing the pinned scientific/streaming/confidence
   and durability changes, then the existing v2 publication and canonical-validation
   reuse work. Preserve source branches; follow the prior dependency plan.
2. Apply only this discovery-projection delta above the equivalent of `3623bb0f`.
   Re-run compatibility/scientific/recovery checks on that actual base; open a small
   draft PR there. Do not target an earlier sibling and duplicate the union.
3. Separately redesign candidate selection and redundant historical verification,
   with explicit authoritative validation under existing read pins. Hash removal
   needs an integrity proof and its own failure tests.
4. Separately version compact scientific inventories/ancestry and define semantic
   liveness/archive obligations before any retention deletion.

No production activation follows from these tests. The prior model-epoch,
end-to-end downstream durability, family-proof and resource/retention prerequisites
remain in force.

## Reproduction and evidence locations

Use a fresh secondary-disk `CASE_ROOT` and the same Python/dependency environment
for both histories. Run from this implementation worktree, with the pinned review
worktree available read only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REVIEW_WORKTREE/src:." "$CI_PY" docs/research/lean_discovery/measure.py --root "$CASE_ROOT/before" --last 64
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. "$CI_PY" docs/research/lean_discovery/measure.py --root "$CASE_ROOT/after_" --last 64
"$CI_PY" docs/research/lean_discovery/compare.py "$CASE_ROOT" "$CASE_ROOT/comparison"
```

Run those commands sequentially, with `TMPDIR` on the secondary disk. The driver
retains cProfile files and publication/resource JSON. `source_module` in each
discovery result confirms which implementation was imported. Full raw fixtures,
profiles and CI logs for this run remain under
`/home/pompote/oldata/victor/morpion-scratch/lean-discovery-v2-20261021/`.
The first after-measurement is retained in `after-prelint`; final tables use only
the final implementation in `after_`. Incomplete setup attempts are separate and
excluded. Committed evidence includes the exact numerical observations,
scientific fingerprints, ancestry, code scope and validation summaries.

The CI environment was newly created on the secondary disk. Third-party packages
were reused read only; Chipiron and the checked evaluator/checkpoint dependencies
are installed/copied locally. All **471 installed production Python files** match
the implementation source byte for byte; only the three reported files differ
from the review base. See [installed-source evidence](lean_discovery/evidence/installed-source.json).
Full tests use `CHIPIRON_TEST_INSTALLED=1`, `PYTHONPATH=.` (no source override),
`-n 0` and the repository's `not integration and not external_data` selection.
The existing validation-safety plugin redirects default leaderboard writes to
temporary test directories; no scientific assertion is disabled.
