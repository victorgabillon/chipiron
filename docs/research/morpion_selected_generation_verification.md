# Selected-generation verification for Morpion discovery

Ordinary discovery now reads advisory manifests, preserves the existing selection
rules, and verifies the selected generation under the existing lifetime pin.
It no longer verifies every historical capsule on each poll. **Every file in the
selected seal is still freshly hashed**, including old immutable shards, old
updates/descriptors retained by the current format, model dependencies and the
checkpoint. No source format, scientific policy, receipt or checkpoint changes.

Implementation: `a2904d106daccce2f44d7de58d4410dc924e0c79`.
Exact base: `fa9955a2b9fc31ac43a97ed358affda801029a3b`. Architecture review:
`3623bb0f60d784476294f93fa533f7ec2d0e35db`. This is a new isolated worktree at
`/home/pompote/oldata/victor/morpion-scratch/selected-generation-verification-20261021/worktree`,
branch `perf/morpion-selected-generation-verification`. Scratch, fixtures, profiles,
CI environment and output are on the secondary ext4 filesystem. All **29 existing
worktrees** retain their HEAD, status and tracked-diff fingerprints; see
[protection evidence](selected_generation/evidence/protected-worktrees.json). All inputs to
experiments here were newly generated. No historical experiment, gen38 operation,
distributed worker, remote PR, merge or production activation was performed.

## Old and new contract

| Boundary | Before | After |
| --- | --- | --- |
| `load_available_pipeline_manifests` | Decode all manifests, reread authority and verify one complete capsule for every managed manifest | Decode advisory manifests only; malformed JSON/schema still fails |
| Dataset/Training candidate rules | Status, paths, descending generation, existing claims/lower bounds and stale-stage recovery | Same functions and ordering, unchanged |
| Selected stage entry | Check selected manifest against run state and validate its capsule without retaining a stage lifetime pin | Select a pin from the advisory capsule reference/path, acquire it, then read authority, verify lineage/head/manifest identity and hash the entire selected closure |
| Selected scientific source | Existing v1/v2 readers and explicit v2 opt-in | Same readers, Values, certainties, source identities and opt-in gates |
| Dataset output | Existing streaming writer and its atomic replacement | Same writer; recheck the selected closure when the row iterator exhausts, before output replacement |
| Training completion | Existing model/result/status/active-model protocol | Retain selected pin throughout; recheck closure before completion/status/active-model publication |
| Reevaluation | Reader pin ended after node reconstruction, before inference and patch publication | Existing `ExitStack` now spans inference, pre-publication verification, patch publication and cursor save |
| Historical archive audit | Implicit capsule verification during every discovery | Explicit `audit_generation_publications(work_dir)`, which also covers committed roots whose discovery manifests are missing |

The new loader is deliberately **not a validation certificate**. A successful
metadata load alone must never authorize consuming a managed source. Worker stage
entrypoints use `pinned_generation_manifest`; direct selected validation remains
available through `validate_manifest_publication`. Direct bounded/full sharded
readers retain `generation_read_pin`. No caller receives a reusable validity token,
process-global cache or new source of authority.

The selected pin is the existing stable `capsule_lease` flock inode. The discovery
reference chooses which pin to acquire; it cannot commit anything. Inside the pin,
`generation_read_pin` rereads `run_state.json`, checks the supported publication
version, unique roots, contiguous generations/parent links, head/checkpoint/source
agreement, selected membership and exact manifest/projection identity. The ancestry
check was moved from the Growth-specific validator into the existing shared root
parser, rather than duplicated. The existing complete `validate_capsule` and v2
`validate_committed_identity` functions are unchanged. The observed run state is
synced through the existing durability primitive before exposure.

A newer committed head can appear while an older selected capsule remains pinned.
The selected record must still belong to that committed lineage. There is no
rollback to an older checkpoint or reuse of newer receipts with an older head.
Failed verification does not trigger selection of another generation. Reevaluation
continues to skip missing unmanaged legacy source paths as before, but a missing
managed capsule source fails closed instead of falling back to another generation.

Historical discovery without the new marker remains readable. Old adopted v1
baselines with external legacy pointers need an initial root lookup to find their
pin; authority is then observed again **after** acquisition. Pre-protocol history
and receipt-only workspaces keep their existing interpretation; this change does
not invent sealed scientific identities for unmanaged legacy files. The complete
closure claim applies to the committed capsule inventory. Adoption's legacy
external-pointer compatibility is unchanged, not a new immutable-storage guarantee.

An unrelated old checkpoint may be damaged while the selected current closure
remains intact. Ordinary selected work can proceed; explicit audit rejects the old
capsule. By contrast, damage through an old hardlink to a record required by the
selected capsule rejects the selected operation. Deleting only a redundant old
hardlink does not delete the selected dependency; tests distinguish these cases.
All historical files listed in the selected seal remain dependencies even when a
future layout could avoid retaining them.

For a deliberate full capsule archive audit:

```python
from pathlib import Path
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    audit_generation_publications,
)

audit_generation_publications(Path("/WORKSPACE/TO/AUDIT"))
```

This uses existing root, pin and capsule validators. It audits the committed root
list observed at entry, re-adopting each under its pin. Later appended generations
belong to a subsequent audit. It does not repair or delete anything. Growth's
existing receipt/recovery checks remain separate and unchanged; the audit is not
a new transaction authority or a replacement for those checks.

## Integrity and scientific boundaries

Byte hashes are recomputed; filenames, mtimes, parent hashes and SQLite cache
presence never substitute for them. The existing seal inventory, path containment,
symlink rejection, size/hash checks, exact file-set check and scientific identity
checks remain intact. Producer, scientific contract and generation identity remain
bound to the sealed source/checkpoint. Reader index reuse retains its own integrity
checks. Pins prevent cooperative collection, not out-of-protocol writes; source
rechecks at output boundaries are retained or strengthened accordingly. There is
no claim of atomic immunity to arbitrary external writes between individual file
reads and publication. The existing cooperative immutable-writer protocol and
SHA-256 trust assumption remain required.

All 465 untouched production Python files match the exact base, including writer,
reader/parser/index, Dataset target/confidence/family logic, neural evaluator,
serialization, checkpoint/RNG and patch receipt implementations. AST checks also
confirm that the capsule/integrity validators, lease implementation, six eligibility
and selection functions, Dataset diagnostics and stale-stage recovery function are
unchanged. See [source fingerprints](selected_generation/evidence/source-fingerprints.json).
This is supplementary evidence; scientific output regression tests remain required.

Dataset v1 retains all nine policies. Confidence v2 retains the five supported
policies and all four `pv_exact_then_*` gates. FORCED(35) stays exact for its state;
selected neural ESTIMATE(42) stays estimated; certified terminal targets cannot be
overwritten by uncertified family predictions. No cross-state proof is introduced.
Checkpoint representations, selector state, model identity, cursor order/wrap,
Reevaluation patch contents, target provenance and training row serialization are
unchanged. The unresolved scientific activation prerequisites remain prerequisites.

## Matched G8–G64 evidence

[The driver](selected_generation/measure.py) reuses the existing four-node history
fixture and `StageMeasurement`, `/proc` CPU/I/O counters, cProfile and tracemalloc.
N=4, K=0 after genesis; same legal states, graph, scientific producer and patch
sequence. The old and new discovery processes read **the very same generated
files**, serially, at each generation. The old implementation is the exact base
worktree; new implementation is this isolated worktree. The explicit audit is a
third fresh process. No benchmark overlaps CI or another benchmark.

The comparison includes metadata loading, the existing Dataset candidate selector,
and selected-entry validation: old discovery plus `load_generation_manifest` versus
new discovery plus `pinned_generation_manifest`. Thus the old measurement validates
G+1 capsules. The user's previous G64 discovery-only baseline of 6,816 hashes /
55,070,410 bytes excludes that extra selected-entry validation. The explicit audit
here independently reproduces those exact historical hash counts/bytes.

| G | Capsules old → new | Run-state reads old → new | Hash calls old → new | Hash bytes old → new |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 9 → 1 | 9 → 1 | 213 → 33 | 1,151,395 → 201,652 |
| 16 | 17 → 1 | 17 → 1 | 609 → 57 | 3,844,242 → 399,099 |
| 32 | 33 → 1 | 33 → 1 | 1,977 → 105 | 14,198,144 → 821,993 |
| 64 | 65 → 1 | 65 → 1 | 7,017 → 201 | 56,848,396 → 1,777,986 |

| G | Profiled wall seconds old → new | CPU seconds old → new | Actual process peak RSS MiB old → new |
| ---: | ---: | ---: | ---: |
| 8 | 0.163 → 0.046 | 0.163 → 0.029 | 283.91 → 283.89 |
| 16 | 0.470 → 0.066 | 0.470 → 0.049 | 283.99 → 283.77 |
| 32 | 1.524 → 0.103 | 1.523 → 0.086 | 284.27 → 284.00 |
| 64 | 5.500 → 0.186 | 5.497 → 0.168 | 285.71 → 284.69 |

G64 hashes fall 97.1% by call count and 96.9% by bytes. Profiled wall time is about
29.6 times lower in this single matched sample. These are warm-cache synthetic
measurements, not production throughput forecasts or a substantial RSS-reduction
claim. Physical read bytes were zero; logical reads are recorded separately in
[scaling evidence](selected_generation/evidence/scaling.json). RSS is measured
before the separate allocation pass and includes imports/runtime overhead.

| G | Manifest-only retained allocations old → new | Full operation retained traced bytes old → new |
| ---: | ---: | ---: |
| 8 | 1,276 → 1,280 | 113,312 → 113,494 |
| 16 | 2,528 → 2,526 | 210,544 → 210,608 |
| 32 | 5,020 → 5,019 | 404,362 → 404,544 |
| 64 | 9,957 → 9,952 | 8,481,647 → 792,186 |

These allocation counts describe retained allocations, not every allocation over
time. Metadata projection bytes do not change in this PR; the small allocation
count variations are not a new memory contract. At each G the three processes have
identical selected scientific identities and run-state SHA-256. Assertions verify
that every selected inventory path is in the actual hash-call trace, including old
record/update files and model/checkpoint dependencies. Old entry hash bytes/calls
are exactly new selected-entry work plus full-history audit work. Raw profiles and
measurements remain under `history-final` in the new scratch root (the preliminary
run is also preserved separately under `history`); committed evidence includes their
fingerprints and selected path traces.

These are **entry-boundary** costs. Complete Dataset, Training and Reevaluation
operations can verify the selected capsule more than once: nested existing readers,
patch provenance lookup and pre-publication rechecks have distinct scopes. This PR
does not pretend that a full operation performs only one hash pass. There is no
history-wide validation in the ordinary discovery loader. Removing redundant
same-scope checks with a validated-pin result remains a separate review item.
Dataset's internal stage measurement now starts after the outer pin entry; the
measurements above include that entry explicitly. Scientific metadata is unchanged;
wall-clock completion timestamps and resource observations are measurements, not
byte-parity assertions.

## Tests and CI

All **2,215 distinct selected CI test identities** now have final outcomes:
**2,213 passed and 2 CUDA-unavailable skips**, with the four normal CI marker
exclusions. This is an honest aggregate of the full attempt and focused reruns,
**not a single clean full-suite invocation**. The full installed serial run reached
2,182 passes, five failures and two skips, then stalled in the supervised-training
smoke test. Its temporary path exceeded the Unix socket path limit: the failures
reported `AF_UNIX path too long`, and the DataLoader waited for a failed worker.
The stalled process was interrupted, retaining its test identities and outcomes.
No scientific assertion failed.

A shorter temporary directory on the same secondary disk fixed the harness issue.
All five failed and 25 unfinished tests passed unchanged in 37.58 seconds. The
final generation-label guard then received a new regression and an installed
**188-case affected-suite rerun**, all passing in 106.18 seconds. All 29 new
selected-verification cases pass. The final strict mypy/pyright/Pylint ratchet has
206 inherited diagnostics against the 209-entry reviewed baseline, **zero new
errors**. Ruff, toolchain checks, sdist/wheel build and installed-distribution smoke
checks pass; all 471 installed production Python files match this checkout.
[Validation summary](selected_generation/evidence/validation.json) and
[every final test outcome](selected_generation/evidence/test-outcomes.json) retain
the interruption, failed identities, retries and exact evidence fingerprints.

New regressions cover v1/v2 selection without capsule reads, exactly one selected
entry verification, authority observed after the pin, stale projection/source/head
identities, broken ancestry, relabeling G1 discovery as G2, selected seal/model/checkpoint/source damage, missing
dependencies, historical shared-record damage, unrelated checkpoint damage detected
by explicit audit even with missing discovery, concurrent append while an old
reader is pinned, exclusive-lock exclusion, Reevaluation inference/publication/
cursor pin lifetime, and a source change during inference. Dataset rechecks before
JSONL replacement; interrupted rows are not published. Training retains the same
selected lifetime scope.

The previous projection tests now invoke selected validation for their corruption
assertions rather than treating advisory metadata loading as a certificate. Their
rejection expectations remain intact. Existing status/claim ordering, stale recovery,
missing-manifest repair, publication crash/fsync, exact receipt conflict, source
substitution, model dependency, scientific reference/indexed JSONL, checkpoint/
selector continuation and Reevaluation patch/cursor tests are included in CI.

## Complexity and scope left for later PRs

Let D_g be total selected sealed dependency bytes (scientific source, checkpoint,
models, configuration), F_g its file/path count, R_g cumulative receipts and M_g all
discovery metadata bytes at generation g. Ignoring implementation-specific path
sorting costs, old ordinary discovery plus entry reads/hashes
`Θ(M_g + (g+1)(g+R_g) + Σ(i≤g) D_i + D_selected)`; the new entry reads
`Θ(M_g + g+R_g + D_selected)`. Existing candidate sorting is O(g log g), claim/status
inspection remains O(g), and validation traverses O(F_selected) inventory/path
entries. Adopted legacy baselines may need one extra root lookup. No-work polls
load metadata and claims but do not validate an unselected capsule.

RAM still includes O(M_g + g+R_g + F_selected), existing bounded hash/parser buffers,
and selected reader windows/caches. There is no new O(N) tree or dictionary, but
neither manifests, run-state JSON nor inventories are independent of history. For
fixed-size discovery records M_g=Θ(g); arbitrary diagnostics can be larger.
This PR adds no persistent data structure and deletes no artifacts. Retained disk
and cumulative publication cost are unchanged.

For the current scientific-v2 layout with fixed-size node states, all old updates
and descriptors remain in the selected capsule. The review's all-history byte model
`Θ((N+E)g² + g³)` becomes selected-closure `Θ((N+E)g + g²)` per entry, plus checkpoint,
model and run-state terms. This is **not O(1)**. The G² descriptor-prefix term, Θ(G²)
retained namespace/inventory entries, authoritative receipt/run-state rewrites,
retained checkpoints/models/rows/indexes and absence of semantic retirement remain.
Use sums over actual N_i/E_i/bytes when nodes grow by K or replay lengths grow.

Full audit intentionally retains history-wide work and repeated root reads; it is
not a background worker or a hot poll. Durable stage result transactions, TTL claim
fencing, active-model handoff, compact scientific inventories and safe retention
remain separate tasks. No parent-hash trust shortcut or age-based deletion is
introduced.

## Code size and dependency strategy

Production Git numstat is **479 added / 315 removed, net +164 lines** across six
modules. Ignoring whitespace-only changes gives **207 added / 43 removed**, the
same net +164. Replaced lines count on both sides. The exact per-file delta is in [Git numstat evidence](selected_generation/evidence/code-delta.json).
Most conventional Reevaluation diff churn is indentation: its existing `ExitStack`
now encloses the existing inference/publication tail. Both conventional and
whitespace-insensitive counts are reported, rather than hiding that review surface.
There are no new production modules/classes, databases, journals, caches, schedulers
or locks. Small scope helpers make the pin lifetime explicit; the row iterator
places verification before the existing atomic writer's commit. The ancestry check
has one implementation, and the per-manifest historical validation loop is removed.

All pinned prior stacks are already ancestors; nothing was cherry-picked twice.
See [exact ancestry](selected_generation/evidence/ancestry.json). Local sequence:

```text
1f335acf (reviewed implementation/report)
  → 3623bb0f (architecture review)
  → a0666f45 → fa9955a2 (thin versioned discovery)
  → this selected-generation verification delta
```

The base still contains the local merge of sibling scientific/confidence and
durability stacks. No locally known remote branch contains fa9955a2; this report
makes no claim about a newly fetched GitHub state. Keep this delta local until a
reviewed common base contains those prerequisites and thin discovery. Then apply
only this delta onto the exact accepted thin-discovery head and rerun parity and
recovery CI before opening a small stacked draft. Do not open a duplicate-code PR
against either sibling. Subsequent PRs should separately address scoped validation
reuse, a versioned compact dependency layout, and reachability-based retention.

## Reproduction

Use a fresh ext4 output directory, the pinned Python/dependency environment, and a
checkout of fa9955a2 as `MORPION_DISCOVERY_BASE`. Set `TMPDIR` to a **short** path on
the secondary disk (leave room below the 108-byte Unix socket path limit).
From this worktree, with this checkout also on `PYTHONPATH` for the probe utilities:

```bash
MORPION_DISCOVERY_BASE=/PINNED_BASE_CHECKOUT \
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/SHORT_SECONDARY_TMP \
PYTHONPATH=/PINNED_BASE_CHECKOUT/src:. \
python docs/research/selected_generation/measure.py \
  --root /SECONDARY_DISK/NEW_SYNTHETIC_HISTORY --last 64
```

The generator uses the pinned base's publication code. Each observation launches
fresh old/new/audit readers serially against the same files. Keep CI and other
benchmarks idle during resource measurements. Standard project CI runs installed
packages; use `-n 0`, `CHIPIRON_TEST_INSTALLED=1`, existing output-directory overrides
and the existing validation-safety plugin for the synthetic test run. This does
not authorize a production worker or historical experiment.
