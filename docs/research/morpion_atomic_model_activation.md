# Explicit atomic Morpion model activation

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

## Result and exact foundation

Implemented an explicit accepted-Training M1 → M2 operation for **fresh, independently
validated checkpoint-format-3 workspaces**. Training completion does not call it.
The operation owns one private search runtime, performs one complete Anemone
scientific refresh, and publishes through the existing closed-capsule/run-state
transaction. It introduces no additional authority, journal, receipt ledger,
cache, tree representation or scientific writer.

This supersedes the checkpoint stop condition in
[`morpion_atomic_model_activation_preflight.md`](morpion_atomic_model_activation_preflight.md).
That report and its unchanged regression witnesses remain available as evidence.
Format 2 is explicitly rejected, never relabelled or migrated.

| Role | Exact commit |
| --- | --- |
| Reviewed Chipiron base | `448d65970ca6c7acdac54e792e54ceb8f9a6f715` |
| Preflight diagnostics | `c605d9d22823c28fecbfc356fc077a928f744076` |
| Preflight implementation base | `f1a4a6e61bcfd5925e5151c7caa1b7e2daced72c` |
| Inherited lean CI | `b67bd94572a1398bd82cb6187e70c0e20240a06c` |
| Exact corrected Anemone | `90c7ef4454edf52459d7f7051b6c904ddf7f7a35` |
| A: strict format-3 integration | `d5eb78977b35efa1b5f0071cb21de4a7ceb5017b` |
| B: atomic activation | `cf47ff8c7f63d9c50521f3db523c87f749851be1` |
| C: certificate/crash/concurrency hardening | `9955bb2c0289a5471ca1a21df72e545390bef2aa` |
| Final executable measurement harness | `331413412317d7b58691936718f0852e22d39e4e` |

All production sources are unchanged after C. The final harness descendant also
contains `e6b09398370b079f9729090196ebe401e4413b4a`; both changes affect research
instrumentation only. Validation and benchmark SHAs are recorded separately from
the final documentation/evidence commit.

The isolated branch is `feat/morpion-atomic-model-activation`, at
`/home/pompote/oldata/victor/morpion-scratch/atomic-activation-impl-20261024/worktree`.
Its sibling `venv`, `wheels`, `logs`, `measurements` and short temporary directory
`/home/pompote/oldata/victor/ma24` are on the secondary filesystem. No protected
branch was rebased or edited. No historical fixture, gen38, production activation,
distributed deployment, package publication or merge was performed.
The protection audit compares all 43 pre-existing Chipiron and 13 Anemone worktree
registrations, including 26/12 present worktrees' HEAD, status and tracked diff
digest. Pre-existing absent `/tmp` registrations remain untouched. See
[`protected_worktrees.json`](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence/protected_worktrees.json).

The exact Anemone archive was built and installed independently. Its unchanged
package version `0.2.26` is insufficient to identify this dependency. Wheel SHA-256:
`5f3015c1a07d7a838a729896a388fad2b31f666f9abea7e82eb6f3381cb52890`.
All 289 installed Anemone Python files and 477 installed Chipiron Python files
match their recorded sources. See
[`dependencies.json`](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence/dependencies.json) and
[`installed_source.json`](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence/installed_source.json).
The latter also records the built Chipiron wheel/sdist digests. No package was
published; consumers must integrate the pinned dependency before this branch.

## Request and admitted contract

The small public domain entry points are
`prepare_model_activation_request(paths, training_generation=...)` and
`activate_model(paths, request)` in
[`model_activation.py`](../../src/chipiron/environments/morpion/bootstrap/model_activation.py).
Preparing a request captures authority read-only under the existing Growth lock.
Calling `activate_model` is the explicit authorization; there is no automatic
worker routing or persistent request queue.

Request schema 1 binds:

- Workspace and research epoch, expected current capsule/head and complete M1 identity.
- Original accepted Training descriptor and exact selected candidate, including
  its full bundle digest. Validation reuses accepted-result/Dataset dependency
  validation; an earlier accepted generation is admissible without substituting
  the latest mutable model directory.
- Frozen bootstrap configuration digest and effective runtime configuration.
- Default-max/Linoo alternating/all-children search, checkpoint format 3,
  `selected_value_v2`, Growth numeric schema 2, refresh scope `all`, batch size 64.
- Accepted cursor digest and the digest of the **entire exact receipt ledger**.

Activation reacquires the one Growth lock and compares the complete request to
freshly validated authority. A changed head, epoch, model, candidate, configuration,
policy, cursor or receipt frontier refuses the request. The model's objective
and all additional producer identity fields are retained; only its evaluator
identity is authorized to change. The explicit configuration supports bounded
rollout or disabled rollout. Custom backup policies and arbitrary external
certificate types are outside the admitted contract.

Fresh genesis requires explicit `model_activation=True`, closed scientific
publication, atomic Reevaluation with exact scoring, fenced Dataset and durable
Training. It cold-restores the actual model/checkpoint and verifies certificates
before adopting the contract. Initialization remains opt-in. Format-2 and missing
checkpoint version headers fail explicitly in typed/JSON-zst and sharded loaders.
No historical workspace is automatically adopted.

## Authority and state machine

The existing version-3 publication metadata and sealed capsule add
`evaluator_contract: 1` and an explicit `active_evaluator` identity. Exactly one
frozen model pin must match it; the scientific producer's evaluator digest must
match too. The authority/descriptor contract markers must agree. Retained M1
receipt dependencies may coexist with M2 pins. Neither pin position nor a mutable
model filename chooses the evaluator.

```text
committed M1 + exact request
  ├─ invalid request / pending proposal → refuse; authority unchanged
  └─ Growth lock + private cold-restored M1
       ├─ admissible pending M1 proposal
       │    → ordinary complete M1 checkpoint/science/receipt/cursor commit
       │    → discard runtime; cold-restore that new M1 boundary
       └─ verified M2 load → full scientific refresh → checkpoint + closed science
            → validate / hash / fsync / immutable capsule rename
            → ONE existing atomic run_state replacement makes M2 authoritative
            → repair discovery and advisory active_model.json
```

An optional M1 drain is its own existing transaction. The M2 transition itself
has exactly one authority replacement. Inference/output failure discards the
private runtime. A precommit orphan M2 capsule is not authority. Before commit,
restart restores the latest committed M1 boundary; after commit it restores the
complete M2 checkpoint, scientific state and evaluator together.

The sealed descriptor records the request, actual previous head/model producer
and refresh result. A request digest is stored once in the **existing generation
publication entry**, not in another ledger. Exact committed-request retry verifies
that record and repairs projections/acknowledgements without refreshing again.
Retrying an older M2 request after a later M3 activation does not reactivate M2.

Growth restoration and Reevaluation resolve the same sealed active binding.
`active_model.json` is rebuildable discovery only. A missing/stale projection cannot
change the model. Reevaluation loads the physical immutable bundle while retaining
the logical identity used in patches. The owned final producer fence prevents a
late M1 computation from publishing work after M2 is authoritative. A producer
adopts/fsyncs an observed authority before publishing work derived from that head.

## Pending M1 proposals and exact receipts

The existing session internals run under the caller-owned Growth lock; they do
not reacquire it. An admissible M1 proposal is fully committed and acknowledged
before switching. Invalid/conflicting pending work blocks activation and remains
present. There is no convenience deletion, fabricated receipt or cursor reset.

A crash after draining M1 leaves the complete new M1 boundary and its exact
receipt/cursor. The old activation request is now stale: the caller must prepare
a new explicit request. A crash after M2 commit permits exact idempotent retry.
Existing arbitrary receipt UUID/content pairs remain exact; same UUID with altered
bytes is a conflict. Retained old receipts are not compressed to a watermark.

The first M2 traversal follows the existing new-model cursor rule while binding
the retained accepted-cursor digest. The next real Growth acceptance advances it
normally. Tests cover that full producer/acceptance sequence, old-byte retries,
conflicting retries, missing projections and M1 → M2 → M3.

## Scientific evidence and strict proof boundary

Certificate preflight uses strict bounded runtime traversal, not the diagnostic
iterator that can materialize a list or suppress errors. For a terminal node,
actual game terminality and the complete direct Value must agree with the
independent Morpion terminal evaluator. Anemone validates reconstructible default-max
tree proofs. Unsupported external nonterminal direct FORCED certificates are
refused before M2 loading; an enum/version/exporter identity is not proof.

The one private runtime is cold-restored directly with committed M1. M2 loading
uses `torch.random.fork_rng(devices=[])`, then existing
`refresh_with_evaluator(scope="all", batch_size=64)`. Chipiron supplies no second
backup implementation. Anemone replaces estimates, recomputes backups and selection,
and preserves terminal/proven certificates, ordered DAG links, unopened branches,
PV, selector count, search RNG and rollout RNG under its format-3 contract.
Actual node IDs/canonical references/creation generations remain stable; workspace
and research epoch do not change. The closed writer permits this explicit
evaluator-only producer transition while retaining other scientific identities.

New integration tests compare real accepted-weight activation against the existing
independent small-tree max oracle on a legal DAG, the real endgame and a partially
opened graph. They compare complete decoded checkpoint payloads and **bytewise all
three scientific files** against the oracle-checked offline refresh. They do not
equate compressed checkpoint bytes with decoded checkpoint semantics.

For each graph, five enabled policies × direct/backed-up selection × exact filtering
give 20 full-reference/bounded Dataset comparisons, including complete metadata
and JSONL bytes. Four exact-family policies × the same options give 16 deliberate
gating checks. Across the three graphs this is 540 paired rows, 60 enabled
comparisons and 48 gated combinations. Canonical references, all incoming/outgoing links and immutable
creation records are compared explicitly.

The real endgame starts with direct ESTIMATE(42), tree/effective FORCED(35), and
terminal 35. Direct preference never certifies 42; exact filtering excludes its
estimated raw target. The terminal row remains TERMINAL(35) in every enabled
policy, including heuristic family prediction. The four cross-state exact-family
policies remain disabled. Legacy confidence interpretation and all nine legacy
policy algorithms are unchanged.

The unchanged earlier witnesses additionally exercise seven-node propagation,
transpositions, equal-score/exactness ties, refresh windows 1/7/64, exact numerical
scoring, selector and both RNGs, uninterrupted/restored continuation, and the
previously failing 1k format-3 ordering/representative-parent round trip in both
checkpoint layouts. No assertion or scientific expected value was weakened.

## Validation

| Validation | Result | Evidence |
| --- | --- | --- |
| FAST during development | 147 passed, 84.95 s | `fast.xml` |
| PR on final production sources | 2,610 passed, 2 skipped, 864.42 s | `pr.xml` |
| Earlier research + new scientific matrix | 109 passed, 126.52 s | `scientific-final.xml` |
| Final strengthened bytewise scientific publication assertions | 6 passed, 31.13 s | `activation-science-final.xml` |
| Added death boundaries and final edge cases | 13 passed, 68.58 s | `activation-extra.xml` |
| FULL at `33141341` | **2,757 passed, 2 skipped, 1,438.19 s (23m58s)** | `full.xml` |
| Ruff check and format | Pass | local logs |
| Static baseline comparison | 206 current vs 209 baseline; **0 new**, 3 removed | `static_summary.json` |
| Wheel/sdist and installed-source checks | Pass; all 477/289 modules match | `installed_source.json` |

JUnit and compact results are in [`atomic_activation/evidence/`](https://github.com/victorgabillon/chipiron/tree/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence).
The 109 research cases comprise 45 prior policy cases, 56 corrected-refresh
integration cases, two unchanged checkpoint witnesses and six new activation
cases. The six new cases were rerun after strengthening the bytewise assertions.
These overlapping runs are reported separately, not summed as unique coverage.
FULL includes all 50 new normal-suite cases: four strict format-3/continuation
cases, two activation layouts, 22 process-death cases and 22 edge/concurrency
cases. It also runs the complete existing affected checkpoint, durable publication,
receipt/cursor, Dataset, Training and numerical-policy matrices, without the PR
target's `exhaustive` exclusion. Per-file counts and complete JUnit identities are
in [`validation.json`](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence/validation.json) and `full.xml`.
The execution recorder reports 2,759 unique selected identities, four deselected
by the existing integration/external-data markers, and zero missing/duplicate
executions. The only skips are the two CUDA-only tests.

Overall combined statement/branch coverage is 80.40% (83.38% statements, 68.58%
branches). The new activation module covers all 111 statements and six branches.
Coverage is execution evidence, not a substitute for the independent scientific
oracles or durability invariants. The precise tested executable head is
`331413412317d7b58691936718f0852e22d39e4e`; its production tree is identical to C.

The existing Make targets use two local pytest workers and preserve their normal
marker selections (`not integration and not external_data` for FULL). PR selected
2,612 identities and deselected 151; its execution recorder found no missing or
duplicate test executions. Its two skips are the existing CUDA-only cases on this
CPU environment. No scientific/crash assertion was skipped to accommodate activation.
Static tools retain inherited debt rather than silently changing their baseline:
mypy 89, pyright 73 and pylint 44 current findings. Ruff passes and 694 files pass
format checking; toolchain-version consistency also passes.

Run from the isolated worktree with its sibling `venv/bin/python`,
`PYTHONPATH=.`, `CHIPIRON_TEST_INSTALLED=1`, `QT_QPA_PLATFORM=offscreen`, and the
short secondary-disk TMPDIR. The final command is:

```bash
make test-full PYTHON=../venv/bin/python \
  TEST_ARGS='-p docs.research.benchmarks.morpion_training_export_20261020.validation_safety --basetemp=/home/pompote/oldata/victor/ma24/full --junitxml=../full.xml --cov-report=json:../coverage.json'
```

The existing safety plugin redirects optional leaderboard effects into generated
fixtures. `external_data/trees` was populated only from tracked `tests/data/trees`,
as in the existing test environment. Research tests run separately from `tests/`
to avoid their pre-existing `scripts` import-name collision. Complete command logs,
coverage and CI identity manifests remain in the isolated scratch root. No FULL
rerun is needed for the documentation/evidence-only descendant.

The new activation process-death matrix terminates actual child interpreters with
`os._exit(73)`: before/during model load, inference, backup, selector, checkpoint
shard writing, scientific writing, output/hash validation, file/directory fsync,
before/after immutable rename, before/partial/immediately after authority replacement,
authority-directory fsync, discovery and active-model repair. Additional cases cover
death after M1 drain, producer restart, and repeated death during recovery repair.
Every recovery checks the complete valid M1 or M2 boundary before retry.

Concurrency tests exercise competing activation requests and a paused late M1
producer under the real existing ownership boundary. Corruption tests cover the
selected bundle, missing model, checkpoint, scientific source, accepted Training
descriptor, unsupported certificates and conflicting pending work. These are local
process-death/filesystem tests; they do not establish guarantees beyond the existing
fsync/local-filesystem contract or simulate storage-device power loss.

## Measurements and resource limits

| Fresh process operation | Nodes | Wall seconds | Peak RSS MiB | CPU user / system seconds |
| --- | ---: | ---: | ---: | ---: |
| Small activation | 5 | 1.304 | 301.12 | 0.417 / 0.058 |
| Small cold recovery + certificate check | 5 | 0.152 | 296.47 | 0.129 / 0.006 |
| Final matched activation | 1,003 | 14.875 | 307.75 | 13.454 / 0.119 |
| Final matched cold recovery + certificate check | 1,003 | 1.221 | 303.40 | 1.193 / 0.010 |
| Same-input instrumented activation | 1,003 | 30.132 | 309.03 | 28.757 / 0.136 |

The approximate-1k growth request finishes at 1,003 legal nodes, of which 1,002
need neural estimates. Setup with accepted Training took 21.929 s / 381.93 MiB;
it is a different process and is not included in activation. The earlier final
production 1,000-node fixture measured 15.495 s / 309.36 MiB and recovery 1.281 s.
That different legal population is retained as evidence, not used to infer a speedup.
No 5k extrapolation or extra performance change was necessary for this task.
These timing fixtures have no pending proposal. An admissible drain adds one
existing complete M1 commit and subsequent cold restore; that branch is covered
by correctness/crash tests, not a separate resource benchmark here.

| Instrumented boundary (nested wall seconds) | Calls | Seconds |
| --- | ---: | ---: |
| Committed M1 load | 1 | 0.00356 |
| Candidate M2 load | 1 | 0.00165 |
| Cold checkpoint restoration, including M1 load | 1 | 0.281 |
| Independent certificate preflight | 1 | 2.292 |
| Complete Anemone refresh | 1 | 24.751 |
| Neural feature/tensor preparation, inside refresh | 1,002 | 18.756 |
| Neural model forward, inside refresh | 1,002 | 0.0624 |
| Backup calls, inside refresh | 67 | 0.467 |
| Selector refresh/checkpoint preparation | 2 | 0.0247 |
| Checkpoint writing | 1 | 0.332 |
| Closed scientific snapshot writing | 1 | 1.389 |
| Complete output scientific validation, inside publication | 1 | 0.542 |
| Sealing, validation, commit and projection repair | 1 | 0.915 |
| File digest calls across phases | 1,142 | 0.0346 |
| fsync calls across phases | 115 | 0.280 |

The unprofiled 1,003-node activation read 57,971,746 logical bytes (`rchar`) and
wrote 6,599,894 (`wchar`). `/proc` physical I/O counters report zero read bytes
(OS page cache) and 6,938,624 written bytes. This is a fresh-process cold restore,
not a dropped-OS-cache disk benchmark. Recovery reads 4,190,381 logical bytes and
writes none. The complete fixture grows from 2,551,808 to 3,764,224 inode-aware
allocated bytes; the sampled private staging peak is 3,928,064 allocated bytes
(3,877,881 logical). Tiny activation's sampled staging peak is 208,896 allocated
bytes. These totals include existing source/result dependencies and are not a
per-node storage forecast.

[`matched_inputs.json`](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/atomic_activation/evidence/matched_inputs.json) contains
the identical input request, model identity, source record, input checkpoint and
scientific digests, and the equal resulting checkpoint/scientific digests for the
profiled/unprofiled pair. Resource JSON files preserve all counters. The raw profile
is retained at the scratch root's `measurements/phases-1k-profile.prof`.

The research harness reuses `StageMeasurement`, existing `/proc`/resource counters,
cProfile/pstats and temporary mock wrappers around phase boundaries. It adds no
production profiling framework. Setup, activation and recovery use fresh serial
processes; setup includes tiny Training and is excluded from activation RSS/time.
The profiled/unprofiled 1k pair is copied from the same newly generated M1 workspace
before activation, binding identical graph, state complexity, accepted weights and
request. Fingerprints prove checkpoint/scientific result equality. Earlier v1
short-state benchmarks are not used as performance comparators.

The 100 ms staging sampler measures a lower bound on peak temporary allocation,
with hard links counted once per inode. It is read-only and runs in both timing
modes. Profiled phase times include instrumentation overhead and are nested, not
additive. Direct wall-clock boundaries are used because cProfile parent cumulative
times were inconsistent on this Python/Torch toolchain. Feature preparation remains
the dominant demonstrated cost; it is intentionally not optimized here.

Working memory is one existing O(N+E) search graph, the bounded 256-state resolver
and 64-node refresh window, plus existing serializer/writer working structures.
Existing checkpoint payload construction and closed-export ordered node references
still have O(N) components; this task does not claim a constant-memory whole Growth
runtime or introduce a second TrainingTreeSnapshot. Time includes state-history
decode/feature cost and existing backup/selection, checkpoint, canonical validation
and hashing. A 1k measurement is not evidence of constant large-tree cost.

Publication records and exact UUID receipts remain cumulative O(G)/O(receipts).
Request hashing and retry scanning inherit that history cost. Immutable capsules
and retained model dependencies also retain existing disk costs. No retention,
receipt compaction, artifact deletion or speculative cache is introduced.

## Code size, review and safe integration

Against `448d6597`, production changes are **639 added, 11 removed, net +628 lines
across 11 existing/new files**. One new domain module has 305 lines. Its responsibility
is binding, admission, drain and orchestration; existing publication/session code
still performs staging, hashing, freezing, fsync, rename, commit and recovery.
The other changes connect the shared sealed evaluator identity to Growth/producer
restoration and narrowly authorize the scientific producer transition.

Existing older workspace routes remain where unrelated callers require them. After
a separately validated migration, candidates for deletion are mutable active-model
authority, zero/one-pin model inference, fixed-genesis-model publication restrictions,
future-only model replacement paths and obsolete publication adapters. They cannot
be removed safely by merely changing a checkpoint version. No new permanent
dual-format checkpoint adapter was added.

Review/install order: exact Anemone source/wheel → A → B → C. The final research
timing commits and this report follow without production changes. Integrate this
local stack only after its existing unmerged ancestry has a clean review base;
no duplicate-code remote PR is opened here. Validate a **fresh** opted-in workspace,
then require an explicit request to activate a selected accepted candidate.

Remaining admission limits are deliberate: format 3 only, supported default-max/
Linoo configuration, verified built-in model/scoring contract, selected-value-v2,
five supported family policies, reconstructible certificates, and cold restoration.
There is no warm-checkpoint dependency, lazy mixed-model policy, generic external
FORCED proof contract, four-policy exact-family enablement, or production activation.

For graphical review, open the prepared workspace:

```bash
code /home/pompote/oldata/victor/morpion-scratch/atomic-activation-impl-20261024/review/atomic-activation.code-workspace
```

It contains the reviewed base's 11 changed production files alongside the new
worktree. The base side of the new `model_activation.py` is deliberately empty.
In the Explorer, select a base file with **Select for Compare**, then the same
implementation file with **Compare with Selected**. Whitespace differences remain
visible. For the main transaction directly:

```bash
review_root=/home/pompote/oldata/victor/morpion-scratch/atomic-activation-impl-20261024
code --diff "$review_root/review/base-448d6597/src/chipiron/environments/morpion/bootstrap/model_activation.py" "$review_root/worktree/src/chipiron/environments/morpion/bootstrap/model_activation.py"
```

Review `generation_publication.py`, `generation_publication_io.py`,
`checkpoint_patch_receipts.py` and `reevaluation_worker.py` next for the authority
and producer boundaries. `git diff 448d6597..33141341 -- src tests` is the complete
implementation/test review; the final report commit adds evidence only.
