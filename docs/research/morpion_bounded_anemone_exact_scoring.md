# Committed evaluator restoration and exact Reevaluation scoring

> Historical generated outputs are available through the [evidence archive](evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

This is a local integration on `1da1c59ae37d901f5c5277a6a2c04149046c49d0`.
It prepares model activation; it does not implement, authorize or enable activation.
All fixtures, environments and outputs are new, on the secondary ext4 partition.

## Reviewed sources and independent changes

- Chipiron base: `1da1c59ae37d901f5c5277a6a2c04149046c49d0`, with implementation
  `7e29c9f58f2d9559e28565afc9371e963e570311` in its ancestry.
- Anemone: `c015194533857a52d07bafdd603b762314692cfb`, directly following
  `5f43821744c4c4d3944c5da2b20def47cde833b9` and the scientific correction
  `74bbd1250f31b72497166e624659d547324a0e5a`.
- Restore implementation: `1b83dea480d0c9a3a3d53eac02e6737871906f10`.
- Numeric protocol implementation: `ecccef061fc8ed0198cefaccef7ab95f61d8d33d`.

Anemone was built from an archive of that exact commit and installed with no
resolution of dependencies in a copied, independent environment. Its unchanged
version string, `0.2.26`, is not an implementation identity. Wheel SHA256:
`16aaa5b3a3ac339b8aef34ef75a93134136ed07010e9c2da0816d412dd109c47`.
All 289 production Python modules match the archived source byte-for-byte.
The installed Chipiron wheel also matches all 476 checkout production modules.
Exact module hashes, interpreter, wheel and dependency identities are in
[dependencies.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_anemone_integration/evidence/dependencies.json).

## A: committed restoration contract

For a scientific publication v2 or closed publication v3, the existing authoritative
run state selects the head. Its generation, checkpoint and scientific pointer must
agree with the publication record. Existing capsule validation verifies the seal,
complete dependency inventory, scientific identity, frozen configuration and model
bundle hashes. The configuration hash must also equal the current run-state binding.
Zero model pins means the committed default evaluator; one pin means that exact
bundle. Multiple pins are rejected as ambiguous. A requested alternative bundle
must have the same complete digest; a missing/corrupt sealed dependency is fatal.

Growth resolves the model from this head before loading, even if `active_model.json`
is absent, corrupt or advertises a different result. The model's logical origin
remains available for receipt provenance; restoration loads its immutable physical
copy. Merely accepting Training or editing its discovery projection cannot activate
it. The existing Growth ownership lease still serializes the transition.

The verified evaluator goes directly into Anemone's existing typed/sharded loader.
The committed path performs no `set_evaluator`, refresh or version increment.
It retains the existing generation read pin during restoration. An explicit
`reevaluate_tree=True` request is rejected on this path: a restart is not an
activation operation. A non-head capsule cannot silently become the live Growth
head. Default Anemone cache capacity is inherited; no caller cache setting is added.

The unmanaged checkpoint/model replacement path keeps its existing explicit
attachment behavior. It does not provide a committed model identity and is not a
safe model-activation API. Its post-restore attachment can eventually be deleted
when unmanaged replacement is retired or replaced by the separately reviewed
activation transaction. No migration or historical rewrite is performed here.

## B: explicitly versioned numerical policy

Fresh genesis may additionally request `atomic_reevaluation=True, growth_scoring=True`
(with closed scientific snapshots, as already required by the atomic protocol).
The existing proposal authority becomes schema 2 and contains
`numeric_policy = anemone_depth_scaled_nonterminal_v1`. This extra identity travels
in the existing proposal envelope and participates in acceptance and duplicate
validation. The receipt layout, exact UUID/content digest and accepted-cursor digest
remain unchanged. The default is still the legacy schema 1, raw-score behavior.
Initializing the new policy in an existing workspace is refused.

Only freshly inferred nonterminal scores call the public Anemone
`scale_nonterminal_score(score, TrainingNodeSnapshot.depth)` helper. No formula is
duplicated in production. The row records the declared policy and checkpoint-relative
depth. Its certainty remains estimated even if a different Value on the same node
is FORCED. Terminal rows retain their certified score; saved-value fallback rows
retain their existing values and explicit source. Historical patches are never
rescaled. Built-in inference or explicitly selected saved-value fallback are the
only production adapters admitted by the new protocol; arbitrary custom adapters
cannot be silently relabeled. A proposal must use the committed Growth model.

Acceptance rejects an old envelope, changed policy, undeclared row source, missing
fresh-score policy/depth or anything other than explicit false certainty and
terminal flags on a fresh neural estimate. It also preserves existing source,
model, epoch, cursor, ownership and exact-content
checks. This validates a declared trusted-producer contract; it is not a cryptographic
proof of neural inference. As with the existing protocol, arbitrary untrusted edits
to the authority or executable producer are outside its trust model.

The actual move-34 endgame has checkpoint depth zero: its new direct estimate is
42, its tree/effective proof remains FORCED(35), and its terminal child remains 35.
Positive, negative, zero and signed-zero estimates, ties and multiply-parented DAG
states are compared bitwise with the actual Growth direct values. This intentionally
changes nonzero-depth fresh scores under the new contract; no rounding hides the
change. The old scientific comparison that characterizes raw policy remains valid
for the unchanged schema-1 adapter.

The saved-value fallback retains the separately established PR #82 limitation
([certainty report](morpion_value_certainty_2026-10-20.md)): one node-level flag can
certify both a saved estimated direct scalar and a different exact tree scalar.
This task deliberately preserves that existing fallback behavior and never rescales
its values. The numeric schema-2 envelope does **not** turn it into a per-value
certainty proof. Fresh neural rows are the path covered by the new estimate-only
guarantee. Keep saved-value fallback disabled for scientific activation until a
separately validated per-value update contract exists; its default remains disabled.
This is an inherited, explicit activation blocker, not a newly demonstrated proof
that fallback values can be safely re-certified.

## Scientific and recovery evidence

The focused checks cover both checkpoint loaders, cold neural restore, direct
versions, independent Value certainties, effective source, complete DAG edge sets,
PV, selector decisions, search and rollout RNG, and subsequent Growth. Mutable
projection changes cannot choose a different Growth evaluator. Missing/corrupt
sealed weights, supplied substitute weights, changed configuration binding and an
implicit restore refresh are refused.

New protocol tests run the real producer in full-reference and bounded modes and
compare complete proposal bytes, then exercise acceptance and exact retry. Existing
process-death tests are reused at all 12 pre-commit and seven post-commit consumer
boundaries and three producer publication boundaries. The observed authority is
old or complete new state; retries do not replay effects or rewind the cursor.
The independent seven-node propagation witness, explicit refusal of unavailable
proofs, five enabled confidence-v2 policies, four exact-family gates and all nine
legacy policy expectations remain covered by the existing scientific suite.

The known representative-parent and child-link iteration discrepancies are not
fixed. Edge-set scientific parity and ordered-link identity are recorded separately.
The model activation plan must retain that limitation rather than normalize it away.

### Completed validation

The final complete run passed against implementation `ecccef06`, including the
two additional null-flag regressions:

- Full serial CI: **2,689 passed, 2 skipped, 4 deselected, zero failures/errors**,
  141 warnings, **2,235.57 s** (37m15s); driver wall time 2,237.40 s. The two skips
  require unavailable CUDA. The four deselections are the existing chess checkmate
  integration cases under the unchanged `not integration and not external_data`
  expression. No Morpion scientific test was excluded.
- Coverage: 80.2764% combined line/branch, 83.2749% statements, 68.4264% branches.
- Toolchain coherence, Ruff lint/format, source and wheel build, and installed
  distribution smoke all pass. Static ratchet: **zero new diagnostics**, 206 current
  versus the inherited 209 baseline; the baseline was not modified.
- Focused existing scientific/atomic suite: 143 passed. Final committed-restore and
  default-cache suite: 12 passed, including full scientific reference equality.
  Numeric-policy/real-death suite: 39 passed after the strict-flag correction.
  Previous independent scientific, seven-node, confidence-policy and restore
  research suite: 63 passed.
- The final integrity audit retains all 40 pre-existing Chipiron registrations and
  12 Anemone registrations with identical HEAD/registration, Git status and tracked
  diff hashes. Installed source hashes remain identical after CI.

The initial complete run against `690201ef` also passed (2,687 passed, two skipped,
four deselected, 2,243.52 s). It is preserved separately in `validation-initial.json`
and `full-ci-initial.xml`; it is not used as the final implementation's CI result.

Exact skipped/excluded identities, commands, coverage and artifact hashes are in
[validation-summary.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_anemone_integration/evidence/validation-summary.json),
with the complete [JUnit result](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_anemone_integration/evidence/full-ci.xml).
Focused outcomes and logs are fingerprinted in `targeted-validation.json`.
The local CI uses the installed wheel, the repository's normal test selection and
the same quality/static gates. Tests run serially (`-n 0`) instead of CI's two
pytest processes. The existing `validation_safety` fixture redirects only default
leaderboard destinations into fresh temporary fixtures; record logic is unchanged.
The exact environment is retained in `ci-environment.json`. This is local gate
validation, not a remote CI run or a test against the ambiguously versioned public
Anemone package.
Development reruns are retained and explained in `development-reruns.json`: they
corrected new-fixture setup/expectations (timestamp, existing exception class,
durable test mutation, and the actual `moves + 0.125` fixture scorer), without
changing any pre-existing scientific expectation or masking a source difference.

The final review added two malformed-proposal regressions. Both fail on the initial
implementation: truthiness allowed `None` flags, while Anemone's legacy update API
can inherit an existing certificate when exactness is unspecified. The final numeric
commit requires `is_exact is False` and `is_terminal is False` for fresh model rows.
Both regressions and all 39 numeric/recovery cases now pass. This rejects ambiguous
input without changing valid scores, receipts or legacy fallback behavior. The
failing-before and passing-after log identities are in `null-flag-regression.json`.
The initial commit remains reachable through the local evidence tag
`evidence/morpion-scoring-initial-20261024`; protected branch history is unchanged.

Resource measurements were taken on `690201ef`, before that two-line admission
hardening. `benchmark-dependencies.json` preserves its exact installed source
hashes. The measured refresh/restore/writer/reader code and Anemone wheel are
unchanged by the guard, which only rejects malformed scaled proposals. The final
package identity is recorded independently in `dependencies.json`; the benchmark
is not mislabeled as a new measurement of the amended commit.

## Measurement method and remaining memory

`bounded_anemone_integration/measure.py` adapts the existing profiling driver and
uses its cProfile, process RSS, CPU, I/O and codec counters. It adds aggregate resolver
hit/miss/eviction counts, two successive complete public refreshes and subsequent
Growth comparison. It creates no profiling framework, per-node cache or alternative
backup. Both runs use inference windows of 64 and real neural feature conversion.

Each size has one newly generated legal graph, copied byte-for-byte for the old
integration (Anemone 74bbd12) and the candidate (c015194). Both use the same freshly
accepted two-row/two-candidate synthetic Training result. Runs are fresh processes
and serial. A fresh process does not imply a cold filesystem page cache. The same
canonical histories are used; earlier small-reference v1 timing is not a comparator.
The reused driver holds its declared synthetic producer label constant across the
pair (`dc142573-model-activation-research`) so complete files can be compared. This
is a fixture label, not a claim about the executing revision; actual implementation
commits and installed source hashes are recorded independently in the manifests.

The 256-entry LRU bounds retained decoded states, not the entire Growth runtime.
The live graph, N node/handle records, E links, checkpoint payload catalog and selector
remain resident; state sizes and anchor/delta reconstruction depend on history length.
An active decoding chain and inference window add bounded-by-depth/window transient
memory. More evictions can increase codec CPU. No second Python tree or full scientific
reference snapshot is introduced into the measured refresh path.

## Measured results

Both pairs match every checkpoint shard and every closed scientific file byte-for-byte, including full Values, certainties, provenance, canonical state references and topology. Subsequent Growth fingerprints, selector signatures and RNG hashes also match. The source populations are 1,003 nodes / 1,114 edges / maximum depth 20 and 5,013 nodes / 5,931 edges / maximum depth 62.

| Nodes | Implementation | First refresh s | Repeated refresh s | First estimates/s | Peak process RSS MiB | Retained decoded states |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,003 | Previous integration | 23.319 | 21.594 | 42.93 | 311.266 | 1,003 |
| 1,003 | Bounded Anemone | 23.099 | 23.291 | 43.33 | 306.340 | 256 |
| 5,013 | Previous integration | 132.851 | 121.226 | 37.67 | 409.973 | 5,013 |
| 5,013 | Bounded Anemone | 136.554 | 136.659 | 36.65 | 337.684 | 256 |

Times include profiling/instrumentation; these are matched measurements, not production service-level guarantees. CPU seconds are within 0.04 s of the corresponding wall seconds. One pair per size does not establish a statistically significant speedup. At 5k, total process peak RSS decreases by 72.289 MiB (17.63%); first refresh wall time increases 2.79%, repeated refresh 12.73%. The 1k RSS decrease is only 4.926 MiB (1.58%). Neither figure is the much larger percentage from the earlier dependency-only experiment. Keep capacity 256.

| Nodes / implementation / pass | Hits | Misses | Anchor decodes | Delta decodes | Evictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 / control / 1 | 52,038 | 1,003 | 268 | 735 | 0 |
| 1000 / control / 2 | 51,038 | 0 | 0 | 0 | 0 |
| 1000 / candidate / 1 | 52,026 | 1,047 | 280 | 767 | 791 |
| 1000 / candidate / 2 | 50,765 | 1,018 | 273 | 745 | 1,018 |
| 5000 / control / 1 | 262,568 | 5,013 | 1,372 | 3,641 | 0 |
| 5000 / control / 2 | 254,834 | 0 | 0 | 0 | 0 |
| 5000 / candidate / 1 | 262,433 | 5,499 | 1,507 | 3,992 | 5,243 |
| 5000 / candidate / 2 | 253,330 | 5,422 | 1,504 | 3,918 | 5,422 |

These eviction counts apply to refresh only: decoded insertions minus the change in retained entries. Canonical `load_state_ref` calls during refresh are zero because checkpoints use the anchor/delta codec; this does not mean zero state decoding. Complete writer/reader validation still performs 2,006 and 10,026 canonical loads respectively in both versions. No canonical validation was removed. The whole-operation codec counters include those independent validation decodes and must not be used as resolver eviction counters.

Both implementations retain the known ordered-link difference relative to the source checkpoint (`ordered_link_parity=false` at both sizes). Their restored ordered links and subsequent scientific outputs agree with each other. Edge-set equivalence does not erase that separate serialization/representative-parent issue.

The bounded 5k cProfile covers two complete refreshes (273.213 s): actual neural batch-interface work is 223.273 s; candidate-segment feature generation is 180.621 s nested inside it; resolver calls total 28.789 s; actual model `forward` is 0.653 s. These nested figures must not be added. Real feature preparation dominates, while eviction increases reconstruction relative to the warm unbounded second pass.

| Nodes / implementation | Checkpoint load s | Checkpoint write+sync s | Closed snapshot write+sync s | Indexed validation s |
| --- | ---: | ---: | ---: | ---: |
| 1000 / control | 0.236 | 0.332 | 3.496 | 2.368 |
| 1000 / candidate | 0.235 | 0.340 | 5.520 | 2.362 |
| 5000 / control | 0.935 | 1.133 | 44.181 | 29.635 |
| 5000 / candidate | 0.934 | 1.111 | 58.275 | 29.759 |

The large resource measurements use sharded checkpoints. Both typed and sharded default-cache bounds and neural continuation are additionally tested on small generated graphs; these measurements do not establish a typed-loader RSS reduction of the same size. No 10k run is needed to establish the 256-entry invariant.

The old integration control reproduces both caller differences on new typed and sharded capsules: expected evaluator version 1 becomes 2 after restore; the correction restores version 1. At checkpoint depth 1, a fresh raw estimate of 42 is `0x1.5000000000000p+5`; the explicitly selected Growth policy yields `0x1.500000385f172p+5`. Root depth zero remains unchanged. The control and candidate observations are retained as exact JSON, not rounded comparisons.

Raw measurements, complete paired hash results, profiler attribution and `/usr/bin/time` observations are retained in [the evidence directory](https://github.com/victorgabillon/chipiron/tree/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_anemone_integration/evidence). Raw checkpoints, models, logs and `.prof` files remain under the isolated parent `bench/` and `logs/` directories.

Continuation scope: the small neural fixtures compare uninterrupted and restored execution directly, including both RNGs. The large 1k/5k runs compare two restored implementations after refresh; they do not prove arbitrary large-tree uninterrupted continuation in the presence of the separately recorded representative-parent/order discrepancy.

The 5k profile comparison attributes most additional repeated-refresh cost to
state reconstruction: the resolver's cumulative time increases from 13.000 to
28.448 s across two passes. Candidate-segment work is 177.888 versus 180.621 s;
neural `forward` is 0.610 versus 0.653 s. Thus this workload does not justify an
additional Chipiron cache or a capacity override. Exact profile counters are in
[profile-comparison5000.json](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_anemone_integration/evidence/profile-comparison5000.json).

## Production size and abstractions

| Commit | Production files | Added | Removed | Net |
| --- | ---: | ---: | ---: | ---: |
| A: committed restore | 4 | 130 | 27 | +103 |
| B: numeric identity/scoring | 3 | 79 | 4 | +75 |
| Total | 7 | 209 | 31 | +178 |

Counts include docstrings and reindentation of retained legacy branches. No production
module was added. The capsule helper is necessary to choose a validated authoritative
evaluator independently of mutable discovery. Two private loader arguments pass that
evaluator to existing Anemone APIs. The one genesis flag and numeric identity extend
the existing transaction version instead of adding a registry. The remaining changes
route Growth through the committed model and apply/check the shared score contract.
The committed path's redundant post-restore attachment is bypassed; its unmanaged
counterpart remains explicitly outside the new committed-restore contract.

Research instrumentation and tests are accounted for separately from production.
There is no caller-side LRU implementation or configuration option, and no second
backup algorithm. Any further consolidation of capsule validation should preserve
actual dependency verification; a parent hash or cache marker alone is insufficient.

## Activation boundary and next minimal transition

No activation is authorized or performed. The smallest next implementation should:

1. Under existing Growth ownership, bind explicit authorization to the current
   head, expected old model, workspace/epoch, accepted Training descriptor, selected
   candidate, full bundle digest, configuration and numerical policy.
2. Resolve pending old-model proposals deterministically: commit admissible work
   before switching, or design an explicit durable supersession and producer cursor
   transition. Preserve exact receipt evidence; never retag an old proposal.
3. Cold restore the verified current runtime with its current evaluator. Load the
   accepted new evaluator and invoke Anemone's complete scientific refresh with
   bounded windows on that single tree. Refuse unavailable proof reconstruction.
4. Write and validate a new complete checkpoint and closed scientific snapshot;
   freeze the new model/configuration and commit all identities through the existing
   run-state replacement. Until that commit, the old generation remains authoritative.
5. Recover only complete old or new state across model loading, inference, backup,
   selector refresh, writes, validation, publication, authority fsync and discovery
   repair. Restart the producer against the same committed model/numeric epoch.

Current publication deliberately enforces a fixed model per scientific epoch.
This must be replaced by the reviewed activation transition, not bypassed. Model
activation also still requires an explicit accepted-result/authorization binding,
old-proposal policy, certified-proof coverage, and independent M1-to-M2 interruption
validation. A release must identify the corrected Anemone implementation beyond its
unchanged version string. The new numerical protocol has a fresh-workspace boundary;
there is no implicit migration of historical receipts or producers. Parity here is
not evidence that those unfinished activation requirements have been satisfied.

The zero/one-model-pin restore rule is deliberately limited to today's fixed-model
scientific epoch. If future activation stores both an active model and retained
old-model receipt dependencies in one capsule, it needs an explicit active-evaluator
binding in that existing versioned descriptor. Selecting the first or last model
pin is not an activation contract. No such field, registry or migration is added
here; ambiguous current capsules fail closed.

Integration order remains explicit: identify/release the corrected Anemone source
with an unambiguous dependency identity, then review restore commit A, then numeric
protocol commit B, then the separate authorized activation transaction. The current
package version string alone cannot reproduce these results: the published or
otherwise installed `0.2.26` may lack `scale_nonterminal_score`. This report records
local installed-wheel CI against the exact corrected wheel, not remote CI against
an unspecified distribution with the same version. No compatibility shim silently
substitutes a different formula or reader.

## Review and reproduction

Worktree:
`/home/pompote/oldata/victor/morpion-scratch/bounded-anemone-chipiron-20261024/worktree`

Branch: `fix/morpion-committed-evaluator-scoring`. No protected branch was rebased
or rewritten. No PR, remote publication, merge, historical mutation or production
activation was performed. Existing cumulative receipt/publication metadata is
unchanged; the bounded resolver does not make historical discovery history-independent.

The two production commits can be reviewed independently in VS Code:

```bash
cd /home/pompote/oldata/victor/morpion-scratch/bounded-anemone-chipiron-20261024/worktree
code docs/research/bounded_anemone_integration/review.code-workspace
TMPDIR=../tmp git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt 1da1c59a 1b83dea4 -- src tests
TMPDIR=../tmp git -c diff.tool=vscode \
  -c 'difftool.vscode.cmd=code --wait --diff "$LOCAL" "$REMOTE"' \
  difftool --no-prompt 1b83dea4 ecccef06 -- src tests
```

Review the report/evidence commit separately. No extra production cache,
configuration hierarchy, journal, model registry, storage format or backup algorithm
was introduced. The one new resolver helper is in the existing capsule I/O module:
it binds a model to the existing authoritative head and reuses its validators.
Full capsule validation remains deliberately conservative; this work does not
introduce a cache to avoid repeated head validation.
