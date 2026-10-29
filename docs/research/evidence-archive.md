# Archived generated research evidence

F1 removes 765 generated files (45,784,532 bytes) from the current source tree.
The original bytes remain in the preserved, merged
[PR #84 source commit](https://github.com/victorgabillon/chipiron/commit/d8c94d6c07dd072e6e25620ea194861644fa42e5).
No history is rewritten. This Git ancestor is the initial archive; local scratch
directories and expiring CI artifacts are not the archive.

The [CSV manifest](evidence-manifest.csv) records the source commit, original path,
Git blob ID, byte size and SHA-256 for each removed file. All 765 entries were
unchanged on implementation base `3981eae24547291e3dc147e19b45faca6681f957`.
Reports retain their scientific results, qualifications and reproduction commands.
Their evidence links now identify the full immutable source commit.

## Retrieve and verify

Run from a normal Git checkout with Python 3.10 or newer. If the clone is shallow
or does not yet contain the archive commit, fetch it first:

```sh
git fetch --no-tags origin d8c94d6c07dd072e6e25620ea194861644fa42e5
python scripts/research_evidence.py
python scripts/research_evidence.py --destination /path/to/NEW-evidence-restore
```

The destination must not exist. The command checks every original commit/path
against its blob ID, size and SHA-256, writes the original repository-relative
paths beneath the new destination, and rereads every restored file. It refuses
to overwrite an existing destination. Verification without `--destination` does
not write evidence. Successful output is `Verified 765 files / 45784532 bytes`.

To retrieve one file without a script, substitute a manifest path:

```sh
git show d8c94d6c07dd072e6e25620ea194861644fa42e5:docs/research/e2e_bootstrap/evidence/sources.json > sources.json
sha256sum sources.json
```

Compare that hash with the manifest entry. Immutable browser links use
`https://github.com/victorgabillon/chipiron/blob/` followed by the manifest's full
`source_commit`, `/`, and `path`; directory links use `tree` instead of `blob`.
For example, browse the original
[E2E evidence](https://github.com/victorgabillon/chipiron/tree/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/e2e_bootstrap/evidence).

## Dependency review and retained material

The allowlist is exactly the 765 tracked files below `docs/research/**/evidence/`:
568 JSON, 66 XML, 53 TXT, 50 LOG, 20 TIME, four GZ and four JSONL files.
Production code, tests, scripts, research harnesses and documentation were searched
for original paths, path suffixes, evidence-directory access and dynamically built
paths before removal. None of the selected outputs is an executable test fixture.

The following executable dependencies remain at their existing paths:

| Consumer | Retained input / behavior |
| --- | --- |
| `tests/environments/test_morpion_dataset_target_confidence.py` | Imports `dataset_confidence_v2/audit.py`; generates the actual confidence cases in pytest scratch directories. |
| `tests/environments/test_morpion_operational_validation.py` | Imports `e2e_bootstrap/worker.py` in a subprocess and runs `operational_bootstrap/reproduce.py`; creates fresh operational history in pytest scratch directories. |
| E2E comparison and receipt/model research harnesses | Read evidence beneath explicit run-directory arguments; use newly generated runs or restored archive outputs when reproducing historical comparisons. |
| Frozen-source CI and scientific regression tests | Source contracts, expected results, source verification and all existing regression tests remain unchanged. |

All Python research helpers, concise Markdown reports, datasets/fixtures outside
the output allowlist and existing reproduction commands remain. There are no
file-level exceptions within the allowlist. Historical absolute experiment paths
in reports describe the original runs; cleanup does not modify those directories.

## Keep generated outputs outside source control

Use separate runtime directories for new runs and restorations. Scoped Git ignore
rules reserve research `evidence/` directories for generated outputs. The
stdlib-only `python scripts/check_generated_artifacts.py` checks the Git index,
including force-added files, for research evidence, known runtime directories and
caches. It runs through the existing toolchain gate in CI and pre-commit without
changing shared workflows. Stage intended deletions before running it locally.
JSON/YAML fixtures outside reserved output locations remain allowed; extensions
are not used to prohibit scientific fixtures.
Existing `.gitkeep` directory placeholders remain allowed, including the four
tracked script-output placeholders; they are not generated research evidence.

Docker context rules exclude generated research evidence, local Morpion run roots,
tensor caches and CI outputs while retaining source, helpers, test data and package
resources. Removing current-tree files does not shrink the preserved Git history.

## Cleanup validation

On CPython 3.14.10, full test collection succeeds with 3,070 tests. The two named
consumers and archive/guard regressions pass together: 73 tests. All 765 original
files were restored and reread with matching blob identities, sizes and SHA-256
hashes. The staged deletion set equals the manifest exactly. The 197 repaired
evidence links and remaining local links in affected reports resolve (286 checked
links, no broken links).

Ruff, the toolchain/output guard, unchanged frozen-source verification, installed
resource/evaluator/checkpoint checks, `pip check`, wheel/sdist inspection and strict
Twine metadata checks pass. Both distributions exclude the archived evidence and
runtime/cache outputs and retain required package resources. Hosted project CI and
the exact final review commit are recorded in the cleanup PR.
