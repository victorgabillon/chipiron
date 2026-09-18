# Dependency and result provenance

Normal development dependencies are declared in `pyproject.toml`; there are no
moving Git-branch Python dependencies there. Coral, Anemone, Atomheart and Valanga
currently have exact PyPI versions. Parsley (`parsley-coco`) has a lower bound.
The Python imports are `coral`, `anemone`, `atomheart`, `valanga`, and `parsley`.
Quality-tool versions are pinned and checked by `scripts/check_toolchain_versions.py`.
Docker additionally fixes its Torch version; record the actual environment used
for each result, rather than assuming a local editable install matches PyPI.

Before a meaningful run, write the small dependency record alongside the existing
external experiment manifest:

```bash
python scripts/record_dependency_state.py --output /absolute/external/run/dependencies.json --require-clean
```

It records Chipiron/source Git SHAs, each package's installed version and actual
import path, Coral/Anemone/Parsley/Atomheart/Valanga source state, Python, Torch,
platform, installer commit metadata and installed RECORD hashes. Dirty checkout
status and changed-file SHA-256 values are included; URL credentials and queries
are removed. `--require-clean` fails for dirty/missing/unpinned VCS sources while
still writing the diagnostic record. For intentional dirty experiments, first
archive the actual source/patch externally; hashes alone cannot reconstruct it.
Use this JSON with the existing bootstrap `metadata` or value-training
`provenance` field. No new experiment coordinator is introduced.

Tests no longer insert arbitrary sibling checkouts into `sys.path`. Use explicit
editable installs or `PYTHONPATH` when developing against siblings, and record
the resulting import paths. Clean-package checks must also run without these
overrides; otherwise unpublished sibling changes can hide release failures.

Model URIs ending in `@main` in shipped chess examples are moving model references,
not immutable research identifiers. The existing model-bundle mechanism supports
revisioned URIs and explicit weights. Record an immutable model revision plus
bundle-file hashes in the experiment manifest; keep the convenient example
references unchanged during maintenance. New weights/data remain outside Git.

## September integration dependency gate

The validated local Coral evaluator-v1 implementation is commit
`40e4e76088f93764bedd01d19a16563f00ef20cb`. On September 18, PyPI's latest
`algorhino-coral==0.1.14` and Coral's remote `main`
`dc1c1f7e602584eaf4aeb96809e56747a3ab3307` did **not** include that implementation;
the validated commit was not available from GitHub. A clean published-package
smoke raises `TypeError` for `relation_bias_scale` when building Transformer-v1.
The corrected additive attention-mask inference behavior is also required.

Publishing the reviewed Coral implementation is a prerequisite to declaring this
canonicalization releasable. This sprint is not authorized to modify/push sibling
repositories. It therefore does not replace the evaluator with a fallback, vendor
Coral, disable compatibility tests, or point package metadata at an unpublished
commit. Once Coral is published, update its exact dependency to that validated
release (or a reachable immutable revision for research), rerun clean installation
and the complete quality gate, and update this status. Local passing tests alone
do not satisfy that release gate.
