# Static-analysis no-regression gate

Static analysis currently contains inherited debt; CI prevents new debt.
The initial baseline has 91 Mypy, 73 Pyright and 45 Pylint diagnostics. All
209 file/rule/message identities (including duplicate counts) match the earlier
maintenance audit with zero additions/removals. Chipiron runtime source is
unchanged from `fea2ff42e8c88f4fea22b764c44bb3cf8c8bae47`. This baseline was
measured with published Coral 0.1.15 and Anemone 0.2.22, Python 3.13 and Torch
2.9.0+cpu in a fresh environment; analyzer versions are recorded in the manifest.

A ratchet pass is not a claim of zero Mypy, Pyright or Pylint errors.
Every current diagnostic is printed, with per-tool baseline/current/new/removed
counts and raw machine-readable reports under the tox environment's temporary
directory. The strict developer commands `tox -e lint,typecheck` remain available.

The default `tox` environments run the toolchain guard, strict installed-wheel
smoke/tests, strict repository-wide Ruff/format, and `static`. No new source
suppressions or analyzer exclusions are used to accommodate inherited debt.

`diagnostics.json` records the source commit, exact analyzer versions and a hash
of their configuration. Diagnostics use tool, relative file, rule, lexical scope,
source text and normalized message. Counts are preserved: a second occurrence of
an existing diagnostic is new debt. Unrelated line insertions do not create new
identities. Pylint's project-wide duplicate-code message already contains the
actual duplicate files/text; its incidental last-visited reporting module is
excluded from that identity. Changed duplicate text or extra occurrences fail.

The checker fails on new identities, malformed reports, failed tools, version or
configuration drift, and baseline growth. The baseline cannot exceed either its
first committed version or the PR base's potentially smaller baseline. CI fetches
full Git history and supplies the PR base SHA. It never updates the manifest.

When diagnostics disappear, remove only those exact entries in a reviewed change.
CI reports reductions without writing files. Analyzer upgrades require an
explicit reviewed tooling/migration change; do not regenerate the manifest to
approve new diagnostics. Initial recording is local-only and refuses to overwrite
an existing manifest:

```bash
python scripts/static_analysis_ratchet.py --mypy-strict \
  --output /absolute/external/static-report --record-baseline
```

The checker has regression tests for line movement, extra duplicate occurrences,
changed scopes/messages, failed analyzers, malformed JSON, missing Git history,
baseline growth and restoration of previously removed debt.

The initial GitHub run reports the same 91 Mypy and 73 Pyright errors, but
44 Pylint messages: `W0231` for Torch `Sampler.__init__` is absent with its
newer Torch build. The local Torch 2.9.0 CPU environment still reports that
message, so its baseline entry remains visible. CI reports one removed and
zero new diagnostics; source debt is not silently expanded or suppressed.
