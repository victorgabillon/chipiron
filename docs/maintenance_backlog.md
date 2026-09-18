# Maintenance backlog

Concrete items retained from the repository audit and the former hidden TODOs.
No new model architecture, search policy, or research campaign is scheduled here.

1. **HIGH — Publish and validate compatible sibling dependencies.** Publish/review Coral evaluator-v1 commit
   `40e4e76088f93764bedd01d19a16563f00ef20cb`, then update Chipiron's declared
   release and verify a clean install. Resolve the Anemone checkpoint-API wheel/source
   mismatch as well; see [dependency gate](maintenance/reproducibility.md).
2. **HIGH — Protect canonical main after review.** Require PRs and successful CI;
   disallow force pushes/deletion. Merge and branch administration belong to Victor.
3. **MEDIUM — Resolve inherited static typing debt.** Baseline strict Pyright has
   73 diagnostics and mypy 91 errors in the audited local environment, primarily
   lazy-export typing and supervised batch protocols; Pylint also reports inherited
   compatibility-facade/annotation warnings. The strict tox gates remain enabled.
   Compare changes against that
   baseline; do not silence the whole gate. Reproduce with declared dependencies.
4. **MEDIUM — Clarify parser/circular-import contracts.** Replace ad-hoc forward
   reference workarounds with explicit package API tests; document normal script
   argument parsing. This consolidates the actionable `src/chipiron/todo` items.
5. **MEDIUM — Validate legacy packaged chess bundle completeness.** Determine
   whether `prelu_no_bug/param.yaml` can be retired after the remaining package-URI
   training examples migrate to complete model bundles. Keep it until then.
6. **LOW — Refresh legacy Sphinx API pages.** Several generated `.rst` pages
   still describe moved `chess_env`/`treevalue` packages. Regenerate in a dedicated
   documentation pass and check links without restoring old implementations.
