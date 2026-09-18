# Developer operations

Use Python 3.13 and the pinned optional dependencies in `pyproject.toml`.
Normal checks are `python -m tox`; the fast test selection is
`python -m pytest -m "not integration and not external_data"`.
Integration installation scripts may download engines/datasets and are separate
from the ordinary offline unit-test gate.

The default tox environments are `tooling`, `py313`, `ruff` and `static`.
`py313` validates the installed wheel with published dependencies and no sibling
source overrides. `ruff` is strict; `static` prints inherited Pylint/mypy/Pyright
debt and rejects new diagnostics using the
[reviewed baseline](static_analysis_baseline/README.md). The strict developer
commands `tox -e lint,typecheck` remain available and can fail on inherited debt.

For MLflow, select an external SQLite database explicitly:

```bash
mlflow ui --backend-store-uri sqlite:////absolute/external/run/mlruns.db
```

Build distributions with `python -m build`. Version-tag releases run the quality
gate before the existing trusted PyPI publisher and the separate Docker job.
PR validation never publishes an image. Docker releases build and smoke-test
locally before authentication/publication, following the
[Docker test-before-push workflow](https://docs.docker.com/build/ci/github-actions/test-before-push/).

For a local image, use `docker build -t chipiron-x11 .`; GUI users may pass their
existing DISPLAY/X11 mount when running it. Only `.env.example` defaults are copied
into the image. Never include credentials in the Docker build context.

The documentation source is under `docs/`. Its API generator uses the current
src layout: `sphinx-apidoc -o docs src/chipiron`. Review generated changes before
committing; old API pages are a documented backlog item. Build with
`make -C docs html`. The former `Notes.md` mixed obsolete Python-3.12 paths and
manual release commands; these maintained instructions replace it.
