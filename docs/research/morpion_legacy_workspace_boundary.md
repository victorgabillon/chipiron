# Legacy bootstrap workspace boundary

The exported legacy loop previously validated the requested execution mode but
not the destination's persisted protocol. Default single-process arguments could
overwrite a scientific workspace's configuration even with `max_cycles=0`.
The direct single-cycle API could also reach the runner using scientific state.

Both entrypoints now reject durable publication authority before creating
directories, writing configuration, starting memory diagnostics or invoking the
runner. The guard reads the existing run state, checks the caller's state for
the single-cycle API, and rejects publication namespaces even if genesis was
interrupted before a run state was committed. It does not repair, adopt or
rewrite scientific workspaces. Ordinary legacy operation is unchanged.

The standalone patch starts at the validated Python 3.13 report commit
`7e5d8e22728ea0421a580fdfd9af00dbba640034`, before merging Python modernization.
The branch `fix/morpion-legacy-workspace-boundary` contains only this guard,
regression tests and this note. It can be cherry-picked independently onto a
development branch descended from the frozen scientific foundation. It requires
no dependency or interpreter update. Never apply it to an active experiment's
pinned checkout or installed environment in place; any adoption needs a separate
reviewed executable identity and the existing experiment's recovery rules.

The new regression file creates a fresh real continual scientific genesis,
tests both public APIs (including zero cycles and a stale caller state), and
compares every workspace file's bytes and all file/directory modification times.
It also covers interrupted publication versions 1, 2 and 3, incomplete authority
markers, no runner calls and ordinary legacy zero-cycle operation. The pre-fix
Python 3.13 run fails all eleven rejection cases; its legacy control passes.
The existing legacy loop suite covers normal multi-cycle training and restart.
With the guard, all 66 new and existing loop tests pass on Python 3.13.5 in
35.63 seconds. Ruff lint and formatting checks pass for all three changed Python
files using the Python 3.13 target.

Evidence is retained outside the repository in
`/home/pompote/oldata/victor/python314-chipiron-20261028/p0-baseline-red.log`
and `p0-python313.log`. Validation uses the existing Python 3.13 interpreter and
dependencies read-only, imports this branch's source, disables bytecode writes,
and directs mutable runtime output into the migration evidence directory.
