# Parsley 0.3.0 consumer compatibility

Chipiron pins the published `parsley-coco==0.3.0` distribution; its import remains
`parsley`. This upgrade retains the existing chess parser integrations. It does
not change model/checkpoint codecs, Morpion recipes, historical environments, or
historical Git/replay contracts. Only the forward CI constraint and exact package
member hashes advance. The verifier still rejects missing, altered or additional
installed members.

Published release: [v0.3.0](https://github.com/victorgabillon/parsley/releases/tag/v0.3.0),
commit `1c30745fbbf423777cbe2082d7e017032f226420`.
The wheel SHA-256 is
`aec44095acdbe77e0f3f7e484333774e7e11d9edcc03b6a077216cca8ccbdd3a`;
the sdist SHA-256 is
`b7ee9a7843a5f2028ecacfd772c878f97547033d1446e06429e9aa05c37d1999`.
Both downloaded artifacts were independently checked against PyPI metadata.

## Inventory and comparison

The comparison starts at Chipiron main `f6ff2fba` and uses the same Python 3.14.10
CPU dependencies with Parsley 0.2.0 and the published 0.3.0 wheel. Raw resolved
objects, failures, package identities and logs are retained outside Git. Only
`BaseScriptArgs`' invocation timestamp is normalized when comparing values.

| Consumer | Public integration retained | Finding |
| --- | --- | --- |
| General scripts | `create_parsley`, `parse_arguments`, `Script.initiate` | Source precedence remains extra_args > CLI > YAML > defaults. |
| Chess supervised/scratch training | Script dataclasses, packaged/test YAML, architecture references | Complete supported configurations and omitted scientific factory defaults match. Partial custom-factory fields require attention below. |
| Player presets | `PlayerConfigTag.get_players_args`, YAML resolution | All 17 preset outputs match. |
| Match presets | `MatchConfigTag.get_match_settings_args`, `resolve_extended_object` | All four outputs match, including nested game/player resolution. |
| Game presets | `GameConfigTag.get_match_settings_args` | Both direct loaders now accept the declared nullable move-limit default; previously they asserted. Values agree with game args resolved through match presets. |
| Chess evaluator wiring | `resolve_yaml_file_to_base_dataclass` | Existing bundle/YAML coverage retained. |
| GUI and programmatic overrides | Generated partial dataclasses, presets and `_overwrite` fields | Complete inputs retain precedence; custom-factory partials follow the corrected nested rules. |

The 58-case inventory has 44 unchanged outcomes and 14 changed outcomes. Changes
include corrected direct game loading, more precise missing-field errors,
required-root construction now succeeding when supplied, and the documented
nested/custom-factory rules. Six custom factories were found among 60 reachable
argument dataclasses. None of these reachable schemas has a nullable field with
no default. Required fields remain required; no blanket `= None` changes were
made. The existing `MatchScriptArgs.match_args` uses `default_factory=MatchArgs`,
which correctly resolves presets under 0.3.0.

Existing problems were distinguished from upgrade regressions:

- `DefaultScriptArgs.base_script_args` is required. League/tree-visualization
  parsing without it already failed; supplied nested input now succeeds.
- The old scratch-training test YAML contains a legacy `RecurZipfBase` selector
  rejected by both releases. It was not rewritten to make the upgrade pass.
- Default chess GUI strength overrides replace the player with an incomplete
  partial object, losing its preset/name. Both releases reject that input
  (0.3.0 reports the missing name more directly). The launcher is outside this
  migration; complete selected-player inputs have regression coverage.

## Explicit nested input and scientific settings

Parsley 0.3.0 composes sources before constructing final dataclasses. A supplied
nested mapping uses that nested type's defaults; it does **not** fill missing
fields from a custom parent factory. This intentionally removes the old
CLI/extra_args versus YAML discrepancy. Chipiron does not emulate the old
source-dependent behavior or change scientific defaults to conceal it.

For example, `LearnNNScriptArgs.dataset_args` has a custom factory with training
and validation filenames. With no dataset override those exact filenames remain.
With only `--dataset_args.train_file_name train.pkl`, 0.2.0 retained the factory's
validation filename, whereas 0.3.0 uses `DataSetArgs.test_file_name = None`, as YAML
already did. To retain that scientific selection, supply both explicitly:

```text
--dataset_args.train_file_name train.pkl
--dataset_args.test_file_name data/datasets/goodgames_plusvariation_stockfish_eval_test
```

Likewise, preprocessing-only input must explicitly include the training filename.
A partial `nn_trainer_args.game_input` must include both `game_kind` and
`representation`; a partial neural-network architecture must provide both its
model arguments and output type. Use a complete resolved dataclass for equivalent
`extra_args`. Explicit YAML null continues to mean None. No in-repository CLI
caller uses the incomplete training forms above; existing supported complete YAML
inputs retain their values.

Custom factories returning already constructed dataclasses are copied without
rerunning their constructors. Returning a `MatchArgs()` containing unresolved
presets now raises `UnresolvedDefaultError`. Use the dataclass type as the factory
(as the existing match script already does), or explicitly return a resolved
object. Do not substitute a new default player or silently drop the preset.

Consumer tests exercise actual Chipiron schemas, complete dataset selections,
explicit null, default isolation, source precedence, final training validation,
GUI input, and preset resolution. Existing chess integration expectations remain
unchanged. This configuration evidence does not qualify a training backend or
change historical numerical findings.

## First Morpion adopter: growth recap

`bootstrap/growth_recap.py` now wraps its existing argparse options in the public
`Parsley` constructor and calls `parse_arguments(argv=argv, strict=True)`. It does
not use `Script.initiate` or initialize an output/work directory. The existing
required `--work-dir` Path, invocation-time worker-cycle environment fallback,
`--rich`/`--no-rich` precedence, tolerant reads and rendering are preserved.
There are no new YAML options or alias rules. Parsley's informational parsing
logs are suppressed at this CLI boundary to preserve recap output.

The private input dataclass is deliberately not frozen: 0.3.0's root CLI schema
extension cannot inherit from a frozen dataclass. Existing frozen recap results,
artifact dataclasses and codecs are unchanged.

Regression coverage runs real subprocess help, errors and successful recap on
read-only fixture directories, comparing contents, modes and modification times.
It also checks missing directories remain absent, explicit argv ignores host
arguments, environment changes apply to successive calls, and CLI overrides win.
The same preservation cases were run against the original argparse implementation
before migration.

Further adoption candidates remain the launcher, operator and recovery CLIs.
They need separate side-effect/routing reviews and are intentionally excluded
here, along with broad parser replacement and new configuration features.
