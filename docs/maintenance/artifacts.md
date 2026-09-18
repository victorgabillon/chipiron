# Tracked data and generated-output audit

The canonical tree excludes machine-generated profiling output, the dated
base-tree-exploration report, the historical pickled replay under the replay
script, both test-output model summaries, and stale production/test model
evaluation YAML reports. No current test consumes those report files; model
evaluation tests build reports under `tmp_path`. The replay integration test
uses its separate checked-in YAML fixture. History is not rewritten.

Retained intentionally:

- `tests/scripts/*/small_dataset.pi`: small deterministic chess training fixtures.
- `tests/integration_test/replay_games/*_game_report.yaml`: replay test input.
- Existing `.gitkeep` placeholders: directory-layout scaffolding, not run data.
- Python modules named cache/checkpoint and their tests: executable runtime,
  training or diagnostic infrastructure, not generated caches/checkpoints.
- `src/chipiron/data/`: shipped player/settings/architecture configuration.
- `data/players/board_evaluators/nn_pytorch/prelu_no_bug/param.yaml` under the
  package (411,458 bytes): historical chess parameter data retained conservatively.
  Shipped scratch-training configs and a test config reference this package
  bundle directory. The current model-bundle resolver normally expects explicit
  `.pt` weights and also supports Hugging Face bundles; the old YAML is not proof
  of a complete modern loadable bundle. Its retirement/conversion needs a
  dedicated compatibility check, so maintenance neither deletes it nor invents
  a replacement remote store.

Use an external experiment directory and the existing configuration-driven
model-bundle paths for new weights, training rows, SQLite databases, caches,
profiles and logs. `.gitignore` covers generated profiles, test outputs and
existing runtime-output paths. These rules deliberately do not hide arbitrary
YAML, `.pi`, or all test fixture directories.
