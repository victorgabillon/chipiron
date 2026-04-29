# Morpion Bootstrap Run & Dashboard

This project runs a bootstrap training loop for **Morpion Solitaire 5T**, combining:

* search (Anemone)
* training cycles
* checkpointing and dataset generation
* a Streamlit dashboard for monitoring

---

## 🚀 1. Setup

```bash
cd ~/oldata/victor/chipiron
conda activate anemone
export PYTHONPATH=src
```

---

## ▶️ 2. Launch a Morpion bootstrap run

```bash
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --max-cycles 100000 \
  --max-growth-steps-per-cycle 5 \
  --save-after-seconds 10 \
  --save-after-tree-growth-factor 1.2 \
  --tree-branch-limit 1000000 \
  2>&1 | tee ~/oldata/victor/morpion_runs/big_run_01/launcher_console.log
```

### 📌 Notes

* `--work-dir` is where all artifacts are stored (checkpoints, models, logs, etc.)
* Make sure this directory is on a **large partition** (e.g. `~/oldata`)
* Logs are saved to `launcher_console.log`

### Run modes

`single_process` is the simplest legacy mode. It runs search, dataset export,
training, and checkpointing in one in-process loop. The command above uses this
mode by default.

`artifact_pipeline` is the Phase 8 file-driven mode. It splits work into staged
launcher invocations and adds a reevaluation patch producer. The recommended
operator setup is two supervised shell loops:

* one loop for `--pipeline-stage loop`
* one loop for `--pipeline-stage reevaluation`

The `loop` stage performs growth and all pending dataset/training work once per
invocation. The `reevaluation` stage produces at most one bounded patch per
invocation. Both commands exit after one pass, so use a shell loop, `systemd`,
or another supervisor to keep them running.

### Recommended artifact-pipeline launch

Terminal 1: growth plus pending dataset/training work.

```bash
while true; do
  python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage loop
  sleep 5
done
```

Terminal 2: reevaluation patch producer.

```bash
while true; do
  python -m chipiron.environments.morpion.bootstrap.launcher \
    --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
    --pipeline-mode artifact_pipeline \
    --pipeline-stage reevaluation \
    --reevaluation-max-nodes-per-patch 10000
  sleep 2
done
```

### Reevaluation patch contract

The reevaluation worker writes only one pending patch at a time:
`pipeline/reevaluation_patch.json`. If that file already exists, reevaluation
skips instead of overwriting it. Growth consumes the pending patch before tree
growth and deletes it only after successful application. If patch application
fails, the patch remains on disk for diagnosis and retry.

This singleton contract avoids multiple huge checkpoint copies and keeps one
checkpoint/tree runtime owner: the growth process.

### Reevaluation batch size

`--reevaluation-max-nodes-per-patch` bounds one patch artifact. The default is
`10000`.

* Smaller values reduce patch size and application latency.
* Larger values improve throughput but can create large JSON artifacts.
* The worker uses `pipeline/reevaluation_cursor.json`, so repeated passes should
  eventually cover the whole tree instead of looping forever on the first N
  nodes.

### Active evaluator behavior

Training writes and updates `pipeline/active_model.json`. Reevaluation uses the
current active model. If the active model changes, the reevaluation cursor
resets so the next pass starts over for that evaluator. The best selected
evaluator remains the source for both future growth and reevaluation.

### Manual/debug stages

Manual stage commands are useful for debugging specific generations, but they
are not the recommended normal operator pattern.

```bash
# One growth pass
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage growth

# One dataset stage for a known generation
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage dataset \
  --pipeline-generation N

# One training stage for a known generation
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage training \
  --pipeline-generation N

# One reevaluation pass
python -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --pipeline-mode artifact_pipeline \
  --pipeline-stage reevaluation \
  --reevaluation-max-nodes-per-patch 10000
```

### Artifact files to inspect

Inside `<work_dir>`:

* `pipeline/active_model.json`
* `pipeline/reevaluation_patch.json`
* `pipeline/reevaluation_cursor.json`
* `pipeline/generation_000001/manifest.json`
* `pipeline/generation_000001/dataset_status.json`
* `pipeline/generation_000001/training_status.json`
* `pipeline/generation_000001/dataset_claim.json`
* `pipeline/generation_000001/training_claim.json`

### Troubleshooting artifact-pipeline runs

* Reevaluation keeps skipping: check whether
  `pipeline/reevaluation_patch.json` exists. Growth may not be running, or patch
  application may be failing.
* No reevaluation patch is produced: check that `pipeline/active_model.json`
  exists and that at least one manifest has a usable tree snapshot.
* Dataset/training blocked: inspect the claim files. Stale claims expire by TTL.
* Patch too large: lower `--reevaluation-max-nodes-per-patch`.
* Growth fails on patch application: do not delete the patch immediately.
  Inspect logs and the patch content first.

### Dependency expectation

Chipiron must run with an Anemone version that provides
`anemone.value_updates.NodeValueUpdate` and
`TreeExploration.apply_node_value_updates`.

---

## 🔄 3. Restart from scratch

```bash
rm -rf ~/oldata/victor/morpion_runs/big_run_01
mkdir -p ~/oldata/victor/morpion_runs/big_run_01
```

---

## 📊 4. Launch the dashboard (GUI)

### ✅ Recommended (robust entrypoint)

We now use a dedicated module entrypoint to avoid Streamlit import issues:

```bash
cd ~/oldata/victor/chipiron
conda activate anemone
export PYTHONPATH=src

python -m streamlit run \
  src/chipiron/environments/morpion/bootstrap/dashboard_streamlit_entry.py \
  -- --work-dir ~/oldata/victor/morpion_runs/big_run_01
```

This entrypoint wraps:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from chipiron.environments.morpion.bootstrap.dashboard_app import run_dashboard_app

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    run_dashboard_app(Path(args.work_dir))

if __name__ == "__main__":
    main()
```

### 🌐 Access

Once launched, open in your browser:

```
http://localhost:8501
```

---

## 🧪 5. Run evaluator sanity checks

Use this when a bootstrap evaluator looks unable to learn from saved data. The
sanity check requires an existing tree export in `tree_exports/`.

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_sanity_check \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --generation 1 \
  --dataset-mode terminal_path \
  --evaluator-name mlp_41 \
  --num-epochs 50 \
  --batch-size 32
```

Artifacts are written under:

```text
<work_dir>/evaluator_sanity/<run_name>/
```

Use `--run-name my_debug_run` for a stable output directory, or omit it to get a
timestamped directory.

Each run also writes `target_diagnostics.json`, which compares `direct_value` and
`backed_up_value` for the extracted rows with delta quantiles, histogram,
correlation, MSE, depth summaries, visit-count summaries, and worst deltas.

Exact/terminal-only overfit probe:

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_sanity_check \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --dataset-mode bootstrap_like \
  --max-rows 200 \
  --require-exact-or-terminal \
  --evaluator-name mlp_41 \
  --num-epochs 1000 \
  --batch-size 200 \
  --learning-rate 0.001 \
  --no-shuffle
```

Fixed-tree fitted-backup probe:

```bash
python -m chipiron.environments.morpion.bootstrap.evaluator_fitted_backup_sanity \
  --work-dir ~/oldata/victor/morpion_runs/big_run_01 \
  --dataset-mode bootstrap_like \
  --max-rows 200 \
  --max-backup-nodes 5000 \
  --backup-selection exact_terminal_plus_prefix \
  --evaluator-name mlp_41 \
  --num-iterations 5 \
  --num-epochs 100 \
  --batch-size 200 \
  --learning-rate 0.001 \
  --no-shuffle \
  --run-name fitted_backup_200x5_5k
```

This freezes one tree export, recomputes backed-up labels from exact/terminal
ground truth plus evaluator predictions, trains, and repeats. Watch
`mean_abs_target_change` and `max_abs_target_change` in `summary.json`; stable
backup targets are the convergence signal. `--max-backup-nodes` limits the
frozen-tree nodes used for prediction and backup, which keeps early diagnostic
runs cheap on large snapshots. `--backup-selection exact_terminal_plus_prefix`
keeps exact/terminal anchors even if they sit beyond the prefix; use
`--backup-selection top_terminal_paths --dataset-mode top_terminal_paths` to
focus the backup universe on the selected terminal paths.

---

## ⚠️ Known issues

### Inotify watch limit (Linux)

If you see:

```
OSError: [Errno 28] inotify watch limit reached
```

Fix it with:

```bash
sudo sysctl fs.inotify.max_user_watches=524288
```

To make it permanent:

```bash
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## 🧠 Tips

* Always store runs under `~/oldata/...` (large partition)
* Use retention (latest 1–2 checkpoints) to avoid disk explosion
* The dashboard reads from the same `--work-dir`
* The new dashboard entrypoint avoids relative-import issues with Streamlit

---

## 📁 Important directories inside a run

* `search_checkpoints/` → runtime snapshots
* `tree_exports/` → training trees
* `rows/` → supervised datasets
* `models/` → trained evaluators
* `latest_status.json` → dashboard state
* `run_state.json` → bootstrap state

---

## ✅ Minimal workflow

```bash
# start run
python -m chipiron.environments.morpion.bootstrap.launcher --work-dir ...

# open dashboard
python -m streamlit run src/chipiron/environments/morpion/bootstrap/dashboard_streamlit_entry.py -- --work-dir ...
```

---

Enjoy exploring Morpion 🚀
