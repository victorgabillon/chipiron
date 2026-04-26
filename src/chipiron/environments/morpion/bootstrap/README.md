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
  --evaluator-name mlp_41 \
  --num-iterations 20 \
  --num-epochs 100 \
  --batch-size 200 \
  --learning-rate 0.001 \
  --no-shuffle
```

This freezes one tree export, recomputes backed-up labels from exact/terminal
ground truth plus evaluator predictions, trains, and repeats. Watch
`mean_abs_target_change` and `max_abs_target_change` in `summary.json`; stable
backup targets are the convergence signal.

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
