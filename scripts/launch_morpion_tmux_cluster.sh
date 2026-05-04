#!/usr/bin/env bash

# Phase 8.9 production-style launcher for the autonomous one-shot artifact
# pipeline workers. One command creates a tmux session with the growth,
# dataset, training, and reevaluation workers running under crash-only loops.
#
# Usage:
#   chmod +x scripts/launch_morpion_tmux_cluster.sh
#   ./scripts/launch_morpion_tmux_cluster.sh

set -u

SESSION_NAME="morpion_bootstrap"
MORPION_WORK_DIR="${MORPION_WORK_DIR:-$HOME/oldata/victor/morpion_runs/big_run_01}"
ANEMONE_REPO_ROOT="${ANEMONE_REPO_ROOT:-$HOME/oldata/victor/anemone}"
PYTHON_BIN="${PYTHON_BIN:-/home/pompote/oldata/conda_envs/anemone/bin/python}"
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$MORPION_WORK_DIR/logs"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH." >&2
  exit 1
fi

attach_or_switch() {
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION_NAME"
  else
    tmux attach-session -t "$SESSION_NAME"
  fi
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  attach_or_switch
  exit 0
fi

mkdir -p "$LOG_DIR"

shared_env="cd \"$REPO_ROOT\" && export PYTHONPATH=\"$ANEMONE_REPO_ROOT/src:$REPO_ROOT/src:\${PYTHONPATH:-}\" && export MORPION_WORK_DIR=\"$MORPION_WORK_DIR\""
python_diagnostics="echo \"python_bin=$PYTHON_BIN\" && which python && \"$PYTHON_BIN\" - <<'PY'
import anemone, chipiron
print(\"anemone:\", anemone.__file__)
print(\"chipiron:\", chipiron.__file__)
PY"

growth_loop="$shared_env && echo \"[GROWTH] work_dir=$MORPION_WORK_DIR\" && echo \"[GROWTH] anemone_repo_root=$ANEMONE_REPO_ROOT\" && $python_diagnostics && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --pipeline-stage growth 2>&1 | tee -a \"$LOG_DIR/growth.log\"; echo \"[GROWTH] worker exited with status \\\$?; restarting in $GROWTH_SLEEP_SECONDS s\"; sleep $GROWTH_SLEEP_SECONDS; done"
dataset_loop="$shared_env && echo \"[DATASET] work_dir=$MORPION_WORK_DIR\" && echo \"[DATASET] anemone_repo_root=$ANEMONE_REPO_ROOT\" && $python_diagnostics && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --pipeline-stage dataset_worker 2>&1 | tee -a \"$LOG_DIR/dataset.log\"; echo \"[DATASET] worker exited with status \\\$?; restarting in $DATASET_SLEEP_SECONDS s\"; sleep $DATASET_SLEEP_SECONDS; done"
training_loop="$shared_env && echo \"[TRAINING] work_dir=$MORPION_WORK_DIR gpu=0\" && echo \"[TRAINING] anemone_repo_root=$ANEMONE_REPO_ROOT\" && $python_diagnostics && while true; do CUDA_VISIBLE_DEVICES=0 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --pipeline-stage training_worker 2>&1 | tee -a \"$LOG_DIR/training.log\"; echo \"[TRAINING] worker exited with status \\\$?; restarting in $TRAINING_SLEEP_SECONDS s\"; sleep $TRAINING_SLEEP_SECONDS; done"
reevaluation_loop="$shared_env && echo \"[REEVALUATION] work_dir=$MORPION_WORK_DIR gpu=1\" && echo \"[REEVALUATION] anemone_repo_root=$ANEMONE_REPO_ROOT\" && $python_diagnostics && while true; do CUDA_VISIBLE_DEVICES=1 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --pipeline-stage reevaluation 2>&1 | tee -a \"$LOG_DIR/reevaluation.log\"; echo \"[REEVALUATION] worker exited with status \\\$?; restarting in $REEVALUATION_SLEEP_SECONDS s\"; sleep $REEVALUATION_SLEEP_SECONDS; done"

tmux new-session -d -s "$SESSION_NAME" -n morpion
tmux split-window -t "$SESSION_NAME":0 -h
tmux split-window -t "$SESSION_NAME":0.0 -v
tmux split-window -t "$SESSION_NAME":0.1 -v
tmux select-layout -t "$SESSION_NAME":0 tiled

tmux select-pane -t "$SESSION_NAME":0.0 -T "GROWTH"
tmux select-pane -t "$SESSION_NAME":0.1 -T "DATASET"
tmux select-pane -t "$SESSION_NAME":0.2 -T "TRAINING"
tmux select-pane -t "$SESSION_NAME":0.3 -T "REEVALUATION"

tmux send-keys -t "$SESSION_NAME":0.0 "bash -lc '$growth_loop'" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "bash -lc '$dataset_loop'" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "bash -lc '$training_loop'" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "bash -lc '$reevaluation_loop'" C-m

attach_or_switch