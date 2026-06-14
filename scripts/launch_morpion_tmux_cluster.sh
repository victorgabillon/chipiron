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
TRAINING_EXPORT_MODE="${TRAINING_EXPORT_MODE:-sharded}"
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5
GROWTH_TREE_BRANCH_LIMIT="${GROWTH_TREE_BRANCH_LIMIT:-1000000}"
GROWTH_MAX_STEPS_PER_CYCLE="${GROWTH_MAX_STEPS_PER_CYCLE:-100}"
GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR="${GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR:-1.2}"
GROWTH_SAVE_AFTER_SECONDS="${GROWTH_SAVE_AFTER_SECONDS:-10}"
MORPION_ROLLOUT_AFTER_OPENING="${MORPION_ROLLOUT_AFTER_OPENING:-1}"
MORPION_ROLLOUT_MAX_EXTRA_STEPS="${MORPION_ROLLOUT_MAX_EXTRA_STEPS:-none}"
MORPION_ROLLOUT_ACTION_SELECTOR_KIND="${MORPION_ROLLOUT_ACTION_SELECTOR_KIND:-random_openable}"
MORPION_ROLLOUT_RANDOM_SEED="${MORPION_ROLLOUT_RANDOM_SEED:-0}"
MORPION_ROLLOUT_STOP_ON_EXISTING_NODE="${MORPION_ROLLOUT_STOP_ON_EXISTING_NODE:-0}"

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
rollout_args="--rollout-max-extra-steps $MORPION_ROLLOUT_MAX_EXTRA_STEPS --rollout-action-selector-kind $MORPION_ROLLOUT_ACTION_SELECTOR_KIND --rollout-random-seed $MORPION_ROLLOUT_RANDOM_SEED"
if [[ "$MORPION_ROLLOUT_AFTER_OPENING" == "1" ]]; then
  rollout_args="$rollout_args --rollout-after-opening"
fi
if [[ "$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE" == "1" ]]; then
  rollout_args="$rollout_args --rollout-stop-on-existing-node"
fi
rollout_echo="echo \"rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE\""

growth_loop="$shared_env && echo \"[GROWTH] work_dir=$MORPION_WORK_DIR\" && echo \"[GROWTH] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[GROWTH] training_export_mode=$TRAINING_EXPORT_MODE\" && echo \"[GROWTH] tree_branch_limit=$GROWTH_TREE_BRANCH_LIMIT\" && echo \"[GROWTH] max_growth_steps_per_cycle=$GROWTH_MAX_STEPS_PER_CYCLE\" && echo \"[GROWTH] save_after_tree_growth_factor=$GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR\" && echo \"[GROWTH] save_after_seconds=$GROWTH_SAVE_AFTER_SECONDS\" && $rollout_echo && $python_diagnostics && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage growth --tree-branch-limit $GROWTH_TREE_BRANCH_LIMIT --max-growth-steps-per-cycle $GROWTH_MAX_STEPS_PER_CYCLE --save-after-tree-growth-factor $GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR --save-after-seconds $GROWTH_SAVE_AFTER_SECONDS $rollout_args 2>&1 | tee -a \"$LOG_DIR/growth.log\"; echo \"[GROWTH] worker exited with status \\\$?; restarting in $GROWTH_SLEEP_SECONDS s\"; sleep $GROWTH_SLEEP_SECONDS; done"
dataset_loop="$shared_env && echo \"[DATASET] work_dir=$MORPION_WORK_DIR\" && echo \"[DATASET] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[DATASET] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage dataset_worker $rollout_args 2>&1 | tee -a \"$LOG_DIR/dataset.log\"; echo \"[DATASET] worker exited with status \\\$?; restarting in $DATASET_SLEEP_SECONDS s\"; sleep $DATASET_SLEEP_SECONDS; done"
training_loop="$shared_env && echo \"[TRAINING] work_dir=$MORPION_WORK_DIR gpu=0\" && echo \"[TRAINING] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[TRAINING] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && while true; do CUDA_VISIBLE_DEVICES=0 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage training_worker $rollout_args 2>&1 | tee -a \"$LOG_DIR/training.log\"; echo \"[TRAINING] worker exited with status \\\$?; restarting in $TRAINING_SLEEP_SECONDS s\"; sleep $TRAINING_SLEEP_SECONDS; done"
reevaluation_loop="$shared_env && echo \"[REEVALUATION] work_dir=$MORPION_WORK_DIR gpu=1\" && echo \"[REEVALUATION] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[REEVALUATION] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && while true; do CUDA_VISIBLE_DEVICES=1 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage reevaluation $rollout_args 2>&1 | tee -a \"$LOG_DIR/reevaluation.log\"; echo \"[REEVALUATION] worker exited with status \\\$?; restarting in $REEVALUATION_SLEEP_SECONDS s\"; sleep $REEVALUATION_SLEEP_SECONDS; done"

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
