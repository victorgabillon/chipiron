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
CORAL_REPO_ROOT="${CORAL_REPO_ROOT:-$HOME/oldata/victor/coral}"
ATOMHEART_REPO_ROOT="${ATOMHEART_REPO_ROOT:-$HOME/oldata/victor/atomheart}"
PYTHON_BIN="${PYTHON_BIN:-/home/pompote/oldata/conda_envs/anemone/bin/python}"
TRAINING_EXPORT_MODE="${TRAINING_EXPORT_MODE:-sharded}"
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5
GROWTH_TREE_BRANCH_LIMIT="${GROWTH_TREE_BRANCH_LIMIT:-1000000}"
GROWTH_MAX_STEPS_PER_CYCLE="${GROWTH_MAX_STEPS_PER_CYCLE:-100}"
MORPION_GROWTH_WORKER_MAX_CYCLES="${MORPION_GROWTH_WORKER_MAX_CYCLES:-20}"
GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR="${GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR:-1.2}"
GROWTH_SAVE_AFTER_SECONDS="${GROWTH_SAVE_AFTER_SECONDS:-10}"
MORPION_ROLLOUT_AFTER_OPENING="${MORPION_ROLLOUT_AFTER_OPENING:-1}"
MORPION_ROLLOUT_MAX_EXTRA_STEPS="${MORPION_ROLLOUT_MAX_EXTRA_STEPS:-none}"
# Example: MORPION_ROLLOUT_ACTION_SELECTOR_KIND=random_legal_prefer_openable
MORPION_ROLLOUT_ACTION_SELECTOR_KIND="${MORPION_ROLLOUT_ACTION_SELECTOR_KIND:-random_legal_prefer_openable}"
MORPION_ROLLOUT_RANDOM_SEED="${MORPION_ROLLOUT_RANDOM_SEED:-0}"
MORPION_ROLLOUT_STOP_ON_EXISTING_NODE="${MORPION_ROLLOUT_STOP_ON_EXISTING_NODE:-0}"
MORPION_CLUSTER_SHOW_RECAP="${MORPION_CLUSTER_SHOW_RECAP:-1}"
MORPION_CLUSTER_RECAP_EVERY_SUCCESS="${MORPION_CLUSTER_RECAP_EVERY_SUCCESS:-1}"
MORPION_CLUSTER_IDLE_LOG_EVERY="${MORPION_CLUSTER_IDLE_LOG_EVERY:-12}"
MORPION_CLUSTER_OPEN_STATUS="${MORPION_CLUSTER_OPEN_STATUS:-1}"
MORPION_CLUSTER_STATUS_REFRESH_SECONDS="${MORPION_CLUSTER_STATUS_REFRESH_SECONDS:-20}"

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

shared_env="cd \"$REPO_ROOT\" && export PYTHONPATH=\"$CORAL_REPO_ROOT/src:$ATOMHEART_REPO_ROOT/src:$ANEMONE_REPO_ROOT/src:$REPO_ROOT/src:\${PYTHONPATH:-}\" && export MORPION_WORK_DIR=\"$MORPION_WORK_DIR\""
python_diagnostics="echo \"python_bin=$PYTHON_BIN\" && which python && \"$PYTHON_BIN\" - <<'PY'
import anemone, atomheart, chipiron, coral
print("anemone:", anemone.__file__)
print("atomheart:", atomheart.__file__)
print("chipiron:", chipiron.__file__)
print("coral:", coral.__file__)
PY"
rollout_args="--rollout-max-extra-steps $MORPION_ROLLOUT_MAX_EXTRA_STEPS --rollout-action-selector-kind $MORPION_ROLLOUT_ACTION_SELECTOR_KIND --rollout-random-seed $MORPION_ROLLOUT_RANDOM_SEED"
if [[ "$MORPION_ROLLOUT_AFTER_OPENING" == "1" ]]; then
  rollout_args="$rollout_args --rollout-after-opening"
fi
if [[ "$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE" == "1" ]]; then
  rollout_args="$rollout_args --rollout-stop-on-existing-node"
fi
rollout_echo="echo \"rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE\""

growth_loop="$shared_env && echo \"[GROWTH] work_dir=$MORPION_WORK_DIR\" && echo \"[GROWTH] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[GROWTH] training_export_mode=$TRAINING_EXPORT_MODE\" && echo \"[GROWTH] tree_branch_limit=$GROWTH_TREE_BRANCH_LIMIT\" && echo \"[GROWTH] max_growth_steps_per_cycle=$GROWTH_MAX_STEPS_PER_CYCLE\" && echo \"[GROWTH] worker_max_cycles=$MORPION_GROWTH_WORKER_MAX_CYCLES\" && echo \"[GROWTH] save_after_tree_growth_factor=$GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR\" && echo \"[GROWTH] save_after_seconds=$GROWTH_SAVE_AFTER_SECONDS\" && $rollout_echo && $python_diagnostics && idle_checks=0 && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage growth --tree-branch-limit $GROWTH_TREE_BRANCH_LIMIT --max-growth-steps-per-cycle $GROWTH_MAX_STEPS_PER_CYCLE --max-cycles $MORPION_GROWTH_WORKER_MAX_CYCLES --save-after-tree-growth-factor $GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR --save-after-seconds $GROWTH_SAVE_AFTER_SECONDS $rollout_args 2>&1 | tee -a \"$LOG_DIR/growth.log\"; status=\${PIPESTATUS[0]}; if [[ \"\$status\" -eq 0 ]] && \"$PYTHON_BIN\" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); sys.exit(0 if p.is_file() and json.loads(p.read_text(encoding=\"utf-8\")).get(\"metadata\", {}).get(\"growth_status\") == \"growth_budget_already_exhausted\" else 1)' \"$MORPION_WORK_DIR/run_state.json\"; then echo \"[GROWTH] worker exited with status \$status; growth_budget_already_exhausted; not restarting\"; break; fi; if [[ \"\$status\" -ne 0 ]]; then echo \"[GROWTH] ERROR worker exited status=\$status; restarting in $GROWTH_SLEEP_SECONDS s\"; sleep $GROWTH_SLEEP_SECONDS; continue; fi; idle_checks=\$((idle_checks + 1)); if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then echo \"[GROWTH] idle; next check in $GROWTH_SLEEP_SECONDS s\"; fi; sleep $GROWTH_SLEEP_SECONDS; done"
dataset_loop="$shared_env && echo \"[DATASET] work_dir=$MORPION_WORK_DIR\" && echo \"[DATASET] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[DATASET] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && idle_checks=0 && while true; do \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage dataset_worker $rollout_args 2>&1 | tee -a \"$LOG_DIR/dataset.log\"; status=\${PIPESTATUS[0]}; if [[ \"\$status\" -ne 0 ]]; then echo \"[DATASET] ERROR worker exited status=\$status; restarting in $DATASET_SLEEP_SECONDS s\"; sleep $DATASET_SLEEP_SECONDS; continue; fi; idle_checks=\$((idle_checks + 1)); if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then echo \"[DATASET] idle; next check in $DATASET_SLEEP_SECONDS s\"; fi; sleep $DATASET_SLEEP_SECONDS; done"
training_loop="$shared_env && echo \"[TRAINING] work_dir=$MORPION_WORK_DIR gpu=0\" && echo \"[TRAINING] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[TRAINING] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && idle_checks=0 && while true; do CUDA_VISIBLE_DEVICES=0 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage training_worker $rollout_args 2>&1 | tee -a \"$LOG_DIR/training.log\"; status=\${PIPESTATUS[0]}; if [[ \"\$status\" -ne 0 ]]; then echo \"[TRAINING] ERROR worker exited status=\$status; restarting in $TRAINING_SLEEP_SECONDS s\"; sleep $TRAINING_SLEEP_SECONDS; continue; fi; idle_checks=\$((idle_checks + 1)); if [[ \"$MORPION_CLUSTER_SHOW_RECAP\" == \"1\" && ( \"$MORPION_CLUSTER_RECAP_EVERY_SUCCESS\" == \"1\" || \"\$idle_checks\" -eq 1 || \$((idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY)) -eq 0 ) ]]; then \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.training_recap --work-dir \"$MORPION_WORK_DIR\" --all-evaluators || true; fi; if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then echo \"[TRAINING] idle; no new training claim; restarting in $TRAINING_SLEEP_SECONDS s\"; fi; sleep $TRAINING_SLEEP_SECONDS; done"
reevaluation_loop="$shared_env && echo \"[REEVALUATION] work_dir=$MORPION_WORK_DIR gpu=1\" && echo \"[REEVALUATION] anemone_repo_root=$ANEMONE_REPO_ROOT\" && echo \"[REEVALUATION] training_export_mode=$TRAINING_EXPORT_MODE\" && $rollout_echo && $python_diagnostics && idle_checks=0 && while true; do CUDA_VISIBLE_DEVICES=1 \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" --pipeline-stage reevaluation --quiet-worker-startup $rollout_args 2>&1 | tee -a \"$LOG_DIR/reevaluation.log\"; status=\${PIPESTATUS[0]}; if [[ \"\$status\" -ne 0 ]]; then echo \"[REEVALUATION] ERROR worker exited status=\$status; restarting in $REEVALUATION_SLEEP_SECONDS s\"; sleep $REEVALUATION_SLEEP_SECONDS; continue; fi; idle_checks=\$((idle_checks + 1)); if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then echo \"[REEVALUATION] idle; next check in $REEVALUATION_SLEEP_SECONDS s\"; fi; sleep $REEVALUATION_SLEEP_SECONDS; done"
status_loop="$shared_env && echo \"[STATUS] work_dir=$MORPION_WORK_DIR\" && echo \"[STATUS] refresh_seconds=$MORPION_CLUSTER_STATUS_REFRESH_SECONDS\" && \"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.training_recap --work-dir \"$MORPION_WORK_DIR\" --all-evaluators --watch \"$MORPION_CLUSTER_STATUS_REFRESH_SECONDS\" --clear"

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

if [[ "$MORPION_CLUSTER_OPEN_STATUS" == "1" ]]; then
  tmux new-window -t "$SESSION_NAME" -n status "bash -lc '$status_loop'"
fi

attach_or_switch
