#!/usr/bin/env bash

# Phase 8.9 production-style launcher for the autonomous one-shot artifact
# pipeline workers using GNOME Terminal. One command opens four terminals for
# growth, dataset, training, and reevaluation under crash-only supervision
# loops.
#
# Usage:
#   chmod +x scripts/launch_morpion_gnome_cluster.sh
#   ./scripts/launch_morpion_gnome_cluster.sh

set -u

MORPION_WORK_DIR="${MORPION_WORK_DIR:-$HOME/oldata/victor/morpion_runs/big_run_01}"
ANEMONE_REPO_ROOT="${ANEMONE_REPO_ROOT:-$HOME/oldata/victor/anemone}"
PYTHON_BIN="${PYTHON_BIN:-/home/pompote/oldata/conda_envs/anemone/bin/python}"
TRAINING_EXPORT_MODE="${TRAINING_EXPORT_MODE:-sharded}"
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5
GROWTH_TREE_BRANCH_LIMIT=1200000
# Controls search expansion batching for each growth worker invocation.
# It is intended as a relaunch-time throughput tuning knob.
GROWTH_MAX_STEPS_PER_CYCLE="${GROWTH_MAX_STEPS_PER_CYCLE:-10}"
GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR="${GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR:-1.2}"
GROWTH_SAVE_AFTER_SECONDS="${GROWTH_SAVE_AFTER_SECONDS:-10}"
MORPION_ROLLOUT_AFTER_OPENING="${MORPION_ROLLOUT_AFTER_OPENING:-1}"
MORPION_ROLLOUT_MAX_EXTRA_STEPS="${MORPION_ROLLOUT_MAX_EXTRA_STEPS:-none}"
# Example: MORPION_ROLLOUT_ACTION_SELECTOR_KIND=random_legal_prefer_openable
MORPION_ROLLOUT_ACTION_SELECTOR_KIND="${MORPION_ROLLOUT_ACTION_SELECTOR_KIND:-random_legal_prefer_openable}"
MORPION_ROLLOUT_RANDOM_SEED="${MORPION_ROLLOUT_RANDOM_SEED:-0}"
MORPION_ROLLOUT_STOP_ON_EXISTING_NODE="${MORPION_ROLLOUT_STOP_ON_EXISTING_NODE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$MORPION_WORK_DIR/logs"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal is required but was not found in PATH." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

launch_worker_terminal() {
  local worker_name="$1"
  local sleep_seconds="$2"
  local extra_prefix="$3"
  local launcher_args="$4"
  local log_name="$5"
  local startup_message="$6"
  local command="cd \"$REPO_ROOT\" && export PYTHONPATH=\"$ANEMONE_REPO_ROOT/src:$REPO_ROOT/src:\${PYTHONPATH:-}\" && export MORPION_WORK_DIR=\"$MORPION_WORK_DIR\" && trap 'echo; echo \"[$worker_name] stopped; terminal kept open\"; exec bash' INT TERM && printf '%b\\n' \"$startup_message\" && echo \"[$worker_name] python_bin=$PYTHON_BIN\" && which python && \"$PYTHON_BIN\" - <<'PY' && while true; do ${extra_prefix}\"$PYTHON_BIN\" -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline --training-export-mode \"$TRAINING_EXPORT_MODE\" ${launcher_args} 2>&1 | tee -a \"$LOG_DIR/$log_name\"; status=\${PIPESTATUS[0]}; if [[ \"$worker_name\" == \"GROWTH\" && \"\$status\" -eq 0 ]] && \"$PYTHON_BIN\" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); sys.exit(0 if p.is_file() and json.loads(p.read_text(encoding=\"utf-8\")).get(\"metadata\", {}).get(\"growth_status\") == \"growth_budget_already_exhausted\" else 1)' \"$MORPION_WORK_DIR/run_state.json\"; then echo \"[$worker_name] worker exited with status \$status; growth_budget_already_exhausted; not restarting\"; break; fi; echo \"[$worker_name] worker exited with status \$status; restarting in $sleep_seconds s\"; sleep $sleep_seconds; done
import anemone, chipiron
print(\"anemone:\", anemone.__file__)
print(\"chipiron:\", chipiron.__file__)
PY"

  gnome-terminal \
    --title="$worker_name" \
    -- bash -lc "$command"
}

ROLLOUT_ARGS="--rollout-max-extra-steps $MORPION_ROLLOUT_MAX_EXTRA_STEPS --rollout-action-selector-kind $MORPION_ROLLOUT_ACTION_SELECTOR_KIND --rollout-random-seed $MORPION_ROLLOUT_RANDOM_SEED"
if [[ "$MORPION_ROLLOUT_AFTER_OPENING" == "1" ]]; then
  ROLLOUT_ARGS="$ROLLOUT_ARGS --rollout-after-opening"
fi
if [[ "$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE" == "1" ]]; then
  ROLLOUT_ARGS="$ROLLOUT_ARGS --rollout-stop-on-existing-node"
fi

GROWTH_ARGS="--pipeline-stage growth --tree-branch-limit $GROWTH_TREE_BRANCH_LIMIT --max-growth-steps-per-cycle $GROWTH_MAX_STEPS_PER_CYCLE --save-after-tree-growth-factor $GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR --save-after-seconds $GROWTH_SAVE_AFTER_SECONDS $ROLLOUT_ARGS"

launch_worker_terminal "GROWTH" "$GROWTH_SLEEP_SECONDS" "" "$GROWTH_ARGS" "growth.log" "[GROWTH] work_dir=$MORPION_WORK_DIR\n[GROWTH] anemone_repo_root=$ANEMONE_REPO_ROOT\n[GROWTH] training_export_mode=$TRAINING_EXPORT_MODE\n[GROWTH] tree_branch_limit=$GROWTH_TREE_BRANCH_LIMIT\n[GROWTH] max_growth_steps_per_cycle=$GROWTH_MAX_STEPS_PER_CYCLE\n[GROWTH] save_after_tree_growth_factor=$GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR\n[GROWTH] save_after_seconds=$GROWTH_SAVE_AFTER_SECONDS\n[GROWTH] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
launch_worker_terminal "DATASET" "$DATASET_SLEEP_SECONDS" "" "--pipeline-stage dataset_worker $ROLLOUT_ARGS" "dataset.log" "[DATASET] work_dir=$MORPION_WORK_DIR\n[DATASET] anemone_repo_root=$ANEMONE_REPO_ROOT\n[DATASET] training_export_mode=$TRAINING_EXPORT_MODE\n[DATASET] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
launch_worker_terminal "TRAINING" "$TRAINING_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=0 " "--pipeline-stage training_worker $ROLLOUT_ARGS" "training.log" "[TRAINING] work_dir=$MORPION_WORK_DIR\n[TRAINING] anemone_repo_root=$ANEMONE_REPO_ROOT\n[TRAINING] training_export_mode=$TRAINING_EXPORT_MODE\n[TRAINING] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
launch_worker_terminal "REEVALUATION" "$REEVALUATION_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=1 " "--pipeline-stage reevaluation $ROLLOUT_ARGS" "reevaluation.log" "[REEVALUATION] work_dir=$MORPION_WORK_DIR\n[REEVALUATION] anemone_repo_root=$ANEMONE_REPO_ROOT\n[REEVALUATION] training_export_mode=$TRAINING_EXPORT_MODE\n[REEVALUATION] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
