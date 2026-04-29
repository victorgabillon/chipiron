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
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5

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
  local command="cd \"$REPO_ROOT\" && export PYTHONPATH=src && export MORPION_WORK_DIR=\"$MORPION_WORK_DIR\" && echo \"[$worker_name] work_dir=$MORPION_WORK_DIR\" && while true; do ${extra_prefix}python -m chipiron.environments.morpion.bootstrap.launcher --work-dir \"$MORPION_WORK_DIR\" --pipeline-mode artifact_pipeline ${launcher_args} 2>&1 | tee -a \"$LOG_DIR/$log_name\"; echo \"[$worker_name] worker exited with status \\$?; restarting in $sleep_seconds s\"; sleep $sleep_seconds; done"

  gnome-terminal \
    --title="$worker_name" \
    -- bash -lc "$command"
}

launch_worker_terminal "GROWTH" "$GROWTH_SLEEP_SECONDS" "" "--pipeline-stage growth" "growth.log"
launch_worker_terminal "DATASET" "$DATASET_SLEEP_SECONDS" "" "--pipeline-stage dataset_worker" "dataset.log"
launch_worker_terminal "TRAINING" "$TRAINING_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=0 " "--pipeline-stage training_worker" "training.log"
launch_worker_terminal "REEVALUATION" "$REEVALUATION_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=1 " "--pipeline-stage reevaluation" "reevaluation.log"