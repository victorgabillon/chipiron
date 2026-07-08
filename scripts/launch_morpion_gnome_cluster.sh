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

# Run identity and local source roots.
# MORPION_WORK_DIR is the self-contained run directory. Change it when starting
# a different production run; keep it off old runs unless intentionally
# resuming them.
MORPION_WORK_DIR="${MORPION_WORK_DIR:-$HOME/oldata/victor/morpion_runs/generic_linoo_fresh_with_bigrun_models_v1}"
# Repository roots used to build worker PYTHONPATH. Override these only when
# testing another local checkout of one dependency.
ANEMONE_REPO_ROOT="${ANEMONE_REPO_ROOT:-$HOME/oldata/victor/anemone}"
CORAL_REPO_ROOT="${CORAL_REPO_ROOT:-$HOME/oldata/victor/coral}"
ATOMHEART_REPO_ROOT="${ATOMHEART_REPO_ROOT:-$HOME/oldata/victor/atomheart}"
# Python interpreter for all workers. Keep this on the anemone environment
# unless intentionally testing a different environment.
PYTHON_BIN="${PYTHON_BIN:-/home/pompote/oldata/conda_envs/anemone/bin/python}"

# Export and checkpoint formats. `sharded` is the production-safe setting for
# large Morpion trees; change only for compatibility testing.
TRAINING_EXPORT_MODE="${TRAINING_EXPORT_MODE:-sharded}"
RUNTIME_CHECKPOINT_FORMAT="${RUNTIME_CHECKPOINT_FORMAT:-sharded}"
MORPION_EVALUATOR_FAMILY="${MORPION_EVALUATOR_FAMILY:-canonical_linear_mlp_entity_transformer_small}"
MORPION_ALLOW_EVALUATOR_CATALOG_EXTENSION="${MORPION_ALLOW_EVALUATOR_CATALOG_EXTENSION:-1}"

# Worker selection. `all` opens growth, dataset, training, and reevaluation;
# use one of `growth`, `dataset`, `training`, or `reevaluation` for focused
# debugging or partial restarts.
MORPION_CLUSTER_WORKERS="${MORPION_CLUSTER_WORKERS:-all}"
MORPION_CLUSTER_SHOW_RECAP="${MORPION_CLUSTER_SHOW_RECAP:-1}"
MORPION_CLUSTER_RECAP_EVERY_SUCCESS="${MORPION_CLUSTER_RECAP_EVERY_SUCCESS:-1}"
MORPION_CLUSTER_IDLE_LOG_EVERY="${MORPION_CLUSTER_IDLE_LOG_EVERY:-12}"
MORPION_CLUSTER_OPEN_STATUS="${MORPION_CLUSTER_OPEN_STATUS:-1}"
MORPION_CLUSTER_STATUS_REFRESH_SECONDS="${MORPION_CLUSTER_STATUS_REFRESH_SECONDS:-20}"

# Optional model seeding. Leave MORPION_SEED_MODELS_FROM_WORK_DIR empty for a
# self-contained run. Set it to another work dir only when bootstrapping a new
# run from that run's `models/` directory. Seeding never copies pipeline state.
MORPION_SEED_MODELS_FROM_WORK_DIR="${MORPION_SEED_MODELS_FROM_WORK_DIR:-}"
# With the default `1`, an existing target `models/` directory is left alone.
# Set to `0` only when you want the script to fail instead of reusing it.
MORPION_SEED_MODELS_IF_MISSING="${MORPION_SEED_MODELS_IF_MISSING:-1}"
# Keep this at `0` for production. Symlinked models can mutate or depend on an
# old run; set to `1` only for deliberate local experiments.
MORPION_ALLOW_SYMLINKED_MODELS="${MORPION_ALLOW_SYMLINKED_MODELS:-0}"
# Allows a fresh-looking work dir to contain a deliberate
# `pipeline/active_model.json`. Keep `1` for seeded fresh runs; use `0` when you
# want any active-model artifact in a fresh dir to be treated as stale state.
MORPION_ALLOW_FRESH_ACTIVE_MODEL="${MORPION_ALLOW_FRESH_ACTIVE_MODEL:-1}"
# These identify the model expected by the startup status line. They do not
# create or select the active model; the pipeline artifact controls that.
MORPION_SEED_ACTIVE_MODEL_GENERATION="${MORPION_SEED_ACTIVE_MODEL_GENERATION:-430}"
MORPION_SEED_ACTIVE_MODEL_EVALUATOR="${MORPION_SEED_ACTIVE_MODEL_EVALUATOR:-mlp_41}"

# Crash-only supervision delays, in seconds. Increase if repeated failures are
# noisy or external resources need longer to settle between retries.
GROWTH_SLEEP_SECONDS=2
DATASET_SLEEP_SECONDS=5
TRAINING_SLEEP_SECONDS=5
REEVALUATION_SLEEP_SECONDS=5

# Growth budget and batching. Lower GROWTH_TREE_BRANCH_LIMIT for smoke tests;
# raise it for longer production runs. The step/save knobs trade throughput
# against checkpoint frequency and recovery granularity.
GROWTH_TREE_BRANCH_LIMIT="${GROWTH_TREE_BRANCH_LIMIT:-1200000}"
GROWTH_MAX_STEPS_PER_CYCLE="${GROWTH_MAX_STEPS_PER_CYCLE:-10}"
GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR="${GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR:-1.2}"
GROWTH_SAVE_AFTER_SECONDS="${GROWTH_SAVE_AFTER_SECONDS:-10}"

# Training worker controls. Leave evaluator names empty to train all configured
# evaluators. Use max rows for quick experiments, and chunk size to trade memory
# for export/training throughput.
MORPION_TRAINING_EVALUATOR_NAMES="${MORPION_TRAINING_EVALUATOR_NAMES:-}"
MORPION_TRAINING_MAX_ROWS="${MORPION_TRAINING_MAX_ROWS:-}"
MORPION_TRAINING_ROW_CHUNK_SIZE="${MORPION_TRAINING_ROW_CHUNK_SIZE:-8192}"
# Diagnostics are small by default. Increase max rows for deeper inspection, or
# set MORPION_SKIP_EVALUATOR_DIAGNOSTICS=1 to skip them during throughput runs.
MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS="${MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS:-60}"
MORPION_SKIP_EVALUATOR_DIAGNOSTICS="${MORPION_SKIP_EVALUATOR_DIAGNOSTICS:-0}"

# Memory guards. The RAM floor prevents workers from starting risky loads on a
# tight machine. Candidate checkpoint headroom leaves forecast slack around
# checkpoint loads before growth starts.
MORPION_MIN_AVAILABLE_RAM_MB="${MORPION_MIN_AVAILABLE_RAM_MB:-1200}"
MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR="${MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR:-60}"
MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB="${MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB:-512}"

# Optional growth memory profiling. Keep disabled in normal production because
# profiling adds overhead; enable temporarily when investigating memory growth.
MORPION_GROWTH_MEMORY_PROFILE="${MORPION_GROWTH_MEMORY_PROFILE:-0}"
MORPION_GROWTH_MEMORY_PROFILE_TOP_N="${MORPION_GROWTH_MEMORY_PROFILE_TOP_N:-20}"
MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES="${MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES:-2000}"

# Growth state eviction. `frontier_cold` with `delta_when_safe` is the current
# production profile: it saves memory while keeping rematerialization bounded.
# Tune depths/windows/cache sizes only when profiling shows a bottleneck.
MORPION_GROWTH_STATE_EVICTION_POLICY="${MORPION_GROWTH_STATE_EVICTION_POLICY:-frontier_cold}"
MORPION_GROWTH_STATE_EVICTION_PAYLOAD_MODE="${MORPION_GROWTH_STATE_EVICTION_PAYLOAD_MODE:-delta_when_safe}"
MORPION_GROWTH_STATE_EVICTION_DELTA_CHAIN_MAX_DEPTH="${MORPION_GROWTH_STATE_EVICTION_DELTA_CHAIN_MAX_DEPTH:-32}"
MORPION_GROWTH_STATE_EVICTION_RECENT_WINDOW="${MORPION_GROWTH_STATE_EVICTION_RECENT_WINDOW:-100}"
MORPION_GROWTH_STATE_REMATERIALIZATION_CACHE_SIZE="${MORPION_GROWTH_STATE_REMATERIALIZATION_CACHE_SIZE:-10000}"
MORPION_GROWTH_STATE_EVICTION_SCAN_INTERVAL_STEPS="${MORPION_GROWTH_STATE_EVICTION_SCAN_INTERVAL_STEPS:-10}"
MORPION_GROWTH_STATE_EVICTION_SCAN_NODE_LIMIT="${MORPION_GROWTH_STATE_EVICTION_SCAN_NODE_LIMIT:-5000}"

# Rollout behavior after opening expansion. The defaults favor legal random
# continuation with reproducibility. Use `none` for unlimited extra steps, or a
# number for bounded smoke tests.
MORPION_ROLLOUT_AFTER_OPENING="${MORPION_ROLLOUT_AFTER_OPENING:-1}"
MORPION_ROLLOUT_MAX_EXTRA_STEPS="${MORPION_ROLLOUT_MAX_EXTRA_STEPS:-none}"
# Safe selector values are the action selector kinds supported by the launcher;
# production currently uses `random_legal_prefer_openable`.
# Example: MORPION_ROLLOUT_ACTION_SELECTOR_KIND=random_legal_prefer_openable
MORPION_ROLLOUT_ACTION_SELECTOR_KIND="${MORPION_ROLLOUT_ACTION_SELECTOR_KIND:-random_legal_prefer_openable}"
MORPION_ROLLOUT_RANDOM_SEED="${MORPION_ROLLOUT_RANDOM_SEED:-0}"
# Set to `1` when rollouts should stop as soon as they reconnect to known tree
# state; keep `0` when exploring through existing nodes is acceptable.
MORPION_ROLLOUT_STOP_ON_EXISTING_NODE="${MORPION_ROLLOUT_STOP_ON_EXISTING_NODE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$MORPION_WORK_DIR/logs"
MODEL_PARAM_PATH="$MORPION_WORK_DIR/models/generation_$(printf "%06d" "$MORPION_SEED_ACTIVE_MODEL_GENERATION")/$MORPION_SEED_ACTIVE_MODEL_EVALUATOR/param.pt"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal is required but was not found in PATH." >&2
  exit 1
fi

should_launch_worker() {
  local name="$1"
  [[ "$MORPION_CLUSTER_WORKERS" == "all" || "$MORPION_CLUSTER_WORKERS" == "$name" ]]
}

validate_worker_selection() {
  case "$MORPION_CLUSTER_WORKERS" in
    all|growth|dataset|training|reevaluation)
      ;;
    *)
      echo "Invalid MORPION_CLUSTER_WORKERS=$MORPION_CLUSTER_WORKERS; expected all, growth, dataset, training, or reevaluation." >&2
      exit 1
      ;;
  esac
}

seed_models_if_requested() {
  local source_models
  local target_models
  if [[ -z "$MORPION_SEED_MODELS_FROM_WORK_DIR" ]]; then
    return
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required for MORPION_SEED_MODELS_FROM_WORK_DIR but was not found in PATH." >&2
    exit 1
  fi
  source_models="$MORPION_SEED_MODELS_FROM_WORK_DIR/models"
  target_models="$MORPION_WORK_DIR/models"
  if [[ ! -d "$source_models" ]]; then
    echo "Seed models source does not exist: $source_models" >&2
    exit 1
  fi
  if [[ -L "$target_models" ]]; then
    if [[ "$MORPION_ALLOW_SYMLINKED_MODELS" != "1" ]]; then
      echo "Refusing to launch: $target_models is a symlink. Use a real copy to avoid mutating the old run." >&2
      exit 1
    fi
    echo "[seed-models] allowing symlinked models because MORPION_ALLOW_SYMLINKED_MODELS=1: $target_models" >&2
    return
  fi
  if [[ -e "$target_models" ]]; then
    if [[ "$MORPION_SEED_MODELS_IF_MISSING" == "1" ]]; then
      echo "[seed-models] target models already exist; not overwriting: $target_models"
      return
    fi
    echo "Target models already exist: $target_models. Refusing to overwrite." >&2
    exit 1
  fi
  mkdir -p "$MORPION_WORK_DIR"
  echo "[seed-models] copying models from $source_models/ to $target_models/"
  rsync -a --info=progress2 "$source_models/" "$target_models/"
}

refuse_stale_active_model_for_fresh_run() {
  local active_model_path="$MORPION_WORK_DIR/pipeline/active_model.json"
  if [[ ! -f "$active_model_path" ]]; then
    return
  fi
  if [[ -f "$MORPION_WORK_DIR/run_state.json" ]]; then
    return
  fi
  if [[ -f "$MORPION_WORK_DIR/history.jsonl" ]]; then
    return
  fi
  if [[ -f "$MORPION_WORK_DIR/latest_status.json" ]]; then
    return
  fi
  if [[ "$MORPION_ALLOW_FRESH_ACTIVE_MODEL" == "1" ]]; then
    validate_fresh_active_model "$active_model_path"
    return
  fi
  echo "Refusing to launch: found $active_model_path in a fresh-looking work dir with no run_state/history/latest status." >&2
  echo "Remove stale pipeline state before a fresh tree run:" >&2
  echo "  rm -rf \"$MORPION_WORK_DIR/pipeline\" \"$MORPION_WORK_DIR/bootstrap_config.json\" \"$MORPION_WORK_DIR/launcher_process_state.json\"" >&2
  echo "Or move models aside and reset the work dir." >&2
  exit 1
}

validate_fresh_active_model() {
  local active_model_path="$1"
  local validation_output
  if ! validation_output="$("$PYTHON_BIN" - "$MORPION_WORK_DIR" "$active_model_path" <<'PY'
import json
import pathlib
import sys

work_dir = pathlib.Path(sys.argv[1])
active_model_path = pathlib.Path(sys.argv[2])

def fail(message: str) -> None:
    print(f"Invalid seeded active model: {message}", file=sys.stderr)
    raise SystemExit(1)

try:
    payload = json.loads(active_model_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    fail(f"{active_model_path} is not valid JSON: {exc}")

if not isinstance(payload, dict):
    fail(f"{active_model_path} must contain a JSON object")

evaluator_name = payload.get("evaluator_name")
generation = payload.get("generation")
model_bundle_path = payload.get("model_bundle_path")
updated_at_utc = payload.get("updated_at_utc")

if not isinstance(evaluator_name, str):
    fail("field `evaluator_name` must be a string")
if isinstance(generation, bool) or not isinstance(generation, int):
    fail("field `generation` must be an integer")
if not isinstance(model_bundle_path, str):
    fail("field `model_bundle_path` must be a string")
if not isinstance(updated_at_utc, str):
    fail("field `updated_at_utc` must be a string")

bundle_path = pathlib.PurePosixPath(model_bundle_path)
if bundle_path.is_absolute():
    fail("field `model_bundle_path` must be relative")
if ".." in bundle_path.parts:
    fail("field `model_bundle_path` must not contain '..'")

param_path = work_dir / pathlib.Path(*bundle_path.parts) / "param.pt"
if not param_path.is_file():
    fail(f"model parameter file does not exist: {param_path}")

print(f"{evaluator_name}\t{generation}\t{model_bundle_path}")
PY
  )"; then
    exit 1
  fi
  IFS=$'\t' read -r seeded_evaluator seeded_generation seeded_bundle_path <<< "$validation_output"
  echo "[seed-models] seeded active model accepted evaluator=$seeded_evaluator generation=$seeded_generation model_bundle_path=$seeded_bundle_path"
}

validate_worker_selection
seed_models_if_requested
refuse_stale_active_model_for_fresh_run
if [[ -L "$MORPION_WORK_DIR/models" && "$MORPION_ALLOW_SYMLINKED_MODELS" != "1" ]]; then
  echo "Refusing to launch: $MORPION_WORK_DIR/models is a symlink. Use a real copy to avoid mutating the old run." >&2
  exit 1
fi

launch_worker_terminal() {
  local worker_name="$1"
  local sleep_seconds="$2"
  local extra_prefix="$3"
  local launcher_args="$4"
  local log_name="$5"
  local startup_message="$6"
  local command
  command=$(cat <<EOF
cd "$REPO_ROOT" &&
export PYTHONPATH="$CORAL_REPO_ROOT/src:$ATOMHEART_REPO_ROOT/src:$ANEMONE_REPO_ROOT/src:$REPO_ROOT/src:\${PYTHONPATH:-}" &&
export MORPION_WORK_DIR="$MORPION_WORK_DIR" &&
mkdir -p "$LOG_DIR" &&
trap 'echo; echo "[$worker_name] stopped; terminal kept open"; exec bash' INT TERM &&
printf '%b\n' "$startup_message" &&
echo "[$worker_name] python_bin=$PYTHON_BIN" &&
which python &&
"$PYTHON_BIN" - <<'PY' &&
import anemone, atomheart, chipiron, coral
print(\"anemone:\", anemone.__file__)
print(\"atomheart:\", atomheart.__file__)
print(\"chipiron:\", chipiron.__file__)
print(\"coral:\", coral.__file__)
PY
idle_checks=0
while true; do
  ${extra_prefix}"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.launcher --work-dir "$MORPION_WORK_DIR" --pipeline-mode artifact_pipeline --training-export-mode "$TRAINING_EXPORT_MODE" ${launcher_args} 2>&1 | tee -a "$LOG_DIR/$log_name"
  status=\${PIPESTATUS[0]}
  if [[ "$worker_name" == "GROWTH" && "\$status" -eq 0 ]] && "$PYTHON_BIN" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); sys.exit(0 if p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("growth_status") == "growth_budget_already_exhausted" else 1)' "$MORPION_WORK_DIR/run_state.json"; then
    echo "[$worker_name] worker exited with status \$status; growth_budget_already_exhausted; not restarting"
    break
  fi
  if [[ "\$status" -ne 0 ]]; then
    echo "[$worker_name] ERROR worker exited status=\$status; restarting in $sleep_seconds s"
    sleep $sleep_seconds
    continue
  fi
  idle_checks=\$((idle_checks + 1))
  if [[ "$worker_name" == "TRAINING" ]]; then
    if [[ "$MORPION_CLUSTER_SHOW_RECAP" == "1" && ( "$MORPION_CLUSTER_RECAP_EVERY_SUCCESS" == "1" || "\$idle_checks" -eq 1 || \$((idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY)) -eq 0 ) ]]; then
      "$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.training_recap --work-dir "$MORPION_WORK_DIR" || true
    fi
    if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then
      echo "[TRAINING] idle; no new training claim; restarting in $sleep_seconds s"
    fi
    sleep $sleep_seconds
    continue
  fi
  if [[ "$worker_name" == "REEVALUATION" ]]; then
    if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then
      echo "[REEVALUATION] idle; next check in $sleep_seconds s"
    fi
    sleep $sleep_seconds
    continue
  fi
  if (( idle_checks == 1 || idle_checks % $MORPION_CLUSTER_IDLE_LOG_EVERY == 0 )); then
    echo "[$worker_name] idle; next check in $sleep_seconds s"
  fi
  sleep $sleep_seconds
done
EOF
)

  gnome-terminal \
    --title="$worker_name" \
    -- bash -lc "$command"
}

launch_status_terminal() {
  local command
  command=$(cat <<EOF
cd "$REPO_ROOT" &&
export PYTHONPATH="$CORAL_REPO_ROOT/src:$ATOMHEART_REPO_ROOT/src:$ANEMONE_REPO_ROOT/src:$REPO_ROOT/src:\${PYTHONPATH:-}" &&
export MORPION_WORK_DIR="$MORPION_WORK_DIR" &&
echo "[STATUS] work_dir=$MORPION_WORK_DIR" &&
echo "[STATUS] refresh_seconds=$MORPION_CLUSTER_STATUS_REFRESH_SECONDS" &&
trap 'echo; echo "[STATUS] stopped; terminal kept open"; exec bash' INT TERM &&
"$PYTHON_BIN" -m chipiron.environments.morpion.bootstrap.training_recap --work-dir "$MORPION_WORK_DIR" --watch "$MORPION_CLUSTER_STATUS_REFRESH_SECONDS" --clear
EOF
)

  gnome-terminal \
    --title="STATUS" \
    -- bash -lc "$command"
}

ROLLOUT_ARGS="--rollout-max-extra-steps $MORPION_ROLLOUT_MAX_EXTRA_STEPS --rollout-action-selector-kind $MORPION_ROLLOUT_ACTION_SELECTOR_KIND --rollout-random-seed $MORPION_ROLLOUT_RANDOM_SEED"
if [[ "$MORPION_ROLLOUT_AFTER_OPENING" == "1" ]]; then
  ROLLOUT_ARGS="$ROLLOUT_ARGS --rollout-after-opening"
fi
if [[ "$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE" == "1" ]]; then
  ROLLOUT_ARGS="$ROLLOUT_ARGS --rollout-stop-on-existing-node"
fi

RAM_GUARD_ARGS=""
if [[ -n "$MORPION_MIN_AVAILABLE_RAM_MB" ]]; then
  RAM_GUARD_ARGS="--min-available-ram-mb $MORPION_MIN_AVAILABLE_RAM_MB"
fi
EVALUATOR_CATALOG_ARGS="--evaluator-family $MORPION_EVALUATOR_FAMILY"
if [[ "$MORPION_ALLOW_EVALUATOR_CATALOG_EXTENSION" == "1" ]]; then
  EVALUATOR_CATALOG_ARGS="$EVALUATOR_CATALOG_ARGS --allow-evaluator-catalog-extension"
fi
GROWTH_PROFILE_ARGS=""
if [[ "$MORPION_GROWTH_MEMORY_PROFILE" == "1" ]]; then
  GROWTH_PROFILE_ARGS="--growth-memory-profile --growth-memory-profile-top-n $MORPION_GROWTH_MEMORY_PROFILE_TOP_N --growth-memory-profile-sample-nodes $MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES"
fi

GROWTH_STATE_EVICTION_ARGS="--growth-state-eviction-policy $MORPION_GROWTH_STATE_EVICTION_POLICY --growth-state-eviction-payload-mode $MORPION_GROWTH_STATE_EVICTION_PAYLOAD_MODE --growth-state-eviction-delta-chain-max-depth $MORPION_GROWTH_STATE_EVICTION_DELTA_CHAIN_MAX_DEPTH --growth-state-eviction-recent-window $MORPION_GROWTH_STATE_EVICTION_RECENT_WINDOW --growth-state-rematerialization-cache-size $MORPION_GROWTH_STATE_REMATERIALIZATION_CACHE_SIZE --growth-state-eviction-scan-interval-steps $MORPION_GROWTH_STATE_EVICTION_SCAN_INTERVAL_STEPS --growth-state-eviction-scan-node-limit $MORPION_GROWTH_STATE_EVICTION_SCAN_NODE_LIMIT"
RUNTIME_CHECKPOINT_ARGS="--runtime-checkpoint-format $RUNTIME_CHECKPOINT_FORMAT"
GROWTH_ARGS="--pipeline-stage growth $RUNTIME_CHECKPOINT_ARGS --tree-branch-limit $GROWTH_TREE_BRANCH_LIMIT --max-growth-steps-per-cycle $GROWTH_MAX_STEPS_PER_CYCLE --save-after-tree-growth-factor $GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR --save-after-seconds $GROWTH_SAVE_AFTER_SECONDS --candidate-checkpoint-load-headroom-factor $MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR --candidate-checkpoint-load-min-headroom-mb $MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB $GROWTH_STATE_EVICTION_ARGS $ROLLOUT_ARGS $RAM_GUARD_ARGS $GROWTH_PROFILE_ARGS $EVALUATOR_CATALOG_ARGS"
DATASET_ARGS="--pipeline-stage dataset_worker $RUNTIME_CHECKPOINT_ARGS $GROWTH_STATE_EVICTION_ARGS $ROLLOUT_ARGS $RAM_GUARD_ARGS $EVALUATOR_CATALOG_ARGS"
TRAINING_ARGS="--pipeline-stage training_worker $RUNTIME_CHECKPOINT_ARGS --training-row-chunk-size $MORPION_TRAINING_ROW_CHUNK_SIZE --evaluator-diagnostics-max-rows $MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS $GROWTH_STATE_EVICTION_ARGS $ROLLOUT_ARGS $RAM_GUARD_ARGS $EVALUATOR_CATALOG_ARGS"
REEVALUATION_ARGS="--pipeline-stage reevaluation --quiet-worker-startup $RUNTIME_CHECKPOINT_ARGS $GROWTH_STATE_EVICTION_ARGS $ROLLOUT_ARGS $RAM_GUARD_ARGS $EVALUATOR_CATALOG_ARGS"
if [[ -n "$MORPION_TRAINING_EVALUATOR_NAMES" ]]; then
  TRAINING_ARGS="$TRAINING_ARGS --training-evaluator-names $MORPION_TRAINING_EVALUATOR_NAMES"
fi
if [[ -n "$MORPION_TRAINING_MAX_ROWS" ]]; then
  TRAINING_ARGS="$TRAINING_ARGS --training-max-rows $MORPION_TRAINING_MAX_ROWS"
fi
if [[ "$MORPION_SKIP_EVALUATOR_DIAGNOSTICS" == "1" ]]; then
  TRAINING_ARGS="$TRAINING_ARGS --skip-evaluator-diagnostics"
fi

if should_launch_worker growth; then
  launch_worker_terminal "GROWTH" "$GROWTH_SLEEP_SECONDS" "" "$GROWTH_ARGS" "growth.log" "[GROWTH] work_dir=$MORPION_WORK_DIR\n[GROWTH] anemone_repo_root=$ANEMONE_REPO_ROOT\n[GROWTH] coral_repo_root=$CORAL_REPO_ROOT\n[GROWTH] atomheart_repo_root=$ATOMHEART_REPO_ROOT\n[GROWTH] training_export_mode=$TRAINING_EXPORT_MODE\n[GROWTH] runtime_checkpoint_format=$RUNTIME_CHECKPOINT_FORMAT\n[GROWTH] min_available_ram_mb=${MORPION_MIN_AVAILABLE_RAM_MB:-disabled}\n[GROWTH] memory_profile=$MORPION_GROWTH_MEMORY_PROFILE\n[GROWTH] memory_profile_top_n=$MORPION_GROWTH_MEMORY_PROFILE_TOP_N\n[GROWTH] memory_profile_sample_nodes=$MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES\n[GROWTH] candidate_checkpoint_load_headroom_factor=$MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR\n[GROWTH] candidate_checkpoint_load_min_headroom_mb=$MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB\n[GROWTH] tree_branch_limit=$GROWTH_TREE_BRANCH_LIMIT\n[GROWTH] max_growth_steps_per_cycle=$GROWTH_MAX_STEPS_PER_CYCLE\n[GROWTH] save_after_tree_growth_factor=$GROWTH_SAVE_AFTER_TREE_GROWTH_FACTOR\n[GROWTH] save_after_seconds=$GROWTH_SAVE_AFTER_SECONDS\n[GROWTH] state_eviction_policy=$MORPION_GROWTH_STATE_EVICTION_POLICY\n[GROWTH] state_eviction_payload_mode=$MORPION_GROWTH_STATE_EVICTION_PAYLOAD_MODE\n[GROWTH] state_eviction_delta_chain_max_depth=$MORPION_GROWTH_STATE_EVICTION_DELTA_CHAIN_MAX_DEPTH\n[GROWTH] state_eviction_recent_window=$MORPION_GROWTH_STATE_EVICTION_RECENT_WINDOW\n[GROWTH] state_rematerialization_cache_size=$MORPION_GROWTH_STATE_REMATERIALIZATION_CACHE_SIZE\n[GROWTH] state_eviction_scan_interval_steps=$MORPION_GROWTH_STATE_EVICTION_SCAN_INTERVAL_STEPS\n[GROWTH] state_eviction_scan_node_limit=$MORPION_GROWTH_STATE_EVICTION_SCAN_NODE_LIMIT\n[GROWTH] seed_models_from=${MORPION_SEED_MODELS_FROM_WORK_DIR:-disabled}\n[GROWTH] seed_model_param_exists=$(if [[ -f "$MODEL_PARAM_PATH" ]]; then echo yes; else echo no; fi) path=$MODEL_PARAM_PATH\n[GROWTH] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
fi
if should_launch_worker dataset; then
  launch_worker_terminal "DATASET" "$DATASET_SLEEP_SECONDS" "" "$DATASET_ARGS" "dataset.log" "[DATASET] work_dir=$MORPION_WORK_DIR\n[DATASET] anemone_repo_root=$ANEMONE_REPO_ROOT\n[DATASET] coral_repo_root=$CORAL_REPO_ROOT\n[DATASET] atomheart_repo_root=$ATOMHEART_REPO_ROOT\n[DATASET] training_export_mode=$TRAINING_EXPORT_MODE\n[DATASET] min_available_ram_mb=${MORPION_MIN_AVAILABLE_RAM_MB:-disabled}\n[DATASET] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
fi
if should_launch_worker training; then
  launch_worker_terminal "TRAINING" "$TRAINING_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=0 " "$TRAINING_ARGS" "training.log" "[TRAINING] work_dir=$MORPION_WORK_DIR\n[TRAINING] anemone_repo_root=$ANEMONE_REPO_ROOT\n[TRAINING] coral_repo_root=$CORAL_REPO_ROOT\n[TRAINING] atomheart_repo_root=$ATOMHEART_REPO_ROOT\n[TRAINING] training_export_mode=$TRAINING_EXPORT_MODE\n[TRAINING] evaluator_names=${MORPION_TRAINING_EVALUATOR_NAMES:-all}\n[TRAINING] max_rows=${MORPION_TRAINING_MAX_ROWS:-all}\n[TRAINING] row_chunk_size=$MORPION_TRAINING_ROW_CHUNK_SIZE\n[TRAINING] evaluator_diagnostics_max_rows=$MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS\n[TRAINING] skip_evaluator_diagnostics=$MORPION_SKIP_EVALUATOR_DIAGNOSTICS\n[TRAINING] min_available_ram_mb=${MORPION_MIN_AVAILABLE_RAM_MB:-disabled}\n[TRAINING] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
fi
if should_launch_worker reevaluation; then
  launch_worker_terminal "REEVALUATION" "$REEVALUATION_SLEEP_SECONDS" "CUDA_VISIBLE_DEVICES=1 " "$REEVALUATION_ARGS" "reevaluation.log" "[REEVALUATION] work_dir=$MORPION_WORK_DIR\n[REEVALUATION] anemone_repo_root=$ANEMONE_REPO_ROOT\n[REEVALUATION] coral_repo_root=$CORAL_REPO_ROOT\n[REEVALUATION] atomheart_repo_root=$ATOMHEART_REPO_ROOT\n[REEVALUATION] training_export_mode=$TRAINING_EXPORT_MODE\n[REEVALUATION] min_available_ram_mb=${MORPION_MIN_AVAILABLE_RAM_MB:-disabled}\n[REEVALUATION] rollout: enabled=$MORPION_ROLLOUT_AFTER_OPENING max_extra_steps=$MORPION_ROLLOUT_MAX_EXTRA_STEPS action_selector=$MORPION_ROLLOUT_ACTION_SELECTOR_KIND random_seed=$MORPION_ROLLOUT_RANDOM_SEED stop_on_existing_node=$MORPION_ROLLOUT_STOP_ON_EXISTING_NODE"
fi

if [[ "$MORPION_CLUSTER_OPEN_STATUS" == "1" ]]; then
  launch_status_terminal
fi
