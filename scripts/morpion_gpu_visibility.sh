#!/usr/bin/env bash
# Shared GPU visibility for the Morpion tmux and GNOME cluster launchers.
# Source this after PYTHON_BIN is configured. ROCm PyTorch also uses torch.cuda.

morpion_resolve_gpu_visibility() {
  local training_request="${MORPION_TRAINING_CUDA_VISIBLE_DEVICES-auto}"
  local reevaluation_request="${MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES-auto}"
  local request

  for request in "$training_request" "$reevaluation_request"; do
    case "$request" in
      auto|cpu|"") ;;
      *)
        if [[ ! "$request" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
          printf "Invalid Morpion CUDA_VISIBLE_DEVICES setting: %s\n" "$request" >&2
          printf "Expected auto, cpu, empty, or GPU IDs (e.g. 0 or 0,1).\n" >&2
          return 2
        fi
        ;;
    esac
  done

  local gpu_count=0
  if [[ "$training_request" == "auto" || "$reevaluation_request" == "auto" ]]; then
    if [[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]]; then
      gpu_count=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || gpu_count=0
    fi
    [[ "$gpu_count" =~ ^[0-9]+$ ]] || gpu_count=0
  fi

  case "$training_request" in
    auto)
      if (( gpu_count >= 1 )); then
        MORPION_TRAINING_CUDA_VISIBLE_DEVICES="0"
      else
        MORPION_TRAINING_CUDA_VISIBLE_DEVICES=""
      fi
      ;;
    cpu) MORPION_TRAINING_CUDA_VISIBLE_DEVICES="" ;;
    *) MORPION_TRAINING_CUDA_VISIBLE_DEVICES="$training_request" ;;
  esac
  case "$reevaluation_request" in
    auto)
      if (( gpu_count >= 2 )); then
        MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES="1"
      elif (( gpu_count == 1 )); then
        MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES="0"
      else
        MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES=""
      fi
      ;;
    cpu) MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES="" ;;
    *) MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES="$reevaluation_request" ;;
  esac
}
