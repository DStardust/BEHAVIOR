#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${DELTASG_ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Run one OmniGibson process on exactly one physical GPU. DELTASG_GPU accepts
# an explicit index or "auto". Auto mode waits for a candidate with enough
# free VRAM and no existing CUDA workload, then takes a per-GPU advisory lock.
# DELTASG_GPU_SLOTS (default 1) sets how many processes may share one GPU.
GPU_REQUEST="${DELTASG_GPU:-0}"
GPU_CANDIDATES="${DELTASG_GPU_CANDIDATES:-0,1,2}"
MIN_FREE_MIB="${DELTASG_MIN_FREE_MIB:-16000}"
LOCK_TIMEOUT="${DELTASG_GPU_LOCK_TIMEOUT:-21600}"
EXTERNAL_WAIT_TIMEOUT="${DELTASG_GPU_EXTERNAL_WAIT_TIMEOUT:-21600}"
POLL_INTERVAL="${DELTASG_GPU_POLL_INTERVAL:-5}"
CHILD_TIMEOUT="${DELTASG_CHILD_TIMEOUT:-0}"
ALLOW_EXTERNAL_GPU_PROCESSES="${DELTASG_ALLOW_EXTERNAL_GPU_PROCESSES:-0}"
GPU_SLOTS="${DELTASG_GPU_SLOTS:-1}"
CUDA_ROOT="${DELTASG_CUDA_HOME:-/usr/local/cuda}"

if ! [[ "$CHILD_TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "[single-gpu] DELTASG_CHILD_TIMEOUT must be a non-negative integer" >&2
  exit 64
fi
if ! [[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[single-gpu] DELTASG_GPU_POLL_INTERVAL must be a positive integer" >&2
  exit 64
fi
if [[ "$ALLOW_EXTERNAL_GPU_PROCESSES" != "0" && "$ALLOW_EXTERNAL_GPU_PROCESSES" != "1" ]]; then
  echo "[single-gpu] DELTASG_ALLOW_EXTERNAL_GPU_PROCESSES must be 0 or 1" >&2
  exit 64
fi
if ! [[ "$GPU_SLOTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[single-gpu] DELTASG_GPU_SLOTS must be a positive integer" >&2
  exit 64
fi

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "[single-gpu] invalid CUDA root (missing bin/nvcc): $CUDA_ROOT" >&2
  exit 78
fi

gpu_blockers() {
  local gpu_uuid="$1"
  local rows=()
  while IFS=, read -r raw_uuid raw_pid; do
    local uuid="${raw_uuid// /}"
    local pid="${raw_pid// /}"
    [[ "$uuid" == "$gpu_uuid" && "$pid" =~ ^[0-9]+$ ]] || continue
    local cmd
    cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    rows+=("$pid:$cmd")
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)
  printf '%s\n' "${rows[@]}"
}

gpu_pid_count() {
  local gpu_uuid="$1"
  local count
  count="$(gpu_blockers "$gpu_uuid" | cut -d: -f1 | sort -u | grep -c '^[0-9]\+$' || true)"
  printf '%s\n' "${count:-0}"
}

select_gpu() {
  if [[ "$GPU_REQUEST" != "auto" ]]; then
    printf '%s\n' "$GPU_REQUEST"
    return
  fi
  local best=""
  local best_free=-1
  IFS=',' read -r -a candidates <<< "$GPU_CANDIDATES"
  for index in "${candidates[@]}"; do
    index="${index// /}"
    [[ "$index" =~ ^[0-9]+$ ]] || continue
    local row uuid free
    row="$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits -i "$index" 2>/dev/null || true)"
    uuid="${row%%,*}"
    free="${row##*,}"; free="${free// /}"
    [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_MIB" ]] || continue
    concurrent="$(gpu_pid_count "$uuid")"
    (( concurrent < GPU_SLOTS )) || continue
    if (( free > best_free )); then
      best="$index"
      best_free="$free"
    fi
  done
  printf '%s\n' "$best"
}

deadline=$((SECONDS + EXTERNAL_WAIT_TIMEOUT))
gpu_index=""
while [[ -z "$gpu_index" ]]; do
  gpu_index="$(select_gpu)"
  if [[ -n "$gpu_index" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[single-gpu] no compatible GPU became available" >&2
    exit 75
  fi
  echo "[single-gpu] waiting candidates=$GPU_CANDIDATES min_free=${MIN_FREE_MIB}MiB" >&2
  sleep "$POLL_INTERVAL"
done

# Acquire one of GPU_SLOTS shared slots for this GPU. Each slot is an independent
# flock, so up to DELTASG_GPU_SLOTS DeltaSG processes may share one physical GPU.
LOCK_FILE="${DELTASG_GPU_LOCK:-/tmp/deltasg_omnigibson_gpu${gpu_index}.lock}"
acquired_fd=""
lock_deadline=$((SECONDS + LOCK_TIMEOUT))
while [[ -z "$acquired_fd" ]]; do
  for ((slot = 0; slot < GPU_SLOTS; slot++)); do
    slot_lock="${LOCK_FILE}.${slot}"
    exec {fd}>>"$slot_lock"
    if flock -n "$fd"; then
      acquired_fd=$fd
      echo "[single-gpu] acquired lock=$slot_lock slot=$slot/$GPU_SLOTS command=$*" >&2
      break
    fi
    exec {fd}>&-
  done
  [[ -n "$acquired_fd" ]] && break
  if (( SECONDS >= lock_deadline )); then
    echo "[single-gpu] lock timeout base=$LOCK_FILE slots=$GPU_SLOTS" >&2
    exit 75
  fi
  echo "[single-gpu] waiting lock=$LOCK_FILE slots=$GPU_SLOTS command=$*" >&2
  sleep "$POLL_INTERVAL"
done

# Recheck after taking a slot. Explicit GPU requests wait until fewer than
# GPU_SLOTS concurrent CUDA processes remain on the card; auto selection
# restarts so another free candidate can be chosen. DELTASG_ALLOW_EXTERNAL_GPU_PROCESSES
# bypasses this check entirely (any number of foreign CUDA processes allowed).
while true; do
  if [[ "$ALLOW_EXTERNAL_GPU_PROCESSES" == "1" ]]; then
    echo "[single-gpu] shared mode gpu=$gpu_index; external CUDA processes allowed" >&2
    break
  fi
  gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index" 2>/dev/null | head -n 1 || true)"
  concurrent="$(gpu_pid_count "$gpu_uuid")"
  if (( concurrent < GPU_SLOTS )); then
    break
  fi
  if (( SECONDS >= deadline )); then
    printf '[single-gpu] external Kit wait timeout gpu=%s concurrent=%s slots=%s\n' "$gpu_index" "$concurrent" "$GPU_SLOTS" >&2
    exit 75
  fi
  printf '[single-gpu] waiting gpu=%s concurrent=%s slots=%s\n' "$gpu_index" "$concurrent" "$GPU_SLOTS" >&2
  sleep "$POLL_INTERVAL"
done

echo "[single-gpu] acquired gpu=$gpu_index pid=$$ run_token=${DELTASG_RUN_TOKEN:-none} command=$*" >&2
# SSH sessions can leave a stale forwarded DISPLAY behind. Kit initializes the
# GLFW input plugin even with --no-window and can then block before GPU startup.
# Project batch jobs are headless, so remove display variables in the child only.
if (( CHILD_TIMEOUT > 0 )); then
  exec env -u ALL_PROXY -u all_proxy -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
    CUDA_HOME="$CUDA_ROOT" \
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    timeout --signal=TERM --kill-after=30 "$CHILD_TIMEOUT" "$@"
fi
exec env -u ALL_PROXY -u all_proxy -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
  CUDA_HOME="$CUDA_ROOT" \
  CUDA_VISIBLE_DEVICES="$gpu_index" \
  "$@"
