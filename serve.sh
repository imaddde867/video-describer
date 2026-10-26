#!/usr/bin/env bash
# Model server lifecycle, one lane per role.
#
#   ./serve.sh start  [vision|audio|all]   start detached, wait until ready
#   ./serve.sh stop   [vision|audio|all]   stop it
#   ./serve.sh status [vision|audio|all]   is it up, and on which port?
#   ./serve.sh fg     vision|audio         run one in the foreground
#
# Both lanes share one GPU, so their memory budgets are sized to fit together:
# about 18 GB for vision and 11 GB for audio on a 32 GB card. The audio lane is
# served in FP8 for that reason; set AUDIO_QUANT="" to serve it in bf16 on a
# card with room to spare.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs

ACTION="${1:-start}"
ROLE="${2:-all}"

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "$ACTION" in
  -h | --help | help)
    usage
    exit 0
    ;;
  start | stop | status | fg) ;;
  *)
    echo "unknown command: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac

case "$ROLE" in
  vision | audio | all) ;;
  *)
    echo "unknown lane: $ROLE (use vision, audio or all)" >&2
    exit 2
    ;;
esac

roles() {
  if [[ "$ROLE" == "all" ]]; then
    printf 'vision audio'
  else
    printf '%s' "$ROLE"
  fi
}

# Per-lane configuration. Everything is overridable from the environment.
role_env() {
  BIND_HOST="${VLM_BIND_HOST:-127.0.0.1}"
  READY_TIMEOUT="${VLM_READY_TIMEOUT:-300}"
  case "$1" in
    vision)
      MODEL_DIR="${VLM_MODEL_DIR:-$HERE/models/Qwen3-VL-8B-Instruct-FP8}"
      SERVED="${VLM_SERVED_NAME:-qwen3-vl-8b}"
      RPORT="${VLM_PORT:-8000}"
      # 0.46 leaves ~2 GB beyond weights and KV cache, which the image encoder
      # needs when a window arrives as 25+ frames at once.
      UTIL="${VLM_GPU_UTIL:-0.46}"
      # 45k holds 36 frames of a portrait 720p window (~33k) plus a full reply.
      MAXLEN="${VLM_MAX_MODEL_LEN:-45056}"
      SEQS="${VLM_MAX_NUM_SEQS:-4}"
      # Frames are the bulk of every prompt, so the KV cache is quantised to
      # double the context that fits beside the audio lane. Set VLM_KV_DTYPE=auto
      # to keep the model's own dtype when memory is not the constraint.
      KV_DTYPE="${VLM_KV_DTYPE:-fp8}"
      MM='{"image": 64, "video": 1}'
      MM_KWARGS="{\"max_pixels\": ${MAX_PIXELS:-1280512}}"
      EXTRA=()
      ;;
    audio)
      MODEL_DIR="${AUDIO_MODEL_DIR:-$HERE/models/Qwen2-Audio-7B-Instruct}"
      SERVED="${AUDIO_SERVED_NAME:-qwen2-audio-7b}"
      RPORT="${AUDIO_PORT:-8001}"
      UTIL="${AUDIO_GPU_UTIL:-0.40}"
      # Qwen2-Audio's position budget is 8192 and its KV cache is ~0.75 MB per
      # token, so the context is kept to what the windows actually need: a 25 s
      # clip is ~625 audio tokens, and 4096 leaves room for longer windows.
      MAXLEN="${AUDIO_MAX_MODEL_LEN:-3072}"
      SEQS="${AUDIO_MAX_NUM_SEQS:-4}"
      KV_DTYPE="${AUDIO_KV_DTYPE:-auto}"
      # Eager execution skips CUDA-graph memory, which the second lane on a
      # 32 GB card cannot spare; audio calls are short either way.
      EAGER="${AUDIO_EAGER:-1}"
      MM='{"audio": 1}'
      MM_KWARGS=""
      if [[ "${AUDIO_QUANT-unset}" == "unset" ]]; then
        EXTRA=(--quantization fp8_per_tensor)
      elif [[ -n "${AUDIO_QUANT}" ]]; then
        EXTRA=(--quantization "${AUDIO_QUANT}")
      else
        EXTRA=()
      fi
      ;;
  esac
  LOG="logs/vllm-$1.log"
  PIDFILE="logs/vllm-$1.pid"
}

endpoint() { printf 'http://%s:%s/v1/models' "$BIND_HOST" "$RPORT"; }

is_up() { curl -sf --max-time 4 "$(endpoint)" >/dev/null 2>&1; }

# The distro nvcc in /usr/bin is CUDA 12.0 here, which cannot compile kernels
# for SM 12.x. DeepGEMM JIT-builds FP8 kernels at startup and aborts with
# "NVCC version must be at least 12.9" if it picks that compiler.
select_cuda() {
  if [[ -n "${CUDA_HOME:-}" ]]; then
    return
  fi
  for candidate in \
    /usr/local/cuda-13.3 /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.9; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      export PATH="$candidate/bin:$PATH"
      return
    fi
  done
}

build_args() {
  ARGS=(
    --model "$MODEL_DIR"
    --served-model-name "$SERVED"
    --host "$BIND_HOST"
    --port "$RPORT"
    --gpu-memory-utilization "$UTIL"
    --max-model-len "$MAXLEN"
    --max-num-seqs "$SEQS"
    --kv-cache-dtype "$KV_DTYPE"
    --limit-mm-per-prompt "$MM"
    --trust-remote-code
  )
  if [[ -n "$MM_KWARGS" ]]; then
    ARGS+=(--mm-processor-kwargs "$MM_KWARGS")
  fi
  if [[ "${VLM_DISABLE_DEEPGEMM:-0}" == "1" ]]; then
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_MOE_USE_DEEP_GEMM=0
  fi
  if [[ "${EAGER:-0}" == "1" ]]; then
    ARGS+=(--enforce-eager)
  fi
  ARGS+=("${EXTRA[@]}")
}

start_role() {
  role_env "$1"
  if is_up; then
    echo "$1: already up at $(endpoint)"
    return 0
  fi
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "error: $1 has no model at $MODEL_DIR (run ./install.sh)" >&2
    return 1
  fi
  select_cuda
  build_args

  echo "$1: $MODEL_DIR"
  echo "$1: serving $SERVED on $(endpoint) (gpu_util=$UTIL, max_model_len=$MAXLEN)"
  setsid .venv/bin/vllm serve "${ARGS[@]}" >"$LOG" 2>&1 &
  local pid=$!
  echo "$pid" >"$PIDFILE"
  echo "$1: started pid $pid, waiting up to ${READY_TIMEOUT}s for readiness..."

  for _ in $(seq 1 "$READY_TIMEOUT"); do
    if is_up; then
      echo "$1: ready at $(endpoint)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "error: $1 server exited during startup. Last log lines:" >&2
      tail -n 20 "$LOG" >&2
      rm -f "$PIDFILE"
      return 1
    fi
    sleep 1
  done
  echo "error: $1 not ready after ${READY_TIMEOUT}s; see $LOG" >&2
  rm -f "$PIDFILE"
  return 1
}

stop_role() {
  role_env "$1"
  if [[ ! -f "$PIDFILE" ]]; then
    echo "$1: no pid file at $PIDFILE"
    return 0
  fi
  local pid
  pid="$(cat "$PIDFILE")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$1: process $pid is gone; removing stale pid file"
    rm -f "$PIDFILE"
    return 0
  fi
  # Started with setsid, so the process group id equals its pid; signalling the
  # group also stops the engine-core workers.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "$1: stopped"
}

status_role() {
  role_env "$1"
  if is_up; then
    echo "$1: up at $(endpoint)"
    [[ -f "$PIDFILE" ]] && echo "$1: pid $(cat "$PIDFILE")"
    return 0
  fi
  echo "$1: down (nothing answering at $(endpoint))"
  return 1
}

rc=0
for lane in $(roles); do
  case "$ACTION" in
    start) start_role "$lane" || rc=1 ;;
    stop) stop_role "$lane" || rc=1 ;;
    status) status_role "$lane" || rc=1 ;;
    fg)
      role_env "$lane"
      select_cuda
      build_args
      echo "$lane: $SERVED on $(endpoint)"
      exec .venv/bin/vllm serve "${ARGS[@]}"
      ;;
  esac
done
exit "$rc"
