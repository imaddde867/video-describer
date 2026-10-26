#!/usr/bin/env bash
# Launch the local vision-language model as an OpenAI-compatible endpoint.
#
# Tuned for a single 32 GB Blackwell card (RTX 5090, sm_120). The FP8 weights
# are ~10 GB, leaving room for the KV cache and a realistic frame budget.
#
# Usage:
#   ./serve_vlm.sh            # run in the foreground (Ctrl-C to stop)
#   ./serve_vlm.sh start      # start detached and wait until ready
#   ./serve_vlm.sh stop       # stop the detached server
#   ./serve_vlm.sh status     # is it up, and on which port?
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p logs
LOG="logs/vllm.log"
PIDFILE="logs/vllm.pid"

# Intentionally VLM_-prefixed: this environment exports PORT=0, which would
# silently override a plain $PORT and move the server to a random port.
MODEL_DIR="${VLM_MODEL_DIR:-$HERE/models/Qwen3-VL-8B-Instruct-FP8}"
SERVED_NAME="${VLM_SERVED_NAME:-qwen3-vl-8b}"
PORT="${VLM_PORT:-8000}"
BIND_HOST="${VLM_BIND_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_PIXELS="${MAX_PIXELS:-1280512}"
MAX_NUM_SEQS="${VLM_MAX_NUM_SEQS:-8}"
READY_TIMEOUT="${VLM_READY_TIMEOUT:-300}"

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

endpoint() { printf 'http://%s:%s/v1/models' "$BIND_HOST" "$PORT"; }

is_up() { curl -sf --max-time 4 "$(endpoint)" >/dev/null 2>&1; }

# --------------------------------------------------------------------------- #
# Subcommands that do not need the model
# --------------------------------------------------------------------------- #

case "${1:-fg}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  status)
    if is_up; then
      echo "up: $(endpoint)"
      [[ -f "$PIDFILE" ]] && echo "pid: $(cat "$PIDFILE")"
      exit 0
    fi
    echo "down: nothing answering at $(endpoint)"
    exit 1
    ;;
  stop)
    if [[ ! -f "$PIDFILE" ]]; then
      echo "no pid file at $PIDFILE; nothing recorded to stop" >&2
      exit 1
    fi
    pid="$(cat "$PIDFILE")"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "process $pid is not running; removing stale pid file"
      rm -f "$PIDFILE"
      exit 0
    fi
    # The server is started with setsid, so its process group id equals its
    # pid; signalling the group also stops the engine-core workers.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped"
    exit 0
    ;;
  fg|start) ;;
  *)
    echo "unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac

# --------------------------------------------------------------------------- #
# Everything below needs the weights
# --------------------------------------------------------------------------- #

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  cat >&2 <<EOF
error: no model at $MODEL_DIR

Download it with:
  ./install.sh
or:
  HF_HOME=\$PWD/models .venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct-FP8 \\
    --local-dir $MODEL_DIR
EOF
  exit 1
fi

# The distro nvcc in /usr/bin is CUDA 12.0 here, which cannot compile kernels
# for SM 12.x. DeepGEMM JIT-builds FP8 kernels at startup and aborts with
# "NVCC version must be at least 12.9" if it picks that compiler.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
    /usr/local/cuda-13.3 /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.9; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      export PATH="$candidate/bin:$PATH"
      break
    fi
  done
fi

if [[ "${VLM_DISABLE_DEEPGEMM:-0}" == "1" ]]; then
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
fi

ARGS=(
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$BIND_HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  # Each request carries many sampled frames; the default limit of one media
  # item per prompt would reject them. Videos are passed as sampled images.
  --limit-mm-per-prompt "{\"image\": 64, \"video\": 1}"
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}"
  --trust-remote-code
)

# CUDA graph capture is occasionally fragile on new architectures; EAGER=1
# falls back to eager execution if startup fails there.
if [[ "${EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

echo "model:    $MODEL_DIR"
echo "nvcc:     ${CUDA_HOME:-<from PATH>}/bin/nvcc"
echo "serving:  $SERVED_NAME on $(endpoint)"
echo "context:  max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_MEM_UTIL"

if [[ "${1:-fg}" == "fg" ]]; then
  exec .venv/bin/vllm serve "${ARGS[@]}"
fi

if is_up; then
  echo "already up at $(endpoint); run './serve_vlm.sh stop' first" >&2
  exit 1
fi

setsid .venv/bin/vllm serve "${ARGS[@]}" > "$LOG" 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"
echo "started pid $pid; waiting up to ${READY_TIMEOUT}s for readiness..."

for _ in $(seq 1 "$READY_TIMEOUT"); do
  if is_up; then
    echo "ready: $(endpoint)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "error: server exited during startup. Last log lines:" >&2
    tail -n 25 "$LOG" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 1
done

echo "error: not ready after ${READY_TIMEOUT}s; see $LOG" >&2
exit 1
