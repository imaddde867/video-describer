#!/usr/bin/env bash
# Run a named experiment: both model lanes plus the pipeline, results recorded.
#
#   ./run_experiment.sh init-9 ~/footage/*.mp4
#   VLM_FPS=3 VLM_NO_AUDIO=1 ./run_experiment.sh silent ~/clips
#
# Tuning comes from the environment and lands in each video's meta.json:
#   VLM_FPS VLM_CHUNK VLM_MAX_FRAMES VLM_WIDTH VLM_WORKERS VLM_FOCUS
#   VLM_SKIP_EXISTING=1  resume a batch
#   VLM_NO_SUMMARY=1     skip the aggregate pass
#   VLM_NO_AUDIO=1       vision only, skip the audio lane
#   VLM_AUTOSTART=0      do not start the models if they are down
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

NAME="$1"
shift

if [[ "$NAME" == */* || "$NAME" == *..* || -z "$NAME" ]]; then
  echo "error: experiment name must be a plain directory name, got: $NAME" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "error: no videos given" >&2
  usage >&2
  exit 2
fi

OUT="results/$NAME"

if [[ "${VLM_AUTOSTART:-1}" == "1" ]]; then
  if ./serve.sh status >/dev/null 2>&1; then
    ./serve.sh status
  else
    echo "models down: starting both lanes"
    ./serve.sh start
  fi
else
  echo "models: not checked (VLM_AUTOSTART=0)"
fi

ARGS=(--out-dir "$OUT")
[[ -n "${VLM_FPS:-}" ]] && ARGS+=(--fps "$VLM_FPS")
[[ -n "${VLM_CHUNK:-}" ]] && ARGS+=(--chunk-seconds "$VLM_CHUNK")
[[ -n "${VLM_MAX_FRAMES:-}" ]] && ARGS+=(--max-frames "$VLM_MAX_FRAMES")
[[ -n "${VLM_WIDTH:-}" ]] && ARGS+=(--width "$VLM_WIDTH")
[[ -n "${VLM_WORKERS:-}" ]] && ARGS+=(--workers "$VLM_WORKERS")
[[ -n "${VLM_FOCUS:-}" ]] && ARGS+=(--focus "$VLM_FOCUS")
[[ "${VLM_SKIP_EXISTING:-0}" == "1" ]] && ARGS+=(--skip-existing)
[[ "${VLM_NO_SUMMARY:-0}" == "1" ]] && ARGS+=(--no-summary)
[[ "${VLM_NO_AUDIO:-0}" == "1" ]] && ARGS+=(--no-audio)

echo "experiment:  $NAME"
echo "results:     $OUT"
echo

rc=0
./describe_video.py "${ARGS[@]}" "$@" || rc=$?

echo
if [[ -f "$OUT/RESULTS.md" ]]; then
  echo "index:       $OUT/RESULTS.md"
  echo "record:      $OUT/run.json"
fi
exit "$rc"
