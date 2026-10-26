#!/usr/bin/env bash
# One-command setup: virtualenv, dependencies, and model weights.
#
#   ./install.sh          # runtime only
#   ./install.sh --dev    # also pytest and ruff
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
MODEL_DIR="${VLM_MODEL_DIR:-$HERE/models/Qwen3-VL-8B-Instruct-FP8}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: uv is required.

  curl -LsSf https://astral.sh/uv/install.sh | sh

(uv is used because the system Python here is 3.14, which vLLM does not
support; uv can fetch a 3.12 interpreter on demand.)
EOF
  exit 1
fi

for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool is required (e.g. sudo apt install ffmpeg)" >&2
    exit 1
  }
done

if [[ -x .venv/bin/python ]]; then
  echo "==> reusing existing .venv ($(.venv/bin/python -V 2>&1))"
else
  echo "==> creating a python $PYTHON_VERSION environment"
  uv venv --python "$PYTHON_VERSION" .venv
fi

echo "==> dependencies (this pulls a CUDA build of torch, several GB)"
uv pip install --python .venv/bin/python -r requirements.txt

if [[ "${1:-}" == "--dev" ]]; then
  echo "==> development dependencies"
  uv pip install --python .venv/bin/python -r requirements-dev.txt
fi

if [[ -f "$MODEL_DIR/config.json" ]]; then
  echo "==> model already present at $MODEL_DIR"
else
  echo "==> downloading $MODEL_REPO (~10 GB) to $MODEL_DIR"
  mkdir -p models
  HF_HOME="$HERE/models" .venv/bin/hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
fi

cat <<EOF

Setup complete.

  ./serve_vlm.sh start     # start the model server, wait until ready
  ./describe_video.py VIDEO.mp4

EOF
