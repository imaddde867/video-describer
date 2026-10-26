# video-describer

Exact, timestamped descriptions of video files, produced by a vision-language
model running locally on a single GPU. No cloud API, no account, no API key.

[![CI](https://github.com/imaddde867/video-describer/actions/workflows/ci.yml/badge.svg)](https://github.com/imaddde867/video-describer/actions/workflows/ci.yml)

Give it recorded footage and it returns, per video:

- a whole-video overview and a merged timeline with absolute timestamps,
- a dense caption for every window, covering events, entities, on-screen text
  and camera movement,
- an explicit *uncertainty* section saying what the frames do not show.

## Why this exists

NVIDIA's [Video Search and Summarization blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
solves this problem for datacentre hardware. Its models do not fit a 32 GB
consumer card — the default VLM needs more than 56 GB even in FP8 — and its
full stack is ~18 containers sized for 128 GB of RAM. This tool keeps the part
that matters, NVIDIA's chunk → dense-caption → aggregate pattern, and runs it
against a model that fits. Background and sources:
[docs/blueprint-notes.md](docs/blueprint-notes.md).

## Requirements

- An NVIDIA GPU with ~12 GB of free VRAM (developed on an RTX 5090, 32 GB).
  Blackwell / sm_120 works; see [Troubleshooting](#troubleshooting) for the
  toolchain caveat.
- `ffmpeg` and `ffprobe` on `PATH`.
- [uv](https://docs.astral.sh/uv/), which supplies a supported Python 3.12
  interpreter (vLLM does not support 3.14).

## Install

```bash
./install.sh          # or ./install.sh --dev, to also get pytest and ruff
```

This creates `.venv`, installs vLLM and the OpenAI client, and downloads
`Qwen/Qwen3-VL-8B-Instruct-FP8` (~10 GB) into `models/`.

## Usage

Start the model server once, then describe as many videos as you like:

```bash
./serve_vlm.sh start      # detached; waits until the endpoint answers
./describe_video.py clip.mp4
less outputs/clip/description.md

./serve_vlm.sh status
./serve_vlm.sh stop
```

`./serve_vlm.sh` with no argument runs in the foreground instead.

Multiple videos, directories and globs are accepted, processed in sequence:

```bash
./describe_video.py footage/*.mp4
./describe_video.py ~/Videos/dashcam --skip-existing
```

### Tuning

| Flag | Default | Purpose |
|---|---|---|
| `--fps` | `1.0` | Frames sampled per second. **The main accuracy/speed knob.** |
| `--chunk-seconds` | `30` | Analysis window. Longer means fewer, larger model calls. |
| `--max-frames` | `48` | Frame cap per window; lowers `--fps` automatically. |
| `--width` | `896` | Frame width. Raise to 1200–1400 to read small on-screen text. |
| `--workers` | `3` | Windows analysed in parallel. |
| `--focus` | — | What you care about, e.g. `"safety equipment"`. |
| `--skip-existing` | off | Resume a batch without redoing finished videos. |
| `--no-summary` | off | Skip the aggregation pass; keep only per-window captions. |
| `--keep-frames` | off | Keep sampled JPEGs, for checking a claim against pixels. |

Typical choices:

```bash
# fast action: more frames, shorter windows
./describe_video.py sport.mp4 --fps 3 --chunk-seconds 15 --width 1280

# slides or whiteboards: sharp frames, long windows
./describe_video.py talk.mp4 --fps 0.5 --chunk-seconds 120 --width 1400

# one question in mind
./describe_video.py site.mp4 --focus "how many people wear hard hats" --fps 2
```

## Output

`outputs/<video-name>/` contains:

| File | Contents |
|---|---|
| `description.md` | The report: overview, timeline, recurring elements, notable details, per-window captions. |
| `chunk_captions.json` | Each window's caption with its exact source time range. |
| `meta.json` | Probe results, sampling parameters, timings. |

## How it works

1. `ffprobe` reads duration, geometry and frame rate.
2. The runtime is split into equal windows, so the last one is never a stub.
3. `ffmpeg` samples frames from each window at the requested rate.
4. Each window's frames, in chronological order, go to the VLM with an explicit
   frame→timestamp list, producing a dense caption anchored to those timestamps.
5. The captions are aggregated into one report for the whole video.

Windows are captioned concurrently; vLLM batches them on the GPU.

## Performance

Measured on 1080p clips with an RTX 5090 at the defaults (1 fps, 30 s windows):
a 10 s clip takes about 3 s, a 40 s clip 12–24 s — roughly 3–6× faster than
realtime. Server start-up is about 3 minutes the first time (FP8 kernels are
compiled) and about 20 s afterwards.

## Limitations

- **Audio is ignored.** Qwen3-VL is vision-only; speech and off-screen sound
  contribute nothing. Videos with an audio track are flagged as such.
- **Sampling is sparse.** At 1 fps, 29 of every 30 frames are never seen, so a
  brief event can be missed. Raise `--fps` for fast action.
- **The model can be overconfident.** It often reports no uncertainty. Treat
  that section as a hint and spot-check with `--keep-frames`.
- **It may anticipate an ending.** An action can be described as complete when
  the last sampled frame only shows it starting.
- **Text reading degrades** on small or low-contrast on-screen text at the
  default `--width`.
- **One GPU, one video at a time.** Run batches sequentially.

## Troubleshooting

**`Engine core initialization failed` / `NVCC version must be at least 12.9`.**
The FP8 path compiles kernels at start-up, and an older `nvcc` on `PATH` cannot
target sm_120. `serve_vlm.sh` selects `/usr/local/cuda-13.x` automatically when
`CUDA_HOME` is unset. If it still fails:

```bash
VLM_DISABLE_DEEPGEMM=1 ./serve_vlm.sh    # slower FP8 kernels, no compiler needed
```

**The server comes up on an unexpected port.** Some shells export `PORT`, which
collides with common defaults. This launcher uses `VLM_PORT`:

```bash
VLM_PORT=8001 ./serve_vlm.sh start
```

**`Connection refused` from `describe_video.py`.** Check the server and the
endpoint:

```bash
./serve_vlm.sh status
curl -s http://127.0.0.1:8000/v1/models
```

**Out of memory when raising `--fps` or `--width`.** Lower `--max-frames`, or
restart with a smaller context: `MAX_MODEL_LEN=16384 ./serve_vlm.sh start`.

**Use a different model.** Any OpenAI-compatible VLM endpoint works:

```bash
VLM_MODEL_DIR=/path/to/other-vlm ./serve_vlm.sh start
./describe_video.py clip.mp4 --model other-vlm
```

## Development

```bash
./install.sh --dev
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Tests generate their own media with `ffmpeg` and never call a model, so they run
without a GPU and are what CI executes. `tests/` covers window planning,
timestamp formatting, probing, frame extraction, report rendering and CLI
validation.

## Layout

```
describe_video.py         pipeline and CLI
serve_vlm.sh              model server lifecycle (start/stop/status/fg)
install.sh                environment and weight setup
tests/                    unit tests, no GPU required
docs/blueprint-notes.md   why this exists, and how it maps to NVIDIA VSS
```

## License

MIT — see [LICENSE](LICENSE).
