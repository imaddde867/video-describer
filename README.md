# video-describer

Exact, timestamped descriptions of video files, produced by vision **and audio**
models running locally on a single GPU. No cloud API, no account, no API key.

[![CI](https://github.com/imaddde867/video-describer/actions/workflows/ci.yml/badge.svg)](https://github.com/imaddde867/video-describer/actions/workflows/ci.yml)

Give it recorded footage and it returns, per video:

- a whole-video overview and a merged timeline with absolute timestamps,
- a dense caption for every window, covering events, entities, on-screen text
  and camera movement,
- an analysis of what the same window *sounds* like — what is said and what it
  means, how it is said, music and tool sounds, and sound whose source is off
  screen,
- a fused report that states where the two readings agree and where they
  diverge, plus explicit gaps in each of them.

## The harness

Four local runners, one results tree:

| Runner | Role |
|---|---|
| [`install.sh`](install.sh) | environment, dependencies, both sets of model weights |
| [`serve.sh`](serve.sh) | the models — `start` / `stop` / `status` / `fg` per lane |
| [`describe_video.py`](describe_video.py) | the pipeline — probe → chunk → caption → listen → fuse → report |
| [`run_experiment.sh`](run_experiment.sh) | a named experiment over a set of videos, recorded under `results/<name>/` |

Every run indexes the directory it writes to, producing `RESULTS.md` and
`run.json` from the per-video `meta.json` files. So a results tree states what
was analysed, with which settings, how long each video took, and whether any
report came back incomplete — whether it was filled by one batch or by many
single runs. Curated experiments are committed under [`results/`](results/);
ad-hoc runs land in the gitignored `outputs/`.

## Why this exists

NVIDIA's [Video Search and Summarization blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
solves this problem for datacentre hardware. Its models do not fit a 32 GB
consumer card — the default VLM needs more than 56 GB even in FP8 — and its
full stack is ~18 containers sized for 128 GB of RAM. This tool keeps the part
that matters, NVIDIA's chunk → dense-caption → aggregate pattern, and runs it
against models that fit. Background and sources:
[docs/blueprint-notes.md](docs/blueprint-notes.md).

## Why two lanes

A single small omni model can take frames and audio together, but on a 32 GB
card it has to give up either temporal resolution or vision quality to do so.
Two specialised lanes cost the same GPU and keep both: the vision model samples
~36 frames per window, the audio model hears the whole window at 16 kHz, and
neither is told what the other reported. The fusion pass then compares them,
which is what surfaces the useful contradictions — narration describing
something the frames never show, or an on-screen claim the audio undermines.

## Requirements

- An NVIDIA GPU with ~28 GB free for both lanes together (developed on an
  RTX 5090, 32 GB). The vision lane takes ~18 GB and the audio lane ~11 GB.
  Blackwell / sm_120 works; see [Troubleshooting](#troubleshooting).
- `ffmpeg` and `ffprobe` on `PATH`.
- [uv](https://docs.astral.sh/uv/), which supplies a supported Python 3.12
  interpreter (vLLM does not support 3.14).
- ~30 GB of disk for the weights.

## Install

```bash
./install.sh          # or ./install.sh --dev, to also get pytest and ruff
```

This creates `.venv`, installs vLLM and the OpenAI client, and downloads
`Qwen/Qwen3-VL-8B-Instruct-FP8` (~10 GB, vision) and
`Qwen/Qwen2-Audio-7B-Instruct` (~17 GB, audio) into `models/`.

## Usage

Start both lanes once, then describe as many videos as you like:

```bash
./serve.sh start          # detached; waits until both endpoints answer
./describe_video.py clip.mp4
less outputs/clip/description.md

./serve.sh status
./serve.sh stop
```

`./serve.sh fg vision` runs one lane in the foreground. Lanes can be started
individually — `./serve.sh start audio` — which is useful when only one is being
worked on. Multiple videos, directories and globs are accepted, processed in
sequence:

```bash
./describe_video.py footage/*.mp4
./describe_video.py ~/Videos/dashcam --skip-existing
```

Audio is analysed whenever the source has a soundtrack. `--no-audio` turns the
second lane off, which halves the model calls for silent footage or when only
the picture matters.

### Named experiments

`run_experiment.sh` starts whichever lanes are down, runs the pipeline into
`results/<name>/`, and records the settings in each video's `meta.json`:

```bash
./run_experiment.sh warehouse ~/footage/*.mp4

VLM_FPS=1.5 VLM_CHUNK=25 VLM_MAX_FRAMES=36 VLM_WIDTH=720 VLM_WORKERS=3 \
  ./run_experiment.sh init-9 ~/Downloads/OneDrive_1_10-26-2026
```

`VLM_FPS`, `VLM_CHUNK`, `VLM_MAX_FRAMES`, `VLM_WIDTH`, `VLM_WORKERS` and
`VLM_FOCUS` map to the flags below; `VLM_SKIP_EXISTING=1` resumes a batch,
`VLM_NO_SUMMARY=1` skips the fusion pass, `VLM_NO_AUDIO=1` skips the audio lane,
and `VLM_AUTOSTART=0` leaves the model servers alone.

To rebuild `RESULTS.md` and `run.json` for an existing results directory —
without a GPU, and without touching the reports:

```bash
./describe_video.py --reindex results/init-9
```

### Tuning

| Flag | Default | Purpose |
|---|---|---|
| `--fps` | `1.0` | Frames sampled per second. **The main accuracy/speed knob.** |
| `--chunk-seconds` | `30` | Analysis window. Longer means fewer, larger model calls; the audio lane splits on the same windows. |
| `--max-frames` | `48` | Frame cap per window; lowers `--fps` automatically. |
| `--width` | `896` | Frame width. Raise to 1200–1400 to read small on-screen text. |
| `--workers` | `3` | Windows analysed in parallel, in each lane. |
| `--focus` | — | What you care about, e.g. `"safety equipment"`. |
| `--no-audio` | off | Skip the audio lane. |
| `--skip-existing` | off | Resume a batch without redoing finished videos. |
| `--no-summary` | off | Skip the fusion pass; keep the per-window readings only. |
| `--keep-frames` | off | Keep sampled frames and audio clips, for checking a claim against pixels. |

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

`outputs/<video-name>/` (or `results/<experiment>/<video-name>/`) contains:

| File | Contents |
|---|---|
| `description.md` | The report: overview, timeline, spoken content, voice and sound, where sound and picture meet, recurring elements, notable details, gaps — then every window's picture and sound readings. |
| `chunk_captions.json` | Each window's vision caption and audio analysis with their exact source time range. |
| `meta.json` | Probe results, requested and effective sampling parameters, timings for each lane. |

The directory root gets two generated files, `RESULTS.md` and `run.json`, with
the same content in human and machine form. They are rewritten on every run.

## How it works

1. `ffprobe` reads duration, geometry, frame rate and whether there is audio.
2. The runtime is split into equal windows, so the last one is never a stub.
3. `ffmpeg` samples frames from each window at the requested rate, and cuts the
   same window's audio to a mono 16 kHz WAV.
4. Each window's frames, in chronological order, go to the vision model with an
   explicit frame→timestamp list, producing a dense caption anchored to those
   timestamps.
5. The window's audio goes to the audio model, which is asked what is heard and
   what it means — not for a transcript.
6. The two readings are fused into one report that keeps them apart where they
   disagree.

Windows are processed concurrently within each lane; vLLM batches them on the
GPU. Replies are checked before use: a reading that stops on the token limit is
retried once with double the budget, and one that collapses into a repetition
loop is retried with a mild repetition penalty. Reports missing a section they
were asked for are flagged in `RESULTS.md` rather than passed off as complete.

## Performance

Measured on an RTX 5090 with both lanes resident. The initial experiment in
[`results/init-9`](results/init-9) describes nine portrait phone clips (7.0 min
of footage, 720×1280 @ 30 fps) at 1.5 fps and 25 s windows. Numbers and per-clip
timings are in its [`RESULTS.md`](results/init-9/RESULTS.md).

Server start-up is a few minutes on the first run (FP8 kernels are compiled for
both lanes) and much faster afterwards. Cost tracks generated tokens rather than
sampled frames, and the audio lane adds one call per window that is cheap by
comparison — a 25 s window is a handful of audio tokens.

## Limitations

- **Audio understanding is interpretation.** The audio model hears speech, tone,
  music and events, but it is a 7B model: treat descriptions of emotion, intent
  or "who is speaking" as inference, not measurement.
- **No word-level alignment.** Sound is analysed per window, not per lip
  movement; quoting a sentence does not give you its exact onset time.
- **Sampling is sparse.** At 1 fps, 29 of every 30 frames are never seen, so a
  brief visual event can be missed. Raise `--fps` for fast action. The audio
  lane hears every second, so an event the eye misses may still appear.
- **Timestamps drift in fast cuts.** Anchoring is per sampled frame, but a busy
  stretch can be off by 1–3 s at 1.5 fps. Raise `--fps` and shorten
  `--chunk-seconds` together to stay inside the frame cap for tighter timing.
- **The vision model can be overconfident.** It often reports no uncertainty.
  Treat that section as a hint and spot-check with `--keep-frames`.
- **It may anticipate an ending.** An action can be described as complete when
  the last sampled frame only shows it starting.
- **Text reading degrades** on small or low-contrast on-screen text at the
  default `--width`.
- **A section can go missing.** The fusion pass occasionally drops one of its
  headings, usually when generated text ran long; `RESULTS.md` flags it.
- **One GPU, one video at a time.** Run batches sequentially.

## Troubleshooting

**`Engine core initialization failed` / `NVCC version must be at least 12.9`.**
The FP8 path compiles kernels at start-up, and an older `nvcc` on `PATH` cannot
target sm_120. `serve.sh` selects `/usr/local/cuda-13.x` automatically when
`CUDA_HOME` is unset. If it still fails:

```bash
VLM_DISABLE_DEEPGEMM=1 ./serve.sh start
```

**Out of memory when both lanes are up.** They are sized for 28 GB of a 32 GB
card. Give them less if something else holds VRAM:

```bash
VLM_GPU_UTIL=0.5 AUDIO_GPU_UTIL=0.3 ./serve.sh start
```

or run one lane at a time (`./serve.sh start audio`) and use `--no-audio` on the
other. On a card with room to spare, `AUDIO_QUANT= ./serve.sh start audio` serves
the audio lane in bf16 instead of FP8.

**The server comes up on an unexpected port.** Some shells export `PORT`, which
collides with common defaults. The lanes use `VLM_PORT` and `AUDIO_PORT`:

```bash
VLM_PORT=8000 AUDIO_PORT=8001 ./serve.sh start
```

**`error: audio endpoint ... is not answering`.** The pipeline checks the audio
lane before starting work. `./serve.sh status` says which lane is down.

**`Connection refused`.** Check both endpoints directly:

```bash
./serve.sh status
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8001/v1/models
```

**`error: the 'openai' package is missing`.** The script runs itself under
`.venv` when another interpreter starts it, so this means `.venv` is absent or
incomplete — run `./install.sh`.

**`warning: the model ran out of tokens`.** That reading was cut. Lower `--fps`
or `--chunk-seconds` so each call carries less, or raise `CAPTION_MAX_TOKENS` in
`describe_video.py`.

**Use a different model.** Any OpenAI-compatible endpoint works:

```bash
VLM_MODEL_DIR=/path/to/other-vlm ./serve.sh start vision
./describe_video.py clip.mp4 --model other-vlm --no-audio
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
timestamp formatting, probing, frame and audio extraction, report rendering,
prompt assembly for both lanes, the model client's truncation and repetition
handling, input expansion, the results index and CLI validation.

## Layout

```
describe_video.py         pipeline, two lanes, fusion, index and CLI
serve.sh                  model lifecycle for both lanes
run_experiment.sh         named experiment runner
install.sh                environment and weight setup
tests/                    unit tests, no GPU required
results/init-9/           the initial experiment: reports, index, notes
docs/blueprint-notes.md   why this exists, and how it maps to NVIDIA VSS
outputs/                  scratch results from ad-hoc runs (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
