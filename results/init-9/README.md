# init-9 — initial experiment

The first real run of the harness: nine portrait phone clips of maker and
AI-hardware projects, described to check that the pipeline returns exact,
checkable reports rather than plausible-sounding ones.

## Inputs

Nine clips, 30–55 s each (7.0 min total), all 720×1280 @ 30 fps, mixed HEVC and
H.264. The source footage is not committed — only the reports derived from it —
so `source` paths in each `meta.json` point at the original files on this
machine.

## Settings

| Setting | Value | Why |
|---|---|---|
| `--chunk-seconds` | 25 | One window is one model call; short windows keep a caption focused. |
| `--fps` | 1.5 | These clips cut every ~2 s, so 1 fps would miss transitions. |
| `--max-frames` | 36 | Caps each prompt; the effective rate is 36/25 ≈ 1.44 fps on the longest windows. |
| `--width` | 720 | Native width, no upscale. |
| `--workers` | 3 | Windows captioned concurrently; vLLM batches them. |
| Model | `Qwen3-VL-8B-Instruct-FP8` on vLLM | Fits a 32 GB card with room for the KV cache. |
| `MAX_MODEL_LEN` | 65536 | Portrait frames cost ~1175 visual tokens each, so 32 k would reject a full window. |

These are recorded verbatim in each `meta.json` under `config`, and repeated in
`RESULTS.md`.

## Reproduce

```bash
VLM_FPS=1.5 VLM_CHUNK=25 VLM_MAX_FRAMES=36 VLM_WIDTH=720 VLM_WORKERS=3 \
  ./run_experiment.sh init-9 ~/Downloads/OneDrive_1_10-26-2026
```

`VLM_SKIP_EXISTING=1` resumes a partial run. `./describe_video.py --reindex
results/init-9` rebuilds the index without a GPU.

## Contents

| Path | Contents |
|---|---|
| `<video>/description.md` | The report: whole-video overview, timeline, recurring elements, notable details, gaps, then every window caption. |
| `<video>/chunk_captions.json` | Each window's caption with its exact source time range. |
| `<video>/meta.json` | Probe results, requested and effective sampling parameters, timings. |
| `RESULTS.md` | Generated index of the directory. |
| `run.json` | The same record in machine-readable form. |

## Outcome

22 windows and 9 reports, all complete: every caption and every report contains
each section it was asked for. Wall clock was 11.8 min for 7.0 min of footage
(0.6× realtime), 27 s to 3.0 min per clip.

The run needed the harness's recovery paths: 5 replies stopped on the token
limit and were retried with double the budget (2 of those were still cut inside
a long on-screen-text list), and 5 collapsed into repetition loops and were
retried with a repetition penalty — all 5 recovered. Both behaviours are
reported on stderr as they happen; without them this run would have shipped
captions missing their closing sections.

Content was spot-checked against extracted frames rather than trusted.
`20261025_164935000_iOS` claims a Waveshare product page priced at $29.99 with
the "Standard version" label (correct), an Instagram post showing a 3D render of
the round board (correct), and a red plush toy with glowing eyes beside a white
robot labelled "Mac" (correct).

Two limits apply to every report here:

- **Audio was not analysed.** The model is vision-only and all nine clips are
  narrated, so spoken explanation is absent unless it appears as burned-in text.
- **Timestamps drift in fast cuts.** Anchoring is per sampled frame, but at
  1.5 fps a busy stretch can be off by 1–3 s — in `164935000_iOS` the plush-toy
  action is placed at 0:44.6 and actually plays around 0:47. Identification and
  wording stay reliable; only the timing is soft. Finer timing needs more frames:
  `VLM_CHUNK=12 VLM_FPS=3` keeps 36 frames per call but samples twice as densely,
  at about twice the number of model calls.
