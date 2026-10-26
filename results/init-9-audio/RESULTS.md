# Results: init-9-audio

- **Videos:** 2 (1.5 min of footage)
- **Processing:** 7.0 min (0.2x realtime)
- **Model:** `qwen3-vl-8b` via http://127.0.0.1:8000/v1
- **Requested settings:** 1.0 fps sampled, 30.0 s windows, 48 frames/window max, width 896, 3 worker(s)
- **Audio:** `qwen2-audio-7b` in 2 of 2 videos

| Video | Duration | Windows | Frames | Time | Notes |
|---|---|---|---|---|---|
| [20261025_164350000_iOS](20261025_164350000_iOS/description.md) | 0:49.4 | 2 | 50 | 2.8 min | — |
| [20261025_164610000_iOS](20261025_164610000_iOS/description.md) | 0:42.8 | 2 | 42 | 4.1 min | missing: Recurring elements, Notable details, Gaps and uncertainties |

`Notes` lists sections the aggregate pass was asked for and did not
produce. A `~` before the frame count marks an estimate from the
sampling rate. Rebuild this file and `run.json` with
`./describe_video.py --reindex results/init-9-audio`.
