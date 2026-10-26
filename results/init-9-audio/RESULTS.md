# Results: init-9-audio

[Study overview, source links, reproduction notes and limits](README.md) · [Independent GLM comparison](INDEPENDENT_GLM.md)

- **Videos:** 9 (7.0 min of footage)
- **Processing:** 12.0 min (0.6x realtime)
- **Model:** `qwen3-vl-8b` via http://127.0.0.1:8000/v1
- **Requested settings:** 1.5 fps sampled, 25.0 s windows, 36 frames/window max, width 720, 3 worker(s)
- **Audio:** `qwen2-audio-7b` in 9 of 9 videos
- **Transcript:** `small` (faster-whisper) in 7 of 9 videos
- **Independent comparison:** GLM-5.3 Flash completed 2 of 9 separate analyses; seven more attempts stopped at the provider free-usage limit. See [INDEPENDENT_GLM.md](INDEPENDENT_GLM.md) and the `GLM_*.md` attempt-status notes. Do not treat those seven as GLM results.

| Video | Duration | Windows | Frames | Time | Notes |
|---|---|---|---|---|---|
| [20261025_164350000_iOS](20261025_164350000_iOS/description.md) | 0:49.4 | 2 | 72 | 1.2 min | — |
| [20261025_164610000_iOS](20261025_164610000_iOS/description.md) | 0:42.8 | 2 | 64 | 3.1 min | — |
| [20261025_164620000_iOS](20261025_164620000_iOS/description.md) | 0:50.8 | 3 | 75 | 48 s | — |
| [20261025_164935000_iOS](20261025_164935000_iOS/description.md) | 0:50.8 | 3 | 75 | 1.5 min | — |
| [20261025_165244000_iOS](20261025_165244000_iOS/description.md) | 0:54.9 | 3 | 81 | 1.5 min | — |
| [20261025_165416000_iOS](20261025_165416000_iOS/description.md) | 0:39.7 | 2 | 60 | 40 s | — |
| [20261025_165443000_iOS](20261025_165443000_iOS/description.md) | 0:30.6 | 2 | 46 | 1.1 min | — |
| [20261025_165745000_iOS](20261025_165745000_iOS/description.md) | 0:55.4 | 3 | 84 | 54 s | — |
| [20261025_173551000_iOS](20261025_173551000_iOS/description.md) | 0:46.9 | 2 | 70 | 1.3 min | — |

`Notes` lists sections the aggregate pass was asked for and did not
produce. A `~` before the frame count marks an estimate from the
sampling rate. Rebuild this file and `run.json` with
`./describe_video.py --reindex results/init-9-audio`.
