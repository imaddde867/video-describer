# Results: init-9

- **Videos:** 9 (7.0 min of footage)
- **Processing:** 11.8 min (0.6x realtime)
- **Model:** `qwen3-vl-8b` via http://127.0.0.1:8000/v1
- **Requested settings:** 1.5 fps sampled, 25.0 s windows, 36 frames/window max, width 720, 3 worker(s)
- **Audio:** not analysed (the model is vision-only)

| Video | Duration | Windows | Frames | Time | Notes |
|---|---|---|---|---|---|
| [20261025_164350000_iOS](20261025_164350000_iOS/description.md) | 0:49.4 | 2 | 72 | 35 s | — |
| [20261025_164610000_iOS](20261025_164610000_iOS/description.md) | 0:42.8 | 2 | 64 | 1.5 min | — |
| [20261025_164620000_iOS](20261025_164620000_iOS/description.md) | 0:50.8 | 3 | 75 | 28 s | — |
| [20261025_164935000_iOS](20261025_164935000_iOS/description.md) | 0:50.8 | 3 | 75 | 27 s | — |
| [20261025_165244000_iOS](20261025_165244000_iOS/description.md) | 0:54.9 | 3 | 81 | 3.0 min | — |
| [20261025_165416000_iOS](20261025_165416000_iOS/description.md) | 0:39.7 | 2 | 60 | 1.7 min | — |
| [20261025_165443000_iOS](20261025_165443000_iOS/description.md) | 0:30.6 | 2 | 46 | 1.4 min | — |
| [20261025_165745000_iOS](20261025_165745000_iOS/description.md) | 0:55.4 | 3 | 84 | 1.4 min | — |
| [20261025_173551000_iOS](20261025_173551000_iOS/description.md) | 0:46.9 | 2 | 70 | 1.3 min | — |

`Notes` lists sections the aggregate pass was asked for and did not
produce. A `~` before the frame count marks an estimate from the
sampling rate. Audio is never part of these results. Rebuild this file
and `run.json` with
`./describe_video.py --reindex results/init-9`.
