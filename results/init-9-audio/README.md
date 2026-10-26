# Maker-project video study — `init-9-audio`

This package analyzes nine short maker videos (7 min 1 sec total) about AI desk companions, robots, a Mac automation assistant, and a DIY robot vacuum. It is intended as **early-stage project research**, not as validated build documentation: use it to identify candidate architectures, parts, software, and questions to investigate before sourcing or building.

## What is useful here

The multimodal pipeline connects sampled visual events with a separate audio interpretation, then combines those readings into timestamped per-video reports. That makes it practical to locate where a creator names a part or describes a step and compare the narration with what is on screen. Transcript files preserve spoken wording where the local ASR lane completed. The run manifest records models, settings, timing, and report paths.

For developing a similar project, start with the [results index](RESULTS.md), open the matching report, and use its timestamps to inspect the relevant moment in the [source video](videos/). Treat details as leads to verify—not a substitute for schematics, source code, datasheets, a bill of materials, or hands-on tests. Promotional reels may omit wiring, power, firmware, safety details, costs, or failure cases, and claims about capabilities are not performance evidence.

## Inventory

| Video | Topic at a glance | System report |
|---|---|---|
| [20261025_164350000_iOS.MP4](videos/20261025_164350000_iOS.MP4) | Atticus: voice-operated macOS automation and shortcut workflows | [Description](20261025_164350000_iOS/description.md) |
| [20261025_164610000_iOS.MP4](videos/20261025_164610000_iOS.MP4) | Mac AI buddy: 3D-printed body, ESP32-S3 camera, servos, Cursor workflow | [Description](20261025_164610000_iOS/description.md) |
| [20261025_164620000_iOS.MP4](videos/20261025_164620000_iOS.MP4) | Rocky AI desk robot: build-guide files, printing, wiring and open-source code | [Description](20261025_164620000_iOS/description.md) |
| [20261025_164935000_iOS.MP4](videos/20261025_164935000_iOS.MP4) | Deskimon NEO: modular desk toy, round display module and build resources | [Description](20261025_164935000_iOS/description.md) |
| [20261025_165244000_iOS.MP4](videos/20261025_165244000_iOS.MP4) | Interactive AI desk device with animated display and laptop workflow | [Description](20261025_165244000_iOS/description.md) |
| [20261025_165416000_iOS.MP4](videos/20261025_165416000_iOS.MP4) | Mac shoulder companion: OLED, ESP32-S3 camera, 3D-printed enclosure, voice model | [Description](20261025_165416000_iOS/description.md) |
| [20261025_165443000_iOS.MP4](videos/20261025_165443000_iOS.MP4) | StackChan: expressive cube robot and configurable AI interactions | [Description](20261025_165443000_iOS/description.md) |
| [20261025_165745000_iOS.MP4](videos/20261025_165745000_iOS.MP4) | OOMWOO: open-source, modular robot vacuum with LiDAR and ROS 2 | [Description](20261025_165745000_iOS/description.md) |
| [20261025_173551000_iOS.MP4](videos/20261025_173551000_iOS.MP4) | Low-cost desk buddy: microcontroller, IMU, OLED and I²C interaction | [Description](20261025_173551000_iOS/description.md) |

Each report is accompanied by sampled-window captions and metadata; seven videos also have a `transcript.json`. See [the machine-readable run record](run.json) for exact per-video coverage and settings.

## Analysis methods and coverage

- **System:** local `qwen3-vl-8b` sampled-frame vision lane plus `qwen2-audio-7b` audio lane; 1.5 fps requested, 25-second windows, up to 36 frames per window, 720-pixel analysis width, three workers.
- **Coverage:** 9/9 videos received vision and audio analysis. Faster-whisper `small` produced a transcript for 7/9; the first two have audio-model analysis but no transcript output.
- **Compute:** 421.3 seconds of footage; recorded pipeline processing time 720.5 seconds (about 12 minutes; 0.6× footage duration).
- **Independent comparison:** two fresh GLM-5.3 Flash analyses were completed. Attempts for the other seven were stopped by the provider's free-usage limit. The [comparison report](INDEPENDENT_GLM.md) documents what was independently assessed and what could not be compared; the seven `GLM_*.md` files are status notes, not analyses.

## Limits and confidence

Frame sampling can miss quick actions and small on-screen text; audio interpretation and speech recognition can mishear. The pipeline can confidently state unsupported details, and creators may describe capabilities that the clip does not demonstrate. Model summaries are not independently fact-checked. The GLM comparison covers only two clips, and its inputs were reduced-resolution re-encodes, not original-quality attachments. Do not treat either model's output as verified engineering requirements or a complete build recipe.

A stronger next step for any selected concept is to locate its original project repository and design files, verify the exact component part numbers against datasheets, reconstruct wiring/software from those primary sources, then bench-test the behavior and costs.

## Reproducing the pipeline run

The original MP4s are included under [`videos/`](videos/) and tracked using Git LFS. From this directory, the project root is `video-describer/`; install its dependencies and model weights as described in the [project README](../../README.md), then run:

```bash
cd ../..
./run_experiment.sh init-9-audio results/init-9-audio/videos/*.MP4
```

That command regenerates the experiment output. The run record describes the captured invocation; model generation may vary between runs. To rebuild just the human and JSON indexes from existing per-video metadata, without model execution:

```bash
cd ../..
./describe_video.py --reindex results/init-9-audio
```

## Data and redistribution

The source videos were provided by the user, who confirmed permission to publish them with this public repository. Video files are stored as Git LFS objects; the reports and metadata remain ordinary Git files. SHA-256 checksums are in [`SHA256SUMS.txt`](SHA256SUMS.txt); verify them from this directory with `sha256sum -c SHA256SUMS.txt`. The videos total approximately 400.9 MiB; downloading them uses the repository owner's Git LFS bandwidth allowance.
