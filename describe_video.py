#!/usr/bin/env python3
"""Describe videos exactly, using a local vision-language model.

The pipeline mirrors NVIDIA's long-video-summarization pattern: split the
runtime into windows, caption each window densely from sampled frames, then
aggregate the captions into one report for the whole video.

Usage:
    ./describe_video.py clip.mp4
    ./describe_video.py recordings/*.mp4 --fps 2 --focus "safety equipment"
    ./describe_video.py --out-dir results/init-9 recordings/*.mp4
    ./describe_video.py --reindex results/init-9

Every run indexes whatever is in its output directory, writing RESULTS.md and
run.json from the per-video meta.json files, so a results directory records
what was done whether it was filled by one batch or by many single runs.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

__version__ = "0.3.0"

VIDEO_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
}

CAPTION_SYSTEM = (
    "You are a meticulous video analyst. You describe only what is visibly "
    "present in the frames you are given. You never guess identities, names, "
    "brands, places, or intent, and you never invent detail you cannot see. "
    "When something is ambiguous or illegible, you say so explicitly."
)

CAPTION_PROMPT = """\
These {n} frames are consecutive samples from ONE continuous segment of a video.
They are in chronological order, sampled at {fps:g} frame(s) per second.
The segment starts at {start_ts} in the source video and covers {dur:g} seconds.

EXACT FRAME TIMESTAMPS — frame 1 is the first image supplied, frame 2 the
second, and so on:
{frame_times}

These are the ONLY valid timestamps for this segment. Sample times are exactly
{step:g} second(s) apart, so the frames do NOT show everything that happened in
between. Do not invent intermediate times, and do not assume consecutive frames
are a fraction of a second apart. Anchor every event to one of the timestamps
above.

Describe this segment in exact, literal detail. Use these sections:

## Segment summary
2-4 sentences covering what happens from the first frame to the last.

## Events
A bullet list of the distinct actions and changes you can see, in order.
Format each as `<timestamp> - <what happens>`, using ABSOLUTE timestamps taken
from the exact frame timestamp list above (i.e. within {start_ts} to {end_ts}).
Merge anything that repeats into ONE line describing the cycle and its period
(for example: `0:03.0 - a red indicator alternates between lit and dark once per
second for the rest of the segment`). Do NOT write one line per frame, and do
not repeat an unchanged state from frame to frame. Whole-frame static segments
get a single line saying they are static. Aim for the shortest list that still
covers every distinct change.

On-screen text is NOT an event. Captions, subtitles, and graphics that type
themselves out change on almost every frame; describing each change floods this
list and hides the real action. Mention such text at most once, saying it
updates continuously, and put the actual wording under "On-screen text".

## Visible entities
People (count, posture, clothing colours), animals, objects, vehicles,
machinery, text, signage, products, food. Be specific and literal.

## On-screen text
Transcribe the text, labels, captions and UI elements you can read, verbatim.
Keep it to distinct messages: merge repeated appearances and the fragments of a
sentence that types itself out into one entry, and do not list near-identical
values (successive filenames, counters, timestamps, progress percentages) one
per frame. If there is none, write "none".

## Camera and framing
Static, pan, tilt, zoom, handheld, tracking, cut between shots. Note any
change in framing or lighting during the segment.

## Uncertainty
Anything you cannot determine from these {n} frames alone: what happens in the
gaps between frames, off-screen context, indistinct objects, unreadable text.
"""

AGGREGATE_SYSTEM = (
    "You are a meticulous video analyst. You summarise only what the supplied "
    "captions state. You never add detail that is not in them, and you flag "
    "gaps instead of filling them in."
)

AGGREGATE_PROMPT = """\
Below are captions for consecutive {n} segments of a single video, produced by
a vision model that saw sampled frames from each segment. The video is
{duration}, split into {n} windows of about {chunk:g} seconds.

Write one coherent report describing the whole video exactly. Use these sections:

## Overview
One paragraph: what the video is and what happens over its full length.

## Timeline
A single merged, chronological bullet list covering the entire video, in the
form `<timestamp> - <what happens>`, using absolute timestamps (`M:SS` or
`H:MM:SS`). Merge duplicate, overlapping, and repetitive events into one line
each. Cover the whole runtime. If the video is largely static, a short timeline
is the correct answer — do not pad it to look thorough.

List what happens, not what is written. Do not turn caption or subtitle wording
into timeline entries; put that wording under Notable details instead.

## Recurring elements
People, objects, settings, or actions that appear more than once, and how they
change over the video.

## Notable details
Concrete, checkable specifics: on-screen text, counts, colours, quantities,
transitions, anything a viewer would want quoted. Quote the wording that
matters rather than every variant of it, and group near-identical entries.

Write every section, even when it is short; use "none" or "not determinable"
rather than dropping a heading.

## Gaps and uncertainties
What the captions leave unclear, disagree about, or could not see.

Do not invent anything that is not supported by the captions below.

--- CAPTIONS ---
{captions}
"""

# Captions of text-dense footage run well past 1400 tokens and were being cut
# mid-sentence; the aggregate pass needs room to reach its last sections. Budget
# that goes unused costs nothing, so both limits are deliberately generous.
CAPTION_MAX_TOKENS = 4000
AGGREGATE_BASE_TOKENS = 2000
AGGREGATE_TOKENS_PER_WINDOW = 800
AGGREGATE_MAX_TOKENS = 12000

# A repeating cycle is at most a quarter of the n-grams it produces; ordinary
# prose runs far above that.
DEGENERATE_MIN_WORDS = 40
DEGENERATE_MAX_UNIQUE_RATIO = 0.25

FOCUS_SUFFIX = (
    "\nThe person who requested this analysis cares most about: {focus}\n"
    "Cover that in as much concrete detail as the frames support, inside the "
    "relevant sections above — do not drop the other sections.\n"
)


# --------------------------------------------------------------------------- #
# Process helpers
# --------------------------------------------------------------------------- #


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a command, capturing output as text."""
    return subprocess.run(list(cmd), capture_output=True, text=True)


def require_binaries(*names: str) -> None:
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise RuntimeError(
            f"missing required tool(s): {', '.join(missing)}. "
            "Install ffmpeg (e.g. `sudo apt install ffmpeg`)."
        )


# --------------------------------------------------------------------------- #
# Probing and planning (pure apart from the ffprobe call)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int | None
    height: int | None
    fps: float
    has_audio: bool


def _parse_rate(rate: str | None) -> float:
    """Parse an ffprobe frame-rate like '30000/1001' into a float."""
    if not rate or rate == "0/0":
        return 0.0
    num, _, den = rate.partition("/")
    try:
        n = float(num)
        d = float(den) if den else 1.0
    except ValueError:
        return 0.0
    return n / d if d else 0.0


def probe_video(path: Path) -> VideoInfo:
    """Read duration, geometry, frame rate and audio presence from a video."""
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}:\n{proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream found in {path}")

    duration = 0.0
    for candidate in (data.get("format", {}).get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break
    if duration <= 0:
        raise RuntimeError(f"could not determine the duration of {path}")

    return VideoInfo(
        path=path,
        duration=duration,
        width=video.get("width"),
        height=video.get("height"),
        fps=round(
            _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")), 3
        ),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def plan_windows(
    duration: float,
    chunk_seconds: float,
    fps: float,
    max_frames: int,
) -> tuple[int, float, float]:
    """Split a runtime into equal windows.

    Windows are of equal length so the last one is never a stub, which keeps the
    per-window frame counts (and therefore the model calls) uniform.

    Returns (window_count, seconds_per_window, effective_fps).
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    windows = max(1, math.ceil(duration / chunk_seconds))
    chunk = duration / windows
    effective = min(fps, max_frames / chunk)
    return windows, chunk, effective


def fmt_ts(seconds: float, force_hours: bool = False) -> str:
    """Format seconds as M:SS.s, or H:MM:SS.s when an hour or more is present."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rest = seconds % 60
    if hours or force_hours:
        return f"{hours}:{minutes:02d}:{rest:04.1f}"
    return f"{minutes}:{rest:04.1f}"


def frame_timestamps(start: float, fps: float, count: int) -> list[str]:
    """Absolute timestamps for frames sampled at `fps` from `start`.

    ffmpeg's fps filter emits frames at 0, 1/F, 2/F, ... from the seek point, so
    spacing is 1/F regardless of how many frames survived rounding.
    """
    step = 1.0 / fps if fps > 0 else 0.0
    return [fmt_ts(start + i * step) for i in range(count)]


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #


def extract_frames(
    video: Path,
    start: float,
    duration: float,
    fps: float,
    width: int,
    dest: Path,
) -> list[Path]:
    """Sample `fps` frames per second from [start, start+duration)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps:g},scale={width}:-2"]
    cmd += ["-q:v", "3", "-f", "image2", str(dest / "f_%04d.jpg")]

    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{proc.stderr.strip()}")

    frames = sorted(dest.glob("f_*.jpg"))
    if not frames:
        raise RuntimeError(
            f"no frames extracted from {video.name} at {start:.1f}s "
            f"(fps={fps:g}, duration={duration:.1f}s)"
        )
    return frames


# --------------------------------------------------------------------------- #
# Model client
# --------------------------------------------------------------------------- #


def looks_degenerate(text: str) -> bool:
    """True when a reply collapsed into a repetition loop.

    The model occasionally locks onto a short cycle and repeats it to the token
    limit, which is worth catching because the result reads like a valid answer.
    Comparing distinct word n-grams against the number of n-grams catches that
    cheaply, including a short phrase repeated many times.
    """
    words = text.split()
    if len(words) < DEGENERATE_MIN_WORDS:
        return False
    for size in (4, 8):
        grams = [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]
        if grams and len(set(grams)) / len(grams) < DEGENERATE_MAX_UNIQUE_RATIO:
            return True
    return False


def aggregate_token_budget(windows: int) -> int:
    """Output room for a report, which grows with the number of windows."""
    scaled = AGGREGATE_BASE_TOKENS + AGGREGATE_TOKENS_PER_WINDOW * windows
    return min(AGGREGATE_MAX_TOKENS, scaled)


class Describer:
    """Thin wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(self, client, model: str, attempts: int = 4) -> None:
        self._client = client
        self._model = model
        self._attempts = attempts

    def chat(
        self,
        system: str,
        user: str,
        images: Iterable[Path] | None = None,
        max_tokens: int = CAPTION_MAX_TOKENS,
    ) -> str:
        content: list[dict] = []
        for image in images or ():
            payload = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                }
            )
        content.append({"type": "text", "text": user})

        text, truncated = self._complete(system, content, max_tokens)

        # Greedy decoding can walk into a loop and stay there. Retrying with a
        # mild repetition penalty and a little jitter breaks the tie; more room
        # would only produce a longer loop, so this comes before widening.
        if looks_degenerate(text):
            print(
                "warning: the model collapsed into a repetition loop; retrying once",
                file=sys.stderr,
            )
            text, truncated = self._complete(
                system, content, max_tokens, breaking_loops=True
            )
            if looks_degenerate(text):
                print(
                    "warning: the retry repeated itself too; its output may be "
                    "unusable",
                    file=sys.stderr,
                )

        # A caption that stops on the limit is missing its closing sections.
        # Giving that one call more room costs nothing when it goes unused.
        if truncated:
            print(
                f"warning: the answer stopped at {max_tokens} tokens; retrying "
                "with a larger budget",
                file=sys.stderr,
            )
            widened, still_truncated = self._complete(system, content, max_tokens * 2)
            if still_truncated:
                print(
                    f"warning: still truncated at {max_tokens * 2} tokens; "
                    "sections may be incomplete",
                    file=sys.stderr,
                )
            if looks_degenerate(widened):
                print(
                    "warning: the longer answer repeated itself; keeping the "
                    "shorter one",
                    file=sys.stderr,
                )
            else:
                text = widened

        return text

    def _complete(
        self,
        system: str,
        content: list[dict],
        max_tokens: int,
        breaking_loops: bool = False,
    ) -> tuple[str, bool]:
        """One successful call, retrying transport errors with backoff.

        Returns the reply and whether it stopped on the token limit.
        """
        sampling: dict = {"temperature": 0.2 if breaking_loops else 0.0}
        if breaking_loops:
            sampling["extra_body"] = {"repetition_penalty": 1.05}

        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=max_tokens,
                    **sampling,
                )
                choice = response.choices[0]
                return (
                    (choice.message.content or "").strip(),
                    choice.finish_reason == "length",
                )
            except Exception as error:  # noqa: BLE001 - reported after retries
                last_error = error
                if attempt + 1 < self._attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"model call failed after {self._attempts} attempts: {last_error}"
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class WindowCaption:
    index: int
    start: float
    end: float
    frames: int
    caption: str

    def as_dict(self, force_hours: bool = False) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "start_ts": fmt_ts(self.start, force_hours),
            "end_ts": fmt_ts(self.end, force_hours),
            "frames": self.frames,
            "caption": self.caption,
        }


@dataclass
class Config:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3-vl-8b"
    chunk_seconds: float = 30.0
    fps: float = 1.0
    max_frames: int = 48
    width: int = 896
    workers: int = 3
    focus: str = ""
    summarise: bool = True
    keep_frames: bool = False
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def render_markdown(
    info: VideoInfo,
    captions: Sequence[WindowCaption],
    summary: str,
    cfg: Config,
    chunk: float,
    sampled_fps: float,
) -> str:
    """Render the human-readable report."""
    hours = info.duration >= 3600
    # Frames per window vary by one either side of the nominal count, so report
    # what was actually sent to the model rather than a computed approximation.
    counts = [c.frames for c in captions] or [0]
    frame_note = (
        f"{counts[0]} frames"
        if len(set(counts)) == 1
        else f"{min(counts)}-{max(counts)} frames"
    )
    lines = [
        f"# Video description: {info.path.name}",
        "",
        f"- **Source:** `{info.path}`",
        f"- **Duration:** {fmt_ts(info.duration, hours)}",
        f"- **Resolution:** {info.width}x{info.height} @ {info.fps} fps",
        f"- **Window:** {len(captions)} x {chunk:.1f}s, {sampled_fps:g} fps sampled "
        f"({frame_note} per window)",
        f"- **Model:** `{cfg.model}` via {cfg.base_url}",
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    if cfg.focus:
        lines.append(f"- **Focus:** {cfg.focus}")
    if not info.has_audio:
        lines.append("- **Note:** source has no audio track")
    lines.append("")

    if summary:
        lines += ["## Whole-video report", "", summary, ""]

    lines += ["---", "", "## Per-window captions", ""]
    for caption in captions:
        lines += [
            f"### {fmt_ts(caption.start, hours)} - {fmt_ts(caption.end, hours)} "
            f"({caption.frames} frames)",
            "",
            caption.caption,
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _caption_prompt(
    info: VideoInfo,
    start: float,
    span: float,
    fps: float,
    frame_count: int,
    focus: str,
    force_hours: bool,
) -> str:
    stamps = frame_timestamps(start, fps, frame_count)
    prompt = CAPTION_PROMPT.format(
        n=frame_count,
        fps=fps,
        start_ts=fmt_ts(start, force_hours),
        end_ts=fmt_ts(start + span, force_hours),
        dur=span,
        step=1.0 / fps if fps > 0 else span,
        frame_times=", ".join(f"{i + 1}={ts}" for i, ts in enumerate(stamps)),
    )
    if focus:
        prompt += FOCUS_SUFFIX.format(focus=focus)
    return prompt


def describe_video(path: Path, cfg: Config, describer: Describer) -> Path:
    """Describe one video; returns the output directory."""
    info = probe_video(path)
    hours = info.duration >= 3600
    windows, chunk, sampled_fps = plan_windows(
        info.duration, cfg.chunk_seconds, cfg.fps, cfg.max_frames
    )
    workers = max(1, min(cfg.workers, windows))

    print(f"video      {path.name}")
    print(
        f"duration   {fmt_ts(info.duration, hours)}  "
        f"{info.width}x{info.height} @ {info.fps} fps"
        f"{'' if info.has_audio else '  (no audio track)'}"
    )
    print(
        f"windows    {windows} x {chunk:.1f}s, {sampled_fps:g} fps each "
        f"(~{int(sampled_fps * chunk)} frames)"
    )
    print(f"endpoint   {cfg.base_url}  model={cfg.model}")

    out_dir = cfg.out_dir.expanduser().resolve() / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(tempfile.mkdtemp(prefix="video-describer-"))
    started = time.time()

    try:
        spans = [
            (i * chunk, min(chunk, info.duration - i * chunk)) for i in range(windows)
        ]

        # 1. Extract frames up front: fast, sequential, and keeps memory flat.
        frames_per_window: list[list[Path]] = []
        for index, (start, span) in enumerate(spans):
            frames = extract_frames(
                path, start, span, sampled_fps, cfg.width, frames_root / f"w{index:04d}"
            )
            print(
                f"  window {index + 1}/{windows}  {fmt_ts(start, hours)}  "
                f"{len(frames)} frames",
                flush=True,
            )
            frames_per_window.append(frames)

        # 2. Caption windows in parallel; the server batches them.
        def caption(index: int) -> tuple[int, str]:
            start, span = spans[index]
            frames = frames_per_window[index]
            text = describer.chat(
                CAPTION_SYSTEM,
                _caption_prompt(
                    info, start, span, sampled_fps, len(frames), cfg.focus, hours
                ),
                images=frames,
            )
            print(
                f"  captioned {index + 1}/{windows}  ({fmt_ts(start, hours)})",
                flush=True,
            )
            return index, text

        print(f"captioning with {workers} parallel worker(s)...")
        captured: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, text in pool.map(caption, range(windows)):
                captured[index] = text

        captions = [
            WindowCaption(
                index=i,
                start=spans[i][0],
                end=spans[i][0] + spans[i][1],
                frames=len(frames_per_window[i]),
                caption=captured[i],
            )
            for i in range(windows)
        ]
        caption_seconds = time.time() - started

        # 3. Aggregate. A single window is already a complete description.
        summary = ""
        if cfg.summarise and len(captions) > 1:
            joined = "\n\n".join(
                f"### Window {c.index + 1} "
                f"({fmt_ts(c.start, hours)} - {fmt_ts(c.end, hours)})\n{c.caption}"
                for c in captions
            )
            prompt = AGGREGATE_PROMPT.format(
                n=len(captions),
                duration=fmt_ts(info.duration, hours),
                chunk=chunk,
                captions=joined,
            )
            if cfg.focus:
                prompt += f"\nThe requester cares most about: {cfg.focus}\n"
            print("aggregating into a single report...")
            summary = describer.chat(
                AGGREGATE_SYSTEM,
                prompt,
                max_tokens=aggregate_token_budget(len(captions)),
            )

        elapsed = time.time() - started

        # 4. Persist.
        (out_dir / "chunk_captions.json").write_text(
            json.dumps(
                {
                    "video": str(path),
                    "windows": [c.as_dict(hours) for c in captions],
                },
                indent=2,
            )
            + "\n"
        )
        (out_dir / "description.md").write_text(
            render_markdown(info, captions, summary, cfg, chunk, sampled_fps)
        )
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "video": str(path),
                    "probe": {
                        "duration": round(info.duration, 3),
                        "width": info.width,
                        "height": info.height,
                        "fps": info.fps,
                        "has_audio": info.has_audio,
                    },
                    "config": {
                        "chunk_seconds": cfg.chunk_seconds,
                        "fps": cfg.fps,
                        "max_frames": cfg.max_frames,
                        "width": cfg.width,
                        "workers": cfg.workers,
                        "focus": cfg.focus or None,
                        "summarise": cfg.summarise,
                    },
                    "windows": windows,
                    "frames_per_window": [len(frames) for frames in frames_per_window],
                    "chunk_seconds": round(chunk, 3),
                    "sampled_fps": round(sampled_fps, 3),
                    "width": cfg.width,
                    "model": cfg.model,
                    "base_url": cfg.base_url,
                    "focus": cfg.focus or None,
                    "audio_analyzed": False,
                    "caption_seconds": round(caption_seconds, 1),
                    "total_seconds": round(elapsed, 1),
                },
                indent=2,
            )
            + "\n"
        )

        print(f"\ndone in {elapsed:.0f}s")
        for name in ("description.md", "chunk_captions.json", "meta.json"):
            print(f"  {out_dir / name}")
        return out_dir
    finally:
        if cfg.keep_frames:
            print(f"frames kept in {frames_root}")
        else:
            shutil.rmtree(frames_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Experiment index
# --------------------------------------------------------------------------- #

# Sections the aggregate pass is asked to produce. A missing heading means the
# model dropped part of its own reporting contract, which is worth surfacing
# rather than passing off as silence.
REQUIRED_REPORT_SECTIONS = (
    "Overview",
    "Timeline",
    "Recurring elements",
    "Notable details",
    "Gaps and uncertainties",
)


def check_report(text: str) -> list[str] | None:
    """Aggregate sections that were asked for but are absent from a report.

    Returns None when there is nothing to check: a single-window video, or a run
    with --no-summary, produces no whole-video report.
    """
    if "## Whole-video report" not in text:
        return None
    present = set(re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    return [name for name in REQUIRED_REPORT_SECTIONS if name not in present]


def fmt_span(seconds: float) -> str:
    """Compact duration for an index table: '35 s', '6.4 min', '1.2 h'."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def collect_records(out_dir: Path) -> list[dict]:
    """Read every per-video record under a results directory, sorted by name."""
    records: list[dict] = []
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as error:
            print(f"warning: skipping {meta_path}: {error}", file=sys.stderr)
            continue

        folder = meta_path.parent
        name = folder.name
        probe = meta.get("probe") or {}
        frames = meta.get("frames_per_window") or []
        # Runs recorded before frames_per_window existed only allow a nominal
        # frame count; flag it so the index never passes an estimate as fact.
        estimated = False
        if not frames and meta.get("sampled_fps") and meta.get("chunk_seconds"):
            per_window = meta["sampled_fps"] * meta["chunk_seconds"]
            frames = [round(per_window)] * (meta.get("windows") or 1)
            estimated = True
        width, height = probe.get("width"), probe.get("height")
        report = folder / "description.md"
        report_text = report.read_text() if report.exists() else ""
        captions = folder / "chunk_captions.json"

        records.append(
            {
                "name": name,
                "source": meta.get("video"),
                "duration": probe.get("duration"),
                "resolution": f"{width}x{height}" if width and height else None,
                "source_fps": probe.get("fps"),
                "has_audio": probe.get("has_audio"),
                "audio_analyzed": meta.get("audio_analyzed", False),
                "model": meta.get("model"),
                "base_url": meta.get("base_url"),
                "config": meta.get("config"),
                "windows": meta.get("windows"),
                "frames_per_window": frames or None,
                "frames": sum(frames) or None,
                "frames_estimated": estimated,
                "chunk_seconds": meta.get("chunk_seconds"),
                "sampled_fps": meta.get("sampled_fps"),
                "width": meta.get("width"),
                "caption_seconds": meta.get("caption_seconds"),
                "total_seconds": meta.get("total_seconds"),
                "missing_sections": check_report(report_text) if report_text else None,
                "report": f"{name}/{report.name}" if report.exists() else None,
                "captions": f"{name}/{captions.name}" if captions.exists() else None,
            }
        )
    return records


def shared_config(records: Sequence[dict]) -> dict | None:
    """The one requested configuration behind a directory, if it is uniform."""
    configs = [r["config"] for r in records if r.get("config")]
    if not configs or any(config != configs[0] for config in configs):
        return None
    return configs[0]


def _describe_config(config: dict | None) -> str:
    if not config:
        return (
            "not recorded (the run predates config recording); effective settings "
            "are in each video's meta.json"
        )
    parts = [
        f"{config.get('fps')} fps sampled",
        f"{config.get('chunk_seconds')} s windows",
        f"{config.get('max_frames')} frames/window max",
        f"width {config.get('width')}",
        f"{config.get('workers')} worker(s)",
    ]
    if config.get("focus"):
        parts.append(f"focus: {config['focus']}")
    if config.get("summarise") is False:
        parts.append("no aggregate pass")
    return ", ".join(parts)


def render_index_markdown(out_dir: Path, records: Sequence[dict]) -> str:
    """Render the index of everything in a results directory."""
    footage = sum(record["duration"] or 0.0 for record in records)
    processing = sum(record["total_seconds"] or 0.0 for record in records)
    speed = f" ({footage / processing:.1f}x realtime)" if processing else ""

    models = ", ".join(
        f"`{model}`" for model in sorted({r["model"] for r in records if r["model"]})
    )
    endpoints = ", ".join(sorted({r["base_url"] for r in records if r["base_url"]}))
    model_line = f"{models or 'unknown'}" + (f" via {endpoints}" if endpoints else "")

    lines = [
        f"# Results: {out_dir.name}",
        "",
        f"- **Videos:** {len(records)} ({fmt_span(footage)} of footage)",
        f"- **Processing:** {fmt_span(processing)}{speed}",
        f"- **Model:** {model_line}",
        f"- **Requested settings:** {_describe_config(shared_config(records))}",
        "- **Audio:** not analysed (the model is vision-only)",
        "",
        "| Video | Duration | Windows | Frames | Time | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        name = record["name"]
        label = f"[{name}]({record['report']})" if record["report"] else name
        duration = fmt_ts(record["duration"]) if record["duration"] is not None else "—"
        elapsed = (
            fmt_span(record["total_seconds"])
            if record["total_seconds"] is not None
            else "—"
        )
        missing = record["missing_sections"]
        notes = "missing: " + ", ".join(missing) if missing else "—"
        frames = record["frames"]
        frames_cell = f"~{frames}" if record["frames_estimated"] else (frames or "—")
        lines.append(
            f"| {label} | {duration} | {record['windows'] or '—'} | "
            f"{frames_cell} | {elapsed} | {notes} |"
        )

    lines += [
        "",
        "`Notes` lists sections the aggregate pass was asked for and did not",
        "produce. A `~` before the frame count marks an estimate from the",
        "sampling rate. Audio is never part of these results. Rebuild this file",
        "and `run.json` with",
        f"`./describe_video.py --reindex {os.path.relpath(out_dir)}`.",
        "",
    ]
    return "\n".join(lines)


def write_index(out_dir: Path) -> tuple[Path, Path] | None:
    """Write run.json and RESULTS.md for a results directory.

    Both are rebuilt from the per-video meta.json files on every run, so the
    directory describes itself whether it was filled by one batch or by many
    single runs. Returns (run.json, RESULTS.md), or None when there is nothing
    to index.
    """
    records = collect_records(out_dir)
    if not records:
        return None

    run_path = out_dir / "run.json"
    index_path = out_dir / "RESULTS.md"
    run_path.write_text(
        json.dumps(
            {
                "experiment": out_dir.name,
                "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
                "videos": len(records),
                "footage_seconds": round(sum(r["duration"] or 0.0 for r in records), 1),
                "processing_seconds": round(
                    sum(r["total_seconds"] or 0.0 for r in records), 1
                ),
                "requested": shared_config(records),
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )
    index_path.write_text(render_index_markdown(out_dir, records))
    return run_path, index_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def expand_inputs(inputs: Sequence[str]) -> list[Path]:
    """Resolve files, directories and globs into a de-duplicated video list."""
    found: dict[Path, None] = {}
    for raw in inputs:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            matches = sorted(
                p
                for p in candidate.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
            )
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = sorted(
                Path(p) for p in glob.glob(raw, recursive=True) if Path(p).is_file()
            )
        if not matches:
            raise RuntimeError(f"no video found matching: {raw}")
        for match in matches:
            found[match.resolve()] = None
    return list(found)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="describe_video.py",
        description=(
            "Describe videos exactly, using a local vision-language model "
            "served over an OpenAI-compatible API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ./describe_video.py clip.mp4\n"
            "  ./describe_video.py recordings/ --fps 2\n"
            "  ./describe_video.py clip.mp4 --focus 'safety equipment'\n"
            "  ./describe_video.py --out-dir results/init-9 recordings/*.mp4\n"
            "  ./describe_video.py --reindex results/init-9\n"
        ),
    )
    parser.add_argument(
        "videos",
        nargs="*",
        metavar="VIDEO",
        help="video files, directories, or glob patterns",
    )
    parser.add_argument(
        "--reindex",
        type=Path,
        default=None,
        metavar="DIR",
        help="rebuild DIR/RESULTS.md and DIR/run.json from existing results; "
        "runs no model and needs no video",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="root output directory (default: outputs)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible endpoint (default: $(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLM_API_KEY", "EMPTY"),
        help="API key for the endpoint (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLM_MODEL", "qwen3-vl-8b"),
        help="served model name (default: $(default)s)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="length of each analysis window (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="frames sampled per second of video (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="hard cap on frames per window (default: 48)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=896,
        help="frame width in px; height keeps aspect (default: 896)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="windows analysed in parallel (default: 3)",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="what you care about most, e.g. 'people and signage'",
    )
    parser.add_argument(
        "--no-summary",
        dest="summarise",
        action="store_false",
        help="skip the whole-video aggregation pass",
    )
    parser.add_argument(
        "--keep-frames", action="store_true", help="keep sampled frames for inspection"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip videos that already have a description.md",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.reindex is not None:
        target = args.reindex.expanduser().resolve()
        if not target.is_dir():
            print(f"error: no such results directory: {target}", file=sys.stderr)
            return 1
        index = write_index(target)
        if index is None:
            print(
                f"error: no per-video meta.json found under {target}", file=sys.stderr
            )
            return 1
        print(f"rebuilt {index[1].name} and {index[0].name} in {target}")
        return 0

    if not args.videos:
        print(
            "error: no videos given. Pass files, directories or glob patterns, "
            "or use --reindex DIR.",
            file=sys.stderr,
        )
        return 1

    try:
        require_binaries("ffmpeg", "ffprobe")
        videos = expand_inputs(args.videos)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    stems: dict[str, Path] = {}
    for video in videos:
        if video.stem in stems:
            print(
                f"error: {stems[video.stem].name} and {video.name} share the "
                f"name '{video.stem}' and would overwrite each other. "
                "Rename one, or run them separately with --out-dir.",
                file=sys.stderr,
            )
            return 1
        stems[video.stem] = video

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print(
            "error: the 'openai' package is missing. Run ./install.sh "
            "or `uv pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        chunk_seconds=args.chunk_seconds,
        fps=args.fps,
        max_frames=args.max_frames,
        width=args.width,
        workers=args.workers,
        focus=args.focus.strip(),
        summarise=args.summarise,
        keep_frames=args.keep_frames,
        out_dir=args.out_dir,
    )

    for label, value in (
        ("--chunk-seconds", cfg.chunk_seconds),
        ("--fps", cfg.fps),
        ("--width", cfg.width),
    ):
        if value <= 0:
            print(f"error: {label} must be positive", file=sys.stderr)
            return 1
    if cfg.max_frames < 1:
        print("error: --max-frames must be at least 1", file=sys.stderr)
        return 1

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=900.0)
    describer = Describer(client, cfg.model)

    failures = 0
    for position, video in enumerate(videos, start=1):
        if len(videos) > 1:
            print(f"\n=== [{position}/{len(videos)}] {video.name} ===")
        target = cfg.out_dir.expanduser().resolve() / video.stem / "description.md"
        if args.skip_existing and target.exists():
            print(f"skipping {video.name}: {target} already exists")
            continue
        try:
            describe_video(video, cfg, describer)
        except (RuntimeError, OSError) as error:
            print(f"error: {video.name}: {error}", file=sys.stderr)
            failures += 1

    index = write_index(cfg.out_dir.expanduser().resolve())
    if index is not None:
        print(f"\nindex: {index[1]}")

    if failures:
        print(f"\n{failures} of {len(videos)} video(s) failed", file=sys.stderr)
        return 1
    return 0


def runs_in_venv(venv_dir: str, prefix: str) -> bool:
    """Whether an interpreter reporting this sys.prefix is the given venv."""
    return os.path.realpath(prefix) == os.path.realpath(venv_dir)


def _ensure_venv() -> None:
    """Re-exec under the project virtualenv if started with another interpreter.

    Keeps `./describe_video.py` working without activating anything. The check is
    sys.prefix, never the executable path: `.venv/bin/python` is a symlink to the
    system 3.12 binary, so a plain `python3` picked up from PATH can share its
    real path while carrying none of the venv's packages. Deliberately not run at
    import time, so importing this module has no side effects.
    """
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    venv_python = os.path.join(venv_dir, "bin", "python")
    if not os.path.isfile(venv_python) or runs_in_venv(venv_dir, sys.prefix):
        return
    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_venv()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
