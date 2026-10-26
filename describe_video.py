#!/usr/bin/env python3
"""Describe videos exactly, using local vision and audio models.

The pipeline mirrors NVIDIA's long-video-summarization pattern: split the
runtime into windows, caption each window densely from sampled frames, then
aggregate the captions into one report for the whole video.

Audio takes a second lane. Each window's soundtrack goes to a model that can
hear while the frames go to one that can see, and neither is told what the other
reported; the fusion pass then states where the two agree and where they
diverge. Keeping the lanes separate means each modality is handled by a model
that specialises in it, instead of one model splitting its capacity.

A third lane transcribes the spoken words. faster-whisper runs in-process on
each window's audio and records every utterance with the time it starts, so
what was literally said — names, numbers, prices — reaches the aggregate pass
verbatim instead of through an audio model's paraphrase, and the report
carries it as plain text.

A video with more windows than roughly a couple of dozen cannot feed its
readings to one pass whole: the middle of the video gets squeezed out. Those
readings are condensed first — consecutive batches become digests that keep
the picture and sound accounts separate — and the report reads the digests
instead. Whichever route the readings take, the finished report is stitched
section by section in reading order, and a section any pass failed to write
is recovered by one extra completion pass rather than missing silently.

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
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

__version__ = "0.5.0"

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

AUDIO_SYSTEM = (
    "You are a meticulous audio analyst. You describe only what can be heard in "
    "the clip you are given, as understanding rather than transcription. You "
    "never guess identities, names, brands, or intent, and you never invent "
    "words or sounds you cannot make out. When something is unclear or "
    "inaudible, you say so explicitly."
)

AUDIO_PROMPT = """\
This is {span:g} seconds of audio, from {start_ts} to {end_ts} of a video.

Describe what can be heard. Use these sections:

## Audio summary
3-5 sentences on what a listener hears from beginning to end: who speaks, what
the speech is about, and what else carries the segment. Judge only what you
hear, not what a video like this usually contains.

## Speech
First say what is being said and what it means, in your own words: the claim,
argument, joke or instruction. Then quote at most two short phrases whose exact
wording matters (a name, a number, a price, a product). Do NOT transcribe the
clip, and do not guess what a product is beyond how it is described. If nobody
speaks, write "no speech".

## Delivery and tone
Who is speaking and how: one person or several, conversational or
presentational, pace, emphasis, hesitation, excitement, deadpan, laughter,
audible cuts between takes. Note any change during the segment. Say so plainly
if the delivery is too flat or muffled to judge.

## Music and effects
Music presence, style and any change in it; sound effects; device, tool and
material sounds; ambient noise. Name them concretely (solder sizzle, keyboard
clicks, motor whirr, cardboard handling, room tone). State whether the only
sound is room tone, and write "silence" only if there is nothing at all.

## Sound off screen
Sounds whose source is not in a typical frame: speech from another room, pets,
traffic, notifications, tools being handled out of view. A viewer watching only
the picture would not know these happened, so they matter.

## Uncertainty
Anything you cannot make out: muffled or overlapping speech, unclear words,
sounds you cannot identify, audio too quiet to judge.
"""

AGGREGATE_SYSTEM = (
    "You are a meticulous video analyst. You summarise only what the supplied "
    "readings state. You never add detail that is not in them, you reconcile "
    "what is seen against what is heard, and you flag gaps instead of filling "
    "them in."
)

# The aggregate pass is assembled from these parts so the sound sections appear
# only when audio was actually analysed.
AGGREGATE_SECTIONS_OPENING = (
    (
        "Overview",
        "One paragraph: what the video is, what happens over its full length, and "
        "what it sounds like.",
    ),
    (
        "Timeline",
        "A single merged, chronological bullet list covering the entire video, in "
        "the form `<timestamp> - <what happens>`, using absolute timestamps "
        "(`M:SS` or `H:MM:SS`). Merge duplicate, overlapping, and repetitive "
        "events into one line each. Cover the whole runtime, but not at a fixed "
        "interval: an entry belongs where something changes. A clip under a "
        "minute needs about ten to twenty entries. If the video is largely "
        "static, a short timeline is the correct answer — do not pad it to look "
        "thorough.\n\n"
        "List what happens, not what is written. Do not turn caption or subtitle "
        "wording into timeline entries; put that wording under Notable details "
        'instead. Wrong: `0:05 - text changes to "and."`. Right: one line for '
        "the event those frames belong to — a walkthrough of on-screen text is "
        "not a timeline.",
    ),
)

AGGREGATE_SECTIONS_SOUND = (
    (
        "Spoken content",
        "What is actually being said and what it means — the claim, argument, "
        "joke or instruction, in your own words. Quote wording only where it "
        "carries the point (a name, number, price, product). Never transcribe "
        'the clip. Say who speaks and when they start or stop; use "no speech" '
        "if nobody does.",
    ),
    (
        "Voice and sound",
        "How it is said and what else is heard: delivery and tone, changes "
        "between takes, music and its role, tool and material sounds, ambience. "
        "Name concrete sounds, and note any whose source is never visible.",
    ),
    (
        "Sound and picture together",
        "What the two readings agree on, then where they genuinely diverge: "
        "narration that describes something the frames do not show, a sound that "
        "explains an otherwise unclear action, on-screen text that contradicts "
        "what is said, a moment where the sound carries the meaning. Group "
        "similar cases into one point and give at most five. A sound reading only "
        "covers what was heard, so it not mentioning a detail is narrower "
        "coverage, not a disagreement — never list items as divergences merely "
        "because the sound reading omits them.",
    ),
)

AGGREGATE_SECTIONS_CLOSING = (
    (
        "Recurring elements",
        "People, objects, settings, or actions that appear more than once, and "
        "how they change over the video. At most eight entries.",
    ),
    (
        "Notable details",
        "Concrete, checkable specifics: on-screen text, counts, colours, "
        "quantities, prices, names, transitions, anything a viewer would want "
        "quoted. Quote the wording that matters rather than every variant of it, "
        "group near-identical entries, and give at most ten.",
    ),
    (
        "Gaps and uncertainties",
        "At most six entries, and only what the readings themselves could not "
        "resolve: an unclear or overlapping word, a sound you cannot identify, a "
        "detail the two readings disagree about. Never phrase an entry as an "
        'absence — not "no X", "X is not shown", "X is not referenced", "no '
        'confirmation of X". A topic the video simply does not cover is not a gap.',
    ),
)

AGGREGATE_INTRO_SOUND = """\
Below are two independent readings of the same {n} consecutive segments of one
video ({duration} long, windows of about {chunk:g} seconds): CAPTIONS from a
vision model that saw sampled frames and could not hear, and AUDIO from an
audio model that heard each window and could not see. Neither model knew what
the other reported. Both readings cover the whole video.
"""

AGGREGATE_INTRO_VISION = """\
Below are captions for consecutive {n} segments of a single video, produced by
a vision model that saw sampled frames from each segment. The video is
{duration}, split into {n} windows of about {chunk:g} seconds.
"""

AGGREGATE_RULES = """\
Write every section, even when it is short; use "none" or "not determinable"
rather than dropping a heading. Make each point once: a point belongs to one
section, and restating it elsewhere in the report is wrong.

Never phrase an entry as an absence. Reporting that a reading does not mention
something adds nothing — an unmentioned topic is not a finding — and listing
those omissions is how this report has gone wrong before.

Do not invent anything that is not supported by the readings below.
"""

AGGREGATE_TRANSCRIPT_NOTICE = """\
A TRANSCRIPT block follows: the video's spoken words as an automatic system
heard them, one utterance per line with the time it starts. Use it for exact
wording — names, numbers, prices — and for when speech starts; the audio
reading paraphrases what the words mean, and is bound by what its own system
can make out.\n"""

# Long-video digest: the prompt condensing a batch of window readings into a
# digest the later passes can read whole. The digest keeps the two accounts
# separate, so the fusion pass can still weigh one against the other.
DIGEST_INTRO = """\
Below are two readings of {n} consecutive windows of one video ({duration}
long, windows of about {chunk:g} seconds): CAPTIONS from a vision model that
saw sampled frames and could not hear, and AUDIO from an audio model that
heard each window and could not see. Neither model knew what the other
reported. You are condensing one span of {span:g} seconds into one digest.\n"""

DIGEST_PREAMBLE = (
    "You are condensing part of one video. The final report reads your digest, "
    "never the windows themselves, so every observation has to reach it."
)

DIGEST_SECTIONS = (
    (
        "Picture",
        "A chronological bullet list `<timestamp> - <what happens>` covering the "
        "whole span with the windows' own ABSOLUTE timestamps — they are the "
        "only anchor the final report will have. Merge repeated events into one "
        "line with its period. After the list, one short paragraph on what the "
        "frames are mostly made of: setting and entities across the span, camera "
        "changes, anything dense in on-screen text. Quote the on-screen wording "
        "that matters verbatim instead of paraphrasing it.",
    ),
    (
        "Sound",
        "What is said, with when it starts when the transcript makes it knowable, "
        "and the arguments' conclusions kept separate from the exact wording; "
        "delivery changes; music and foreground sounds with their spans. Quote "
        "the names, numbers and prices the transcript carries verbatim.",
    ),
    (
        "Divergence",
        "At most three points where the sound reading and picture reading "
        "genuinely disagree, or where one explains the other. The sound reading "
        "covering less than the picture is narrower coverage, not a "
        'disagreement. Write "none" when there is nothing worth noting.',
    ),
)

DIGEST_RULES = """\
Timestamps stay absolute throughout the digest. Keep the distinct wording of
quoted speech and on-screen text; a digest that paraphrases a name or price is
losing exactly what the final report needs from it.
"""

# The report is written in three bounded calls instead of one. Asked for all
# eight sections at once from two long readings, the model padded, ran past its
# budget and looped on its last line; each pass below finishes on its own, and
# the fusion section is prose because a list of near-identical comparisons is
# what invited the loop. Budgets come from measuring that behaviour: the opening
# pass stops itself well inside its room, and the closing pass stops at about
# half of it.
AGGREGATE_OPENING_BASE_TOKENS = 1200
AGGREGATE_OPENING_TOKENS_PER_WINDOW = 500
AGGREGATE_FUSION_TOKENS = 900
AGGREGATE_CLOSING_BASE_TOKENS = 800
AGGREGATE_CLOSING_TOKENS_PER_WINDOW = 200
AGGREGATE_MAX_TOKENS = 14000

AGGREGATE_FUSION_PREAMBLE = (
    "You are writing the fusion part of one report on the whole video, from the "
    "same two readings. Write it as two or three short paragraphs of prose — no "
    "bullet lists — naming at most five points."
)

# The section list is introduced by this line for every pass: without it the
# model answers its preamble instead, echoing a word from it as its heading.
AGGREGATE_SECTIONS_INTRO = (
    "Write exactly these sections, in this order, and nothing else:\n"
)


@dataclass(frozen=True)
class ReportPass:
    """One aggregate call: the sections it writes and the room it gets."""

    name: str
    preamble: str
    sections: tuple[tuple[str, str], ...]
    max_tokens: int


def report_passes(with_audio: bool, windows: int) -> tuple[ReportPass, ...]:
    """The aggregate calls, in the order the report is read."""
    opening = AGGREGATE_SECTIONS_OPENING
    if with_audio:
        opening += AGGREGATE_SECTIONS_SOUND[:2]
    passes = [
        ReportPass(
            "opening",
            "You are writing the first part of one report on the whole video.",
            opening,
            min(
                AGGREGATE_MAX_TOKENS,
                AGGREGATE_OPENING_BASE_TOKENS
                + AGGREGATE_OPENING_TOKENS_PER_WINDOW * windows,
            ),
        )
    ]
    if with_audio:
        passes.append(
            ReportPass(
                "fusion",
                AGGREGATE_FUSION_PREAMBLE,
                AGGREGATE_SECTIONS_SOUND[2:],
                AGGREGATE_FUSION_TOKENS,
            )
        )
    passes.append(
        ReportPass(
            "closing",
            "You are writing the closing part of the same report.",
            AGGREGATE_SECTIONS_CLOSING,
            min(
                AGGREGATE_MAX_TOKENS,
                AGGREGATE_CLOSING_BASE_TOKENS
                + AGGREGATE_CLOSING_TOKENS_PER_WINDOW * windows,
            ),
        )
    )
    return tuple(passes)


def report_sections(with_audio: bool) -> tuple[str, ...]:
    """The sections the aggregate pass is asked for, in reading order."""
    sections = AGGREGATE_SECTIONS_OPENING
    if with_audio:
        sections += AGGREGATE_SECTIONS_SOUND
    sections += AGGREGATE_SECTIONS_CLOSING
    return tuple(name for name, _ in sections)


def section_specs(with_audio: bool) -> dict[str, str]:
    """The instruction each report section carries, by name."""
    sections = AGGREGATE_SECTIONS_OPENING
    if with_audio:
        sections += AGGREGATE_SECTIONS_SOUND
    sections += AGGREGATE_SECTIONS_CLOSING
    return dict(sections)


REPAIR_BASE_TOKENS = 500
REPAIR_TOKENS_PER_SECTION = 400


def repair_pass(sections: tuple[str, ...], with_audio: bool) -> ReportPass | None:
    """The completion pass rewriting the sections earlier passes dropped.

    Asked to write everything at once, the model padded and looped — the
    report is only ever extended after a pass was cut. A repair pass gets a
    small, focused job, so its budget is per section rather than per window.
    """
    if not sections:
        return None
    specs = section_specs(with_audio)
    return ReportPass(
        "repair",
        (
            "You are completing part of one report on the whole video. Passes "
            "before yours were asked for more and failed to write the following "
            "sections; write them now, and nothing else."
        ),
        tuple((name, specs[name]) for name in sections),
        min(
            AGGREGATE_MAX_TOKENS,
            REPAIR_BASE_TOKENS + REPAIR_TOKENS_PER_SECTION * len(sections),
        ),
    )


SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$")


def split_pass_output(text: str) -> dict[str, str]:
    """One pass output split into `{section: body}`, first occurrence wins.

    Anything before the first heading — a preamble the model echoed instead of
    a heading — has no section to belong to and is left out.
    """
    found: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = SECTION_HEADING.match(line)
        if match:
            found.append((match.group(1), []))
        elif found:
            found[-1][1].append(line)
    collected: dict[str, str] = {}
    for name, body in found:
        cleaned = "\n".join(body).strip()
        if name not in collected and cleaned:
            collected[name] = cleaned
    return collected


def assemble_report(
    passes: Sequence[ReportPass], outputs: Mapping[str, str]
) -> tuple[str, list[str]]:
    """Stitch pass outputs into one report, in the sections' reading order.

    Each section's body comes from a pass that was asked for it — a heading
    echoed in another pass's output cannot supply a section it does not own —
    and each section stands in its canonical position once, whatever order the
    calls ran in. A section no pass wrote is returned as missing; one a later
    completion pass recovered is not, however early its own writer failed.
    """
    bodies: dict[str, str] = {}
    for step in passes:
        blocks = split_pass_output(outputs.get(step.name, ""))
        for name, _ in step.sections:
            if name in blocks and name not in bodies:
                bodies[name] = blocks[name]
    written: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for step in passes:
        for name, _ in step.sections:
            if name in seen:
                continue
            seen.add(name)
            if name in bodies:
                written.append(f"## {name}\n\n{bodies[name]}")
            else:
                missing.append(name)
    return "\n\n".join(written), missing


# Beyond a couple of dozen windows the readings no longer fit one pass's
# prompt whole, and the middle of the video is what gets squeezed out. Those
# readings are condensed first: consecutive batches become digests that keep
# the picture and sound accounts apart, and the report reads the digests.
AGGREGATE_MAX_RAW_WINDOWS = 24
DIGEST_BATCH_WINDOWS = 12

DIGEST_BASE_TOKENS = 800
DIGEST_TOKENS_PER_WINDOW = 250
DIGEST_MAX_TOKENS = 4000

# Captions of text-dense footage run well past 1400 tokens and were being cut
# mid-sentence; the aggregate pass needs room to reach its last sections. Budget
# that goes unused costs nothing, so both limits are deliberately generous.
CAPTION_MAX_TOKENS = 4000
# The audio model's own context is small, and its KV cache is ~0.75 MB per token,
# so its reply budget is kept well inside it.
AUDIO_MAX_TOKENS = 1400

# A repeating cycle is at most a quarter of the n-grams it produces; ordinary
# prose runs far above that.
DEGENERATE_MIN_WORDS = 40
DEGENERATE_MAX_UNIQUE_RATIO = 0.25
# A loop can also repeat a long block: its many distinct n-grams keep the ratio
# above the threshold while the same whole lines come back over and over. Those
# lines are counted separately, ignoring short ones such as "- none.".
DEGENERATE_MIN_LINE_CHARS = 40
DEGENERATE_MIN_LINE_REPEATS = 3

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


def require_endpoint(base_url: str, what: str) -> None:
    """Fail fast when a model server is not answering, instead of per call."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            response.read(1)
    except OSError as error:
        raise RuntimeError(
            f"{what} endpoint {base_url} is not answering ({error}). "
            "Start the servers with `./serve.sh start`."
        ) from error


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


def extract_audio(video: Path, start: float, duration: float, dest: Path) -> Path:
    """Cut one window's audio to the mono 16 kHz WAV the audio model expects."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-t", f"{duration:.3f}", "-vn"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]

    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr.strip()}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"no audio extracted from {video.name} at {start:.1f}s "
            f"(duration={duration:.1f}s)"
        )
    return dest


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
    for size in (4, 8, 16):
        grams = [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]
        if grams and len(set(grams)) / len(grams) < DEGENERATE_MAX_UNIQUE_RATIO:
            return True
    lines = [
        line.strip().lower()
        for line in text.splitlines()
        if len(line.strip()) >= DEGENERATE_MIN_LINE_CHARS
    ]
    if lines:
        repeats = Counter(lines).most_common(1)[0][1]
        if repeats >= DEGENERATE_MIN_LINE_REPEATS:
            return True
    return False


def trim_repeated_lines(text: str) -> str:
    """Drop consecutive repeats of a long line.

    A locked loop emits the same sentence over and over, which the reader gains
    nothing from: the first copy carries the point and the rest are noise.
    Repeats that are *not* consecutive are left alone, because a report
    legitimately revisits a point under a different heading.
    """
    kept: list[str] = []
    previous = ""
    dropped = 0
    for line in text.splitlines():
        key = line.strip().lower()
        if key and len(key) >= DEGENERATE_MIN_LINE_CHARS and key == previous:
            dropped += 1
            continue
        kept.append(line)
        previous = key
    if dropped:
        print(
            f"warning: dropped {dropped} repeated line(s) from the answer",
            file=sys.stderr,
        )
    return "\n".join(kept).rstrip()


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
        audio: Path | None = None,
        max_tokens: int = CAPTION_MAX_TOKENS,
        loop_resistant: bool = False,
        widen: bool = True,
    ) -> str:
        """One reply, with the loop and truncation recovery described below.

        `loop_resistant` starts the very first attempt with the sampling that
        breaks loops, for prompts that reliably collapse under greedy decoding.
        `widen` retries a cut answer with twice the room; the report passes turn
        it off, because they fill whatever room they are given and the retry only
        produces a longer, padded answer.
        """
        content: list[dict] = []
        for image in images or ():
            payload = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                }
            )
        if audio is not None:
            payload = base64.b64encode(audio.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{payload}"},
                }
            )
        content.append({"type": "text", "text": user})

        text, truncated = self._complete(
            system, content, max_tokens, breaking_loops=loop_resistant
        )

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
        if truncated and widen:
            print(
                f"warning: the answer stopped at {max_tokens} tokens; retrying "
                "with a larger budget",
                file=sys.stderr,
            )
            try:
                widened, still_truncated = self._complete(
                    system, content, max_tokens * 2, breaking_loops=loop_resistant
                )
            except RuntimeError:
                # Two lanes share one GPU, so the larger budget may not fit the
                # context. A cut answer is better than a failed video.
                print(
                    "warning: no room to widen the budget; keeping the cut answer",
                    file=sys.stderr,
                )
            else:
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

        elif truncated:
            print(
                f"warning: the answer stopped at {max_tokens} tokens; keeping it "
                "as it is (this call fills any budget it is given)",
                file=sys.stderr,
            )

        # A loop that survived both retries is not an answer, and shipping it
        # would put thousands of identical lines in the report. The repeated run
        # carries no information, so it is dropped and what remains is re-checked.
        if looks_degenerate(text):
            text = trim_repeated_lines(text)
            if looks_degenerate(text):
                print(
                    "warning: the answer still reads as a loop after trimming; "
                    "treat that part of the report with suspicion",
                    file=sys.stderr,
                )

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
# Transcript (local ASR)
# --------------------------------------------------------------------------- #


ASR_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_asr_model(name: str, device: str):
    """Load and cache one faster-whisper model for the process lifetime.

    Deliberately not at import time: the pipeline must still run with the
    package absent (--no-transcript, or an environment that never installed
    it), and per-call loading would re-read the weights every window.
    """
    from faster_whisper import WhisperModel

    if (name, device) not in ASR_MODEL_CACHE:
        # int8 keeps the small model quick on the CPU where the GPU is busy
        # with the lanes; cuda gets half precision.
        compute = {"cpu": "int8", "cuda": "float16"}.get(device)
        options: dict = {"device": device}
        if compute:
            options["compute_type"] = compute
        ASR_MODEL_CACHE[(name, device)] = WhisperModel(name, **options)
    return ASR_MODEL_CACHE[(name, device)]


def transcribe_window(
    model, audio_path: Path, start: float, language: str = ""
) -> list[dict]:
    """One window's speech as utterances with absolute start and end times.

    Whisper's offsets are relative to the clip it is played, so the window's
    place in the source video is added back. Iterating the result is what
    drives the transcription.
    """
    segments, _info = model.transcribe(
        str(audio_path), language=language or None, vad_filter=True
    )
    speech = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            speech.append(
                {
                    "start": round(start + segment.start, 3),
                    "end": round(start + segment.end, 3),
                    "text": text,
                }
            )
    return speech


def transcript_lines(speech: Sequence[dict], force_hours: bool = False) -> str:
    """Utterances as `M:SS - text` lines; empty input reads as no speech."""
    if not speech:
        return "no speech detected"
    return "\n".join(f"{fmt_ts(s['start'], force_hours)} - {s['text']}" for s in speech)


def segments_in_window(speech: Sequence[dict], start: float, end: float) -> list[dict]:
    """The utterances that begin inside [start, end), with absolute times."""
    return [s for s in speech if start <= s["start"] < end]


def asr_available() -> bool:
    """Whether the optional package the transcript lane needs is installed."""
    from importlib.util import find_spec

    return find_spec("faster_whisper") is not None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class WindowCaption:
    """One window's vision caption, its analysed sound, and its verbatim speech."""

    index: int
    start: float
    end: float
    frames: int
    caption: str
    audio: str | None = None
    speech: list[dict] | None = None

    def as_dict(self, force_hours: bool = False) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "start_ts": fmt_ts(self.start, force_hours),
            "end_ts": fmt_ts(self.end, force_hours),
            "frames": self.frames,
            "caption": self.caption,
            "audio": self.audio,
            "speech": self.speech,
        }


@dataclass
class Config:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3-vl-8b"
    audio: bool = True
    audio_base_url: str = "http://127.0.0.1:8001/v1"
    audio_model: str = "qwen2-audio-7b"
    transcript: bool = True
    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_language: str = ""
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
    analysed = any(c.audio for c in captions)
    audio_line = (
        f"`{cfg.audio_model}` via {cfg.audio_base_url}" if analysed else "not analysed"
    )
    spoken = [segment for c in captions for segment in (c.speech or [])]
    transcript_header = (
        f"- **Transcript:** `{cfg.asr_model}` (faster-whisper, in-process)"
        if spoken
        else ""
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
        f"- **Audio:** {audio_line}",
    ]
    if transcript_header:
        lines.append(transcript_header)
    lines += [
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    if cfg.focus:
        lines.append(f"- **Focus:** {cfg.focus}")
    if not info.has_audio:
        lines.append("- **Note:** source has no audio track")
    lines.append("")

    if summary:
        lines += ["## Whole-video report", "", summary, ""]

    if spoken:
        lines += [
            "## Transcript (verbatim)",
            "",
            "Spoken words transcribed locally with faster-whisper; each line starts "
            "at the time the utterance was heard. Gaps between lines are video, "
            "music or silence, not lost speech.",
            "",
            transcript_lines(spoken, hours),
            "",
        ]

    lines += ["---", "", "## Per-window captions", ""]
    for caption in captions:
        lines += [
            f"### {fmt_ts(caption.start, hours)} - {fmt_ts(caption.end, hours)} "
            f"({caption.frames} frames)",
            "",
        ]
        if caption.audio is None:
            lines += [caption.caption, ""]
        else:
            lines += ["#### Picture", "", caption.caption, ""]
            lines += ["#### Sound", "", caption.audio, ""]
        if caption.speech:
            lines += [
                "#### Speech (transcript)",
                "",
                transcript_lines(caption.speech, hours),
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


def audio_prompt(start: float, span: float, force_hours: bool) -> str:
    return AUDIO_PROMPT.format(
        span=span,
        start_ts=fmt_ts(start, force_hours),
        end_ts=fmt_ts(start + span, force_hours),
    )


def keyed_readings(
    captions: Sequence[WindowCaption],
    texts: Sequence[str],
    force_hours: bool,
    label: str = "Window",
) -> str:
    """Readings grouped under one heading per caption, in order."""
    return "\n\n".join(
        f"### {label} {c.index + 1} ({fmt_ts(c.start, force_hours)} - "
        f"{fmt_ts(c.end, force_hours)})\n{text}"
        for c, text in zip(captions, texts, strict=True)
    )


def transcript_reading(captions: Sequence[WindowCaption], force_hours: bool) -> str:
    """The windows' verbatim speech as one keyed block."""
    return keyed_readings(
        captions,
        [transcript_lines(c.speech or [], force_hours) for c in captions],
        force_hours,
    )


def aggregate_prompt(
    captions: Sequence[WindowCaption],
    chunk: float,
    duration: str,
    force_hours: bool,
    step: ReportPass,
    transcript: bool = False,
) -> str:
    """Build the prompt for one pass, carrying the readings every time.

    Both readings are repeated in full, because a pass cannot see what an
    earlier one read. Earlier passes are deliberately not quoted back: shown its
    own text so far, the model expands and restates instead of moving on. The
    transcript rides along as the ground truth for what was literally said.
    """
    with_audio = any(c.audio for c in captions)

    audio_block = ""
    if with_audio:
        audio_block = "\n--- AUDIO ---\n" + keyed_readings(
            captions, [c.audio or "not analysed" for c in captions], force_hours
        )

    transcript_block = ""
    if transcript:
        transcript_block = "\n--- TRANSCRIPT ---\n" + transcript_reading(
            captions, force_hours
        )

    intro = (AGGREGATE_INTRO_SOUND if with_audio else AGGREGATE_INTRO_VISION).format(
        n=len(captions), duration=duration, chunk=chunk
    )
    if transcript:
        intro += f"\n{AGGREGATE_TRANSCRIPT_NOTICE}"
    body = "\n".join(f"## {name}\n{text}\n" for name, text in step.sections)
    readings = (
        "\n--- CAPTIONS ---\n"
        + keyed_readings(captions, [c.caption for c in captions], force_hours)
        + audio_block
        + transcript_block
    )
    return (
        f"{intro}\n{step.preamble}\n\n{AGGREGATE_SECTIONS_INTRO}{body}\n"
        f"{AGGREGATE_RULES}{readings}"
    )


# --------------------------------------------------------------------------- #
# Long-video digest
# --------------------------------------------------------------------------- #


def readings_route(windows: int) -> str:
    """Whether the report reads the windows directly or through digests."""
    return "digests" if windows > AGGREGATE_MAX_RAW_WINDOWS else "windows"


def digest_batches(count: int) -> list[range]:
    """Consecutive groups of DIGEST_BATCH_WINDOWS windows; the last is smaller."""
    return [
        range(at, min(at + DIGEST_BATCH_WINDOWS, count))
        for at in range(0, count, DIGEST_BATCH_WINDOWS)
    ]


def digest_max_tokens(window_count: int) -> int:
    return min(
        AGGREGATE_MAX_TOKENS,
        DIGEST_BASE_TOKENS + DIGEST_TOKENS_PER_WINDOW * window_count,
    )


def digest_prompt(
    captions: Sequence[WindowCaption],
    chunk: float,
    duration: str,
    force_hours: bool,
) -> str:
    """Build the prompt condensing one batch of window readings into a digest.

    The transcript is carried with the windows it belongs to, so quoted names
    and numbers survive the condensing with their wording intact.
    """
    with_transcript = any(c.speech for c in captions)
    span = captions[-1].end - captions[0].start

    readings = "\n--- CAPTIONS ---\n" + keyed_readings(
        captions, [c.caption for c in captions], force_hours
    )
    if any(c.audio for c in captions):
        readings += "\n--- AUDIO ---\n" + keyed_readings(
            captions, [c.audio or "not analysed" for c in captions], force_hours
        )
    if with_transcript:
        readings += "\n--- TRANSCRIPT ---\n" + transcript_reading(captions, force_hours)

    intro = DIGEST_INTRO.format(
        n=len(captions), duration=duration, chunk=chunk, span=span
    )
    if with_transcript:
        intro += f"\n{AGGREGATE_TRANSCRIPT_NOTICE}"
    body = "\n".join(f"## {name}\n{text}\n" for name, text in DIGEST_SECTIONS)
    return (
        f"{intro}\n{DIGEST_PREAMBLE}\n\n{AGGREGATE_SECTIONS_INTRO}{body}\n"
        f"{DIGEST_RULES}{readings}"
    )


def digest_windows(
    captions: Sequence[WindowCaption],
    chunk: float,
    duration: str,
    force_hours: bool,
    describer: Describer,
    focus: str = "",
) -> list[WindowCaption]:
    """Condense window readings into batch digests that keep the modalities apart.

    Each batch becomes a pseudo-window whose caption is the picture digest and
    whose audio reading carries the sound digest plus the batch's divergence
    notes; the rest of the report reads these exactly as it read the windows.
    Digests do not carry speech themselves: the transcript has already done its
    job inside the digest.
    """
    batches = digest_batches(len(captions))
    digests: list[WindowCaption] = []
    for number, take in enumerate(batches, start=1):
        batch = [captions[i] for i in take]
        start, end = batch[0].start, batch[-1].end
        prompt = digest_prompt(
            batch, chunk=chunk, duration=duration, force_hours=force_hours
        )
        if focus:
            prompt += f"\nThe requester cares most about: {focus}\n"
        answer = describer.chat(
            AGGREGATE_SYSTEM,
            prompt,
            max_tokens=digest_max_tokens(len(batch)),
            loop_resistant=True,
            widen=False,
        )
        print(
            f"  digested batch {number}/{len(batches)}  ({fmt_ts(start, force_hours)})",
            flush=True,
        )
        blocks = split_pass_output(answer)
        picture = blocks.get("Picture", "")
        sound = blocks.get("Sound", "")
        divergence = blocks.get("Divergence", "")
        # A digest that ignored the format is still worth carrying whole: it is
        # a condensation, just one whose headings cannot be split apart.
        if not picture and not sound:
            picture = answer
        caption_text = f"## Picture\n\n{picture}" if picture else ""
        sound_parts = []
        if sound:
            sound_parts.append(f"## Sound\n\n{sound}")
        if divergence:
            sound_parts.append(f"## Divergence\n\n{divergence}")
        digests.append(
            WindowCaption(
                index=number - 1,
                start=start,
                end=end,
                frames=sum(w.frames for w in batch),
                caption=caption_text or answer,
                audio="\n\n".join(sound_parts) or None,
                speech=None,
            )
        )
    return digests


def describe_video(
    path: Path,
    cfg: Config,
    describer: Describer,
    audio_describer: Describer | None = None,
) -> Path:
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
    analyse_audio = cfg.audio and info.has_audio and audio_describer is not None
    if analyse_audio:
        print(f"audio      {cfg.audio_base_url}  model={cfg.audio_model}")
    elif info.has_audio:
        print("audio      skipped")
    transcript_available = cfg.transcript and info.has_audio and asr_available()
    if cfg.transcript and info.has_audio and not transcript_available:
        print(
            "transcript skipped: the faster-whisper package is missing; "
            "run ./install.sh to transcribe speech"
        )
    elif transcript_available:
        print(f"transcript {cfg.asr_model} on {cfg.asr_device} (in-process)")

    out_dir = cfg.out_dir.expanduser().resolve() / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(tempfile.mkdtemp(prefix="video-describer-"))
    started = time.time()

    try:
        spans = [
            (i * chunk, min(chunk, info.duration - i * chunk)) for i in range(windows)
        ]

        # 1. Extract frames and audio up front: fast, sequential, flat on
        #    memory, and it leaves the models nothing but requests to answer.
        frames_per_window: list[list[Path]] = []
        audio_per_window: list[Path | None] = []
        for index, (start, span) in enumerate(spans):
            frames = extract_frames(
                path, start, span, sampled_fps, cfg.width, frames_root / f"w{index:04d}"
            )
            audio_path: Path | None = None
            if analyse_audio or transcript_available:
                audio_path = extract_audio(
                    path, start, span, frames_root / "audio" / f"w{index:04d}.wav"
                )
            print(
                f"  window {index + 1}/{windows}  {fmt_ts(start, hours)}  "
                f"{len(frames)} frames"
                + (
                    f"  {audio_path.stat().st_size // 1024} KiB audio"
                    if audio_path
                    else ""
                ),
                flush=True,
            )
            frames_per_window.append(frames)
            audio_per_window.append(audio_path)

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

        caption_seconds = time.time() - started

        # 3. Listen to the same windows with a model that can hear. Separate
        #    phases keep the two servers busy in turn instead of competing for
        #    the same GPU.
        heard: dict[int, str] = {}
        audio_seconds = 0.0
        if analyse_audio and audio_describer is not None:
            audio_started = time.time()

            def listen(index: int) -> tuple[int, str]:
                start, span = spans[index]
                text = audio_describer.chat(
                    AUDIO_SYSTEM,
                    audio_prompt(start, span, hours),
                    audio=audio_per_window[index],
                    max_tokens=AUDIO_MAX_TOKENS,
                )
                print(
                    f"  heard {index + 1}/{windows}  ({fmt_ts(start, hours)})",
                    flush=True,
                )
                return index, text

            print(f"analysing audio with {workers} parallel worker(s)...")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, text in pool.map(listen, range(windows)):
                    heard[index] = text
            audio_seconds = time.time() - audio_started

        # 4. Transcribe the spoken words in the same windows. The transcript is
        #    a third, verbatim reading: the aggregate pass uses it for exact
        #    wording instead of trusting the audio model's paraphrase. One
        #    window failing to transcribe must not fail the video.
        speech: list[list[dict] | None] = [None] * windows
        transcript_seconds = 0.0
        if transcript_available:
            transcript_started = time.time()
            print(f"transcribing with {cfg.asr_model} on {cfg.asr_device}...")

            def hear_words(index: int) -> tuple[int, list[dict]]:
                try:
                    model = load_asr_model(cfg.asr_model, cfg.asr_device)
                    segments = transcribe_window(
                        model,
                        audio_per_window[index],
                        spans[index][0],
                        cfg.asr_language,
                    )
                except Exception as error:  # noqa: BLE001 - reported, then skipped
                    print(
                        f"warning: transcription of window {index + 1} failed: {error}",
                        file=sys.stderr,
                    )
                    segments = []
                print(
                    f"  transcribed {index + 1}/{windows}  "
                    f"({fmt_ts(spans[index][0], hours)})",
                    flush=True,
                )
                return index, segments

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, segments in pool.map(hear_words, range(windows)):
                    speech[index] = segments
            transcript_seconds = time.time() - transcript_started

        captions = [
            WindowCaption(
                index=i,
                start=spans[i][0],
                end=spans[i][0] + spans[i][1],
                frames=len(frames_per_window[i]),
                caption=captured[i],
                audio=heard.get(i),
                speech=speech[i],
            )
            for i in range(windows)
        ]

        # 5. Whole-video report: condense first when the video is long, then
        #    write it in passes and stitch the sections in reading order — a
        #    section a pass failed to write is recovered by a completion pass
        #    instead of missing silently. A single window is already complete.
        summary = ""
        if cfg.summarise and len(captions) > 1:
            groups = captions
            group_chunk = chunk
            if readings_route(len(captions)) == "digests":
                print(
                    f"condensing {len(captions)} windows into batches of "
                    f"{DIGEST_BATCH_WINDOWS} before the report..."
                )
                groups = digest_windows(
                    captions,
                    chunk=chunk,
                    duration=fmt_ts(info.duration, hours),
                    force_hours=hours,
                    describer=describer,
                    focus=cfg.focus,
                )
                group_chunk = DIGEST_BATCH_WINDOWS * chunk

            with_audio = any(g.audio for g in groups)
            steps = list(report_passes(with_audio, len(groups)))
            print(f"aggregating into a report in {len(steps)} passes...")
            outputs: dict[str, str] = {}
            for step in steps:
                prompt = aggregate_prompt(
                    groups,
                    chunk=group_chunk,
                    duration=fmt_ts(info.duration, hours),
                    force_hours=hours,
                    step=step,
                    transcript=any(g.speech for g in groups),
                )
                if cfg.focus:
                    prompt += f"\nThe requester cares most about: {cfg.focus}\n"
                print(f"  pass {step.name} ({step.max_tokens} tokens)")
                outputs[step.name] = describer.chat(
                    AGGREGATE_SYSTEM,
                    prompt,
                    max_tokens=step.max_tokens,
                    loop_resistant=True,
                    widen=False,
                )
            summary, missing = assemble_report(steps, outputs)
            if missing:
                print(f"  recovering dropped sections: {', '.join(missing)}")
                fix = repair_pass(tuple(missing), with_audio)
                if fix is not None:
                    prompt = aggregate_prompt(
                        groups,
                        chunk=group_chunk,
                        duration=fmt_ts(info.duration, hours),
                        force_hours=hours,
                        step=fix,
                        transcript=any(g.speech for g in groups),
                    )
                    if cfg.focus:
                        prompt += f"\nThe requester cares most about: {cfg.focus}\n"
                    print(f"  pass {fix.name} ({fix.max_tokens} tokens)")
                    outputs[fix.name] = describer.chat(
                        AGGREGATE_SYSTEM,
                        prompt,
                        max_tokens=fix.max_tokens,
                        loop_resistant=True,
                        widen=False,
                    )
                    summary, missing = assemble_report([*steps, fix], outputs)
            if missing:
                print(
                    "warning: the report still lacks: " + ", ".join(missing),
                    file=sys.stderr,
                )

        elapsed = time.time() - started

        # 6. Persist.
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
        if transcript_available:
            utterances = [s for per in speech if per for s in per]
            (out_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "video": str(path),
                        "model": cfg.asr_model,
                        "device": cfg.asr_device,
                        "language": cfg.asr_language or None,
                        "segments": utterances,
                    },
                    indent=2,
                )
                + "\n"
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
                        "audio": cfg.audio,
                        "transcript": cfg.transcript,
                        "asr_model": cfg.asr_model,
                        "asr_device": cfg.asr_device,
                        "asr_language": cfg.asr_language or None,
                    },
                    "windows": windows,
                    "frames_per_window": [len(frames) for frames in frames_per_window],
                    "chunk_seconds": round(chunk, 3),
                    "sampled_fps": round(sampled_fps, 3),
                    "width": cfg.width,
                    "model": cfg.model,
                    "base_url": cfg.base_url,
                    "focus": cfg.focus or None,
                    "audio_analyzed": analyse_audio,
                    "audio_model": cfg.audio_model if analyse_audio else None,
                    "audio_base_url": cfg.audio_base_url if analyse_audio else None,
                    "audio_seconds": round(audio_seconds, 1),
                    "caption_seconds": round(caption_seconds, 1),
                    "total_seconds": round(elapsed, 1),
                    "transcript_model": cfg.asr_model if transcript_available else None,
                    "transcript_seconds": round(transcript_seconds, 1),
                },
                indent=2,
            )
            + "\n"
        )

        print(f"\ndone in {elapsed:.0f}s")
        written = ["description.md", "chunk_captions.json", "meta.json"]
        if transcript_available:
            written.insert(2, "transcript.json")
        for name in written:
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


def check_report(text: str, sections: Sequence[str] | None = None) -> list[str] | None:
    """Sections that were asked for but are absent from a report.

    A missing heading means the model dropped part of its own reporting
    contract, which is worth surfacing rather than passing off as silence.
    Returns None when there is nothing to check: a single-window video, or a run
    with --no-summary, produces no whole-video report. The expected sections
    depend on whether audio was part of the run.
    """
    if "## Whole-video report" not in text:
        return None
    required = report_sections(False) if sections is None else sections
    present = set(re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    return [name for name in required if name not in present]


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
        audio = bool(meta.get("audio_analyzed"))
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
                "audio_analyzed": audio,
                "audio_model": meta.get("audio_model"),
                "transcript_model": meta.get("transcript_model"),
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
                "missing_sections": (
                    check_report(report_text, report_sections(audio))
                    if report_text
                    else None
                ),
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
    if config.get("audio") is False:
        parts.append("no audio analysis")
    if config.get("transcript") is False:
        parts.append("no transcript")
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

    heard = sum(1 for r in records if r["audio_analyzed"])
    audio_models = ", ".join(
        f"`{m}`"
        for m in sorted({r["audio_model"] for r in records if r["audio_model"]})
    )
    audio_line = (
        f"{audio_models or 'analysed'} in {heard} of {len(records)} videos"
        if heard
        else "not analysed"
    )

    transcribed = sum(1 for r in records if r.get("transcript_model"))
    asr_models = ", ".join(
        f"`{m}`"
        for m in sorted(
            {r["transcript_model"] for r in records if r["transcript_model"]}
        )
    )
    transcript_line = (
        f"{asr_models or 'analysed'} in {transcribed} of {len(records)} videos"
        if transcribed
        else "not analysed"
    )

    lines = [
        f"# Results: {out_dir.name}",
        "",
        f"- **Videos:** {len(records)} ({fmt_span(footage)} of footage)",
        f"- **Processing:** {fmt_span(processing)}{speed}",
        f"- **Model:** {model_line}",
        f"- **Requested settings:** {_describe_config(shared_config(records))}",
        f"- **Audio:** {audio_line}",
        f"- **Transcript:** {transcript_line}",
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
        "sampling rate. Rebuild this file and `run.json` with",
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
            "  ./describe_video.py clip.mp4 --no-audio\n"
            "  ./describe_video.py clip.mp4 --no-transcript\n"
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
        "--no-audio",
        dest="audio",
        action="store_false",
        help="skip the audio lane even when the source has a soundtrack",
    )
    parser.add_argument(
        "--audio-base-url",
        default=os.environ.get("AUDIO_BASE_URL", "http://127.0.0.1:8001/v1"),
        help="OpenAI-compatible endpoint for the audio lane (default: $(default)s)",
    )
    parser.add_argument(
        "--audio-model",
        default=os.environ.get("AUDIO_MODEL", "qwen2-audio-7b"),
        help="served audio model name (default: $(default)s)",
    )
    parser.add_argument(
        "--no-transcript",
        dest="transcript",
        action="store_false",
        help="skip the local speech transcript even when the source has sound",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("ASR_MODEL", "small"),
        help="faster-whisper model size or local path (default: $(default)s)",
    )
    parser.add_argument(
        "--asr-device",
        default=os.environ.get("ASR_DEVICE", "cpu"),
        help="ASR device: cpu, cuda, or auto (default: $(default)s)",
    )
    parser.add_argument(
        "--asr-language",
        default=os.environ.get("ASR_LANGUAGE", ""),
        help="ASR language code, e.g. 'en'; empty auto-detects (default: auto)",
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
        audio=args.audio,
        audio_base_url=args.audio_base_url,
        audio_model=args.audio_model,
        transcript=args.transcript,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_language=args.asr_language.strip(),
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

    audio_describer: Describer | None = None
    if cfg.audio:
        try:
            require_endpoint(cfg.audio_base_url, "audio")
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        audio_describer = Describer(
            OpenAI(base_url=cfg.audio_base_url, api_key=cfg.api_key, timeout=900.0),
            cfg.audio_model,
        )

    failures = 0
    for position, video in enumerate(videos, start=1):
        if len(videos) > 1:
            print(f"\n=== [{position}/{len(videos)}] {video.name} ===")
        target = cfg.out_dir.expanduser().resolve() / video.stem / "description.md"
        if args.skip_existing and target.exists():
            print(f"skipping {video.name}: {target} already exists")
            continue
        try:
            describe_video(video, cfg, describer, audio_describer)
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
