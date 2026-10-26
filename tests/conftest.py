"""Shared fixtures. Media is generated at test time, never committed."""

from __future__ import annotations

import subprocess

import pytest


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> str:
    """A small synthetic MP4 with an audio track, built with ffmpeg."""
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    proc = _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-shortest",
            # mpeg4 and aac are built into every ffmpeg build, unlike libx264.
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ]
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not build the test clip: {proc.stderr.strip()}")
    return str(path)


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> str:
    """A small synthetic MP4 with no audio stream."""
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    proc = _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5:duration=2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-y",
            str(path),
        ]
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not build the test clip: {proc.stderr.strip()}")
    return str(path)
