"""Unit tests. No GPU, no model server: every test uses generated media."""

from __future__ import annotations

from pathlib import Path

import pytest

import describe_video as dv

# --------------------------------------------------------------------------- #
# Timestamp formatting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0:00.0"),
        (9.5, "0:09.5"),
        (65.4, "1:05.4"),
        (3599.9, "59:59.9"),
    ],
)
def test_fmt_ts_under_an_hour(seconds, expected):
    assert dv.fmt_ts(seconds) == expected


def test_fmt_ts_rolls_over_to_hours():
    assert dv.fmt_ts(3600.0) == "1:00:00.0"
    assert dv.fmt_ts(3725.5) == "1:02:05.5"


def test_fmt_ts_can_force_hours():
    assert dv.fmt_ts(65.4, force_hours=True) == "0:01:05.4"


def test_fmt_ts_clamps_negatives():
    assert dv.fmt_ts(-3.0) == "0:00.0"


# --------------------------------------------------------------------------- #
# Window planning
# --------------------------------------------------------------------------- #


def test_plan_windows_splits_evenly():
    assert dv.plan_windows(120.0, 30.0, 1.0, 48) == (4, 30.0, 1.0)


def test_plan_windows_avoids_a_stub_final_window():
    windows, chunk, fps = dv.plan_windows(40.1, 30.0, 1.0, 48)
    assert windows == 2
    assert chunk == pytest.approx(20.05)
    assert fps == 1.0


def test_plan_windows_short_video_is_one_window():
    windows, chunk, _ = dv.plan_windows(4.0, 30.0, 1.0, 48)
    assert (windows, chunk) == (1, 4.0)


def test_plan_windows_caps_fps_to_the_frame_budget():
    windows, chunk, fps = dv.plan_windows(60.0, 30.0, 2.0, 20)
    assert windows == 2
    assert fps == pytest.approx(20 / 30)
    assert fps * chunk <= 20


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 30.0, 1.0, 48),
        (10.0, 0.0, 1.0, 48),
        (10.0, 30.0, 0.0, 48),
        (10.0, 30.0, 1.0, 0),
    ],
)
def test_plan_windows_rejects_bad_input(args):
    with pytest.raises(ValueError):
        dv.plan_windows(*args)


# --------------------------------------------------------------------------- #
# Frame timestamps
# --------------------------------------------------------------------------- #


def test_frame_timestamps_follow_the_sampling_interval():
    assert dv.frame_timestamps(60.0, 2.0, 3) == ["1:00.0", "1:00.5", "1:01.0"]


def test_frame_timestamps_count_matches_request():
    assert len(dv.frame_timestamps(0.0, 1.0, 48)) == 48


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


def test_probe_video_reads_metadata(sample_video):
    info = dv.probe_video(Path(sample_video))
    assert info.duration == pytest.approx(3.0, abs=0.3)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10.0, abs=0.5)
    assert info.has_audio is True


def test_probe_video_detects_missing_audio(silent_video):
    assert dv.probe_video(Path(silent_video)).has_audio is False


def test_probe_video_rejects_a_non_video(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_text("this is not a video")
    with pytest.raises(RuntimeError):
        dv.probe_video(junk)


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #


def test_extract_frames_samples_at_the_requested_rate(sample_video, tmp_path):
    frames = dv.extract_frames(
        Path(sample_video),
        start=0.0,
        duration=3.0,
        fps=1.0,
        width=160,
        dest=tmp_path / "w0",
    )
    assert len(frames) == 3
    assert all(f.exists() and f.stat().st_size > 0 for f in frames)


def test_extract_frames_seeks_into_the_video(sample_video, tmp_path):
    frames = dv.extract_frames(
        Path(sample_video),
        start=1.0,
        duration=1.0,
        fps=2.0,
        width=160,
        dest=tmp_path / "w1",
    )
    assert len(frames) == 2


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _info(**overrides) -> dv.VideoInfo:
    fields = {
        "path": Path("clip.mp4"),
        "duration": 65.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "has_audio": True,
    }
    fields.update(overrides)
    return dv.VideoInfo(**fields)


def _caption(index=0, start=0.0, end=30.0) -> dv.WindowCaption:
    return dv.WindowCaption(
        index=index,
        start=start,
        end=end,
        frames=30,
        caption="## Segment summary\nA test.",
    )


def test_render_markdown_includes_header_and_captions():
    report = dv.render_markdown(
        _info(),
        [_caption()],
        summary="An overview.",
        cfg=dv.Config(),
        chunk=32.5,
        sampled_fps=1.0,
    )
    assert report.startswith("# Video description: clip.mp4")
    assert "## Whole-video report" in report
    assert "An overview." in report
    assert "## Per-window captions" in report
    assert "### 0:00.0 - 0:30.0 (30 frames)" in report


def test_render_markdown_flags_a_missing_audio_track():
    report = dv.render_markdown(
        _info(has_audio=False),
        [_caption()],
        summary="",
        cfg=dv.Config(),
        chunk=30.0,
        sampled_fps=1.0,
    )
    assert "no audio track" in report
    assert "## Whole-video report" not in report


def test_render_markdown_records_the_focus():
    cfg = dv.Config(focus="hard hats")
    report = dv.render_markdown(
        _info(),
        [_caption()],
        summary="",
        cfg=cfg,
        chunk=30.0,
        sampled_fps=1.0,
    )
    assert "- **Focus:** hard hats" in report


# --------------------------------------------------------------------------- #
# Input expansion
# --------------------------------------------------------------------------- #


def test_expand_inputs_accepts_a_directory(sample_video, tmp_path):
    target = tmp_path / "footage"
    target.mkdir()
    (target / "a.mp4").write_bytes(Path(sample_video).read_bytes())
    (target / "notes.txt").write_text("ignore me")

    found = dv.expand_inputs([str(target)])
    assert [p.name for p in found] == ["a.mp4"]


def test_expand_inputs_deduplicates(sample_video):
    found = dv.expand_inputs([sample_video, sample_video])
    assert len(found) == 1


def test_expand_inputs_rejects_unknown_paths(tmp_path):
    with pytest.raises(RuntimeError):
        dv.expand_inputs([str(tmp_path / "missing.mp4")])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        dv.main(["--version"])
    assert excinfo.value.code == 0
    assert dv.__version__ in capsys.readouterr().out


def test_cli_rejects_a_missing_file():
    assert dv.main(["definitely-not-here.mp4"]) == 1


@pytest.mark.parametrize(
    "flag", [("--fps", "0"), ("--chunk-seconds", "0"), ("--max-frames", "0")]
)
def test_cli_rejects_invalid_numbers(sample_video, flag):
    assert dv.main([sample_video, *flag]) == 1


def test_cli_no_summary_flag_parses():
    args = dv.build_parser().parse_args(["clip.mp4", "--no-summary"])
    assert args.summarise is False


def test_cli_defaults_are_documented_values():
    args = dv.build_parser().parse_args(["clip.mp4"])
    assert (args.fps, args.chunk_seconds, args.max_frames, args.width) == (
        1.0,
        30.0,
        48,
        896,
    )
