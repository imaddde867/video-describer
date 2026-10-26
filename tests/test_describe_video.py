"""Unit tests. No GPU, no model server: every test uses generated media."""

from __future__ import annotations

import json
import types
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


def test_cli_requires_videos_or_a_reindex(capsys):
    assert dv.main([]) == 1
    assert "no videos given" in capsys.readouterr().err


def test_runs_in_venv_compares_sys_prefix(tmp_path):
    assert dv.runs_in_venv(str(tmp_path), str(tmp_path)) is True
    assert dv.runs_in_venv(str(tmp_path), "/usr") is False


# --------------------------------------------------------------------------- #
# Model client
# --------------------------------------------------------------------------- #

LOOPING_REPLY = " ".join(["MjU2MjU2MjU2"] * 300)
PROSE_REPLY = " ".join(f"word{i}" for i in range(300))


class _StubCompletions:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish = self._replies.pop(0)
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=content), finish_reason=finish
        )
        return types.SimpleNamespace(choices=[choice])


class _StubClient:
    """Stands in for the OpenAI client, recording how it was called."""

    def __init__(self, replies):
        self.completions = _StubCompletions(replies)
        self.chat = types.SimpleNamespace(completions=self.completions)


def test_looks_degenerate_catches_a_repetition_loop():
    assert dv.looks_degenerate(LOOPING_REPLY) is True


def test_looks_degenerate_catches_a_repeated_phrase():
    assert dv.looks_degenerate(" ".join(["the robot turns left"] * 60)) is True


def test_looks_degenerate_accepts_prose():
    assert dv.looks_degenerate(PROSE_REPLY) is False


def test_looks_degenerate_ignores_short_replies():
    assert dv.looks_degenerate("a a a a a") is False


def test_aggregate_token_budget_grows_with_the_window_count():
    budgets = [dv.aggregate_token_budget(n) for n in (1, 3, 10, 50)]
    assert budgets == sorted(budgets)
    assert budgets[0] >= dv.AGGREGATE_BASE_TOKENS
    assert budgets[-1] == dv.AGGREGATE_MAX_TOKENS


def test_describer_returns_a_clean_answer_unchanged():
    client = _StubClient([("A clean caption.", "stop")])
    assert dv.Describer(client, "stub").chat("system", "prompt") == "A clean caption."
    assert len(client.completions.calls) == 1


def test_describer_retries_a_repetition_loop(capsys):
    client = _StubClient([(LOOPING_REPLY, "stop"), ("A clean caption.", "stop")])

    assert dv.Describer(client, "stub").chat("system", "prompt") == "A clean caption."
    assert [call["temperature"] for call in client.completions.calls] == [0.0, 0.2]
    assert client.completions.calls[1]["extra_body"] == {"repetition_penalty": 1.05}
    assert "repetition loop" in capsys.readouterr().err


def test_describer_warns_when_the_retry_also_loops(capsys):
    client = _StubClient([(LOOPING_REPLY, "stop"), (LOOPING_REPLY, "stop")])

    assert dv.Describer(client, "stub").chat("system", "prompt") == LOOPING_REPLY
    assert "retry repeated itself" in capsys.readouterr().err


def test_describer_widens_the_budget_when_an_answer_is_cut(capsys):
    client = _StubClient([("Cut off mid", "length"), ("A complete caption.", "stop")])

    text = dv.Describer(client, "stub").chat("system", "prompt", max_tokens=100)
    assert text == "A complete caption."
    assert [call["max_tokens"] for call in client.completions.calls] == [100, 200]
    assert "retrying with a larger budget" in capsys.readouterr().err


def test_describer_reports_an_answer_cut_twice(capsys):
    client = _StubClient([("Cut off mid", "length"), ("Cut off again", "length")])

    text = dv.Describer(client, "stub").chat("system", "prompt", max_tokens=100)
    assert text == "Cut off again"
    assert "still truncated at 200 tokens" in capsys.readouterr().err


def test_describer_prefers_the_shorter_answer_over_a_longer_loop(capsys):
    client = _StubClient([("Cut off mid", "length"), (LOOPING_REPLY, "length")])

    text = dv.Describer(client, "stub").chat("system", "prompt", max_tokens=100)
    assert text == "Cut off mid"
    assert "keeping the shorter one" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Experiment index
# --------------------------------------------------------------------------- #

COMPLETE_REPORT = (
    "# Video description: clip.mp4\n\n"
    "## Whole-video report\n\n"
    "## Overview\nA.\n\n"
    "## Timeline\n- 0:00.0 - b.\n\n"
    "## Recurring elements\nc.\n\n"
    "## Notable details\nd.\n\n"
    "## Gaps and uncertainties\ne.\n"
)


def _folder(root: Path, name: str, *, report: str = COMPLETE_REPORT, **overrides):
    """Create a per-video result folder the way a real run leaves one."""
    folder = root / name
    folder.mkdir(parents=True)
    meta = {
        "video": f"/footage/{name}.mp4",
        "probe": {
            "duration": 49.4,
            "width": 720,
            "height": 1280,
            "fps": 30.0,
            "has_audio": True,
        },
        "config": {
            "chunk_seconds": 25.0,
            "fps": 1.5,
            "max_frames": 36,
            "width": 720,
            "workers": 3,
            "focus": None,
            "summarise": True,
        },
        "windows": 2,
        "frames_per_window": [18, 18],
        "chunk_seconds": 24.7,
        "sampled_fps": 1.5,
        "width": 720,
        "model": "qwen3-vl-8b",
        "base_url": "http://127.0.0.1:8000/v1",
        "audio_analyzed": False,
        "caption_seconds": 21.1,
        "total_seconds": 35.0,
    }
    meta.update(overrides)
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "chunk_captions.json").write_text("{}")
    (folder / "description.md").write_text(report)
    return folder


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(35.0, "35 s"), (384.0, "6.4 min"), (5400.0, "1.5 h")],
)
def test_fmt_span(seconds, expected):
    assert dv.fmt_span(seconds) == expected


def test_check_report_accepts_a_complete_report():
    assert dv.check_report(COMPLETE_REPORT) == []


def test_check_report_flags_a_dropped_section():
    text = COMPLETE_REPORT.replace("## Notable details\nd.\n\n", "")
    assert dv.check_report(text) == ["Notable details"]


def test_check_report_ignores_a_report_without_an_aggregate_pass():
    assert (
        dv.check_report("# Video description: clip.mp4\n\n## Segment summary\nA.")
        is None
    )


def test_write_index_records_every_video(tmp_path):
    _folder(tmp_path, "one")
    _folder(tmp_path, "two")

    run_path, index_path = dv.write_index(tmp_path)

    run = json.loads(run_path.read_text())
    assert run["experiment"] == tmp_path.name
    assert run["videos"] == 2
    assert run["footage_seconds"] == pytest.approx(98.8)
    assert run["processing_seconds"] == pytest.approx(70.0)
    assert run["requested"]["fps"] == 1.5
    assert [r["name"] for r in run["results"]] == ["one", "two"]
    assert run["results"][0]["missing_sections"] == []
    assert run["results"][0]["report"] == "one/description.md"

    index = index_path.read_text()
    assert index.startswith(f"# Results: {tmp_path.name}")
    assert "| Video | Duration | Windows | Frames | Time | Notes |" in index
    assert "[one](one/description.md) | 0:49.4 | 2 | 36 | 35 s" in index
    assert "not recorded" not in index  # the request is known for these runs


def test_write_index_reports_dropped_sections(tmp_path):
    _folder(
        tmp_path,
        "one",
        report=COMPLETE_REPORT.replace("## Notable details\nd.\n\n", ""),
    )

    _, index_path = dv.write_index(tmp_path)
    assert "missing: Notable details" in index_path.read_text()


def test_write_index_notes_an_unrecorded_config(tmp_path):
    _folder(tmp_path, "one", config=None)

    run_path, index_path = dv.write_index(tmp_path)
    assert json.loads(run_path.read_text())["requested"] is None
    assert "not recorded" in index_path.read_text()


def test_write_index_needs_results(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dv.write_index(empty) is None


def test_write_index_skips_an_unreadable_record(tmp_path, capsys):
    _folder(tmp_path, "good")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "meta.json").write_text("{not json")

    run_path, _ = dv.write_index(tmp_path)
    assert [r["name"] for r in json.loads(run_path.read_text())["results"]] == ["good"]
    assert "skipping" in capsys.readouterr().err


def test_cli_reindex_rebuilds_an_existing_index(tmp_path, capsys):
    _folder(tmp_path, "one")

    assert dv.main(["--reindex", str(tmp_path)]) == 0
    assert (tmp_path / "RESULTS.md").exists()
    assert (tmp_path / "run.json").exists()
    assert "rebuilt" in capsys.readouterr().out


def test_cli_reindex_rejects_a_directory_without_results(tmp_path):
    assert dv.main(["--reindex", str(tmp_path)]) == 1


def test_cli_reindex_rejects_a_missing_directory(tmp_path):
    assert dv.main(["--reindex", str(tmp_path / "absent")]) == 1
