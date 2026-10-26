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


def _caption(index=0, start=0.0, end=30.0, audio=None, speech=None) -> dv.WindowCaption:
    return dv.WindowCaption(
        index=index,
        start=start,
        end=end,
        frames=30,
        caption="## Segment summary\nA test.",
        audio=audio,
        speech=speech,
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
# Audio lane
# --------------------------------------------------------------------------- #


def test_extract_audio_writes_a_wav(sample_video, tmp_path):
    clip = dv.extract_audio(Path(sample_video), 0.0, 3.0, tmp_path / "w0.wav")
    assert clip.suffix == ".wav"
    assert clip.stat().st_size > 1000


def test_extract_audio_rejects_a_video_without_sound(silent_video, tmp_path):
    with pytest.raises(RuntimeError):
        dv.extract_audio(Path(silent_video), 0.0, 2.0, tmp_path / "w0.wav")


def test_report_sections_add_sound_only_when_audio_is_analysed():
    assert "Spoken content" not in dv.report_sections(False)
    assert "Spoken content" in dv.report_sections(True)
    assert "Gaps and uncertainties" in dv.report_sections(True)


def test_audio_prompt_names_the_window_and_refuses_transcription():
    prompt = dv.audio_prompt(30.0, 25.0, False)
    assert "0:30.0" in prompt
    assert "0:55.0" in prompt
    assert "Do NOT transcribe" in prompt


def _prompt(step, audio=True):
    captions = [_caption(audio="## Audio summary\nA drill runs." if audio else None)]
    return dv.aggregate_prompt(
        captions, chunk=25.0, duration="0:30.0", force_hours=False, step=step
    )


def test_every_pass_carries_both_readings_and_its_own_sections():
    for step in dv.report_passes(with_audio=True, windows=2):
        prompt = _prompt(step)
        assert "--- CAPTIONS ---" in prompt
        assert "--- AUDIO ---" in prompt
        assert "A drill runs." in prompt
        assert dv.AGGREGATE_SECTIONS_INTRO in prompt
        assert f"## {step.sections[0][0]}" in prompt


def test_no_pass_preamble_reads_as_a_heading():
    # The model answered "You are writing the FUSION part" with a `**FUSION**`
    # heading, so the section it was asked for never appeared in the report.
    for step in dv.report_passes(with_audio=True, windows=2):
        assert not any(
            word.isupper() and len(word) > 2 for word in step.preamble.split()
        )


def test_report_passes_cover_every_section_exactly_once_in_reading_order():
    passes = dv.report_passes(with_audio=True, windows=2)
    written = [name for step in passes for name, _ in step.sections]

    assert written == list(dv.report_sections(True))
    assert [step.name for step in passes] == ["opening", "fusion", "closing"]


def test_report_passes_skip_the_sound_steps_without_audio():
    passes = dv.report_passes(with_audio=False, windows=2)
    written = [name for step in passes for name, _ in step.sections]

    assert written == list(dv.report_sections(False))
    assert [step.name for step in passes] == ["opening", "closing"]


def test_report_pass_budgets_grow_with_the_window_count_and_stay_bounded():
    small = [step.max_tokens for step in dv.report_passes(True, 2)]
    big = [step.max_tokens for step in dv.report_passes(True, 40)]

    assert small[0] < big[0]
    assert max(big) <= dv.AGGREGATE_MAX_TOKENS


def test_the_fusion_pass_is_asked_for_short_prose():
    fusion = next(s for s in dv.report_passes(True, 2) if s.name == "fusion")
    prompt = _prompt(fusion).replace("\n", " ")

    assert "two or three short paragraphs of prose" in prompt
    assert "at most five points" in prompt
    assert "never list items as divergences" in prompt


def test_the_opening_pass_bounds_the_timeline():
    opening = next(s for s in dv.report_passes(True, 2) if s.name == "opening")
    prompt = _prompt(opening).replace("\n", " ")

    assert "a walkthrough of on-screen text is " in prompt
    assert "about ten to twenty entries" in prompt
    assert "restating it elsewhere in the report is wrong" in prompt
    assert "Never phrase an entry as an absence" in prompt


def test_the_closing_pass_bans_gaps_phrased_as_absences():
    closing = next(s for s in dv.report_passes(True, 2) if s.name == "closing")
    prompt = _prompt(closing).replace("\n", " ")

    assert "Never phrase an entry as an absence" in prompt
    assert "is not a gap" in prompt
    assert "At most six entries" in prompt
    assert "give at most ten" in prompt


def test_no_pass_is_shown_its_own_text_so_far():
    # Quoted back its earlier output, the model expanded and padded instead of
    # moving on to the next sections, and lost the last one to the budget.
    closing = next(s for s in dv.report_passes(True, 2) if s.name == "closing")

    assert "WRITTEN SO FAR" not in _prompt(closing)


def test_the_prompt_without_audio_omits_the_sound_sections():
    opening = dv.report_passes(with_audio=False, windows=2)[0]
    prompt = _prompt(opening, audio=False)

    assert "--- AUDIO ---" not in prompt
    assert "Spoken content" not in prompt


# --------------------------------------------------------------------------- #
# Transcript (the third reading)
# --------------------------------------------------------------------------- #


def _speech(seconds: float, text: str) -> dict:
    return {"start": seconds, "end": seconds + 2.0, "text": text}


def test_the_aggregate_pass_receives_the_transcript_verbatim():
    captions = [
        _caption(
            audio="## Audio summary\nSomeone speaks.",
            speech=[_speech(3.2, "hello there")],
        )
    ]
    step = dv.report_passes(with_audio=True, windows=2)[0]
    prompt = dv.aggregate_prompt(
        captions,
        chunk=30.0,
        duration="0:30.0",
        force_hours=False,
        step=step,
        transcript=True,
    )

    assert "--- TRANSCRIPT ---" in prompt
    assert "0:03.2 - hello there" in prompt
    assert "exact" in prompt  # the wording rule rides along with the block


def test_the_aggregate_pass_omits_the_transcript_without_speech():
    captions = [_caption(audio="## Audio summary\nA drill runs.")]
    step = dv.report_passes(with_audio=True, windows=2)[0]
    prompt = dv.aggregate_prompt(
        captions,
        chunk=30.0,
        duration="0:30.0",
        force_hours=False,
        step=step,
    )

    assert "TRANSCRIPT" not in prompt


def test_transcript_lines_format_utterances_with_absolute_times():
    speech = [_speech(0.0, "first"), _speech(3601.0, "second")]

    assert dv.transcript_lines(speech[:1]) == "0:00.0 - first"
    assert "1:00:01.0 - second" in dv.transcript_lines(speech, force_hours=True)
    assert dv.transcript_lines([]) == "no speech detected"


def test_segments_in_window_keeps_what_starts_there():
    speech = [_speech(29.0, "straddles the window"), _speech(35.0, "inside")]

    picked = dv.segments_in_window(speech, start=30.0, end=60.0)
    assert [s["text"] for s in picked] == ["inside"]


def test_render_markdown_carries_the_transcript_verbatim():
    report = dv.render_markdown(
        _info(),
        [_caption(speech=[_speech(4.0, "hello there")])],
        summary="Fused.",
        cfg=dv.Config(),
        chunk=30.0,
        sampled_fps=1.0,
    )

    assert "## Transcript (verbatim)" in report
    assert "0:04.0 - hello there" in report
    assert "#### Speech (transcript)" in report
    assert "`small`" in report  # the header names the ASR model


def test_render_markdown_drops_the_transcript_when_there_is_none():
    report = dv.render_markdown(
        _info(),
        [_caption()],
        summary="",
        cfg=dv.Config(),
        chunk=30.0,
        sampled_fps=1.0,
    )

    assert "Transcript" not in report


def test_a_report_can_be_repaired_after_it_is_written():
    steps = dv.report_passes(with_audio=True, windows=2)
    outputs = {
        "opening": "## Overview\nA short film.\n\n## Timeline\n- 0:00 - opens.\n\n"
        "## Spoken content\nSomebody talks.\n\n## Voice and sound\nQuiet room.",
        "closing": "## Gaps and uncertainties\nnone.",
    }

    report, missing = dv.assemble_report(steps, outputs)

    assert "## Overview" in report
    assert report.index("## Overview") < report.index("## Timeline")
    assert report.index("## Spoken content") < report.index("## Voice and sound")
    # The fusion part was never written, and the closing pass skipped two.
    assert missing == [
        "Sound and picture together",
        "Recurring elements",
        "Notable details",
    ]
    assert "## Sound and picture together" not in report


def test_an_assembled_report_keeps_the_canonical_order_despite_a_wayward_pass():
    steps = dv.report_passes(with_audio=False, windows=2)
    # The closing pass chose its own order; assembly must not carry it over.
    outputs = {
        "opening": "## Overview\nA.\n\n## Timeline\n- 0:00 - b.",
        "closing": "## Gaps and uncertainties\nnone.\n\n## Notable details\nd.\n\n"
        "## Recurring elements\nc.",
    }

    report, _missing = dv.assemble_report(steps, outputs)

    assert report.index("## Recurring elements") < report.index("## Notable details")
    assert report.index("## Notable details") < report.index(
        "## Gaps and uncertainties"
    )


def test_a_section_written_twice_keeps_its_first_body():
    blocks = dv.split_pass_output(
        "## Overview\nfirst.\n\n## Timeline\nt.\n\n## Overview\nsecond."
    )

    assert blocks["Overview"] == "first."


def test_a_wayward_heading_cannot_supply_a_section_it_does_not_own():
    # A closing pass that echoes an opening heading wrote nothing usable for
    # either: the overview is only ever read from the pass asked for it.
    report, missing = dv.assemble_report(
        dv.report_passes(with_audio=False, windows=2),
        {"closing": "## Overview\nrepeated.\n\n## Recurring elements\nc."},
    )

    assert "repeated." not in report
    assert missing == [
        "Overview",
        "Timeline",
        "Notable details",
        "Gaps and uncertainties",
    ]


def test_the_repair_pass_targets_only_the_missing_sections():
    step = dv.repair_pass(
        ("Notable details", "Gaps and uncertainties"), with_audio=True
    )

    assert [name for name, _ in step.sections] == [
        "Notable details",
        "Gaps and uncertainties",
    ]
    assert step.max_tokens <= dv.AGGREGATE_MAX_TOKENS


def test_a_repair_pass_is_not_built_for_a_whole_report():
    assert dv.repair_pass((), with_audio=True) is None


def test_a_section_the_repair_pass_recovered_counts_as_recovered():
    # Closing failed to write one section, the completion pass wrote it, and
    # its position is where the report's reading order puts it, not last.
    steps = dv.report_passes(with_audio=False, windows=2)
    fix = dv.repair_pass(("Notable details",), with_audio=False)
    outputs = {
        "opening": "## Overview\nA.\n\n## Timeline\nb.",
        "closing": "## Recurring elements\nc.\n\n## Gaps and uncertainties\ne.",
        "repair": "## Notable details\nd.",
    }

    report, missing = dv.assemble_report([*steps, fix], outputs)

    assert missing == []
    assert report.index("## Recurring elements") < report.index("## Notable details")
    assert report.index("## Notable details") < report.index(
        "## Gaps and uncertainties"
    )
    text = "# Video description: clip.mp4\n\n## Whole-video report\n\n" + "".join(
        f"## {name}\nA.\n\n" for name in dv.report_sections(False)
    )

    assert dv.check_report(text, dv.report_sections(True)) == [
        "Spoken content",
        "Voice and sound",
        "Sound and picture together",
    ]


def test_render_markdown_shows_both_readings_per_window():
    report = dv.render_markdown(
        _info(),
        [_caption(audio="## Audio summary\nA drill runs.")],
        summary="Fused.",
        cfg=dv.Config(),
        chunk=30.0,
        sampled_fps=1.0,
    )
    assert "#### Picture" in report
    assert "#### Sound" in report
    assert "A drill runs." in report
    assert "`qwen2-audio-7b`" in report


def test_render_markdown_marks_audio_as_not_analysed():
    report = dv.render_markdown(
        _info(),
        [_caption()],
        summary="",
        cfg=dv.Config(),
        chunk=30.0,
        sampled_fps=1.0,
    )
    assert "#### Sound" not in report
    assert "- **Audio:** not analysed" in report


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


def test_cli_no_transcript_flag_parses():
    args = dv.build_parser().parse_args(["clip.mp4", "--no-transcript"])
    assert args.transcript is False


def test_cli_audio_is_on_unless_disabled():
    parser = dv.build_parser()
    assert parser.parse_args(["clip.mp4"]).audio is True
    assert parser.parse_args(["clip.mp4", "--no-audio"]).audio is False


def test_cli_defaults_are_documented_values(monkeypatch):
    for name in ("ASR_MODEL", "ASR_DEVICE", "ASR_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    args = dv.build_parser().parse_args(["clip.mp4"])
    assert (args.fps, args.chunk_seconds, args.max_frames, args.width) == (
        1.0,
        30.0,
        48,
        896,
    )
    # Speech transcription runs unless it is turned off, on the CPU, so the
    # lanes keep the whole GPU to themselves.
    assert args.transcript is True
    assert (args.asr_model, args.asr_device, args.asr_language) == ("small", "cpu", "")


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


# The shape that slipped past the n-gram ratio: a whole block of genuinely
# varied lines repeated, so the cycle itself contributes plenty of unique
# n-grams (with three copies the ratio sits near a third, above the threshold).
def _varied_block(lines: int = 12, words_per_line: int = 10) -> str:
    pool = [f"token{n:03d}" for n in range(lines * words_per_line)]
    return "\n".join(
        " ".join(pool[i * words_per_line : (i + 1) * words_per_line])
        for i in range(lines)
    )


LONG_PERIOD_LOOP = "\n\n".join([_varied_block()] * 3)


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


def test_looks_degenerate_catches_a_long_period_loop():
    assert dv.looks_degenerate(LONG_PERIOD_LOOP) is True


def test_looks_degenerate_tolerates_a_block_that_appears_twice():
    body = _varied_block()

    assert dv.looks_degenerate(f"## Timeline\n{body}\n\n{body}") is False


def test_trim_repeated_lines_collapses_a_locked_loop():
    line = '- Audio says "mac os x 10.12" — visual says "macOS 26+."'
    text = f"## Gaps and uncertainties\n{line}\n" + f"{line}\n" * 42 + "- A real point."

    assert dv.trim_repeated_lines(text).count(line) == 1


def test_trim_repeated_lines_keeps_a_point_made_in_two_sections():
    line = "- Audio says the app needs macOS 26 or later, which the frames confirm."
    text = f"## Sound and picture together\n{line}\n\n## Notable details\n{line}"

    assert dv.trim_repeated_lines(text).count(line) == 2


def test_trim_repeated_lines_leaves_ordinary_prose_alone():
    text = "## Overview\nA hand places the toy on the desk.\n\nThe screen wakes."

    assert dv.trim_repeated_lines(text) == text


def test_describer_trims_a_loop_that_survives_every_retry(capsys):
    loop = '- Audio says "clod pro" — likely a typo for "Claude Pro."'
    degenerate = f"## Gaps\n{loop}\n" + f"{loop}\n" * 40
    client = _StubClient([(degenerate, "stop"), (degenerate, "stop")])

    answer = dv.Describer(client, "stub").chat("system", "prompt")

    assert answer.count(loop) == 1
    assert "dropped 40 repeated line(s)" in capsys.readouterr().err


def test_a_report_pass_does_not_widen_a_cut_answer(capsys):
    client = _StubClient([("## Overview\nA cut report.", "length")])

    answer = dv.Describer(client, "stub").chat(
        "system", "prompt", max_tokens=100, loop_resistant=True, widen=False
    )

    assert answer == "## Overview\nA cut report."
    assert len(client.completions.calls) == 1
    assert "keeping it as it is" in capsys.readouterr().err


def test_a_loop_resistant_pass_starts_with_the_sampling_that_breaks_loops():
    client = _StubClient([("A clean report.", "stop")])

    answer = dv.Describer(client, "stub").chat("system", "prompt", loop_resistant=True)

    assert answer == "A clean report."
    assert client.completions.calls[0]["temperature"] == 0.2
    assert client.completions.calls[0]["extra_body"] == {"repetition_penalty": 1.05}


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


# --------------------------------------------------------------------------- #
# Long-video digest
# --------------------------------------------------------------------------- #


def test_short_videos_are_read_directly_and_long_ones_condensed():
    assert dv.readings_route(24) == "windows"
    assert dv.readings_route(25) == "digests"


def test_digest_batches_cover_every_window_exactly_once():
    batches = dv.digest_batches(37)

    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(37))
    assert all(len(batch) <= dv.DIGEST_BATCH_WINDOWS for batch in batches)
    assert batches[-1][-1] == 36  # the last batch is smaller, not a stub of one
    assert len(batches) == 4


def test_digest_prompt_names_its_span_and_keeps_both_modalities():
    captions = [
        _caption(index=0, start=3600.0, end=3630.0, audio="## Audio summary\nTalk."),
        _caption(index=1, start=3630.0, end=3660.0, audio="## Audio summary\nMusic."),
    ]
    prompt = dv.digest_prompt(
        captions, chunk=30.0, duration="2:00:00.0", force_hours=True
    )

    assert "60 seconds" in prompt  # the batch's own span
    assert "1:00:00.0" in prompt and "1:01:00.0" in prompt  # absolute window times
    assert "## Picture" in prompt
    assert "## Sound" in prompt
    assert "## Divergence" in prompt


def test_the_digest_prompt_carries_the_transcript_with_its_windows():
    captions = [
        _caption(
            index=0,
            start=0.0,
            end=30.0,
            audio="## Audio summary\nTalk.",
            speech=[_speech(5.0, "a price of forty dollars")],
        )
    ]
    prompt = dv.digest_prompt(
        captions, chunk=30.0, duration="0:30.0", force_hours=False
    )

    assert "--- TRANSCRIPT ---" in prompt
    assert "0:05.0 - a price of forty dollars" in prompt


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
