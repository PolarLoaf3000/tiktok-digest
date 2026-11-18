"""Turn chosen intervals into one vertical MP4.

ffmpeg only. The timestamps are already decided; this module cuts each one
with its audio, fits the picture to 9:16, and concatenates in the given
order. It does not pick new ranges when a cut fails — that would hide an
impossible request inside a shorter file.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import ffmpeg

from app.errors import ClipError
from app.select import OUTPUT_FPS

# 1080x1920 is the 9:16 size TikTok publishes. Width and height stay even
# because the pixel format below subsamples chroma in 2x2 blocks.
WIDTH = 1080
HEIGHT = 1920

# Pad, don't crop. Nothing in this app looks at the picture, so a center
# crop could remove the only subject. force_divisible_by keeps the scaled
# frame even before the pad. setsar clears a non-square pixel aspect from
# the phone recording; leaving it would stretch the 9:16 frame in players.
# fps is the same rate selection counted in, so the file duration stays on
# the intervals we promised instead of drifting to another frame grid.
_VIDEO_FILTER = (
    f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:"
    "force_divisible_by=2,"
    f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
    "setsar=1,"
    f"fps={OUTPUT_FPS},"
    "format=yuv420p"
)


def probe_duration(source: Path) -> float:
    """Return how many seconds of ``source`` can be cut.

    The container duration, not a single stream's, is the timeline when
    the file is complete: a stream can report a shorter value when its
    timestamps start late. That number is a lie once ``mdat`` is cut out
    from under an intact ``moov``. The index still names the old duration,
    and a later encode exits 0 with a shorter picture. The last video
    packet still on disk is the limit selection has to hear about, or it
    will place a clip in samples that are not there.
    """

    info = _probe(source)
    return _limit_to_readable_video(source, info, _declared_duration(info))


def render_digest(
    source: Path,
    intervals: Sequence[tuple[float, float]],
) -> bytes:
    """Cut ``intervals`` from ``source`` and return one MP4.

    Intervals are used in list order. An empty or reversed interval is
    refused here because ffmpeg would still exit successfully and write a
    file with no samples, which the client would treat as the digest. A
    file shorter than those intervals is refused for the same reason: a
    truncated source still makes ffmpeg exit 0.
    """

    checked = _require_intervals(intervals)
    has_audio = _source_has_audio(source)

    with tempfile.TemporaryDirectory(prefix="tiktok-digest-") as directory:
        root = Path(directory)
        clips: list[Path] = []
        for index, (start, end) in enumerate(checked):
            clip = root / f"clip-{index:04d}.mp4"
            _cut_clip(source, start, end, clip, has_audio=has_audio)
            clips.append(clip)

        output = root / "digest.mp4"
        _concat(clips, output)
        # Check while the file still exists. The temp directory deletes it
        # on the way out, including when this raises, so a short encode is
        # never what the handler returns.
        _reject_short_output(output, checked)
        data = output.read_bytes()

    # A non-zero ffmpeg status is not enough: a truncated mux can still
    # exit 0 on some builds. An MP4 we are willing to return starts with an
    # ftyp box; anything else is an error body, not a video.
    if len(data) < 8 or data[4:8] != b"ftyp":
        raise ClipError("ffmpeg did not produce a readable MP4.")
    return data


def _require_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        raise ClipError("Cannot render a digest with no fragments.")
    checked: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            raise ClipError(
                "Cannot render a fragment that does not move forward in time."
            )
        checked.append((float(start), float(end)))
    return checked


def _declared_duration(info: Mapping[str, object]) -> float:
    streams = _streams(info)
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise ClipError("The file has no video stream.")

    container = info.get("format")
    if isinstance(container, dict):
        duration = _read_duration(cast(Mapping[str, object], container))
        if duration is not None:
            return duration

    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        duration = _read_duration(stream)
        if duration is not None:
            return duration

    raise ClipError(
        "Could not read a duration from this file. It needs to be a valid MP4."
    )


def _limit_to_readable_video(
    source: Path,
    info: Mapping[str, object],
    declared: float,
) -> float:
    """Cap ``declared`` at the last video packet whose bytes are in the file.

    ``ffprobe`` still reports the indexed frame count and the container
    duration after ``mdat`` is cut in half. The packet list stops at the
    cut. When that list is shorter than the index, or the index has no
    frame count and the timeline stops early, the declared duration would
    let selection ask for samples the encoder cannot produce.
    """

    stream = _longest_video(_streams(info))
    if stream is None:
        return declared
    index = stream.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return declared

    packets = [
        packet
        for packet in _packets(_probe(source, with_packets=True))
        if packet.get("stream_index") == index
    ]
    indexed = _optional_int(stream.get("nb_frames"))
    if indexed is not None and indexed > 0 and not packets:
        raise ClipError(
            "The file has no readable video. It needs to be a complete MP4."
        )
    readable_end = _last_packet_end(packets)
    if readable_end is None:
        return declared

    samples_missing = indexed is not None and len(packets) < indexed
    # No frame count to compare. A one-frame gap is normal when the
    # container follows a slightly longer audio track; a truncated file
    # stops well before that.
    timeline_cut_short = indexed is None and declared - readable_end > 1 / OUTPUT_FPS
    if (samples_missing or timeline_cut_short) and readable_end + (
        0.5 / OUTPUT_FPS
    ) < declared:
        return readable_end
    return declared


def _longest_video(
    streams: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return the video stream with the most indexed frames.

    A poster image is its own video stream and is often the first one.
    Measuring that single frame would make a complete recording look
    truncated. The long stream is the one the cut reads.
    """

    best: Mapping[str, object] | None = None
    best_frames = -1
    for stream in streams:
        if stream.get("codec_type") != "video" or _attached_pic(stream):
            continue
        frames = _optional_int(stream.get("nb_frames"))
        count = frames if frames is not None else 0
        if count >= best_frames:
            best = stream
            best_frames = count
    return best


def _attached_pic(stream: Mapping[str, object]) -> bool:
    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    flag = cast(Mapping[str, object], disposition).get("attached_pic")
    return flag == 1 or flag == "1"


def _reject_short_output(
    path: Path,
    intervals: Sequence[tuple[float, float]],
) -> None:
    """Refuse an encode that did not reach the intervals it was given.

    ffmpeg exits 0 when the source runs out of samples mid-cut and writes
    whatever prefix it could decode. That file plays, and it is shorter
    than the digest the client asked for. A finished encode lands on the
    same frame grid as the intervals, so more than half a frame short is
    a failed cut, not a timestamp wobble.
    """

    expected = sum(end - start for start, end in intervals)
    stream = _longest_video(_streams(_probe(path)))
    actual = _read_duration(stream) if stream is not None else None
    if actual is None or expected - actual > 0.5 / OUTPUT_FPS:
        shown = 0.0 if actual is None else actual
        raise ClipError(
            "Source is shorter than the requested digest: "
            f"the readable video is {shown:.3f}s, {expected:.3f}s requested."
        )


def _probe(source: Path, *, with_packets: bool = False) -> Mapping[str, object]:
    try:
        if with_packets:
            # Demux only. A truncated mdat makes the demuxer stop; the
            # packets in the report are the ones still in the file, which
            # is the comparison the container duration cannot make.
            info = ffmpeg.probe(str(source), show_packets=None)
        else:
            info = ffmpeg.probe(str(source))
    except FileNotFoundError as exc:
        raise ClipError(
            "ffprobe is not available, so the source duration cannot be read."
        ) from exc
    except ffmpeg.Error as exc:
        raise ClipError(_ffmpeg_message("read the source", exc)) from exc
    if not isinstance(info, dict):
        raise ClipError("Could not read the source video.")
    return cast(Mapping[str, object], info)


def _streams(info: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = info.get("streams")
    if not isinstance(raw, list):
        return []
    streams: list[Mapping[str, object]] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            streams.append(cast(Mapping[str, object], item))
    return streams


def _packets(info: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = info.get("packets")
    if not isinstance(raw, list):
        return []
    packets: list[Mapping[str, object]] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            packets.append(cast(Mapping[str, object], item))
    return packets


def _read_duration(item: Mapping[str, object]) -> float | None:
    raw = item.get("duration")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value > 0 and math.isfinite(value):
        return value
    return None


def _finite_float(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _optional_int(raw: object) -> int | None:
    # bool is an int subclass. A flag must not become a frame count.
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return int(raw, 10)
    except ValueError:
        return None


def _last_packet_end(packets: Sequence[Mapping[str, object]]) -> float | None:
    end: float | None = None
    for packet in packets:
        pts = _finite_float(packet.get("pts_time"))
        if pts is None:
            continue
        duration = _finite_float(packet.get("duration_time"))
        if duration is not None and duration > 0:
            pts += duration
        if end is None or pts > end:
            end = pts
    return end


def _source_has_audio(source: Path) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in _streams(_probe(source)))


def _cut_clip(
    source: Path,
    start: float,
    end: float,
    dest: Path,
    *,
    has_audio: bool,
) -> None:
    # -ss on the input seeks near the cut, and because we re-encode,
    # ffmpeg's default accurate seek discards the partial GOP before
    # `start`. Output -t then keeps the fragment the length we selected.
    # Stream copy cannot do that: it can only cut on keyframes, which would
    # change the duration, and it cannot scale or pad.
    inp = ffmpeg.input(str(source), ss=start)
    output_args: dict[str, object] = {
        "t": end - start,
        "vf": _VIDEO_FILTER,
        "vcodec": "libx264",
        "preset": "veryfast",
        "crf": 23,
        "pix_fmt": "yuv420p",
    }
    if has_audio:
        # Every fragment is encoded with the same layout so the concat
        # demuxer can copy packets. A mono source and a stereo source in
        # the same digest would otherwise refuse to join.
        output_args["acodec"] = "aac"
        output_args["audio_bitrate"] = "128k"
        output_args["ar"] = 48000
        output_args["ac"] = 2
    else:
        # Do not invent a silent track. The source had no audio to preserve,
        # and a silent track would make the digest a different recording.
        output_args["an"] = None

    stream = ffmpeg.output(inp, str(dest), **output_args)
    _run(stream, "cut a fragment")


def _concat(clips: Sequence[Path], dest: Path) -> None:
    # One code path for one fragment and for many. A special case that just
    # renamed the single clip would skip the faststart pass the multi-clip
    # file gets, so a one-fragment digest would be slower to start playing.
    list_file = dest.with_suffix(".txt")
    lines = [f"file '{_escape_concat_path(clip)}'" for clip in clips]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    joined = ffmpeg.input(str(list_file), format="concat", safe=0)
    # Copy, because the fragments were already encoded with matching
    # parameters. Re-encoding the join would not change the picture and
    # would add another generation of loss. faststart moves the moov atom
    # forward so a client can begin playback without the whole file.
    stream = ffmpeg.output(joined, str(dest), c="copy", movflags="+faststart")
    _run(stream, "join the fragments")


def _escape_concat_path(path: Path) -> str:
    # The concat demuxer treats the list file as a tiny script. Quotes in a
    # path would close the file token early. Temp names do not contain
    # quotes today; the escape is what keeps a later path change from
    # becoming a second, unintended input.
    return str(path).replace("'", r"'\''")


def _run(stream: ffmpeg.Stream, action: str) -> None:
    try:
        ffmpeg.run(
            stream,
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True,
        )
    except FileNotFoundError as exc:
        raise ClipError(
            "ffmpeg is not available, so the digest cannot be encoded."
        ) from exc
    except ffmpeg.Error as exc:
        raise ClipError(_ffmpeg_message(action, exc)) from exc


def _ffmpeg_message(action: str, exc: ffmpeg.Error) -> str:
    # The full stderr is a banner plus a progress log. The last non-empty
    # line is the reason ffmpeg stopped, which is the part a client can act on.
    stderr = exc.stderr
    text = ""
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"ffmpeg could not {action}."
    return f"ffmpeg could not {action}: {lines[-1]}"
