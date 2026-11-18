"""Encode rules that need a real MP4.

Selection never sees ffmpeg. These tests cover the part it cannot: a
container duration that outlives the bytes, and a complete file that still
comes back at the duration that was asked for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.errors import ClipError
from app.render import probe_duration, render_digest
from app.select import select_intervals


def test_truncated_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_source(source, 4.0)
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(_cut_mdat_in_half(source.read_bytes()))

    usable = probe_duration(truncated)
    assert usable < 3.0
    with pytest.raises(ClipError, match="shorter"):
        select_intervals(usable, 3.0, 1)
    # Even a caller that skips selection must not get a short file back.
    with pytest.raises(ClipError, match="shorter"):
        render_digest(truncated, [(0.0, 3.0)])


def test_complete_source_keeps_requested_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_source(source, 4.0)
    declared = probe_duration(source)
    assert declared == pytest.approx(4.0, abs=0.05)

    intervals = select_intervals(declared, 3.0, 1)
    output = tmp_path / "digest.mp4"
    output.write_bytes(render_digest(source, intervals))
    assert _video_duration(output) == pytest.approx(3.0, abs=0.02)


def _write_source(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=30:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _cut_mdat_in_half(data: bytes) -> bytes:
    """Keep moov, drop the second half of mdat.

    The index still describes the original timeline. The samples it names
    past the cut are gone, which is the file the digest must refuse.
    """

    offset = 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header = 8
        if size == 1 and offset + 16 <= len(data):
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        if size < header:
            break
        if box_type == b"mdat":
            payload = size - header
            return data[: offset + header + payload // 2]
        offset += size
    raise AssertionError("fixture mp4 has no mdat box")


def _video_duration(path: Path) -> float:
    import ffmpeg

    info = ffmpeg.probe(str(path))
    for stream in info["streams"]:
        if stream.get("codec_type") == "video":
            return float(stream["duration"])
    raise AssertionError("digest has no video stream")
