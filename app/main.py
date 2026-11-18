"""HTTP boundary for the digest.

The handler writes the upload to a path, then calls probe, select, and
render. It does not choose timestamps or build a filter graph. ClipError
becomes a client error and no MP4 is returned: a failed request must not
look like a shorter digest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.errors import ClipError
from app.render import probe_duration, render_digest
from app.select import select_intervals

app = FastAPI(title="TikTok digest")


@app.post("/digest")
async def digest(
    file: Annotated[UploadFile, File()],
    duration: Annotated[float, Form()],
    fragments: Annotated[int, Form()],
) -> Response:
    """Cut one MP4 into a vertical digest.

    ``duration`` is the finished video in seconds. ``fragments`` is how many
    non-overlapping pieces it is built from. The response is one MP4, or a
    400 with a plain explanation when the source cannot supply that cut.
    """

    _reject_bad_name(file.filename)

    with tempfile.TemporaryDirectory(prefix="tiktok-digest-") as directory:
        # ffprobe and ffmpeg need a filesystem path. The upload is a stream,
        # and its client-supplied name is never joined onto a directory:
        # that name is untrusted and only the suffix was checked above.
        source = Path(directory) / "source.mp4"
        written = await _write_upload(file, source)
        if written == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        try:
            data = _build(source, duration, fragments)
        except ClipError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="digest.mp4"'},
    )


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


def _reject_bad_name(filename: str | None) -> None:
    # Suffix, not the content type. Browsers often send octet-stream for a
    # real MP4, while a .txt upload should fail before ffmpeg starts.
    # A renamed non-MP4 still fails when the probe cannot read a duration.
    if Path(filename or "").suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")


async def _write_upload(file: UploadFile, dest: Path) -> int:
    # Chunked so a long source is not held in memory beside the encoded
    # digest. The temp directory, not this function, deletes the file,
    # including when a later step raises.
    written = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
    finally:
        await file.close()
    return written


def _build(source: Path, duration: float, fragments: int) -> bytes:
    intervals = select_intervals(probe_duration(source), duration, fragments)
    return render_digest(source, intervals)


if __name__ == "__main__":
    main()
