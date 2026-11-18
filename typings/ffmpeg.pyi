"""Types for the parts of ffmpeg-python this app calls.

The package ships no stubs. Without these, strict pyright treats every
fluent call as unknown and cannot check the render module.
"""

from typing import Any

class Error(Exception):
    stdout: bytes | str | None
    stderr: bytes | str | None

class Stream: ...

def probe(filename: str, cmd: str = ..., **kwargs: Any) -> Any: ...
def input(filename: str, **kwargs: Any) -> Stream: ...
def output(*streams_and_filename: Stream | str, **kwargs: Any) -> Stream: ...
def run(
    stream_spec: Stream,
    cmd: str = ...,
    capture_stdout: bool = ...,
    capture_stderr: bool = ...,
    input: str | bytes | None = ...,
    quiet: bool = ...,
    overwrite_output: bool = ...,
) -> tuple[bytes, bytes]: ...
