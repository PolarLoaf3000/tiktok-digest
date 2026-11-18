"""Choose the source intervals that make up one digest.

Pure functions only. Nothing here reads a file, speaks HTTP, or calls
ffmpeg, so the fit rules can be tested without a media toolchain and the
request path can reject an impossible cut before it writes a video.
"""

from __future__ import annotations

import math
import random

from app.errors import ClipError

# The file we encode is 30 fps. A fragment shorter than one frame cannot
# remain its own shot: the encoder drops it or merges it into a neighbor,
# and the result would no longer contain the requested number of fragments.
# Counting in frames keeps the sum exact at that rate. render.py uses the
# same rate so the timestamps we return survive the encode.
OUTPUT_FPS = 30


def select_intervals(
    source_duration: float,
    target_duration: float,
    fragment_count: int,
    *,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Return ``fragment_count`` non-overlapping ``(start, end)`` intervals.

    Times are seconds from the start of the source. Lengths sum to the
    frame-aligned target duration. The list order is the random selection
    order, not chronological source order: fragments are concatenated in the
    order they were chosen.

    Pass ``rng`` to replay a layout. Without one, the generator is seeded
    from the operating system, so the same settings are free to differ on
    every call.
    """

    source_frames = _to_frames(source_duration, "Source")
    target_frames = _to_frames(target_duration, "Digest")
    count = _require_count(fragment_count)

    if target_frames < 1:
        raise ClipError(
            "Digest duration must cover at least one frame "
            f"({1 / OUTPUT_FPS:.3f}s)."
        )

    # Compare frames, not the raw floats. Probe durations wobble by less
    # than a frame, and the file we can actually emit is an integer number
    # of frames. A source that covers those frames is long enough.
    if source_frames < target_frames:
        raise ClipError(
            "Source is shorter than the requested digest: "
            f"{source_duration:.3f}s source, {target_duration:.3f}s requested."
        )

    # Each fragment needs its own frame. More fragments than frames in the
    # digest would force a zero-length cut, which overlaps its neighbors and
    # disappears in the encoded file.
    if target_frames < count:
        raise ClipError(
            "Cannot cut "
            f"{count} non-overlapping fragments for a "
            f"{target_frames / OUTPUT_FPS:.3f}s digest. Each fragment needs "
            f"its own frame ({1 / OUTPUT_FPS:.3f}s); {count} of them would "
            "run longer than the digest and could only be produced by "
            "overlapping or dropping a cut."
        )

    lengths = _split_frames(target_frames, count)
    if sum(lengths) > source_frames or any(length < 1 for length in lengths):
        # Placement assumes every length is a positive run of frames that
        # fits in the source. Reaching this means the checks above drifted,
        # and continuing would emit overlapping timestamps.
        raise ClipError(
            "Cannot cut "
            f"{count} non-overlapping fragments from a "
            f"{source_frames / OUTPUT_FPS:.3f}s source."
        )

    generator = rng if rng is not None else random.Random()
    placed = _place_frames(source_frames, lengths, generator)
    return [(start / OUTPUT_FPS, end / OUTPUT_FPS) for start, end in placed]


def _to_frames(seconds: float, label: str) -> int:
    # bool is a subclass of int, so a caller can pass True where a duration
    # is expected and it would count as one second. Reject that before the
    # numeric checks, which would otherwise accept it.
    if isinstance(seconds, bool) or not math.isfinite(seconds):
        raise ClipError(f"{label} duration must be a finite number of seconds.")
    if seconds <= 0:
        raise ClipError(f"{label} duration must be positive.")
    # Half-up, with a tiny bias so a value that lands just under an integer
    # because of binary fractions (1.1 * 30 == 32.999999999) still rounds to
    # the frame the caller asked for.
    # A duration around 1e307 is still a finite float, but scaling it into
    # frames overflows to infinity. int() then raises OverflowError, and the
    # request handler only turns ClipError into a client error — anything
    # else becomes a 500. Reject the overflow as a duration we cannot cut.
    scaled = seconds * OUTPUT_FPS + 0.5 + 1e-9
    if not math.isfinite(scaled):
        raise ClipError(f"{label} duration is too large to cut into frames.")
    return int(math.floor(scaled))


def _require_count(fragment_count: int) -> int:
    # bool is an int subclass; True would otherwise slip through as one
    # fragment and False as zero, which is not a count the client sent.
    if isinstance(fragment_count, bool):
        raise ClipError("Fragment count must be a positive integer.")
    if fragment_count < 1:
        raise ClipError("Fragment count must be a positive integer.")
    return fragment_count


def _split_frames(total: int, count: int) -> list[int]:
    """Split ``total`` into ``count`` positive lengths that sum to it.

    Lengths differ by at most one frame. An even split is only exact when
    the total is divisible by the count; giving the spare frames to the
    earliest fragments keeps the sum equal to the digest instead of drifting
    by a fraction of a frame.
    """

    base, extra = divmod(total, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


def _place_frames(
    source_frames: int,
    lengths: list[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Lay lengths on the timeline with random gaps, then shuffle play order.

    Picking starts and retrying on overlap can fail after a layout is still
    possible, and the API would then reject a request it could have served.
    The gaps consume exactly the unused frames, so every request that passed
    the fit checks produces a layout.

    Playback order is shuffled after the geometry is fixed. Overlap does not
    depend on list order; the shuffle is what makes concatenation follow
    selection order rather than left-to-right source order.
    """

    leftover = source_frames - sum(lengths)
    gaps = _random_gaps(leftover, len(lengths) + 1, rng)

    intervals: list[tuple[int, int]] = []
    cursor = gaps[0]
    for index, length in enumerate(lengths):
        start = cursor
        end = start + length
        intervals.append((start, end))
        cursor = end + gaps[index + 1]

    rng.shuffle(intervals)
    return intervals


def _random_gaps(leftover: int, gap_count: int, rng: random.Random) -> list[int]:
    """Split ``leftover`` frames into ``gap_count`` non-negative integers.

    Uniform over compositions via stars and bars. Drawing float weights and
    then repairing the sum biases the longer gaps toward whichever slots
    receive the leftover units; integer cut points do not need a repair.
    """

    if gap_count < 1:
        raise ClipError("Cannot place fragments without a timeline.")
    if gap_count == 1:
        return [leftover]
    if leftover == 0:
        return [0] * gap_count

    # leftover identical frames plus (gap_count - 1) bars.
    slots = leftover + gap_count - 1
    cuts = sorted(rng.sample(range(slots), gap_count - 1))
    gaps: list[int] = []
    previous = -1
    for cut in cuts:
        gaps.append(cut - previous - 1)
        previous = cut
    gaps.append(slots - previous - 1)
    return gaps
