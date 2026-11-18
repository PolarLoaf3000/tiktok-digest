"""The only exception the digest is allowed to raise on purpose.

A failed request must not come back as a shorter or truncated MP4. Callers
map this to a client error and discard any partial encode. Selection and
rendering both raise it so there is one failure channel instead of a family
of exception types to catch in the wrong order.
"""


class ClipError(Exception):
    """The source and the requested settings cannot produce this digest."""
