from __future__ import annotations

import os


def apply_opencv_videoio_env() -> None:
    """
    OpenCV 4.x can select different camera backends. On some systems, the OBSENSOR
    backend may be attempted and can produce confusing "Camera index out of range"
    errors even for normal V4L2 devices.

    This function sets env vars to reduce those surprises.
    """

    # Disable obsensor backend unless explicitly enabled.
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

    # Optional debug (uncomment when needed):
    # os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "1")


