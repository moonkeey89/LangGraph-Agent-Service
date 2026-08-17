"""Current-time business capability."""

from datetime import datetime


def get_current_time() -> str:
    """Return the current local time."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
