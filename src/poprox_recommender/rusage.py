"""
Support code for tracking and reporting resource usage.
"""


def pretty_time(seconds: float) -> str:
    """
    Pretty-print a time, such as total CPU time.
    """
    if seconds < 1:
        return "{: 0.0f}ms".format(seconds * 1000)
    elif seconds > 60 * 60:
        h, m = divmod(seconds, 60 * 60)
        m, s = divmod(m, 60)
        return "{:0.0f}h{:0.0f}m{:0.2f}s".format(h, m, s)
    elif seconds > 60:
        m, s = divmod(seconds, 60)
        return "{:0.0f}m{:0.2f}s".format(m, s)
    else:
        return "{:0.2f}s".format(seconds)
