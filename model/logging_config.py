"""
Centralised logging configuration.

One entrypoint — `setup_logging()` — that the Flask app calls at startup.
Every module in the project just does `logger = logging.getLogger(__name__)`
at the top; their output goes through the format + level configured here.

Environment variables
---------------------
LOG_LEVEL        — DEBUG / INFO / WARNING / ERROR (default: INFO)
LOG_TIMESTAMPS   — '1' (default) to include timestamps; '0' for a plain format
TRANSFORMERS_LOG — override the transformers library log level (default: WARNING)
"""
import logging
import os
import sys


_CONFIGURED = False


def setup_logging():
    """Configure root logger, then quieten third-party loggers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if os.getenv("LOG_TIMESTAMPS", "1") == "1":
        fmt = "[%(asctime)s] %(levelname)-5s %(name)s :: %(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(levelname)-5s %(name)s :: %(message)s"
        datefmt = None

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    # Wipe any default handlers (e.g. ones Flask/werkzeug may have installed)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third parties
    tf_level = os.getenv("TRANSFORMERS_LOG", "WARNING").upper()
    logging.getLogger("transformers").setLevel(getattr(logging, tf_level, logging.WARNING))
    # Werkzeug's request log is useful but verbose — keep at WARNING so our
    # own Flask request hooks do the talking.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # urllib3 + huggingface_hub chatter during model downloads
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured at level %s (transformers=%s)", level_name, tf_level,
    )
