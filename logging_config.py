"""
Central logging configuration for the pipeline.

Every module below just does `logger = logging.getLogger(__name__)` and
logs normally — that alone does NOT make logs appear anywhere useful,
because without a configured handler, Python's logging module falls back
to the "handler of last resort" (WARNING+ only, no formatting, easy to
miss). This module is the ONE place that actually calls
`logging.basicConfig(...)`, and every process entry point (the CLI,
backend/api.py) calls `configure_logging()` once, at startup, before
anything else runs.

Level is controlled by the LOG_LEVEL env var (default INFO) so it can be
turned down to DEBUG in development or up to WARNING in production
without touching code — see .env.example.
"""

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent — safe to call more than once (e.g. if both the CLI and
    a module it imports call it); only the first call takes effect."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        # Bad/unknown LOG_LEVEL value — don't silently ignore it, but
        # don't crash the app over a typo'd env var either.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Invalid LOG_LEVEL=%r, defaulting to INFO", level_name
        )
        _CONFIGURED = True
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Third-party libraries that log verbosely at DEBUG/INFO and would
    # otherwise drown out the pipeline's own logs when LOG_LEVEL=DEBUG.
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))

    _CONFIGURED = True
