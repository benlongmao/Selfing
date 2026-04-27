#!/usr/bin/env python3
"""
DEPRECATED one-off helper: used during early S-44 bring-up to splice methods into
``memory_enhancer.py``. The in-repo ``MemoryEnhancer`` already includes the full
implementation — **do not run** this script against a tree you care about.

Kept only so old references do not 404; ``main()`` is a no-op with a log line.
"""

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    logger.warning(
        "add_methods.main() is a no-op: MemoryEnhancer already ships with "
        "background_process / get_statistics. Remove any CI calls to this script."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
