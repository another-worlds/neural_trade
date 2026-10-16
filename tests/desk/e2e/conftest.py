"""Browser tests: collected only with ``NT_DESK_E2E=1`` and Playwright installed (the CI ``desk`` job)."""
from __future__ import annotations

import importlib.util
import os

collect_ignore_glob = []
if os.environ.get("NT_DESK_E2E") != "1" or importlib.util.find_spec("playwright") is None:
    collect_ignore_glob = ["test_*.py"]
