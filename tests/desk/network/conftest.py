"""Tests that talk to a real exchange: collected only with ``NT_DESK_NETWORK=1``."""
from __future__ import annotations

import os

collect_ignore_glob = []
if os.environ.get("NT_DESK_NETWORK") != "1":
    collect_ignore_glob = ["test_*.py"]
