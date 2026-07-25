"""Pytest config for the Miles-Harbor capture fixture.

Tier 1 (parity logic) runs anywhere. Tier 2 (real oracle trial) is marked
``harbor`` and auto-skips unless Docker + the ``harbor`` package are present —
i.e. it runs on the Nebius/agent-env host, not a laptop.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "harbor: needs Docker + the harbor package (runs a real oracle sandbox trial)",
    )


def _harbor_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        import harbor  # noqa: F401
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _harbor_available():
        return
    skip = pytest.mark.skip(reason="harbor tier: needs Docker + harbor package (run on the agent-env host)")
    for item in items:
        if "harbor" in item.keywords:
            item.add_marker(skip)
