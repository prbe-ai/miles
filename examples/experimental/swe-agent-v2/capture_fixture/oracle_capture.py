"""Run a real Harbor trial through the Miles bridge with NO model or GPU.

Uses Harbor's ``oracle`` agent: it applies the task's ``solve.sh`` *as the agent*,
so ``AGENT_START``/``AGENT_END`` fire (our sandbox-state capture runs), the
verifier produces a real reward, and nothing calls a model. Point it at any
sandbox (Harbor task dir) to exercise the capture pipeline against that sandbox.

Importing this module needs nothing special; *running* a capture needs Docker +
the ``harbor`` package (i.e. the Nebius/agent-env host), because it starts a real
sandbox container.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_parity  # noqa: E402  (sibling module in the fixture dir)

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
_BRIDGE_PATH = _EXAMPLE_DIR / "public_harbor_server.py"


def load_bridge():
    """Load public_harbor_server.py as a module (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location("public_harbor_server_fixture", _BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_oracle_settings(bridge, *, tasks_dir: Path, trials_dir: Path, capture_dir: Path):
    """Bridge Settings for a model-free oracle capture: shadow capture + sandbox state."""
    return bridge.Settings(
        tasks_dir=Path(tasks_dir),
        trials_dir=Path(trials_dir),
        capture_dir=Path(capture_dir),
        capture_mode="shadow",
        sandbox_state=True,
        environment_type="docker",
        delete_environments=True,
        # The oracle never calls back, but the bridge validates the URL host.
        allowed_callback_hosts=frozenset({"localhost", "127.0.0.1", "::1"}),
    )


async def run_oracle_capture(
    bridge,
    settings,
    task_name: str,
    *,
    step_index: int = 0,
    rollout_id: int = 0,
    sample_id: int = 0,
    group_id: int = 0,
    session_id: str = "fixture-session",
    run_id: str | None = None,
) -> Any:
    """Run one oracle trial for ``task_name`` and return the bridge RunResponse.

    The response carries reward, exit_status, agent_metrics, the trial dir, and
    ``capture`` (with ``sandbox_state``). The ``probe.sandbox-state/1`` bundle is
    staged under ``settings.capture_dir`` for the export/watch pipeline.
    """
    payload = rollout_parity.build_oracle_run_request(
        task_name,
        step_index=step_index,
        rollout_id=rollout_id,
        sample_id=sample_id,
        group_id=group_id,
        session_id=session_id,
        run_id=run_id,
    )
    request = bridge.RunRequest(**payload)
    return await bridge.run_public_harbor_trial(request, settings)


def _solution_paths(task_dir: Path) -> tuple[list[str], list[str]]:
    """Best-effort ``(writes, deletes)`` absolute paths out of a task's solve.sh.

    Parses ``> /path`` / ``cp ... /path`` style tokens for writes and ``rm``
    command targets for deletes. Not a shell interpreter — just enough to
    assert the sandbox delta captured them.
    """
    solve = None
    for candidate in (task_dir / "solution" / "solve.sh", task_dir / "solution" / "solve.bat"):
        if candidate.is_file():
            solve = candidate
            break
    if solve is None:
        return [], []
    writes: list[str] = []
    deletes: list[str] = []
    for line in solve.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        tokens = line.split()
        target = deletes if tokens[0] == "rm" else writes
        for token in tokens:
            if token.startswith("/") and "." in token.rsplit("/", 1)[-1]:
                target.append(token)
    return sorted(set(writes)), sorted(set(deletes))


def solution_writes(task_dir: Path) -> list[str]:
    """Absolute paths solve.sh writes — expected present in the end manifest."""
    return _solution_paths(task_dir)[0]


def solution_deletes(task_dir: Path) -> list[str]:
    """Absolute paths solve.sh deletes.

    probe.sandbox-state/1 stores no tombstones: a deletion is derived (path in
    the begin manifest, absent from the end manifest), so these must NOT be
    asserted against the end manifest.
    """
    return _solution_paths(task_dir)[1]
