from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "examples/experimental/swe-agent-v2/swe_agent_function.py"
_SPEC = importlib.util.spec_from_file_location("swe_agent_function", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent_function = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = agent_function

# Keep the HTTP contract test runnable in the small Harbor-only environment.
_HTTP_UTILS_NAME = "miles.utils.http_utils"
_ORIGINAL_HTTP_UTILS = sys.modules.get(_HTTP_UTILS_NAME)
_HTTP_UTILS_STUB = ModuleType(_HTTP_UTILS_NAME)


async def _unpatched_post(*args, **kwargs):
    raise AssertionError("test must monkeypatch swe_agent_function.post")


_HTTP_UTILS_STUB.post = _unpatched_post
sys.modules[_HTTP_UTILS_NAME] = _HTTP_UTILS_STUB
_SPEC.loader.exec_module(agent_function)
if _ORIGINAL_HTTP_UTILS is None:
    del sys.modules[_HTTP_UTILS_NAME]
else:
    sys.modules[_HTTP_UTILS_NAME] = _ORIGINAL_HTTP_UTILS


@pytest.mark.asyncio
async def test_bridge_auth_correlation_and_capture_are_forwarded(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SERVER_URL", "http://bridge.internal:18080")
    monkeypatch.setenv("AGENT_SERVER_AUTH_TOKEN", "bridge-secret")
    monkeypatch.setenv("MILES_SESSION_API_KEY", "session-secret")
    monkeypatch.setenv("AGENT_SERVER_TIMEOUT_SEC", "123")
    seen = {}

    async def fake_post(url, payload, headers=None, **kwargs):
        seen.update(url=url, payload=payload, headers=headers)
        return {
            "reward": 0.75,
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "trial_name": "task__abc",
            "task_id": "task",
            "sandbox_id": "task__abc__env",
            "provider_sandbox_id": "daytona-1",
            "session_id": "session-123",
            "run_id": "run-1",
            "rollout_id": 17,
            "sample_id": 41,
            "group_id": 9,
            "step_index": 17,
            "capture": {"status": "complete", "staged_trial_dir": "/captures/trial-1/trial"},
        }

    monkeypatch.setattr(agent_function, "post", fake_post)
    result = await agent_function.run(
        base_url="http://10.0.0.1:30000/sessions/session-123",
        prompt="fix it",
        metadata={
            "instance_id": "task",
            "run_id": "run-1",
            "rollout_id": 17,
            "sample_id": 41,
            "group_id": 9,
        },
    )

    assert seen["url"] == "http://bridge.internal:18080/run"
    assert seen["headers"] == {"Authorization": "Bearer bridge-secret"}
    assert seen["payload"]["api_key"] == "session-secret"
    assert seen["payload"]["rollout_id"] == 17
    assert result["trial_id"] == "trial-1"
    assert result["step_index"] == 17
    assert result["capture"]["status"] == "complete"
