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
async def test_full_external_origin_session_key_and_capture_are_forwarded(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SERVER_URL", "http://bridge.internal:18080")
    monkeypatch.setenv("AGENT_SERVER_AUTH_TOKEN", "test-bridge-key")
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_BASE_URL", "https://miles-model.example.com/")
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "ignored.example.com")
    monkeypatch.setenv("MILES_SESSION_API_KEY", "test-session-key")
    seen = {}

    async def fake_post(url, payload, headers=None, **kwargs):
        seen.update(url=url, payload=payload, headers=headers)
        return {
            "reward": 1.0,
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "capture": {"status": "complete"},
        }

    monkeypatch.setattr(agent_function, "post", fake_post)
    result = await agent_function.run(
        base_url="http://10.0.0.1:30000/sessions/session-123",
        prompt="fix it",
        request_kwargs={"max_tokens": 1024},
        metadata={
            "instance_id": "task-1",
            "session_server_id": "10.0.0.1:30000",
            "session_server_instance_id": "generation-1",
            "dataset_name": "tb2",
        },
    )

    assert seen["url"] == "http://bridge.internal:18080/run"
    assert seen["headers"] == {"Authorization": "Bearer test-bridge-key"}
    assert seen["payload"]["base_url"] == "https://miles-model.example.com/sessions/session-123/v1"
    assert seen["payload"]["session_server_id"] == "https://miles-model.example.com"
    assert seen["payload"]["api_key"] == "test-session-key"
    assert seen["payload"]["capture_context"] == {"dataset_name": "tb2"}
    assert result["trial_id"] == "trial-1"
    assert result["capture"]["status"] == "complete"


@pytest.mark.asyncio
async def test_legacy_external_host_preserves_symmetric_port(monkeypatch) -> None:
    monkeypatch.delenv("MILES_ROUTER_EXTERNAL_BASE_URL", raising=False)
    monkeypatch.delenv("MILES_SESSION_API_KEY", raising=False)
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "203.0.113.10")
    seen = {}

    async def fake_post(url, payload, headers=None, **kwargs):
        seen.update(payload=payload)
        return {}

    monkeypatch.setattr(agent_function, "post", fake_post)
    await agent_function.run(
        base_url="http://10.0.0.1:32001/sessions/session-123",
        prompt="fix it",
        metadata={"session_server_id": "10.0.0.1:32001"},
    )

    assert seen["payload"]["base_url"] == "http://203.0.113.10:32001/sessions/session-123/v1"
    assert seen["payload"]["session_server_id"] == "203.0.113.10:32001"
    assert seen["payload"]["api_key"] == "dummy"


@pytest.mark.parametrize(
    "value",
    [
        "miles-model.example.com",
        "ftp://miles-model.example.com",
        "https://user:password@miles-model.example.com",
        "https://miles-model.example.com/prefix",
        "https://miles-model.example.com?query=yes",
    ],
)
def test_external_origin_rejects_unsafe_or_ambiguous_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_BASE_URL", value)
    with pytest.raises(ValueError, match="MILES_ROUTER_EXTERNAL_BASE_URL"):
        agent_function._external_origin()


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
            "dataset_name": "swebench-verified",
            "osmosis_mix_id": "customer-mix-a",
            "prompt": "must not be copied into capture context",
        },
    )

    assert seen["url"] == "http://bridge.internal:18080/run"
    assert seen["headers"] == {"Authorization": "Bearer bridge-secret"}
    assert seen["payload"]["api_key"] == "session-secret"
    assert seen["payload"]["rollout_id"] == 17
    assert seen["payload"]["capture_context"] == {
        "dataset_name": "swebench-verified",
        "osmosis_mix_id": "customer-mix-a",
    }
    assert result["trial_id"] == "trial-1"
    assert result["step_index"] == 17
    assert result["capture"]["status"] == "complete"
