from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "examples/experimental/swe-agent-v2/public_harbor_server.py"
_SPEC = importlib.util.spec_from_file_location("public_harbor_server", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


def _task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "tasks" / "hello-world"
    task.mkdir(parents=True)
    return task


def test_resolve_task_path_rejects_traversal(tmp_path: Path) -> None:
    _task_dir(tmp_path)
    with pytest.raises(ValueError, match="Invalid instance_id"):
        server.resolve_task_path(tmp_path / "tasks", "../secret")


def test_callback_url_requires_allowlist(tmp_path: Path) -> None:
    settings = server.Settings(tasks_dir=tmp_path, allowed_callback_hosts=frozenset({"miles.internal"}))
    server.validate_callback_url("http://miles.internal:30000/sessions/abc/v1", settings)
    with pytest.raises(ValueError, match="not allowed"):
        server.validate_callback_url("http://metadata.internal/latest", settings)


def test_session_server_id_requires_allowed_origin(tmp_path: Path) -> None:
    settings = server.Settings(tasks_dir=tmp_path, allowed_callback_hosts=frozenset({"miles.internal"}))
    server.validate_session_server_id("miles.internal:30000", settings)
    with pytest.raises(ValueError, match="not allowed"):
        server.validate_session_server_id("metadata.internal:80", settings)
    with pytest.raises(ValueError, match="without a path"):
        server.validate_session_server_id("http://miles.internal:30000/admin", settings)


def test_extract_session_id() -> None:
    assert server.extract_session_id("http://miles:30000/sessions/session-123/v1") == "session-123"
    assert server.extract_session_id("https://api.example.com/v1") is None


def test_build_mini_swe_agent_configuration() -> None:
    request = server.RunRequest(
        base_url="http://miles:30000/sessions/abc/v1",
        model="openai/model",
        instance_id="hello-world",
        sampling_params={"temperature": 0.8, "max_tokens": 4096},
    )
    env, kwargs = server.build_agent_configuration(request)
    assert env["OPENAI_BASE_URL"] == request.base_url
    assert env["MSWEA_API_KEY"] == "dummy"
    assert kwargs == {"max_tokens": 4096}


def test_build_terminus_configuration_forwards_sampling() -> None:
    request = server.RunRequest(
        base_url="http://miles:30000/sessions/abc/v1",
        model="openai/model",
        instance_id="hello-world",
        agent_name="terminus-2",
        max_seq_len=65_536,
        sampling_params={"temperature": 0.4, "max_tokens": 2048, "top_p": 0.9},
    )
    _, kwargs = server.build_agent_configuration(request)
    assert kwargs["api_base"] == request.base_url
    assert kwargs["temperature"] == 0.4
    assert kwargs["model_info"]["max_input_tokens"] == 65_536
    assert kwargs["llm_call_kwargs"] == {"max_tokens": 2048, "top_p": 0.9}


def test_normalize_trial_result() -> None:
    started = datetime.now(timezone.utc)
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1, "tests": 0.75}),
        exception_info=None,
        agent_result=SimpleNamespace(
            n_input_tokens=100,
            n_cache_tokens=20,
            n_output_tokens=30,
            cost_usd=0.0,
            metadata={"turns": 3},
        ),
        agent_execution=SimpleNamespace(started_at=started, finished_at=started + timedelta(seconds=2)),
        verifier=SimpleNamespace(started_at=started, finished_at=started + timedelta(seconds=1)),
    )
    response = server.normalize_trial_result(result)
    assert response.reward == 1.0
    assert response.exit_status == "Submitted"
    assert response.eval_report == {"reward": 1, "tests": 0.75}
    assert response.agent_metrics["turns"] == 3
    assert response.agent_metrics["agent_run_time"] == 2.0


@pytest.mark.asyncio
async def test_http_contract_and_auth(tmp_path: Path) -> None:
    _task_dir(tmp_path)
    settings = server.Settings(
        tasks_dir=tmp_path / "tasks",
        auth_token="test-token",
        allowed_callback_hosts=frozenset({"miles.internal"}),
    )
    seen = []

    async def fake_runner(request, actual_settings):
        seen.append((request, actual_settings))
        return server.RunResponse(reward=1.0, exit_status="Submitted")

    app = server.create_app(settings, fake_runner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        payload = {
            "base_url": "http://miles.internal:30000/sessions/abc/v1",
            "model": "openai/model",
            "instance_id": "hello-world",
        }
        unauthorized = await client.post("/run", json=payload)
        assert unauthorized.status_code == 401

        response = await client.post("/run", json=payload, headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200
        assert response.json()["reward"] == 1.0
        assert len(seen) == 1


@pytest.mark.asyncio
async def test_http_contract_rejects_unknown_task(tmp_path: Path) -> None:
    settings = server.Settings(tasks_dir=tmp_path / "tasks", allow_any_callback=True)

    async def should_not_run(request, actual_settings):
        raise AssertionError("runner should not be called")

    app = server.create_app(settings, should_not_run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/run",
            json={
                "base_url": "http://localhost:30000/sessions/abc/v1",
                "model": "openai/model",
                "instance_id": "missing",
            },
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_session_monitor_enforces_identity_and_token_limit(tmp_path: Path, monkeypatch) -> None:
    request = server.RunRequest(
        base_url="http://miles.internal:30000/sessions/session-123/v1",
        model="openai/model",
        instance_id="hello-world",
        max_seq_len=3,
        session_server_id="miles.internal:30000",
        session_server_instance_id="server-generation-1",
    )
    settings = server.Settings(tasks_dir=tmp_path, session_poll_interval_sec=0.001)
    requested_urls = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            requested_urls.append(url)
            if url.endswith("/health"):
                return httpx.Response(
                    200,
                    json={"status": "ok", "session_server_instance_id": "server-generation-1"},
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(
                200,
                json={"metadata": {"accumulated_token_ids": [1, 2, 3]}},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    observed = await server._poll_until_sequence_limit(request, settings)
    assert observed == 3
    assert requested_urls == [
        "http://miles.internal:30000/health",
        "http://miles.internal:30000/sessions/session-123",
    ]
