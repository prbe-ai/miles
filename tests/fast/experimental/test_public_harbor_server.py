from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _install_fake_probe(monkeypatch):
    """Install the SDK boundary double; SDK contract behavior is tested upstream."""
    seen = {}

    def stage_trial_export(trial_dir, destination, **kwargs):
        seen.update({"trial_dir": Path(trial_dir), "destination": Path(destination), **kwargs})
        root = Path(destination)
        staged_trial = root / "trial"
        root.mkdir(parents=True)
        shutil.copytree(trial_dir, staged_trial, symlinks=True)
        files = [
            {"path": str(path.relative_to(staged_trial)), "size_bytes": path.stat().st_size}
            for path in staged_trial.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
        manifest_path = root / "capture-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "files": files,
                    "capture": {
                        "completeness": {"status": "complete"},
                        "archive": {"content_hash": "a" * 64},
                    },
                }
            )
        )
        request_path = root / "export-request.json"
        external_key = "probe:v1:harbor:rollout:test"
        descriptor = {"correlation": {"external_key": external_key}}
        request_path.write_text(json.dumps(descriptor))
        archive_path = root / "trial.tar.gz"
        archive_path.write_bytes(b"recovery")
        return SimpleNamespace(
            staged_trial=SimpleNamespace(trial_dir=staged_trial),
            capture_manifest_path=manifest_path,
            request_path=request_path,
            descriptor=descriptor,
            archive_path=archive_path,
        )

    probe_module = ModuleType("probe")
    connectors_module = ModuleType("probe.connectors")
    harbor_module = ModuleType("probe.connectors.harbor")
    harbor_module.stage_trial_export = stage_trial_export
    monkeypatch.setitem(sys.modules, "probe", probe_module)
    monkeypatch.setitem(sys.modules, "probe.connectors", connectors_module)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor", harbor_module)
    return seen


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


def test_stage_trial_capture_delegates_native_values_to_probe_sdk(tmp_path: Path, monkeypatch) -> None:
    seen = _install_fake_probe(monkeypatch)
    trial_dir = tmp_path / "trials" / "task__abc"
    (trial_dir / "agent" / "commands").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "unknown").mkdir()
    (trial_dir / "config.json").write_text(json.dumps({"task": {"name": "task"}}))
    (trial_dir / "lock.json").write_text(json.dumps({"task": {"checksum": "sha256:task"}}))
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "harbor-result-id",
                "trial_name": "task__abc",
                "task_name": "task",
                "task_checksum": "sha256:task",
                "agent_info": {"name": "custom-agent", "version": "1"},
                "verifier_result": {"rewards": {"reward": 0.75, "tests": 1.0}},
                "agent_execution": {"started_at": "2026-07-22T01:00:00Z", "finished_at": "2026-07-22T01:01:00Z"},
            }
        )
    )
    (trial_dir / "agent" / "commands" / "stdout.log").write_text("agent output\n")
    (trial_dir / "verifier" / "reward.json").write_text('{"reward": 0.75}\n')
    native_bytes = b"\x00\xffprivate-fork-data"
    (trial_dir / "unknown" / "native.bin").write_bytes(native_bytes)
    (trial_dir / ".native-state").write_text("preserved in archive")
    (trial_dir / "latest-result").symlink_to("result.json")

    request = server.RunRequest(
        base_url="http://miles.internal:30000/sessions/session-123/v1",
        model="openai/model",
        instance_id="task",
        run_id="probe-run-1",
        miles_run_id="miles-run-1",
        rollout_id=17,
        sample_id=41,
        group_id=9,
        capture_context={"mix": "swe-and-terminal"},
    )
    capture = server.stage_trial_capture(
        trial_dir,
        tmp_path / "durable-captures",
        trial_id="trial-uuid",
        task_id="task",
        environment_type="daytona",
        delete_requested=True,
        sandbox_id="task__abc__env",
        provider_sandbox_id="daytona-123",
        session_id="session-123",
        request=request,
    )

    assert capture.status == "complete"
    staged_trial = Path(capture.staged_trial_dir)
    assert (staged_trial / "unknown" / "native.bin").read_bytes() == native_bytes
    assert (staged_trial / "latest-result").is_symlink()

    assert seen["run_id"] == "probe-run-1"
    assert seen["step_index"] == 17
    assert seen["correlation"] == {
        "miles_run_id": "miles-run-1",
        "rollout_id": 17,
        "sample_id": 41,
        "group_id": 9,
        "session_id": "session-123",
        "trial_id": "trial-uuid",
        "task_id": "task",
    }
    assert seen["context"] == {"mix": "swe-and-terminal"}
    assert seen["environment"]["collected"] == {
        "native_trial_directory": True,
        "staged_after_trial_run_returned": True,
    }


def test_capture_directory_name_cannot_escape_or_collapse_provider_ids() -> None:
    first = server._capture_directory_name("../provider/id")
    second = server._capture_directory_name(".._provider_id")
    assert "/" not in first and first not in {".", ".."}
    assert first != second


@pytest.mark.asyncio
async def test_trial_response_carries_correlation_and_completed_capture(tmp_path: Path, monkeypatch) -> None:
    _install_fake_probe(monkeypatch)
    task_dir = _task_dir(tmp_path)

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TrialConfig(Config):
        def __init__(self, **kwargs):
            super().__init__(trial_name="hello-world__unit", **kwargs)

    class FakeTrial:
        id = "trial-unit-id"

        def __init__(self, config):
            self.config = config
            self.paths = SimpleNamespace(trial_dir=config.trials_dir / config.trial_name)
            self.agent_environment = SimpleNamespace(
                session_id=f"{config.trial_name}__env",
                _sandbox=SimpleNamespace(id="provider-sandbox-id"),
            )

        @classmethod
        async def create(cls, config):
            return cls(config)

        async def run(self):
            self.paths.trial_dir.mkdir(parents=True)
            (self.paths.trial_dir / "agent").mkdir()
            (self.paths.trial_dir / "agent" / "native.log").write_text("native agent log")
            (self.paths.trial_dir / "config.json").write_text("{}")
            (self.paths.trial_dir / "lock.json").write_text("{}")
            result_doc = {
                "id": self.id,
                "trial_name": self.config.trial_name,
                "task_name": "hello-world",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
            (self.paths.trial_dir / "result.json").write_text(json.dumps(result_doc))
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
                agent_result=None,
                agent_execution=None,
                verifier=None,
            )

    for package in ("harbor", "harbor.models", "harbor.models.trial", "harbor.trial"):
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    config_module = ModuleType("harbor.models.trial.config")
    config_module.AgentConfig = Config
    config_module.EnvironmentConfig = Config
    config_module.TaskConfig = Config
    config_module.TrialConfig = TrialConfig
    monkeypatch.setitem(sys.modules, "harbor.models.trial.config", config_module)
    trial_module = ModuleType("harbor.trial.trial")
    trial_module.Trial = FakeTrial
    monkeypatch.setitem(sys.modules, "harbor.trial.trial", trial_module)

    settings = server.Settings(
        tasks_dir=task_dir.parent,
        trials_dir=tmp_path / "trials",
        capture_dir=tmp_path / "captures",
    )
    response = await server.run_public_harbor_trial(
        server.RunRequest(
            base_url="http://miles.internal:30000/sessions/session-unit/v1",
            model="openai/model",
            instance_id="hello-world",
            run_id="run-unit",
            miles_run_id="miles-unit",
            rollout_id=3,
            sample_id=4,
            group_id=5,
        ),
        settings,
    )

    assert response.reward == 1.0
    assert response.trial_id == "trial-unit-id"
    assert response.trial_name == "hello-world__unit"
    assert response.task_id == "hello-world"
    assert response.sandbox_id == "hello-world__unit__env"
    assert response.provider_sandbox_id == "provider-sandbox-id"
    assert response.session_id == "session-unit"
    assert response.run_id == "run-unit"
    assert response.miles_run_id == "miles-unit"
    assert response.external_key.startswith("probe:v1:harbor:rollout:")
    assert response.rollout_id == response.step_index == 3
    assert response.sample_id == 4
    assert response.group_id == 5
    assert response.capture.status == "complete"
    assert Path(response.capture.staged_trial_dir, "agent", "native.log").read_text() == "native agent log"


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
async def test_flush_cancels_inflight_trial_for_session_server_generation(tmp_path: Path) -> None:
    _task_dir(tmp_path)
    settings = server.Settings(
        tasks_dir=tmp_path / "tasks",
        auth_token="run-token",
        admin_secret="admin-token",
        allowed_callback_hosts=frozenset({"miles.internal"}),
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_runner(request, actual_settings):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    app = server.create_app(settings, blocking_runner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        run_task = asyncio.create_task(
            client.post(
                "/run",
                json={
                    "base_url": "http://miles.internal:30000/sessions/abc/v1",
                    "model": "openai/model",
                    "instance_id": "hello-world",
                    "session_server_instance_id": "generation-1",
                },
                headers={"Authorization": "Bearer run-token"},
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        unauthorized = await client.post(
            "/flush",
            json={"session_server_instance_id": "generation-1"},
            headers={"Authorization": "Bearer run-token"},
        )
        assert unauthorized.status_code == 401
        flushed = await client.post(
            "/flush",
            json={"session_server_instance_id": "generation-1"},
            headers={"Authorization": "Bearer admin-token"},
        )

        assert flushed.status_code == 200
        assert flushed.json()["cancelled"] == 1
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await asyncio.gather(run_task, return_exceptions=True)


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
        api_key="session-secret",
        instance_id="hello-world",
        max_seq_len=3,
        session_server_id="miles.internal:30000",
        session_server_instance_id="server-generation-1",
    )
    settings = server.Settings(tasks_dir=tmp_path, session_poll_interval_sec=0.001)
    requested = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            requested.append((url, headers))
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
    assert requested == [
        ("http://miles.internal:30000/health", {"Authorization": "Bearer session-secret"}),
        (
            "http://miles.internal:30000/sessions/session-123",
            {"Authorization": "Bearer session-secret"},
        ),
    ]
