from __future__ import annotations

import asyncio
import dataclasses
import enum
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
    """Install the SDK boundary double for ``probe.connectors.harbor_capture``.

    The facade's own contract (hook install, staging, fail-open, sandbox-state
    recording) is tested upstream in the SDK (research-os-agent PR #99); these
    tests only exercise what Miles still owns — the wiring into ``attach``/
    ``finalize`` and the response mapping. Set ``seen["fail_finalize"] = True``
    to mimic the SDK's never-raise staging failure (``status="failed"``).
    """
    seen = {}

    @dataclasses.dataclass
    class SandboxStateOptions:
        begin_timeout_sec: float = 120.0
        end_timeout_sec: float = 300.0
        hash_files: bool = False
        exclude: tuple = ()
        max_files: int | None = None
        max_delta_bytes: int | None = None

    class FakeHandle:
        def __init__(self, trial, *, correlation, context, capture_mode, sandbox_state):
            self._trial = trial
            self.correlation = correlation
            self.context = context
            self.capture_mode = capture_mode
            self.sandbox_state = sandbox_state
            self.errors = []
            self._sandbox_id = None
            self._provider_sandbox_id = None
            from harbor.trial.hooks import TrialEvent

            trial.add_hook(TrialEvent.AGENT_START, self._capture)
            trial.add_hook(TrialEvent.AGENT_END, self._capture)

        async def _capture(self, _event=None):
            environment = getattr(self._trial, "agent_environment", None)
            self._sandbox_id = getattr(environment, "session_id", None) or self._sandbox_id
            handle = getattr(environment, "_sandbox", None)
            if handle is not None:
                self._provider_sandbox_id = getattr(handle, "object_id", None) or self._provider_sandbox_id

        @property
        def sandbox_ids(self):
            return (self._sandbox_id, self._provider_sandbox_id)

        async def finalize(
            self,
            trial_dir,
            *,
            capture_dir=None,
            run_id=None,
            step_index=None,
            environment=None,
            external_key=None,
            create_archive=True,
        ):
            seen.update(
                {
                    "finalize_trial_dir": Path(trial_dir),
                    "finalize_capture_dir": Path(capture_dir) if capture_dir is not None else None,
                    "run_id": run_id,
                    "step_index": step_index,
                    "environment": environment,
                }
            )
            summary = self.sandbox_state.summary() if self.sandbox_state is not None else None
            if seen.get("fail_finalize"):
                return SimpleNamespace(
                    status="failed",
                    staged_trial_dir=None,
                    archive_path=None,
                    manifest_path=None,
                    export_descriptor_path=None,
                    archive_content_hash=None,
                    external_key=None,
                    file_count=0,
                    size_bytes=0,
                    sandbox_id=self._sandbox_id,
                    provider_sandbox_id=self._provider_sandbox_id,
                    sandbox_state=summary,
                    error="RuntimeError: capture unavailable",
                )
            root = Path(capture_dir) / "staged"
            staged_trial = root / "trial"
            shutil.copytree(trial_dir, staged_trial, symlinks=True)
            files = [path for path in staged_trial.rglob("*") if path.is_file() and not path.is_symlink()]
            return SimpleNamespace(
                status="complete",
                staged_trial_dir=str(staged_trial),
                archive_path=str(root / "trial.tar.gz"),
                manifest_path=str(root / "capture-manifest.json"),
                export_descriptor_path=str(root / "export-request.json"),
                archive_content_hash="a" * 64,
                external_key="probe:v1:harbor:rollout:test",
                file_count=len(files),
                size_bytes=sum(path.stat().st_size for path in files),
                sandbox_id=self._sandbox_id,
                provider_sandbox_id=self._provider_sandbox_id,
                sandbox_state=summary,
                error=None,
            )

    def attach(trial, *, correlation=None, context=None, capture_mode="shadow", sandbox_state=None):
        seen.update(
            {
                "attach_correlation": dict(correlation or {}),
                "attach_context": dict(context or {}),
                "capture_mode": capture_mode,
                "sandbox_state_options": sandbox_state,
            }
        )
        return FakeHandle(
            trial,
            correlation=dict(correlation or {}),
            context=dict(context or {}),
            capture_mode=capture_mode,
            sandbox_state=sandbox_state,
        )

    probe_module = ModuleType("probe")
    connectors_module = ModuleType("probe.connectors")
    harbor_capture_module = ModuleType("probe.connectors.harbor_capture")
    harbor_capture_module.attach = attach
    harbor_capture_module.CAPTURE_MODES = frozenset({"off", "shadow", "required"})
    harbor_runner_module = ModuleType("probe.connectors.harbor_runner")
    harbor_runner_module.SandboxStateOptions = SandboxStateOptions
    monkeypatch.setitem(sys.modules, "probe", probe_module)
    monkeypatch.setitem(sys.modules, "probe.connectors", connectors_module)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_capture", harbor_capture_module)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_runner", harbor_runner_module)
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


def test_capture_mode_defaults_off_and_rejects_unknown_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MILES_HARBOR_CAPTURE_MODE", raising=False)
    monkeypatch.setenv("HARBOR_TRIALS_DIR", str(tmp_path / "trials"))
    settings = server.Settings.from_env()
    assert settings.capture_mode == "off"
    assert settings.capture_dir == tmp_path / "trials-captures"

    with pytest.raises(ValueError, match="MILES_HARBOR_CAPTURE_MODE"):
        server.Settings(capture_mode="best-effort")


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
async def test_trial_response_carries_correlation_and_completed_capture(tmp_path: Path, monkeypatch) -> None:
    seen = _install_fake_probe(monkeypatch)
    task_dir = _task_dir(tmp_path)

    class FakeTrialEvent(enum.Enum):
        AGENT_START = "agent-start"
        AGENT_END = "agent-end"

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
                _sandbox=SimpleNamespace(object_id="provider-sandbox-id"),
            )
            self.hooks = {event: [] for event in FakeTrialEvent}

        @classmethod
        async def create(cls, config):
            return cls(config)

        def add_hook(self, event, hook):
            self.hooks[event].append(hook)

        async def emit(self, event):
            for hook in self.hooks[event]:
                await hook(SimpleNamespace(event=event))

        async def run(self):
            await self.emit(FakeTrialEvent.AGENT_START)
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
            result = SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
                agent_result=None,
                agent_execution=None,
                verifier=None,
            )
            await self.emit(FakeTrialEvent.AGENT_END)
            self.agent_environment._sandbox = None
            return result

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
    hooks_module = ModuleType("harbor.trial.hooks")
    hooks_module.TrialEvent = FakeTrialEvent
    monkeypatch.setitem(sys.modules, "harbor.trial.hooks", hooks_module)

    settings = server.Settings(
        tasks_dir=task_dir.parent,
        trials_dir=tmp_path / "trials",
        capture_dir=tmp_path / "captures",
        capture_mode="shadow",
    )
    request = server.RunRequest(
        base_url="http://miles.internal:30000/sessions/session-unit/v1",
        model="openai/model",
        instance_id="hello-world",
        run_id="run-unit",
        miles_run_id="miles-unit",
        rollout_id=3,
        sample_id=4,
        group_id=5,
        capture_context={"mix": "swe-and-terminal"},
    )
    response = await server.run_public_harbor_trial(request, settings)

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

    # Miles-owned wiring into the SDK facade: correlation/context at attach,
    # run/step/capture-dir/environment at finalize.
    assert seen["capture_mode"] == "shadow"
    assert seen["sandbox_state_options"] is None
    assert seen["attach_correlation"] == {
        "miles_run_id": "miles-unit",
        "rollout_id": 3,
        "sample_id": 4,
        "group_id": 5,
        "session_id": "session-unit",
        "trial_id": "trial-unit-id",
        "task_id": "hello-world",
    }
    assert seen["attach_context"] == {"mix": "swe-and-terminal"}
    assert seen["run_id"] == "run-unit"
    assert seen["step_index"] == 3
    assert seen["finalize_capture_dir"] == settings.capture_dir
    assert seen["environment"] == {"type": "docker", "delete_requested": True}

    seen["fail_finalize"] = True
    off_capture_dir = tmp_path / "off-captures"
    off_response = await server.run_public_harbor_trial(
        request,
        server.Settings(
            tasks_dir=task_dir.parent,
            trials_dir=tmp_path / "off-trials",
            capture_dir=off_capture_dir,
            capture_mode="off",
        ),
    )
    assert off_response.reward == 1.0
    assert off_response.exit_status == "Submitted"
    assert off_response.capture is None
    assert off_response.trial_id is None
    assert not off_capture_dir.exists()

    shadow_failure = await server.run_public_harbor_trial(
        request,
        server.Settings(
            tasks_dir=task_dir.parent,
            trials_dir=tmp_path / "shadow-failure-trials",
            capture_dir=tmp_path / "shadow-failure-captures",
            capture_mode="shadow",
        ),
    )
    assert shadow_failure.reward == 1.0
    assert shadow_failure.exit_status == "Submitted"
    assert shadow_failure.capture.status == "failed"
    assert shadow_failure.capture.error == "RuntimeError: capture unavailable"

    required_response = await server.run_public_harbor_trial(
        request,
        server.Settings(
            tasks_dir=task_dir.parent,
            trials_dir=tmp_path / "required-trials",
            capture_dir=tmp_path / "required-captures",
            capture_mode="required",
        ),
    )
    assert required_response.reward == 1.0
    assert required_response.exit_status == "Error: CaptureRequiredError"
    assert required_response.capture.status == "failed"
    assert required_response.capture.error == "RuntimeError: capture unavailable"


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
        assert health.json()["capture_mode"] == "off"
        assert "capture_dir" not in health.json()

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
