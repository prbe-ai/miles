"""Sandbox-state wiring (probe.sandbox-state/1) in the public Harbor bridge.

The begin/end snapshot protocol itself (ephemerality choreography, timeout
bounds, begin-fail-skips-end, integrity verification, bundle authoring) moved
into the SDK (``probe.connectors.harbor_runner.SandboxStateRecorder``) and is
tested upstream (research-os-agent PR #99). These tests only cover what Miles
still owns: Settings gating, the env-derived ``SandboxStateOptions`` wiring,
the summary's mapping onto the response, and the ``create_app`` preflight.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "examples/experimental/swe-agent-v2/public_harbor_server.py"
_SPEC = importlib.util.spec_from_file_location("public_harbor_server_sbx", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)

#: SDK-shaped summary (probe-research >= 0.23.0): phase statuses are
#: pending/ok/failed, integrity is {begin,end}_verified booleans.
_SDK_SUMMARY = {
    "schema": "probe.sandbox-state/1",
    "status": {"begin": "ok", "end": "ok"},
    "arch": "amd64",
    "integrity": {"begin_verified": True, "end_verified": True},
    "errors": [],
}


class FakeTrialEvent(enum.Enum):
    AGENT_START = "agent-start"
    AGENT_END = "agent-end"


def _install_fake_harbor(monkeypatch) -> None:
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TrialConfig(Config):
        def __init__(self, **kwargs):
            super().__init__(trial_name="hello-world__sbx", **kwargs)

    class FakeTrial:
        id = "trial-sbx-id"

        def __init__(self, config):
            self.config = config
            self.paths = SimpleNamespace(trial_dir=config.trials_dir / config.trial_name)
            self.agent_environment = SimpleNamespace(session_id="sbx__env", _sandbox=None)
            self.hooks = {event: [] for event in FakeTrialEvent}

        @classmethod
        async def create(cls, config):
            return cls(config)

        def add_hook(self, event, hook):
            self.hooks[event].append(hook)

        async def run(self):
            self.paths.trial_dir.mkdir(parents=True)
            for name in ("config.json", "lock.json"):
                (self.paths.trial_dir / name).write_text("{}")
            (self.paths.trial_dir / "result.json").write_text(json.dumps({"id": self.id}))
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
    hooks_module = ModuleType("harbor.trial.hooks")
    hooks_module.TrialEvent = FakeTrialEvent
    monkeypatch.setitem(sys.modules, "harbor.trial.hooks", hooks_module)


def _install_fake_probe(monkeypatch, *, summary: dict | None = _SDK_SUMMARY):
    """SDK boundary double capturing attach()/finalize() wiring."""
    seen = {}

    @dataclasses.dataclass
    class SandboxStateOptions:
        begin_timeout_sec: float | None = 120.0
        end_timeout_sec: float = 300.0
        hash_files: bool = False
        exclude: tuple = ()
        max_files: int | None = None
        max_delta_bytes: int | None = None
        begin_bytes: bool = False
        begin_bytes_ref: str | None = None
        max_begin_bytes: int | None = None

    class FakeRecorder:
        def summary(self):
            return summary

    class FakeHandle:
        def __init__(self, trial, sandbox_state_options):
            self._trial = trial
            self._options = sandbox_state_options
            self.sandbox_state = FakeRecorder() if sandbox_state_options is not None else None
            self.errors = []

        async def finalize(self, trial_dir, **kwargs):
            seen.update({"finalize_trial_dir": Path(trial_dir), **kwargs})
            recorder_summary = self.sandbox_state.summary() if self.sandbox_state is not None else None
            # The SDK (>= 0.26.0) reports capture status on the result; simulate a
            # verified archive whenever begin_bytes was requested.
            return SimpleNamespace(
                status="complete",
                staged_trial_dir=str(trial_dir),
                archive_path=None,
                manifest_path=None,
                export_descriptor_path=None,
                archive_content_hash=None,
                external_key="probe:v1:harbor:rollout:sbx",
                file_count=3,
                size_bytes=6,
                sandbox_id="sbx__env",
                provider_sandbox_id=None,
                sandbox_state=recorder_summary,
                begin_bytes_captured=bool(getattr(self._options, "begin_bytes", False)),
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
        return FakeHandle(trial, sandbox_state)

    probe_module = ModuleType("probe")
    connectors_module = ModuleType("probe.connectors")
    harbor_capture_module = ModuleType("probe.connectors.harbor_capture")
    harbor_capture_module.attach = attach
    harbor_runner_module = ModuleType("probe.connectors.harbor_runner")
    harbor_runner_module.SandboxStateOptions = SandboxStateOptions
    monkeypatch.setitem(sys.modules, "probe", probe_module)
    monkeypatch.setitem(sys.modules, "probe.connectors", connectors_module)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_capture", harbor_capture_module)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_runner", harbor_runner_module)
    return seen


def _settings(tmp_path: Path, **overrides) -> server.Settings:
    defaults = dict(
        tasks_dir=tmp_path / "tasks",
        trials_dir=tmp_path / "trials",
        capture_dir=tmp_path / "captures",
        capture_mode="shadow",
        sandbox_state=True,
    )
    defaults.update(overrides)
    return server.Settings(**defaults)


def _run_request(instance_id: str = "hello-world") -> server.RunRequest:
    return server.RunRequest(
        base_url="http://miles.internal:30000/sessions/session-sbx/v1",
        model="openai/model",
        instance_id=instance_id,
        rollout_id=7,
    )


def test_settings_rejects_sandbox_state_with_capture_off(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MILES_SANDBOX_STATE"):
        _settings(tmp_path, capture_mode="off")


@pytest.mark.asyncio
async def test_sandbox_state_options_are_built_from_settings(tmp_path: Path, monkeypatch) -> None:
    _install_fake_harbor(monkeypatch)
    seen = _install_fake_probe(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)
    settings = _settings(
        tmp_path,
        sandbox_state_begin_timeout_sec=45.0,
        sandbox_state_end_timeout_sec=90.0,
        sandbox_state_hash=True,
        sandbox_state_exclude="/data:/var/cache",
    )

    response = await server.run_public_harbor_trial(_run_request(), settings)

    assert response.reward == 1.0
    options = seen["sandbox_state_options"]
    assert options is not None
    assert options.begin_timeout_sec == 45.0
    assert options.end_timeout_sec == 90.0
    assert options.hash_files is True
    assert options.exclude == ("/data", "/var/cache")


@pytest.mark.asyncio
async def test_recorder_summary_is_mapped_onto_the_response(tmp_path: Path, monkeypatch) -> None:
    _install_fake_harbor(monkeypatch)
    _install_fake_probe(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)

    response = await server.run_public_harbor_trial(_run_request(), _settings(tmp_path))

    assert response.capture is not None
    assert response.capture.status == "complete"
    # Happy-path SDK shape: phase statuses are plain "ok" (exception text lives
    # in errors, never in the status value) and integrity is the two booleans.
    assert response.capture.sandbox_state["status"] == {"begin": "ok", "end": "ok"}
    assert response.capture.sandbox_state["integrity"] == {"begin_verified": True, "end_verified": True}


@pytest.mark.asyncio
async def test_attach_failure_fails_open_in_shadow_mode(tmp_path: Path, monkeypatch) -> None:
    """Missing/broken probe must never block the trial; capture reports failed."""
    _install_fake_harbor(monkeypatch)
    monkeypatch.setitem(sys.modules, "probe", None)  # force ImportError at attach
    monkeypatch.setitem(sys.modules, "probe.connectors", None)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_capture", None)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)

    response = await server.run_public_harbor_trial(_run_request(), _settings(tmp_path))

    assert response.reward == 1.0
    assert response.exit_status == "Submitted"
    assert response.capture.status == "failed"
    assert response.capture.error


def test_create_app_preflight_checks_facade_modules_and_binaries(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)

    # probe missing entirely -> loud startup failure with the install hint.
    monkeypatch.setitem(sys.modules, "probe", None)
    monkeypatch.setitem(sys.modules, "probe.connectors", None)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_capture", None)
    monkeypatch.setitem(sys.modules, "probe.connectors.sandbox_state", None)
    monkeypatch.setitem(sys.modules, "probe.connectors.harbor_runner", None)
    with pytest.raises(RuntimeError, match="probe-research >= 0.23.0"):
        server.create_app(settings)

    # facade importable but snapshot binaries stripped -> still loud.
    _install_fake_probe(monkeypatch)
    sandbox_state_module = ModuleType("probe.connectors.sandbox_state")

    def _missing_binary(arch):
        raise FileNotFoundError(arch)

    sandbox_state_module.snapshot_binary_path = _missing_binary
    monkeypatch.setitem(sys.modules, "probe.connectors.sandbox_state", sandbox_state_module)
    with pytest.raises(RuntimeError, match="probe-research >= 0.23.0"):
        server.create_app(settings)

    # everything present -> app builds.
    sandbox_state_module.snapshot_binary_path = lambda arch: Path(__file__)
    app = server.create_app(settings)
    assert app.title == "Miles Public Harbor Bridge"


# ---------------------------------------------------------------------------
# begin-state bytes (probe-research 0.24.0): per-task shared before-content
# ---------------------------------------------------------------------------
def _fresh_ledger(monkeypatch) -> None:
    """Isolate the module-global election ledger per test."""
    monkeypatch.setattr(server, "_BEGIN_BYTES_LEDGER", server.BeginBytesLedger())


@pytest.mark.asyncio
async def test_begin_bytes_wired_into_options_when_enabled(tmp_path: Path, monkeypatch) -> None:
    _install_fake_harbor(monkeypatch)
    seen = _install_fake_probe(monkeypatch)
    _fresh_ledger(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)

    settings = _settings(tmp_path, sandbox_state_begin_bytes=True, sandbox_state_max_begin_bytes=123)
    await server.run_public_harbor_trial(_run_request("hello-world"), settings)

    options = seen["sandbox_state_options"]
    assert options.begin_bytes is True
    assert options.begin_bytes_ref == "hello-world"  # ref == instance_id (task id)
    assert options.max_begin_bytes == 123
    assert options.hash_files is True  # forced on for begin-bytes validity


@pytest.mark.asyncio
async def test_begin_bytes_off_by_default(tmp_path: Path, monkeypatch) -> None:
    _install_fake_harbor(monkeypatch)
    seen = _install_fake_probe(monkeypatch)
    _fresh_ledger(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)

    await server.run_public_harbor_trial(_run_request("hello-world"), _settings(tmp_path))

    options = seen["sandbox_state_options"]
    assert options.begin_bytes is False
    assert options.begin_bytes_ref is None


@pytest.mark.asyncio
async def test_only_first_rollout_of_a_task_captures_siblings_share_ref(tmp_path: Path, monkeypatch) -> None:
    _install_fake_harbor(monkeypatch)
    _fresh_ledger(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)
    settings = _settings(tmp_path, sandbox_state_begin_bytes=True)

    seen1 = _install_fake_probe(monkeypatch)
    await server.run_public_harbor_trial(_run_request("hello-world"), settings)
    first = seen1["sandbox_state_options"]

    seen2 = _install_fake_probe(monkeypatch)  # sibling rollout, same task+run
    await server.run_public_harbor_trial(_run_request("hello-world"), settings)
    second = seen2["sandbox_state_options"]

    assert first.begin_bytes is True and second.begin_bytes is False  # capture once
    assert first.begin_bytes_ref == second.begin_bytes_ref == "hello-world"  # shared ref


# ---- pure units: ledger, election, capture-signal read ----
def test_settings_begin_bytes_requires_sandbox_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MILES_SANDBOX_STATE_BEGIN_BYTES"):
        _settings(tmp_path, sandbox_state=False, capture_mode="shadow", sandbox_state_begin_bytes=True)


def test_settings_begin_bytes_bumps_begin_timeout_default(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/tmp/tasks")
    monkeypatch.setenv("MILES_HARBOR_CAPTURE_MODE", "shadow")
    monkeypatch.setenv("MILES_SANDBOX_STATE", "1")
    monkeypatch.setenv("MILES_SANDBOX_STATE_BEGIN_BYTES", "1")
    monkeypatch.delenv("MILES_SANDBOX_STATE_TIMEOUT_BEGIN", raising=False)
    s = server.Settings.from_env()
    assert s.sandbox_state_begin_bytes is True
    assert s.sandbox_state_begin_timeout_sec == 600.0
    monkeypatch.setenv("MILES_SANDBOX_STATE_TIMEOUT_BEGIN", "200")
    assert server.Settings.from_env().sandbox_state_begin_timeout_sec == 200.0


@pytest.mark.asyncio
async def test_ledger_elects_one_and_reelects_only_after_failure() -> None:
    ledger = server.BeginBytesLedger()
    assert await ledger.claim("r", "t") is True
    assert await ledger.claim("r", "t") is False  # in-flight -> denied
    await ledger.release("r", "t", succeeded=False)  # first attempt failed
    assert await ledger.claim("r", "t") is True  # a later rollout retries
    await ledger.release("r", "t", succeeded=True)  # this one worked
    assert await ledger.claim("r", "t") is False  # latched: no more captures
    assert await ledger.claim("r", "other") is True  # different task -> own capture


@pytest.mark.asyncio
async def test_election_latches_on_captured_result_from_finalize(tmp_path: Path, monkeypatch) -> None:
    """The elected trial's release reads result.begin_bytes_captured (SDK >=
    0.26.0): a captured run latches the (run, task) slot so siblings don't
    re-capture."""
    _install_fake_harbor(monkeypatch)
    _fresh_ledger(monkeypatch)
    (tmp_path / "tasks" / "hello-world").mkdir(parents=True)
    settings = _settings(tmp_path, sandbox_state_begin_bytes=True)

    _install_fake_probe(monkeypatch)  # fake finalize -> begin_bytes_captured=True
    await server.run_public_harbor_trial(_run_request("hello-world"), settings)

    # Slot latched: a fresh sibling rollout is denied capture.
    captured, ref, key = await server._elect_begin_bytes(settings, _run_request("hello-world"))
    assert captured is False and key is None and ref == "hello-world"
