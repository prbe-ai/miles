"""Sandbox-state hook wiring (probe.sandbox-state/1) in the public Harbor bridge.

Uses the real ``probe.connectors.sandbox_state`` contract helpers (skipped when
probe-research is not installed); Harbor is stubbed at the module boundary like
the rest of this suite.
"""

from __future__ import annotations

import asyncio
import enum
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sandbox_state = pytest.importorskip("probe.connectors.sandbox_state")

_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "examples/experimental/swe-agent-v2/public_harbor_server.py"
_SPEC = importlib.util.spec_from_file_location("public_harbor_server_sbx", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


class FakeTrialEvent(enum.Enum):
    AGENT_START = "agent-start"
    AGENT_END = "agent-end"


def _install_fake_hooks_module(monkeypatch) -> None:
    for package in ("harbor", "harbor.trial"):
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    hooks_module = ModuleType("harbor.trial.hooks")
    hooks_module.TrialEvent = FakeTrialEvent
    monkeypatch.setitem(sys.modules, "harbor.trial.hooks", hooks_module)


def _gzip_jsonl_bytes(records: list[dict]) -> bytes:
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        for record in records:
            handle.write(json.dumps(record).encode() + b"\n")
    return buffer.getvalue()


class FakeEnvironment:
    """Records the ephemerality choreography; serves snapshot outputs on exec."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[str] = []
        self.execs: list[str] = []
        self.exec_kwargs: list[dict] = []
        self.machine = "x86_64"
        self.fail_exec_phase: str | None = None
        self._outputs: dict[str, bytes] = {}

    def _trailer(self, phase: str, files: dict[str, bytes]) -> str:
        payload = {
            "schema": sandbox_state.TRAILER_SCHEMA,
            "phase": phase,
            "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for name, data in files.items()},
            "stats": {"entries": 3, "files_scanned": 2, "added": 1, "modified": 0, "deleted": 0},
            "errors": [],
            "hash_mode": "fast",
        }
        return sandbox_state.TRAILER_PREFIX + json.dumps(payload)

    async def exec(self, command: str, user: str | None = None, **kwargs):
        self.execs.append(command)
        self.exec_kwargs.append(kwargs)
        if command == "uname -m":
            return SimpleNamespace(stdout=self.machine + "\n", stderr="", return_code=0)
        if command.startswith("rm -rf ") or command.startswith("mkdir -p "):
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        phase = "begin" if " begin " in command else "end"
        if self.fail_exec_phase == phase:
            return SimpleNamespace(stdout="", stderr="boom", return_code=3)
        workdir = command.split("--workdir ")[1].split()[0]
        if phase == "begin":
            files = {"begin-manifest.jsonl.gz": _gzip_jsonl_bytes([{"p": "/b"}, {"p": "/a"}])}
        else:
            files = {
                "end-manifest.jsonl.gz": _gzip_jsonl_bytes([{"p": "/a"}, {"p": "/new"}]),
                "end-delta.tar.gz": b"delta-bytes",
            }
        for name, data in files.items():
            self._outputs[f"{workdir}/{name}"] = data
        return SimpleNamespace(stdout="noise\n" + self._trailer(phase, files), stderr="", return_code=0)

    async def upload_file(self, source_path, target_path: str) -> None:
        self.uploads.append((str(source_path), target_path))

    async def download_file(self, source_path: str, target_path) -> None:
        self.downloads.append(source_path)
        Path(target_path).write_bytes(self._outputs[source_path])


class FakeTrial:
    def __init__(self) -> None:
        self.agent_environment = FakeEnvironment()
        self.hooks: dict[FakeTrialEvent, list] = {event: [] for event in FakeTrialEvent}

    def add_hook(self, event, hook) -> None:
        self.hooks[event].append(hook)

    async def emit(self, event) -> None:
        for hook in self.hooks[event]:
            await hook(SimpleNamespace(event=event))


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


def _capture(tmp_path: Path, monkeypatch, **settings_overrides):
    _install_fake_hooks_module(monkeypatch)
    monkeypatch.setenv("PROBE_SANDBOX_SNAPSHOT_BIN", __file__)  # any real file
    trial = FakeTrial()
    host_dir = tmp_path / "host"
    host_dir.mkdir(parents=True)
    capture = server.SandboxStateCapture(trial, _settings(tmp_path, **settings_overrides), host_dir)
    capture.install()
    return trial, capture


def test_settings_rejects_sandbox_state_with_capture_off(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MILES_SANDBOX_STATE"):
        _settings(tmp_path, capture_mode="off")


def test_happy_path_ephemeral_choreography_and_bundle(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    env = trial.agent_environment

    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))
    assert capture.status["begin"] == "ok"
    # Ephemerality: every exec'd workdir was cleaned up in the same phase.
    begin_workdirs = {c.split("--workdir ")[1].split()[0] for c in env.execs if "--workdir" in c}
    cleanups = {c.removeprefix("rm -rf ").strip("'") for c in env.execs if c.startswith("rm -rf")}
    assert begin_workdirs <= cleanups

    asyncio.run(trial.emit(FakeTrialEvent.AGENT_END))
    assert capture.status["end"] == "ok"
    # Fresh binary uploaded per phase (nothing persists between phases).
    binary_uploads = [target for _, target in env.uploads if target.endswith("/snap")]
    assert len(binary_uploads) == 2
    assert len(set(binary_uploads)) == 2, "workdir must differ per phase"
    # Begin manifest re-uploaded for the end diff.
    assert any(target.endswith("/begin.jsonl.gz") for _, target in env.uploads)

    trial_dir = tmp_path / "trial-out"
    summary = capture.write_bundle(trial_dir)
    bundle = trial_dir / "artifacts" / sandbox_state.BUNDLE_DIRNAME
    meta = json.loads((bundle / "meta.json").read_text())
    assert meta["schema"] == sandbox_state.SCHEMA
    assert meta["integrity"] == {
        "begin-manifest.jsonl.gz": True,
        "end-manifest.jsonl.gz": True,
        "end-delta.tar.gz": True,
    }
    assert (bundle / "end-delta.tar.gz").read_bytes() == b"delta-bytes"
    with gzip.open(bundle / "begin-manifest.jsonl.gz", "rb") as handle:
        paths = [json.loads(line)["p"] for line in handle]
    assert paths == ["/a", "/b"], "manifest must be host-sorted"
    assert summary["status"] == {"begin": "ok", "end": "ok"}
    assert not capture._host_dir.exists(), "host scratch must be cleaned up"


def test_all_execs_are_timeout_bounded_and_workdir_precreated(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    env = trial.agent_environment

    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))

    # Every container exec must carry a timeout so a hung sandbox can never
    # stall the trial's own unwind (the two HIGH review findings).
    snapshot_execs = [(cmd, kw) for cmd, kw in zip(env.execs, env.exec_kwargs, strict=True) if cmd != "uname -m"]
    for cmd, kw in snapshot_execs:
        assert kw.get("timeout_sec"), f"exec without timeout_sec: {cmd}"
    # mkdir -p must run before the snapshot exec so docker-compose-cp (which
    # does not create parents) has a workdir to upload into.
    mkdir_idx = next(i for i, c in enumerate(env.execs) if c.startswith("mkdir -p "))
    snap_idx = next(i for i, c in enumerate(env.execs) if "/snap begin " in c)
    assert mkdir_idx < snap_idx
    # The in-container binary is given a self-deadline below the exec timeout.
    assert "--max-seconds" in env.execs[snap_idx]


def test_begin_failure_skips_end_and_never_raises(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    trial.agent_environment.fail_exec_phase = "begin"

    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))
    assert capture.status["begin"].startswith("RuntimeError")

    asyncio.run(trial.emit(FakeTrialEvent.AGENT_END))
    assert "begin snapshot unavailable" in capture.status["end"]

    trial_dir = tmp_path / "trial-out"
    summary = capture.write_bundle(trial_dir)
    meta = json.loads((trial_dir / "artifacts" / sandbox_state.BUNDLE_DIRNAME / "meta.json").read_text())
    assert meta["status"]["begin"].startswith("RuntimeError")
    assert summary["status"]["end"].startswith("RuntimeError")


def test_hook_swallows_arbitrary_env_explosions(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)

    async def explode(*args, **kwargs):
        raise OSError("environment vanished")

    trial.agent_environment.upload_file = explode
    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))
    assert capture.status["begin"].startswith("OSError")


def test_integrity_mismatch_detected(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    env = trial.agent_environment
    original_download = env.download_file

    async def tampered_download(source_path, target_path):
        await original_download(source_path, target_path)
        if source_path.endswith("begin-manifest.jsonl.gz"):
            Path(target_path).write_bytes(b"tampered")

    env.download_file = tampered_download
    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))
    assert capture.status["begin"] == "ok"
    assert capture._integrity["begin-manifest.jsonl.gz"] is False


def test_not_attempted_writes_no_bundle(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    trial_dir = tmp_path / "trial-out"
    summary = capture.write_bundle(trial_dir)
    assert summary["status"] == "not_attempted"
    assert not (trial_dir / "artifacts" / sandbox_state.BUNDLE_DIRNAME).exists()


def test_unknown_arch_falls_back_to_amd64(tmp_path: Path, monkeypatch) -> None:
    trial, capture = _capture(tmp_path, monkeypatch)
    trial.agent_environment.machine = "riscv64"
    asyncio.run(trial.emit(FakeTrialEvent.AGENT_START))
    assert capture.status["begin"] == "ok"
    assert capture.summary()["arch"] == "amd64"
    assert any("unrecognized machine" in err for err in capture.summary()["errors"])
