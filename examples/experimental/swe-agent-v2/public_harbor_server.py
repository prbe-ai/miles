#!/usr/bin/env python3
"""Miles-compatible HTTP bridge backed by the public Harbor package.

Miles creates one session-scoped OpenAI-compatible URL per rollout. This
service turns a ``POST /run`` request containing that URL into one public
Harbor trial, then reduces Harbor's result to the reward/status shape expected
by ``swe_agent_function.py``.

The bridge deliberately owns only integration concerns. Harbor still owns task
loading, agent execution, environments, verification, and trial artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("miles.public_harbor_server")

_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SESSION_PATH = re.compile(r"(?:^|/)sessions/([^/]+)(?:/|$)")
_TIMEOUT_EXCEPTIONS = {
    "AgentSetupTimeoutError",
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "EnvironmentStartTimeoutError",
}
_SEQUENCE_EXCEPTIONS = {
    "ContextLengthExceededError",
    "MaxSeqLenExceededError",
    "SequenceLengthLimitExceeded",
}
_HOST_AGENTS = {"terminus", "terminus-1", "terminus-2"}
_EXPECTED_TRIAL_FILES = ("config.json", "lock.json", "result.json")
_CAPTURE_MODES = frozenset({"off", "shadow", "required"})
_SANDBOX_STATE_OUTPUTS = {
    "begin": ("begin-manifest.jsonl.gz",),
    "end": ("end-manifest.jsonl.gz", "end-delta.tar.gz"),
}


class RunRequest(BaseModel):
    base_url: str
    model: str
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    api_key: str = "dummy"
    instance_id: str = ""
    agent_name: str = "mini-swe-agent"
    max_seq_len: int | None = Field(default=None, gt=0)
    session_server_id: str | None = None
    session_server_instance_id: str | None = None
    run_id: str | None = Field(
        default=None,
        description="Target Probe/Research OS run ID for the pending export request.",
    )
    miles_run_id: str | None = Field(
        default=None,
        description="Optional native Miles job/run identifier used only for correlation.",
    )
    rollout_id: int | str | None = None
    sample_id: int | str | None = None
    group_id: int | str | None = None
    step_index: int | None = None
    capture_context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class FlushRequest(BaseModel):
    session_server_instance_id: str


class CaptureResult(BaseModel):
    status: str = "not_attempted"
    staged_trial_dir: str | None = None
    archive_path: str | None = None
    manifest_path: str | None = None
    export_descriptor_path: str | None = None
    archive_content_hash: str | None = None
    external_key: str | None = None
    file_count: int = 0
    size_bytes: int = 0
    completeness_scope: str = "host_harbor_trial_tree"
    sandbox_state_outside_harbor_outputs: str = "unknown"
    sandbox_state: dict[str, Any] | None = None
    error: str | None = None


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    eval_report: dict[str, Any] = Field(default_factory=dict)
    trial_id: str | None = None
    trial_name: str | None = None
    task_id: str | None = None
    sandbox_id: str | None = None
    provider_sandbox_id: str | None = None
    session_id: str | None = None
    trial_dir: str | None = None
    external_key: str | None = None
    run_id: str | None = None
    miles_run_id: str | None = None
    rollout_id: int | str | None = None
    sample_id: int | str | None = None
    group_id: int | str | None = None
    step_index: int | None = None
    capture: CaptureResult | None = None


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path = Path("/root/harbor_tasks")
    trials_dir: Path = Path("./trials")
    capture_dir: Path = Path("./trial-captures")
    capture_mode: str = "off"
    sandbox_state: bool = False
    sandbox_state_begin_timeout_sec: float = 120.0
    sandbox_state_end_timeout_sec: float = 300.0
    sandbox_state_exclude: str = ""
    sandbox_state_hash: bool = False
    environment_type: str = "docker"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    delete_environments: bool = True
    max_concurrent: int = 8
    request_timeout_sec: float = 14_400.0
    session_poll_interval_sec: float = 5.0
    auth_token: str = ""
    admin_secret: str = ""
    allowed_callback_hosts: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
    allow_any_callback: bool = False

    def __post_init__(self) -> None:
        if self.capture_mode not in _CAPTURE_MODES:
            choices = ", ".join(sorted(_CAPTURE_MODES))
            raise ValueError(f"MILES_HARBOR_CAPTURE_MODE must be one of: {choices}")
        if self.sandbox_state and self.capture_mode == "off":
            raise ValueError("MILES_SANDBOX_STATE=1 requires MILES_HARBOR_CAPTURE_MODE=shadow or required; the sandbox-state bundle ships inside the staged capture")

    @classmethod
    def from_env(cls) -> Settings:
        raw_hosts = os.getenv("MILES_HARBOR_ALLOWED_CALLBACK_HOSTS", "localhost,127.0.0.1,::1")
        raw_environment_kwargs = os.getenv("MILES_HARBOR_ENVIRONMENT_KWARGS_JSON", "{}")
        try:
            environment_kwargs = json.loads(raw_environment_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError("MILES_HARBOR_ENVIRONMENT_KWARGS_JSON must be valid JSON") from exc
        if not isinstance(environment_kwargs, dict):
            raise ValueError("MILES_HARBOR_ENVIRONMENT_KWARGS_JSON must decode to an object")

        trials_dir = Path(os.getenv("HARBOR_TRIALS_DIR", "./trials"))
        default_capture_dir = trials_dir.parent / f"{trials_dir.name}-captures"
        return cls(
            tasks_dir=Path(os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks")),
            trials_dir=trials_dir,
            capture_dir=Path(os.getenv("MILES_HARBOR_CAPTURE_DIR", str(default_capture_dir))),
            capture_mode=os.getenv("MILES_HARBOR_CAPTURE_MODE", "off").strip().lower(),
            sandbox_state=_env_bool("MILES_SANDBOX_STATE", False),
            sandbox_state_begin_timeout_sec=float(os.getenv("MILES_SANDBOX_STATE_TIMEOUT_BEGIN", "120")),
            sandbox_state_end_timeout_sec=float(os.getenv("MILES_SANDBOX_STATE_TIMEOUT_END", "300")),
            sandbox_state_exclude=os.getenv("MILES_SANDBOX_STATE_EXCLUDE", ""),
            sandbox_state_hash=_env_bool("MILES_SANDBOX_STATE_HASH", False),
            environment_type=os.getenv("HARBOR_ENVIRONMENT_TYPE", "docker"),
            environment_kwargs=environment_kwargs,
            delete_environments=_env_bool("HARBOR_DELETE_ENVIRONMENTS", True),
            max_concurrent=int(os.getenv("AGENT_MAX_CONCURRENT", "8")),
            request_timeout_sec=float(os.getenv("MILES_HARBOR_REQUEST_TIMEOUT_SEC", "14400")),
            session_poll_interval_sec=float(os.getenv("MILES_HARBOR_SESSION_POLL_INTERVAL_SEC", "5")),
            auth_token=os.getenv("MILES_HARBOR_AUTH_TOKEN", ""),
            admin_secret=os.getenv("HARBOR_ADMIN_SECRET", ""),
            allowed_callback_hosts=frozenset(host.strip().lower() for host in raw_hosts.split(",") if host.strip()),
            allow_any_callback=_env_bool("MILES_HARBOR_ALLOW_ANY_CALLBACK", False),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_task_path(tasks_dir: Path, instance_id: str) -> Path:
    """Resolve one flat Harbor task ID without allowing path traversal."""
    if not instance_id or not _SAFE_INSTANCE_ID.fullmatch(instance_id):
        raise ValueError(f"Invalid instance_id: {instance_id!r}")

    root = tasks_dir.expanduser().resolve()
    task_path = (root / instance_id).resolve()
    try:
        task_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Task path escapes HARBOR_TASKS_DIR: {instance_id!r}") from exc
    if not task_path.is_dir():
        raise FileNotFoundError(f"Harbor task not found: {task_path}")
    return task_path


def validate_callback_url(base_url: str, settings: Settings) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute http(s) URL")
    if settings.allow_any_callback:
        return
    if parsed.hostname.lower() not in settings.allowed_callback_hosts:
        raise ValueError(f"Callback host {parsed.hostname!r} is not allowed; add it to MILES_HARBOR_ALLOWED_CALLBACK_HOSTS")


def validate_session_server_id(session_server_id: str, settings: Settings) -> None:
    """Validate the separate Miles session-server origin used for token polling."""
    server_url = _session_server_url(session_server_id)
    parsed = urlparse(server_url)
    if parsed.username or parsed.password:
        raise ValueError("session_server_id must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("session_server_id must be an HTTP origin without a path, query, or fragment")
    validate_callback_url(server_url, settings)


def extract_session_id(base_url: str) -> str | None:
    match = _SESSION_PATH.search(urlparse(base_url).path)
    return unquote(match.group(1)) if match else None


def _capture_directory_name(trial_id: str) -> str:
    """Return a collision-resistant basename even for unusual provider IDs."""
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", trial_id).strip("._-")[:80] or "trial"
    suffix = hashlib.sha256(trial_id.encode()).hexdigest()[:12]
    return f"{slug}-{suffix}"


def stage_trial_capture(
    trial_dir: Path,
    capture_root: Path,
    *,
    trial_id: str,
    task_id: str,
    environment_type: str,
    delete_requested: bool,
    sandbox_id: str | None,
    provider_sandbox_id: str | None,
    session_id: str | None,
    request: RunRequest,
) -> CaptureResult:
    """Call the SDK-owned, non-network Harbor producer adapter."""
    from probe.connectors.harbor import stage_trial_export

    resolved_step_index = request.step_index
    if resolved_step_index is None and isinstance(request.rollout_id, int):
        resolved_step_index = request.rollout_id
    environment = {
        "type": environment_type,
        "sandbox_id": sandbox_id,
        "provider_sandbox_id": provider_sandbox_id,
        "delete_requested": delete_requested,
        "collected": {
            "native_trial_directory": True,
            "staged_after_trial_run_returned": True,
        },
    }
    correlation = {
        "miles_run_id": request.miles_run_id,
        "rollout_id": request.rollout_id,
        "sample_id": request.sample_id,
        "group_id": request.group_id,
        "session_id": session_id,
        "trial_id": trial_id,
        "task_id": task_id,
    }
    destination = capture_root.expanduser().resolve() / _capture_directory_name(trial_id)
    staged = stage_trial_export(
        trial_dir,
        destination,
        run_id=request.run_id,
        step_index=resolved_step_index,
        environment=environment,
        correlation=correlation,
        context=request.capture_context,
        expected_paths=_EXPECTED_TRIAL_FILES,
        expand=False,
    )
    manifest = json.loads(staged.capture_manifest_path.read_text())
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    capture = manifest.get("capture") if isinstance(manifest.get("capture"), dict) else {}
    archive = capture.get("archive") if isinstance(capture.get("archive"), dict) else {}
    completeness = capture.get("completeness") if isinstance(capture.get("completeness"), dict) else {}
    descriptor_correlation = staged.descriptor.get("correlation") or {}
    return CaptureResult(
        status=str(completeness.get("status") or "partial"),
        staged_trial_dir=str(staged.staged_trial.trial_dir),
        archive_path=str(staged.archive_path) if staged.archive_path is not None else None,
        manifest_path=str(staged.capture_manifest_path),
        export_descriptor_path=str(staged.request_path),
        archive_content_hash=archive.get("content_hash"),
        external_key=descriptor_correlation.get("external_key"),
        file_count=len(files),
        size_bytes=sum(item.get("size_bytes", 0) for item in files if isinstance(item, dict) and isinstance(item.get("size_bytes"), int)),
    )


class SandboxStateCapture:
    """Ephemeral begin/end sandbox filesystem snapshots (``probe.sandbox-state/1``).

    Registered on Harbor's Trial lifecycle hooks. Every callback is fail-open:
    Harbor's ``_emit`` propagates hook exceptions and ``AGENT_END`` fires inside
    the agent phase's ``finally``, so an unhandled raise here would mask the
    trial's own outcome. ``asyncio.CancelledError`` is deliberately NOT caught —
    a cancelled trial must keep unwinding, and losing the end snapshot there is
    correct.

    Ephemerality: each phase uploads the static snapshot binary into a random
    ``/tmp`` workdir, execs it, downloads the outputs to the host, verifies them
    against the sha256 trailer the binary prints to stdout, and deletes the
    workdir — all inside the awaited hook, so the container is probe-free for
    the entire agent phase. The bundle itself is authored host-side into the
    trial tree, outside agent reach, and rides the existing staged capture.
    """

    def __init__(self, trial: Any, settings: Settings, host_dir: Path) -> None:
        self._trial = trial
        self._settings = settings
        self._host_dir = host_dir
        self._arch: str | None = None
        self._trailers: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, str] = {}
        self._integrity: dict[str, bool] = {}
        self._errors: list[str] = []
        self.status: dict[str, str | None] = {"begin": None, "end": None}

    def install(self) -> None:
        from harbor.trial.hooks import TrialEvent

        self._trial.add_hook(TrialEvent.AGENT_START, self._hook("begin"))
        self._trial.add_hook(TrialEvent.AGENT_END, self._hook("end"))

    def _hook(self, phase: str) -> Callable[[Any], Awaitable[None]]:
        async def callback(_event: Any) -> None:
            timeout = self._settings.sandbox_state_begin_timeout_sec if phase == "begin" else self._settings.sandbox_state_end_timeout_sec
            try:
                await asyncio.wait_for(self._run_phase(phase), timeout=timeout)
                self.status[phase] = "ok"
            except Exception as exc:  # noqa: BLE001 - fail-open into Harbor's _emit
                logger.exception("sandbox-state %s snapshot failed", phase)
                self.status[phase] = f"{type(exc).__name__}: {exc}"

        return callback

    async def _run_phase(self, phase: str) -> None:
        from probe.connectors import sandbox_state

        if phase == "end" and self.status.get("begin") != "ok":
            raise RuntimeError("begin snapshot unavailable; end delta skipped")
        env = self._trial.agent_environment
        phase_timeout = self._settings.sandbox_state_begin_timeout_sec if phase == "begin" else self._settings.sandbox_state_end_timeout_sec
        # exec timeout bounds the HOST wait; the binary's own --max-seconds
        # (set below it) bounds the in-CONTAINER process so a runaway scan
        # exits itself rather than lingering past the agent phase. Both are
        # needed: Harbor's exec on timeout only tears down the host-side client.
        exec_timeout = max(1, int(phase_timeout))
        self_deadline = max(1.0, phase_timeout - 10)
        workdir = f"/tmp/.psbx-{uuid.uuid4().hex}"
        try:
            if self._arch is None:
                self._arch = await self._detect_arch(env, sandbox_state)
            # Pre-create the workdir so Harbor's primary `docker compose cp`
            # upload path (which does not create parents) succeeds and we do
            # not silently depend on the tar-based fallback needing in-image tar.
            await env.exec(f"mkdir -p {shlex.quote(workdir)}", user="root", timeout_sec=exec_timeout)
            await env.upload_file(sandbox_state.snapshot_binary_path(self._arch), f"{workdir}/snap")
            command = f"chmod +x {workdir}/snap && {workdir}/snap {phase} --workdir {shlex.quote(workdir)} --max-seconds {self_deadline:.0f}"
            if phase == "end":
                await env.upload_file(
                    self._host_dir / sandbox_state.BEGIN_MANIFEST,
                    f"{workdir}/begin.jsonl.gz",
                )
                command += f" --begin-manifest {workdir}/begin.jsonl.gz"
            if self._settings.sandbox_state_exclude:
                command += f" --exclude {shlex.quote(self._settings.sandbox_state_exclude)}"
            if self._settings.sandbox_state_hash:
                command += " --hash"
            result = await env.exec(command, user="root", timeout_sec=exec_timeout)
            if result.return_code != 0:
                stderr_tail = (result.stderr or "").strip()[-500:]
                raise RuntimeError(f"snapshot exited {result.return_code}: {stderr_tail}")
            trailer = sandbox_state.parse_trailer(result.stdout or "")
            for name in _SANDBOX_STATE_OUTPUTS[phase]:
                target = self._host_dir / name
                await env.download_file(f"{workdir}/{name}", target)
                declared = trailer.get("files", {}).get(name, {}).get("sha256")
                # Hash off the event loop: manifests can be hundreds of MB and
                # this hook is awaited inline by Harbor's _emit.
                digest = await asyncio.to_thread(sandbox_state.sha256_file, target)
                self._integrity[name] = bool(declared) and digest == declared
            self._trailers[phase] = trailer
            self._timestamps[phase] = datetime.now(timezone.utc).isoformat()
        finally:
            # The container must be left probe-free even when the phase failed;
            # a dead environment makes this raise, which is fine to swallow. The
            # timeout keeps a hung `rm` from stalling the trial's own unwind.
            with contextlib.suppress(Exception):
                await env.exec(f"rm -rf {shlex.quote(workdir)}", user="root", timeout_sec=exec_timeout)

    async def _detect_arch(self, env: Any, sandbox_state_mod: Any) -> str:
        machine = ""
        try:
            result = await env.exec("uname -m", timeout_sec=30)
            machine = (result.stdout or "").strip()
        except Exception as exc:  # noqa: BLE001 - default arch keeps capture best-effort
            self._errors.append(f"arch detection failed, assuming amd64: {exc}")
            return "amd64"
        arch = sandbox_state_mod.machine_to_arch(machine)
        if arch is None:
            self._errors.append(f"unrecognized machine {machine!r}, assuming amd64")
            return "amd64"
        return arch

    def record_install_failure(self, exc: Exception) -> None:
        self._errors.append(f"hook install failed: {type(exc).__name__}: {exc}")

    def attempted(self) -> bool:
        return any(value is not None for value in self.status.values())

    def write_bundle(self, trial_dir: Path) -> dict[str, Any]:
        """Author the bundle into the host trial tree; sync, never raises.

        Runs via ``asyncio.to_thread`` in the staging path. When no hook ever
        fired (environment never started, install failed) no files are
        written — the summary alone records why.
        """
        from probe.connectors import sandbox_state

        try:
            if self.attempted():
                meta = sandbox_state.build_meta(
                    begin_trailer=self._trailers.get("begin"),
                    end_trailer=self._trailers.get("end"),
                    status=self.status,
                    begin_at=self._timestamps.get("begin"),
                    end_at=self._timestamps.get("end"),
                    arch=self._arch,
                    integrity=self._integrity,
                    errors=self._errors,
                )
                sandbox_state.write_bundle(
                    trial_dir / "artifacts" / sandbox_state.BUNDLE_DIRNAME,
                    {name: self._host_dir / name for outputs in _SANDBOX_STATE_OUTPUTS.values() for name in outputs},
                    meta,
                )
        except Exception as exc:  # noqa: BLE001 - bundle loss must not fail staging
            logger.exception("sandbox-state bundle write failed")
            self._errors.append(f"bundle write failed: {type(exc).__name__}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                shutil.rmtree(self._host_dir, ignore_errors=True)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "probe.sandbox-state/1",
            "status": dict(self.status) if self.attempted() else "not_attempted",
            "arch": self._arch,
            "integrity": dict(self._integrity),
            "errors": list(self._errors),
        }


def build_agent_configuration(request: RunRequest) -> tuple[dict[str, str], dict[str, Any]]:
    """Translate the common Miles request fields into public Harbor agent config."""
    api_key = request.api_key or "dummy"
    agent_env = {
        "OPENAI_BASE_URL": request.base_url,
        "OPENAI_API_BASE": request.base_url,
        "OPENAI_API_KEY": api_key,
        "MSWEA_API_KEY": api_key,
        # Retained for older mini-swe-agent configurations.
        "HOSTED_VLLM_API_BASE": request.base_url,
        "HOSTED_VLLM_API_KEY": api_key,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    sampling = request.sampling_params
    agent_kwargs: dict[str, Any] = {}

    max_tokens = sampling.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        if request.agent_name == "mini-swe-agent":
            agent_kwargs["max_tokens"] = max_tokens

    reasoning_effort = sampling.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort:
        agent_kwargs["reasoning_effort"] = reasoning_effort

    if request.agent_name in _HOST_AGENTS:
        agent_kwargs.update(
            {
                "api_base": request.base_url,
                "enable_summarize": False,
                "model_info": {
                    "max_input_tokens": request.max_seq_len or 32_768,
                    "max_output_tokens": max_tokens or 8_192,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            }
        )
        temperature = sampling.get("temperature")
        if isinstance(temperature, int | float) and not isinstance(temperature, bool):
            agent_kwargs["temperature"] = float(temperature)
        forwarded = {key: value for key, value in sampling.items() if key in {"max_tokens", "top_p", "seed", "stop"} and value is not None}
        if forwarded:
            agent_kwargs["llm_call_kwargs"] = forwarded

    return agent_env, agent_kwargs


def _duration_sec(timing: Any) -> float | None:
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    return (finished - started).total_seconds() if started and finished else None


def normalize_trial_result(result: Any) -> RunResponse:
    verifier_result = getattr(result, "verifier_result", None)
    rewards = getattr(verifier_result, "rewards", None) or {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))

    exception = getattr(result, "exception_info", None)
    exception_type = getattr(exception, "exception_type", "") if exception is not None else ""
    if exception_type in _TIMEOUT_EXCEPTIONS:
        exit_status = "TimeLimitExceeded"
    elif exception_type in _SEQUENCE_EXCEPTIONS:
        exit_status = "SequenceLengthLimitExceeded"
    elif exception is not None:
        exit_status = "AgentError"
    elif verifier_result is not None:
        exit_status = "Submitted"
    else:
        exit_status = "Unknown"

    metrics: dict[str, Any] = {}
    agent_result = getattr(result, "agent_result", None)
    if agent_result is not None:
        for field_name in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
            value = getattr(agent_result, field_name, None)
            if value is not None:
                metrics[field_name] = value
        metadata = getattr(agent_result, "metadata", None)
        if isinstance(metadata, dict):
            metrics.update(metadata)
    for metric_name, timing_name in (("agent_run_time", "agent_execution"), ("eval_time", "verifier")):
        duration = _duration_sec(getattr(result, timing_name, None))
        if duration is not None:
            metrics[metric_name] = duration

    return RunResponse(
        reward=reward,
        exit_status=exit_status,
        agent_metrics=metrics,
        eval_report=dict(rewards),
    )


def _session_server_url(session_server_id: str) -> str:
    return session_server_id.rstrip("/") if "://" in session_server_id else f"http://{session_server_id}"


def _nonempty_identifier(value: Any) -> str | None:
    """Normalize an SDK identifier without invoking provider methods."""
    if value is None or callable(value):
        return None
    text = str(value).strip()
    return text or None


def _safe_identifier_attr(value: Any, attribute: str) -> str | None:
    try:
        return _nonempty_identifier(getattr(value, attribute, None))
    except Exception:  # noqa: BLE001 - provider SDK properties are best-effort
        return None


def _provider_sandbox_id(environment: Any) -> str | None:
    """Read a live provider ID across Harbor's supported cloud backends.

    Harbor intentionally keeps provider handles private. The public bridge only
    reads stable SDK identifiers while the environment is alive and retains the
    resulting string; it never calls provider methods or keeps the handle.
    """
    if environment is None:
        return None

    for attribute in ("provider_sandbox_id", "provider_ref"):
        if provider_id := _safe_identifier_attr(environment, attribute):
            return provider_id

    # Daytona and E2B use _sandbox (id and sandbox_id respectively), Modal uses
    # _sandbox.object_id, and Runloop uses _devbox.id.
    for handle_attribute in ("_sandbox", "_devbox"):
        try:
            handle = getattr(environment, handle_attribute, None)
        except Exception:  # noqa: BLE001 - provider SDK properties are best-effort
            continue
        if handle is None:
            continue
        for identifier_attribute in ("sandbox_id", "object_id", "id"):
            if provider_id := _safe_identifier_attr(handle, identifier_attribute):
                return provider_id
    return None


def _sandbox_correlation(trial: Any) -> tuple[str | None, str | None]:
    """Return Harbor's logical sandbox ID and a live best-effort provider ID."""
    environment = getattr(trial, "agent_environment", None)
    logical_id = _safe_identifier_attr(environment, "session_id")
    provider_id = _provider_sandbox_id(environment)
    return (
        logical_id,
        provider_id,
    )


class SandboxCorrelationCapture:
    """Retain sandbox identifiers before Harbor clears provider handles."""

    def __init__(self, trial: Any) -> None:
        self._trial = trial
        self.sandbox_id: str | None = None
        self.provider_sandbox_id: str | None = None

    def install(self) -> None:
        from harbor.trial.hooks import TrialEvent

        self._trial.add_hook(TrialEvent.AGENT_START, self._capture)
        self._trial.add_hook(TrialEvent.AGENT_END, self._capture)

    async def _capture(self, _event: Any) -> None:
        sandbox_id, provider_sandbox_id = _sandbox_correlation(self._trial)
        self.sandbox_id = sandbox_id or self.sandbox_id
        self.provider_sandbox_id = provider_sandbox_id or self.provider_sandbox_id

    def resolved(self) -> tuple[str | None, str | None]:
        """Prefer retained live values, with a post-run fallback for old Harbor."""
        sandbox_id, provider_sandbox_id = _sandbox_correlation(self._trial)
        return (
            self.sandbox_id or sandbox_id,
            self.provider_sandbox_id or provider_sandbox_id,
        )


async def _poll_until_sequence_limit(request: RunRequest, settings: Settings) -> int:
    """Return the observed token count once it reaches the configured limit."""
    assert request.session_server_id is not None
    assert request.max_seq_len is not None
    session_id = extract_session_id(request.base_url)
    if session_id is None:
        raise ValueError("Cannot monitor max_seq_len: base_url has no /sessions/<id> component")

    server_url = _session_server_url(request.session_server_id)
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {"Authorization": f"Bearer {request.api_key}"} if request.api_key else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        if request.session_server_instance_id:
            health = await client.get(f"{server_url}/health", headers=headers)
            health.raise_for_status()
            actual_id = health.json().get("session_server_instance_id")
            if actual_id != request.session_server_instance_id:
                raise RuntimeError(f"Miles session-server identity changed during rollout (expected {request.session_server_instance_id!r}, got {actual_id!r})")

        while True:
            try:
                response = await client.get(f"{server_url}/sessions/{session_id}", headers=headers)
                response.raise_for_status()
                token_ids = response.json().get("metadata", {}).get("accumulated_token_ids", [])
                token_count = len(token_ids) if isinstance(token_ids, list) else 0
                if token_count >= request.max_seq_len:
                    return token_count
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                logger.warning("Session monitor poll failed for %s: %s", session_id, exc)
            await asyncio.sleep(settings.session_poll_interval_sec)


async def run_public_harbor_trial(request: RunRequest, settings: Settings) -> RunResponse:
    """Create and run one trial using the current public Harbor API."""
    try:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial
    except ImportError as exc:
        raise RuntimeError("Public Harbor is not installed; install the 'harbor' package") from exc

    task_path = resolve_task_path(settings.tasks_dir, request.instance_id)
    agent_env, agent_kwargs = build_agent_configuration(request)
    settings.trials_dir.mkdir(parents=True, exist_ok=True)

    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trials_dir=settings.trials_dir,
        agent=AgentConfig(
            name=request.agent_name,
            model_name=request.model,
            env=agent_env,
            kwargs=agent_kwargs,
        ),
        environment=EnvironmentConfig(
            type=settings.environment_type,
            delete=settings.delete_environments,
            kwargs=settings.environment_kwargs,
        ),
    )
    trial = await Trial.create(config)

    sandbox_correlation: SandboxCorrelationCapture | None = None
    if settings.capture_mode != "off":
        sandbox_correlation = SandboxCorrelationCapture(trial)
        try:
            sandbox_correlation.install()
        except Exception:  # noqa: BLE001 - metadata must never block a trial
            logger.exception("sandbox-correlation hook install failed for %s", request.instance_id)

    sandbox_capture: SandboxStateCapture | None = None
    if settings.sandbox_state:
        capture_root = settings.capture_dir.expanduser().resolve()
        capture_root.mkdir(parents=True, exist_ok=True)
        host_dir = Path(tempfile.mkdtemp(prefix=".psbx-host-", dir=capture_root))
        sandbox_capture = SandboxStateCapture(trial, settings, host_dir)
        try:
            sandbox_capture.install()
        except Exception as exc:  # noqa: BLE001 - capture is best-effort, trial must run
            logger.exception("sandbox-state hook install failed for %s", request.instance_id)
            sandbox_capture.record_install_failure(exc)

    trial_task = asyncio.create_task(trial.run(), name=f"harbor-trial-{request.instance_id}")

    monitor_task: asyncio.Task[int] | None = None
    if request.max_seq_len and request.session_server_id:
        monitor_task = asyncio.create_task(
            _poll_until_sequence_limit(request, settings),
            name=f"session-monitor-{request.instance_id}",
        )

    async def wait_for_trial() -> RunResponse:
        if monitor_task is None:
            return normalize_trial_result(await trial_task)

        done, _ = await asyncio.wait({trial_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)
        if trial_task in done:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
            return normalize_trial_result(trial_task.result())

        try:
            observed_tokens = monitor_task.result()
        except Exception:
            trial_task.cancel()
            await asyncio.gather(trial_task, return_exceptions=True)
            raise
        trial_task.cancel()
        await asyncio.gather(trial_task, return_exceptions=True)
        return RunResponse(
            reward=0.0,
            exit_status="SequenceLengthLimitExceeded",
            agent_metrics={"observed_tokens": observed_tokens},
            eval_report={},
        )

    response: RunResponse | None = None
    capture: CaptureResult | None = None
    trial_id: str | None = None
    trial_name: str | None = None
    trial_dir: Path | None = None
    sandbox_id: str | None = None
    provider_sandbox_id: str | None = None
    try:
        response = await asyncio.wait_for(wait_for_trial(), timeout=settings.request_timeout_sec)
    except TimeoutError:
        trial_task.cancel()
        if monitor_task is not None:
            monitor_task.cancel()
        await asyncio.gather(*(task for task in (trial_task, monitor_task) if task is not None), return_exceptions=True)
        response = RunResponse(exit_status="TimeLimitExceeded")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Harbor trial execution failed for %s", request.instance_id)
        response = RunResponse(exit_status=f"Error: {type(exc).__name__}")
    finally:
        if not trial_task.done():
            trial_task.cancel()
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()
        await asyncio.gather(*(task for task in (trial_task, monitor_task) if task is not None), return_exceptions=True)

        if settings.capture_mode != "off":
            trial_id = str(trial.id)
            trial_name = str(trial.config.trial_name)
            trial_dir = Path(trial.paths.trial_dir).expanduser().resolve()
            if sandbox_correlation is not None:
                sandbox_id, provider_sandbox_id = sandbox_correlation.resolved()
            else:
                sandbox_id, provider_sandbox_id = _sandbox_correlation(trial)
            sandbox_state_summary: dict[str, Any] | None = None
            if sandbox_capture is not None:
                # Bundle authored into the trial tree BEFORE staging so the
                # existing stage -> export pipeline carries it untouched.
                sandbox_state_summary = await asyncio.to_thread(sandbox_capture.write_bundle, trial_dir)
                request.capture_context = {
                    **request.capture_context,
                    "sandbox_state": sandbox_state_summary,
                }
            try:
                capture = await asyncio.to_thread(
                    stage_trial_capture,
                    trial_dir,
                    settings.capture_dir,
                    trial_id=trial_id,
                    task_id=request.instance_id,
                    environment_type=settings.environment_type,
                    delete_requested=settings.delete_environments,
                    sandbox_id=sandbox_id,
                    provider_sandbox_id=provider_sandbox_id,
                    session_id=extract_session_id(request.base_url),
                    request=request,
                )
            except Exception as exc:
                logger.exception("Failed to stage Harbor trial capture %s", trial_id)
                capture = CaptureResult(status="failed", error=f"{type(exc).__name__}: {exc}")

    assert response is not None
    if settings.capture_mode == "off":
        return response

    assert trial_id is not None and trial_name is not None and trial_dir is not None
    assert capture is not None
    if sandbox_state_summary is not None:
        capture = capture.model_copy(update={"sandbox_state": sandbox_state_summary})
    resolved_step_index = request.step_index
    if resolved_step_index is None and isinstance(request.rollout_id, int):
        resolved_step_index = request.rollout_id
    if settings.capture_mode == "required" and capture.status != "complete":
        response = response.model_copy(update={"exit_status": "Error: CaptureRequiredError"})
    return response.model_copy(
        update={
            "trial_id": trial_id,
            "trial_name": trial_name,
            "task_id": request.instance_id,
            "sandbox_id": sandbox_id,
            "provider_sandbox_id": provider_sandbox_id,
            "session_id": extract_session_id(request.base_url),
            "trial_dir": str(trial_dir),
            "external_key": capture.external_key,
            "run_id": request.run_id,
            "miles_run_id": request.miles_run_id,
            "rollout_id": request.rollout_id,
            "sample_id": request.sample_id,
            "group_id": request.group_id,
            "step_index": resolved_step_index,
            "capture": capture,
        }
    )


TrialRunner = Callable[[RunRequest, Settings], Awaitable[RunResponse]]


def create_app(
    settings: Settings | None = None,
    trial_runner: TrialRunner = run_public_harbor_trial,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.sandbox_state:
        _install_hint = "MILES_SANDBOX_STATE=1 needs probe-research >= 0.9.1 with the packaged probe-sandbox-snapshot binaries. Install from git per requirements-public-harbor-capture.txt (PyPI's 0.9.0 does not carry the binaries)."
        try:
            from probe.connectors import sandbox_state as _sandbox_state
        except ImportError as exc:
            raise RuntimeError(_install_hint) from exc
        # Fail fast at startup if the binaries were stripped from the install,
        # rather than fail-open once per trial with no bundle.
        try:
            for _arch in ("amd64", "arm64"):
                _sandbox_state.snapshot_binary_path(_arch)
        except FileNotFoundError as exc:
            raise RuntimeError(_install_hint) from exc
    semaphore = asyncio.Semaphore(settings.max_concurrent)
    active_tasks: dict[str, set[asyncio.Task[Any]]] = {}
    app = FastAPI(title="Miles Public Harbor Bridge")

    def require_bearer(http_request: Request, *tokens: str) -> None:
        expected = [f"Bearer {token}" for token in tokens if token]
        if expected and not any(secrets.compare_digest(http_request.headers.get("authorization", ""), item) for item in expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        result = {
            "status": "ok",
            "environment_type": settings.environment_type,
            "max_concurrent": settings.max_concurrent,
            "capture_mode": settings.capture_mode,
            "sandbox_state": settings.sandbox_state,
        }
        if settings.capture_mode != "off":
            result["capture_dir"] = str(settings.capture_dir.expanduser().resolve())
        return result

    @app.post("/run", response_model=RunResponse, response_model_exclude_none=True)
    async def run_trial(payload: RunRequest, http_request: Request) -> RunResponse:
        require_bearer(http_request, settings.auth_token)
        try:
            validate_callback_url(payload.base_url, settings)
            if payload.session_server_id:
                validate_session_server_id(payload.session_server_id, settings)
            resolve_task_path(settings.tasks_dir, payload.instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async with semaphore:
            task = asyncio.current_task()
            instance_id = payload.session_server_instance_id
            if task is not None and instance_id:
                active_tasks.setdefault(instance_id, set()).add(task)
            try:
                return await trial_runner(payload, settings)
            except Exception as exc:
                logger.exception("Harbor trial failed for %s", payload.instance_id)
                return RunResponse(exit_status=f"Error: {type(exc).__name__}")
            finally:
                if task is not None and instance_id:
                    tasks = active_tasks.get(instance_id)
                    if tasks is not None:
                        tasks.discard(task)
                        if not tasks:
                            active_tasks.pop(instance_id, None)

    @app.post("/flush")
    async def flush(payload: FlushRequest, http_request: Request) -> dict[str, Any]:
        """Cancel in-flight trials for one Miles session-server generation."""
        require_bearer(http_request, settings.admin_secret or settings.auth_token)
        tasks = list(active_tasks.get(payload.session_server_instance_id, set()))
        current = asyncio.current_task()
        tasks = [task for task in tasks if task is not current and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "session_server_instance_id": payload.session_server_instance_id,
            "cancelled": len(tasks),
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
