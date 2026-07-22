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
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tarfile
import tempfile
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
_CAPTURE_SCHEMA_VERSION = "1.0"
_EXPECTED_TRIAL_FILES = ("config.json", "lock.json", "result.json")
_TOP_LEVEL_ROLES = {
    "config.json": "config",
    "lock.json": "lock",
    "result.json": "result",
    "reward.json": "reward",
    "trajectory.json": "trajectory",
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
    capture: CaptureResult = Field(default_factory=CaptureResult)


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path = Path("/root/harbor_tasks")
    trials_dir: Path = Path("./trials")
    capture_dir: Path = Path("./trial-captures")
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
        raise ValueError(
            f"Callback host {parsed.hostname!r} is not allowed; add it to MILES_HARBOR_ALLOWED_CALLBACK_HOSTS"
        )


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


def _role_for(relative_path: str) -> str:
    """Use the same fork-tolerant roles as probe.connectors.harbor."""
    parts = Path(relative_path).parts
    if len(parts) == 1 and parts[0] in _TOP_LEVEL_ROLES:
        return _TOP_LEVEL_ROLES[parts[0]]
    if not parts:
        return "other"
    if parts[0] == "agent" or parts[:2] == ("logs", "agent"):
        return "agent_log"
    if parts[0] == "verifier" or parts[:2] == ("logs", "verifier"):
        return "verifier"
    if parts[0] == "output":
        return "output"
    return "other"


def _fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _trial_file_inventory(trial_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    """Inventory every regular file and symlink without following symlinks."""
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    total_size = 0
    for path in sorted(trial_dir.rglob("*"), key=lambda item: item.relative_to(trial_dir).as_posix()):
        relative = path.relative_to(trial_dir).as_posix()
        if path.is_symlink():
            symlinks.append({"path": relative, "target": os.readlink(path)})
        elif path.is_file():
            content_hash, size_bytes = _fingerprint(path)
            files.append(
                {
                    "role": _role_for(relative),
                    "path": relative,
                    "content_hash": content_hash,
                    "size_bytes": size_bytes,
                }
            )
            total_size += size_bytes
    return files, symlinks, total_size


def _phase_timings(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    for name in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        timing = result.get(name)
        if isinstance(timing, dict):
            phases[name] = {key: timing.get(key) for key in ("started_at", "finished_at")}
    return phases


def _capture_external_key(trial_id: str, request: RunRequest) -> str:
    identity = {
        "miles_run_id": request.miles_run_id,
        "rollout_id": request.rollout_id,
        "group_id": request.group_id,
        "sample_id": request.sample_id,
        "harbor_trial_id": trial_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"miles-harbor:{hashlib.sha256(encoded).hexdigest()}"


def _capture_directory_name(trial_id: str) -> str:
    """Return a collision-resistant basename even for unusual provider IDs."""
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", trial_id).strip("._-")[:80] or "trial"
    suffix = hashlib.sha256(trial_id.encode()).hexdigest()[:12]
    return f"{slug}-{suffix}"


def _write_json_durably(path: Path, value: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some shared/network filesystems do not implement directory fsync.
        # File fsync plus atomic rename is the strongest contract they expose.
        pass


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Flush copied bytes and directory entries before publishing the capture."""
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            _fsync_file(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _completed_capture(final_dir: Path, *, trial_id: str, external_key: str) -> CaptureResult | None:
    """Reuse an atomically published capture when staging the same Harbor trial."""
    manifest_path = final_dir / "capture-manifest.json"
    archive_path = final_dir / "trial.tar.gz"
    staged_trial_dir = final_dir / "trial"
    if not (manifest_path.is_file() and archive_path.is_file() and staged_trial_dir.is_dir()):
        return None
    manifest = _load_json(manifest_path)
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    trial = manifest.get("trial") if isinstance(manifest.get("trial"), dict) else {}
    if trial.get("id") != trial_id or source.get("external_key") != external_key:
        return None
    capture = manifest.get("capture") if isinstance(manifest.get("capture"), dict) else {}
    completeness = capture.get("completeness") if isinstance(capture.get("completeness"), dict) else {}
    archive = capture.get("archive") if isinstance(capture.get("archive"), dict) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    return CaptureResult(
        status=str(completeness.get("status") or "complete"),
        staged_trial_dir=str(staged_trial_dir),
        archive_path=str(archive_path),
        manifest_path=str(manifest_path),
        export_descriptor_path=str(final_dir / "export-request.json"),
        archive_content_hash=archive.get("content_hash"),
        external_key=external_key,
        file_count=len(files),
        size_bytes=sum(
            entry.get("size_bytes", 0)
            for entry in files
            if isinstance(entry, dict) and isinstance(entry.get("size_bytes", 0), int)
        ),
    )


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
    """Atomically stage Harbor's native trial tree, inventory, and archive.

    The staged ``trial/`` directory intentionally remains byte/layout compatible
    with ``probe.connectors.harbor.capture_trial``. Unknown and binary files are
    copied without interpretation. Symlinks are preserved in the archive but
    never followed while hashing, preventing a sandbox-created link from reading
    files outside the trial directory.
    """
    source = trial_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Harbor trial directory does not exist: {source}")

    root = capture_root.expanduser().resolve()
    if root == source or root.is_relative_to(source):
        raise ValueError("MILES_HARBOR_CAPTURE_DIR must not be inside a Harbor trial directory")
    root.mkdir(parents=True, exist_ok=True)

    external_key = _capture_external_key(trial_id, request)
    final_dir = root / _capture_directory_name(trial_id)
    if final_dir.exists():
        existing = _completed_capture(final_dir, trial_id=trial_id, external_key=external_key)
        if existing is not None:
            return existing
        raise FileExistsError(f"Conflicting or incomplete capture exists for Harbor trial {trial_id}: {final_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=root))
    try:
        staged_trial_dir = temporary_dir / "trial"
        shutil.copytree(source, staged_trial_dir, symlinks=True, copy_function=shutil.copy2)
        _fsync_tree(staged_trial_dir)
        files, symlinks, size_bytes = _trial_file_inventory(staged_trial_dir)
        discovered_paths = {entry["path"] for entry in files}
        expected_files = [
            {
                "path": path,
                "required": True,
                "state": "present" if path in discovered_paths else "missing",
            }
            for path in _EXPECTED_TRIAL_FILES
        ]
        missing_required = [entry["path"] for entry in expected_files if entry["state"] == "missing"]
        completeness_status = "complete" if not missing_required else "partial"

        archive_path = temporary_dir / "trial.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staged_trial_dir, arcname=source.name, recursive=True)
        _fsync_file(archive_path)
        archive_content_hash, _ = _fingerprint(archive_path)

        result = _load_json(staged_trial_dir / "result.json")
        verifier_result = result.get("verifier_result")
        rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
        reward = None
        if isinstance(rewards, dict) and rewards:
            reward = rewards.get("reward", next(iter(rewards.values())))
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
            "external_key": external_key,
            "probe_run_id": request.run_id,
            "miles_run_id": request.miles_run_id,
            "rollout_id": request.rollout_id,
            "sample_id": request.sample_id,
            "group_id": request.group_id,
            "step_index": resolved_step_index,
            "session_id": session_id,
            "trial_id": trial_id,
        }
        manifest = {
            "schema_version": _CAPTURE_SCHEMA_VERSION,
            "trial": {
                "id": trial_id,
                "name": result.get("trial_name") or source.name,
                "task_name": result.get("task_name") or task_id,
                "task_id": task_id,
                "task_checksum": result.get("task_checksum"),
                "trial_uri": result.get("trial_uri") or source.as_uri(),
            },
            "agent": result.get("agent_info"),
            "verifier": {"reward": reward, "rewards": rewards} if isinstance(rewards, dict) else None,
            "phases": _phase_timings(result),
            "environment": environment,
            "exception": result.get("exception_info"),
            "source": {
                "mode": "bridge-hook",
                **correlation,
                "context": request.capture_context,
            },
            "files": files,
            "capture": {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "completeness": {
                    "status": completeness_status,
                    "scope": "host_harbor_trial_tree",
                    "inventory_complete": True,
                    "expected": expected_files,
                    "missing_required": missing_required,
                    "sandbox_state_outside_harbor_outputs": "unknown",
                    "reason": (
                        "Harbor Trial.run() stops/deletes the sandbox before returning; "
                        "only files Harbor persisted into its host trial tree are observable."
                    ),
                },
                "archive": {
                    "path": "trial.tar.gz",
                    "content_hash": archive_content_hash,
                },
                "symlinks": symlinks,
            },
        }
        manifest_path = temporary_dir / "capture-manifest.json"
        _write_json_durably(manifest_path, manifest)

        # A future exporter can watch these descriptors without importing Probe
        # into the Harbor service. The staged trial path is directly consumable by
        # probe.connectors.harbor.capture_trial or `probe trial add`.
        export_descriptor = {
            "schema_version": "probe-harbor-export/1",
            "request_id": external_key,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "last_error": None,
            "target": {"kind": "probe_run", "run_id": request.run_id},
            "connector": "probe.connectors.harbor.capture_trial",
            "arguments": {
                "trial_dir": "trial",
                "trial_dir_base": "descriptor_dir",
                "step_index": resolved_step_index,
                "environment": environment,
                "source_mode": "bridge-hook",
                "expand": False,
            },
            "correlation": correlation,
            "capture_manifest": "capture-manifest.json",
            "archive": "trial.tar.gz",
        }
        export_descriptor_path = temporary_dir / "export-request.json"
        _write_json_durably(export_descriptor_path, export_descriptor)

        # The directory rename publishes only complete captures. The parent fsync
        # makes that publication durable on filesystems that implement fsync.
        _fsync_directory(temporary_dir)
        os.replace(temporary_dir, final_dir)
        _fsync_directory(root)
        return CaptureResult(
            status=completeness_status,
            staged_trial_dir=str(final_dir / "trial"),
            archive_path=str(final_dir / "trial.tar.gz"),
            manifest_path=str(final_dir / "capture-manifest.json"),
            export_descriptor_path=str(final_dir / "export-request.json"),
            archive_content_hash=archive_content_hash,
            external_key=external_key,
            file_count=len(files),
            size_bytes=size_bytes,
        )
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


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
        forwarded = {
            key: value
            for key, value in sampling.items()
            if key in {"max_tokens", "top_p", "seed", "stop"} and value is not None
        }
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


def _sandbox_correlation(trial: Any) -> tuple[str | None, str | None]:
    """Return Harbor's logical sandbox ID and a best-effort provider ID."""
    environment = getattr(trial, "agent_environment", None)
    logical_id = getattr(environment, "session_id", None)
    native_sandbox = getattr(environment, "_sandbox", None)
    provider_id = getattr(native_sandbox, "id", None)
    return (
        str(logical_id) if logical_id is not None else None,
        str(provider_id) if provider_id is not None else None,
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
                raise RuntimeError(
                    f"Miles session-server identity changed during rollout (expected {request.session_server_instance_id!r}, got {actual_id!r})"
                )

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
    try:
        response = await asyncio.wait_for(wait_for_trial(), timeout=settings.request_timeout_sec)
    except TimeoutError:
        trial_task.cancel()
        if monitor_task is not None:
            monitor_task.cancel()
        await asyncio.gather(
            *(task for task in (trial_task, monitor_task) if task is not None), return_exceptions=True
        )
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
        await asyncio.gather(
            *(task for task in (trial_task, monitor_task) if task is not None), return_exceptions=True
        )

        trial_id = str(trial.id)
        trial_name = str(trial.config.trial_name)
        trial_dir = Path(trial.paths.trial_dir).expanduser().resolve()
        sandbox_id, provider_sandbox_id = _sandbox_correlation(trial)
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
    resolved_step_index = request.step_index
    if resolved_step_index is None and isinstance(request.rollout_id, int):
        resolved_step_index = request.rollout_id
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
    semaphore = asyncio.Semaphore(settings.max_concurrent)
    active_tasks: dict[str, set[asyncio.Task[Any]]] = {}
    app = FastAPI(title="Miles Public Harbor Bridge")

    def require_bearer(http_request: Request, *tokens: str) -> None:
        expected = [f"Bearer {token}" for token in tokens if token]
        if expected and not any(
            secrets.compare_digest(http_request.headers.get("authorization", ""), item) for item in expected
        ):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment_type": settings.environment_type,
            "max_concurrent": settings.max_concurrent,
            "capture_dir": str(settings.capture_dir.expanduser().resolve()),
        }

    @app.post("/run", response_model=RunResponse)
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
