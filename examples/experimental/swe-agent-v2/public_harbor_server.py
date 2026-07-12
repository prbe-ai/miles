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
import json
import logging
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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

    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    eval_report: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path = Path("/root/harbor_tasks")
    trials_dir: Path = Path("./trials")
    environment_type: str = "docker"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    delete_environments: bool = True
    max_concurrent: int = 8
    request_timeout_sec: float = 14_400.0
    session_poll_interval_sec: float = 5.0
    auth_token: str = ""
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

        return cls(
            tasks_dir=Path(os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks")),
            trials_dir=Path(os.getenv("HARBOR_TRIALS_DIR", "./trials")),
            environment_type=os.getenv("HARBOR_ENVIRONMENT_TYPE", "docker"),
            environment_kwargs=environment_kwargs,
            delete_environments=_env_bool("HARBOR_DELETE_ENVIRONMENTS", True),
            max_concurrent=int(os.getenv("AGENT_MAX_CONCURRENT", "8")),
            request_timeout_sec=float(os.getenv("MILES_HARBOR_REQUEST_TIMEOUT_SEC", "14400")),
            session_poll_interval_sec=float(os.getenv("MILES_HARBOR_SESSION_POLL_INTERVAL_SEC", "5")),
            auth_token=os.getenv("MILES_HARBOR_AUTH_TOKEN", ""),
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

    try:
        return await asyncio.wait_for(wait_for_trial(), timeout=settings.request_timeout_sec)
    except TimeoutError:
        trial_task.cancel()
        if monitor_task is not None:
            monitor_task.cancel()
        await asyncio.gather(*(task for task in (trial_task, monitor_task) if task is not None), return_exceptions=True)
        return RunResponse(exit_status="TimeLimitExceeded")


TrialRunner = Callable[[RunRequest, Settings], Awaitable[RunResponse]]


def create_app(
    settings: Settings | None = None,
    trial_runner: TrialRunner = run_public_harbor_trial,
) -> FastAPI:
    settings = settings or Settings.from_env()
    semaphore = asyncio.Semaphore(settings.max_concurrent)
    app = FastAPI(title="Miles Public Harbor Bridge")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment_type": settings.environment_type,
            "max_concurrent": settings.max_concurrent,
        }

    @app.post("/run", response_model=RunResponse)
    async def run_trial(payload: RunRequest, http_request: Request) -> RunResponse:
        if settings.auth_token:
            supplied = http_request.headers.get("authorization", "")
            expected = f"Bearer {settings.auth_token}"
            if not secrets.compare_digest(supplied, expected):
                raise HTTPException(status_code=401, detail="Invalid bearer token")
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
            try:
                return await trial_runner(payload, settings)
            except Exception as exc:
                logger.exception("Harbor trial failed for %s", payload.instance_id)
                return RunResponse(exit_status=f"Error: {type(exc).__name__}")

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
