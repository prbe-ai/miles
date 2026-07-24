"""
Custom agent function for agentic_tool_call.generate.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

_CAPTURE_CONTEXT_KEYS = {
    "client_id",
    "customer_id",
    "data_mix",
    "data_mix_id",
    "data_source_id",
    "dataset",
    "dataset_id",
    "dataset_name",
    "dataset_split",
    "mix",
    "mix_id",
    "osmosis_dataset_id",
    "osmosis_mix_id",
    "split",
    "task_type",
}
_CAPTURE_CONTEXT_PREFIXES = ("data_mix_", "dataset_", "osmosis_")


def _external_origin() -> str:
    """Return and validate the optional public session-server origin."""
    value = os.getenv("MILES_ROUTER_EXTERNAL_BASE_URL", "").strip().rstrip("/")
    if not value:
        return ""

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MILES_ROUTER_EXTERNAL_BASE_URL must be an http(s) origin")
    if parsed.username or parsed.password:
        raise ValueError("MILES_ROUTER_EXTERNAL_BASE_URL must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("MILES_ROUTER_EXTERNAL_BASE_URL must not contain a path, query, or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _externalize_session_url(url: str, external_origin: str, external_host: str) -> str:
    parsed = urlsplit(url)
    if external_origin:
        public = urlsplit(external_origin)
        return urlunsplit((public.scheme, public.netloc, parsed.path, parsed.query, parsed.fragment))
    if external_host:
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return url


def _capture_context(metadata: dict[str, Any]) -> dict[str, Any]:
    """Lift bounded mix/dataset identity into the stack-agnostic descriptor."""
    explicit = metadata.get("capture_context")
    candidates = list(explicit.items()) if isinstance(explicit, dict) else []
    candidates.extend(metadata.items())
    context: dict[str, Any] = {}
    for key, value in candidates:
        if not isinstance(key, str):
            continue
        normalized = key.lower()
        is_explicit = isinstance(explicit, dict) and key in explicit
        if (
            not is_explicit
            and normalized not in _CAPTURE_CONTEXT_KEYS
            and not normalized.startswith(_CAPTURE_CONTEXT_PREFIXES)
        ):
            continue
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            continue
        if len(encoded.encode()) <= 4096:
            context.setdefault(key, value)
    return context


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    model_name = os.getenv(
        "AGENT_MODEL_NAME",
        os.getenv("SWE_AGENT_MODEL_NAME", "model"),
    )
    auth_token = os.getenv(
        "AGENT_SERVER_AUTH_TOKEN",
        os.getenv("MILES_HARBOR_AUTH_TOKEN", ""),
    )
    server_timeout_sec = float(os.getenv("AGENT_SERVER_TIMEOUT_SEC", "14400"))
    session_url = f"{base_url.rstrip('/')}/v1"
    external_origin = _external_origin()
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    session_url = _externalize_session_url(session_url, external_origin, external_host or "")
    session_api_key = os.getenv("MILES_SESSION_API_KEY", "")

    request: dict[str, Any] = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
        "api_key": session_api_key or "dummy",
    }
    if capture_context := _capture_context(metadata):
        request["capture_context"] = capture_context

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None:
        if external_origin:
            session_server_id = external_origin
        elif external_host:
            port = urlsplit(f"http://{session_server_id}").port
            session_server_id = f"{external_host}:{port}"
        request["session_server_id"] = session_server_id

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    try:
        response = await asyncio.wait_for(
            post(
                f"{agent_server_url}/run",
                request,
                headers={"Authorization": f"Bearer {auth_token}"} if auth_token else None,
            ),
            timeout=server_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.error("Agent server call timed out after %ss", server_timeout_sec)
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    result = {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }
    for key in (
        "trial_id",
        "trial_name",
        "task_id",
        "sandbox_id",
        "provider_sandbox_id",
        "session_id",
        "trial_dir",
        "external_key",
        "run_id",
        "miles_run_id",
        "rollout_id",
        "sample_id",
        "group_id",
        "step_index",
        "capture",
    ):
        if key in response:
            result[key] = response[key]
    return result


async def abort(args) -> None:
    """Teardown hook for oversampling abort (called by sglang_rollout.abort).

    When Miles has enough samples and aborts SGLang, the in-flight Harbor trials
    keep looping and hitting SGLang until they hit their own max_seq_len/timeout.
    Flush the agent server so it cancels those ``/run`` tasks and releases their
    containers. No-op unless AGENT_SERVER_URL and session_server_instance_id are
    available.
    """
    agent_server_url = os.getenv("AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL"))
    instance_id = getattr(args, "session_server_instance_id", None)
    if not agent_server_url or not instance_id:
        return

    headers = None
    admin_secret = os.getenv(
        "HARBOR_ADMIN_SECRET",
        os.getenv("AGENT_SERVER_AUTH_TOKEN", os.getenv("MILES_HARBOR_AUTH_TOKEN", "")),
    )
    if admin_secret:
        headers = {"Authorization": f"Bearer {admin_secret}"}

    try:
        result = await post(
            f"{agent_server_url.rstrip('/')}/flush",
            {"session_server_instance_id": instance_id},
            max_retries=3,
            headers=headers,
        )
        logger.info(f"Flushed agent server {agent_server_url}: {result}")
    except Exception as e:
        logger.warning(f"Failed to flush agent server {agent_server_url}: {e}")
