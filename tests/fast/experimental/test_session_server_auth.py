from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "miles/rollout/session/server.py"
_SPEC = importlib.util.spec_from_file_location("session_server_auth_contract", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
session_server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = session_server


def _install_stub(name: str, **attributes):
    original = sys.modules.get(name)
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return original


class _ProxyRequest:
    pass


def _setup_session_routes(app, backend, args):
    @app.get("/health")
    async def health():
        return {"status": "ok"}


_STUBS = {
    "setproctitle": _install_stub("setproctitle", setproctitle=lambda value: None),
    "miles.rollout.session.core": _install_stub("miles.rollout.session.core", ProxyRequest=_ProxyRequest),
    "miles.rollout.session.sessions": _install_stub(
        "miles.rollout.session.sessions", setup_session_routes=_setup_session_routes
    ),
    "miles.utils.logging_utils": _install_stub("miles.utils.logging_utils", configure_logger_raw=lambda value: None),
}
_SPEC.loader.exec_module(session_server)
for _name, _original in _STUBS.items():
    if _original is None:
        del sys.modules[_name]
    else:
        sys.modules[_name] = _original


@pytest.mark.asyncio
async def test_session_server_requires_configured_bearer_token() -> None:
    args = SimpleNamespace(miles_router_timeout=30, session_server_api_key="test-session-key")
    server = session_server.SessionServer(args, backend_url="http://backend.invalid")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://session-server",
    ) as client:
        assert (await client.get("/health")).status_code == 401
        assert (await client.get("/health", headers={"Authorization": "Bearer wrong-test-key"})).status_code == 401
        response = await client.get(
            "/health",
            headers={"Authorization": "Bearer test-session-key"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
