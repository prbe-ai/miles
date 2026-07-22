from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType

from miles.utils.tracking_utils import probe_utils
from miles.utils.tracking_utils.base import TrackingBackend, TrackingManager


class FakeSdkBackend:
    instances: list[FakeSdkBackend] = []

    def __init__(self) -> None:
        self.calls = []
        type(self).instances.append(self)

    def init(self, args, *, primary=True, **kwargs):
        self.calls.append(("init", args, primary, kwargs))

    def log(self, metrics, step=None, **kwargs):
        self.calls.append(("log", metrics, step, kwargs))

    def set_terminal_status(self, status):
        self.calls.append(("status", status))

    def finish(self):
        self.calls.append(("finish",))


def _install_fake_sdk(monkeypatch, *, drain_result=None):
    FakeSdkBackend.instances.clear()
    probe_package = ModuleType("probe")
    integrations_package = ModuleType("probe.integrations")
    miles_module = ModuleType("probe.integrations.miles")
    miles_module.MilesMetricBackend = FakeSdkBackend
    miles_module.drain_miles_metric_queue = lambda *args, **kwargs: (
        drain_result or {"unconfirmed": 0, "args": args, "kwargs": kwargs}
    )
    monkeypatch.setitem(sys.modules, "probe", probe_package)
    monkeypatch.setitem(sys.modules, "probe.integrations", integrations_package)
    monkeypatch.setitem(sys.modules, "probe.integrations.miles", miles_module)


def test_probe_backend_is_a_thin_lazy_sdk_adapter(monkeypatch):
    _install_fake_sdk(monkeypatch)
    args = Namespace()
    backend = probe_utils.ProbeBackend()

    backend.init(args, primary=False, router_addr="router")
    backend.log({"rollout/reward": 0.75}, step=17, step_key="rollout/step")
    backend.set_terminal_status("failed")
    backend.finish()

    calls = FakeSdkBackend.instances[0].calls
    assert calls[0] == ("init", args, False, {"router_addr": "router"})
    assert calls[1] == (
        "log",
        {"rollout/reward": 0.75},
        17,
        {"step_key": "rollout/step"},
    )
    assert calls[-2:] == [("status", "failed"), ("finish",)]


def test_probe_backend_satisfies_tracking_manager_contract(monkeypatch):
    _install_fake_sdk(monkeypatch)
    manager = TrackingManager({"probe": (probe_utils.ProbeBackend, "use_probe")})
    manager.init(Namespace(use_probe=True), primary=True)
    manager.log({"train/loss": 1.0}, step=3)
    manager.finish(status="completed")

    calls = FakeSdkBackend.instances[0].calls
    assert ("log", {"train/loss": 1.0}, 3, {"step_key": None}) in calls
    assert calls[-2:] == [("status", "completed"), ("finish",)]
    assert isinstance(probe_utils.ProbeBackend(), TrackingBackend)


def test_drain_metric_queue_delegates_to_sdk(monkeypatch, tmp_path):
    expected = {"unconfirmed": 2, "queue_dir": str(tmp_path)}
    _install_fake_sdk(monkeypatch, drain_result=expected)
    result = probe_utils.drain_metric_queue(
        tmp_path,
        "run-1",
        base_url="https://probe.test",
        token="secret",
        timeout=12,
    )
    assert result == expected
