from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType

from miles.utils.tracking_utils import probe_utils
from miles.utils.tracking_utils import mlflow_utils
from miles.utils.tracking_utils.base import (
    MlflowBackend,
    TrackingBackend,
    TrackingManager,
    WandbBackend,
)


class FakeSdkBackend:
    instances: list[FakeSdkBackend] = []

    def __init__(self) -> None:
        self.calls = []
        type(self).instances.append(self)

    def init(self, args, *, primary=True, **kwargs):
        self.calls.append(("init", args, primary, kwargs))

    def log(self, metrics, step=None, **kwargs):
        self.calls.append(("log", metrics, step, kwargs))

    def define_step_key_metric_group(self, prefix, step_key):
        self.calls.append(("define", prefix, step_key))

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
    miles_module.drain_miles_metric_queue = lambda *args, **kwargs: drain_result or {"unconfirmed": 0, "args": args, "kwargs": kwargs}
    monkeypatch.setitem(sys.modules, "probe", probe_package)
    monkeypatch.setitem(sys.modules, "probe.integrations", integrations_package)
    monkeypatch.setitem(sys.modules, "probe.integrations.miles", miles_module)


def test_probe_backend_is_a_thin_lazy_sdk_adapter(monkeypatch):
    _install_fake_sdk(monkeypatch)
    args = Namespace()
    backend = probe_utils.ProbeBackend()

    backend.init(args, primary=False, router_addr="router")
    backend.log({"rollout/reward": 0.75}, step=17, step_key="rollout/step")
    backend.define_step_key_metric_group("adapter-a", "adapter-a/step")
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
    assert calls[2] == ("define", "adapter-a", "adapter-a/step")
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


def test_wandb_mlflow_and_probe_receive_identical_metric_fanout(monkeypatch):
    _install_fake_sdk(monkeypatch)
    wandb_logs = []
    wandb_definitions = []
    wandb_module = ModuleType("wandb")
    wandb_module.log = lambda metrics: wandb_logs.append(metrics)
    wandb_module.define_metric = lambda *args, **kwargs: wandb_definitions.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    mlflow_logs = []
    monkeypatch.setattr(
        mlflow_utils,
        "log_metrics",
        lambda metrics, step=None: mlflow_logs.append((metrics, step)),
    )

    probe = probe_utils.ProbeBackend()
    probe.init(Namespace(), primary=True)
    manager = TrackingManager({})
    manager._backends = [WandbBackend(), MlflowBackend(), probe]

    metrics = {
        "rollout/step": 11,
        "rollout/reward": 0.75,
        "perf/tokens_per_gpu_per_sec": 42.0,
    }
    manager.log(metrics, step=11, step_key="rollout/step")
    manager.define_step_key_metric_group("adapter-a", "adapter-a/step")

    assert wandb_logs == [metrics]
    assert mlflow_logs == [(metrics, 11)]
    sdk_calls = FakeSdkBackend.instances[0].calls
    assert ("log", metrics, 11, {"step_key": "rollout/step"}) in sdk_calls
    assert ("define", "adapter-a", "adapter-a/step") in sdk_calls
    assert wandb_definitions == [
        (("adapter-a/step",), {}),
        (("adapter-a/*",), {"step_metric": "adapter-a/step"}),
    ]


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
