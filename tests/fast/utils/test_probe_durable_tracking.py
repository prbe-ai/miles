from __future__ import annotations

import time
from argparse import Namespace

import pytest

from miles.utils.tracking_utils import probe_utils
from miles.utils.tracking_utils.base import TrackingBackend, TrackingManager


class FakeRun:
    def __init__(self, client, data=None, *, fail_logs=False):
        self.client = client
        self.data = data or {
            "id": "probe-run-123",
            "experiment_id": "experiment-456",
            "name": "test run",
        }
        self.fail_logs = fail_logs
        self.logs = []
        self.finishes = []
        self.snapshots = []
        self.links = []

    @property
    def id(self):
        return self.data["id"]

    @property
    def experiment_id(self):
        return self.data["experiment_id"]

    def log(self, metrics, **kwargs):
        if self.fail_logs:
            raise ConnectionError("offline")
        self.logs.append((metrics, kwargs))

    def finish(self, status, **kwargs):
        self.finishes.append((status, kwargs))

    def set_status(self, status, **kwargs):
        self.finishes.append((status, kwargs))

    def snapshot(self, **kwargs):
        self.snapshots.append(kwargs)

    def link(self, **kwargs):
        self.links.append(kwargs)


class FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created_run = None
        self.closed = False
        self.run_calls = []
        type(self).instances.append(self)

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        self.created_run = FakeRun(self)
        return self.created_run

    def get_run(self, run_id):
        return {"id": run_id, "experiment_id": "existing-experiment", "name": "existing"}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    FakeClient.instances.clear()
    for name in ("MILES_RUN_ID", "PROBE_RUN_ID", "RESEARCH_OS_RUN_ID", "PROBE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _args(tmp_path, **overrides):
    values = {
        "probe_fail_open": True,
        "probe_base_url": "https://probe.test",
        "probe_project": "customer-project",
        "probe_experiment": "swe-agent",
        "probe_hypothesis": "capture the run",
        "probe_run_name": "nebius-run",
        "probe_external_id": "miles-native-7",
        "probe_run_id": None,
        "probe_tags": ["nebius", "harbor"],
        "probe_links": ["data_mix=swebench"],
        "probe_queue_dir": str(tmp_path / "queue-root"),
        "probe_export_interval_sec": 0.01,
        "probe_finish_timeout_sec": 2.0,
        "probe_snapshot": True,
        "probe_snapshot_cwd": "/workspace/miles",
        "save": str(tmp_path / "save"),
        "wandb_run_id": "wandb-1",
        "mlflow_run_id": "mlflow-2",
        "wandb_key": "must-not-leak",
        "env_report": {"API_TOKEN": "also-must-not-leak"},
        "callback_url": "https://user:password@example.test/path",
        "webhook_url": "https://example.test/hook?token=must-not-leak&mode=full",
        "presigned_url": "https://bucket.test/object?X-Amz-Signature=must-not-leak&part=1",
        "hf_token": "must-not-leak",
        "aws_access_key_id": "must-not-leak",
        "tokenizer_name": "llama-tokenizer",
        "max_tokens": 2048,
        "rollout_stop_token_ids": [1, 2],
        "calculate_per_token_loss": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_durable_queue_claim_retry_recovery_and_ack(tmp_path):
    queue = probe_utils.DurableMetricQueue(tmp_path / "queue")
    queue.enqueue_metrics(
        {"train/loss": 1.25},
        run_id="run-1",
        external_id="miles-1",
        step=7,
        kind="model",
    )
    assert queue.report()["pending"] == 1
    claimed = queue.claim_next()
    assert claimed is not None and queue.report()["inflight"] == 1
    queue.retry(claimed)
    assert queue.report()["pending"] == 1
    claimed = queue.claim_next()
    queue.recover_inflight()
    assert queue.report()["pending"] == 1
    claimed = queue.claim_next()
    queue.acknowledge(claimed)
    assert queue.report()["unconfirmed"] == 0


def test_background_exporter_confirms_metrics_and_finish(tmp_path):
    queue = probe_utils.DurableMetricQueue(tmp_path / "queue")
    client = FakeClient()
    run = FakeRun(client)
    exporter = probe_utils._MetricExporter(queue, client, run, interval=0.01)
    queue.enqueue_metrics(
        {"rollout/reward": 0.5},
        run_id=run.id,
        external_id="miles-1",
        step=3,
        kind="model",
    )
    queue.enqueue_finish(
        run_id=run.id,
        external_id="miles-1",
        status="completed",
        summary={"capture_status": "complete"},
    )

    report = exporter.drain_and_close(2)

    assert report["unconfirmed"] == 0
    assert run.logs[0][0] == {"rollout/reward": 0.5}
    assert run.logs[0][1]["step"] == 3
    assert run.logs[0][1]["kind"] == "model"
    assert run.logs[0][1]["strict"] is True
    assert run.logs[0][1]["wall_clock"]
    assert run.finishes[0][0] == "completed"
    assert run.finishes[0][1]["summary"] == {"capture_status": "complete"}
    assert run.finishes[0][1]["ended_at"]
    assert client.closed


def test_export_failure_keeps_record_and_reports_reason(tmp_path):
    queue = probe_utils.DurableMetricQueue(tmp_path / "queue")
    client = FakeClient()
    run = FakeRun(client, fail_logs=True)
    exporter = probe_utils._MetricExporter(queue, client, run, interval=0.01)
    queue.enqueue_metrics(
        {"train/loss": 1.0},
        run_id=run.id,
        external_id="miles-1",
        step=1,
        kind="model",
    )
    exporter.wake()
    time.sleep(0.03)

    report = exporter.drain_and_close(0.05)

    assert report["unconfirmed"] == 1
    assert "offline" in report["last_error"]


def test_tracker_uses_existing_manager_contract_and_publishes_run_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_utils, "_load_sdk", lambda: (FakeClient, FakeRun))
    monkeypatch.setenv("PROBE_TOKEN", "secret-token")
    args = _args(tmp_path)
    tracker = probe_utils.ProbeTracker()

    tracker.init(args, primary=True)
    tracker.log(
        {"train/loss": 1.25, "train/step": 7, "nested": {"ignored": True}},
        step=7,
        step_key="train/step",
    )
    tracker.finish(status="failed")

    client = FakeClient.instances[0]
    run = client.created_run
    assert args.probe_run_id == "probe-run-123"
    assert args.research_os_run_id == "probe-run-123"
    assert args.probe_capture_status == "queue_drained"
    assert run.logs[0][0] == {"train/loss": 1.25, "train/step": 7.0}
    assert run.logs[0][1]["step"] == 7
    assert run.logs[0][1]["kind"] == "model"
    assert run.logs[0][1]["wall_clock"]
    assert run.logs[0][1]["strict"] is True
    assert run.finishes[0][0] == "failed"
    assert run.finishes[0][1]["summary"]["capture_completeness"] == "unknown"
    assert "expected_producer_set" in run.finishes[0][1]["summary"]["capture_missing"]
    assert client.run_calls[0]["config"]["wandb_key"] == "<redacted>"
    assert "env_report" not in client.run_calls[0]["config"]
    assert client.run_calls[0]["config"]["callback_url"] == "https://<redacted>@example.test/path"
    assert client.run_calls[0]["config"]["webhook_url"] == ("https://example.test/hook?token=%3Credacted%3E&mode=full")
    assert client.run_calls[0]["config"]["presigned_url"] == (
        "https://bucket.test/object?X-Amz-Signature=%3Credacted%3E&part=1"
    )
    assert client.run_calls[0]["config"]["hf_token"] == "<redacted>"
    assert client.run_calls[0]["config"]["aws_access_key_id"] == "<redacted>"
    assert client.run_calls[0]["config"]["tokenizer_name"] == "llama-tokenizer"
    assert client.run_calls[0]["config"]["max_tokens"] == 2048
    assert client.run_calls[0]["config"]["rollout_stop_token_ids"] == [1, 2]
    assert client.run_calls[0]["config"]["calculate_per_token_loss"] is True
    assert run.snapshots[0]["strict"] is True
    assert run.links[0]["data_mix"] == "swebench"


def test_secondary_only_writes_shared_queue_and_never_imports_sdk(tmp_path, monkeypatch):
    def forbidden_sdk():
        raise AssertionError("secondary must not import the Probe SDK")

    monkeypatch.setattr(probe_utils, "_load_sdk", forbidden_sdk)
    args = _args(
        tmp_path,
        probe_queue_dir=str(tmp_path / "resolved-queue"),
        probe_queue_resolved=True,
        probe_run_id="shared-run",
        research_os_run_id="shared-run",
        probe_experiment_id="shared-experiment",
    )
    tracker = probe_utils.ProbeTracker()

    tracker.init(args, primary=False)
    tracker.log({"rollout/reward": 0.75}, step=4, step_key="rollout/step")
    tracker.finish()

    queue = probe_utils.DurableMetricQueue(args.probe_queue_dir)
    assert queue.report()["pending"] == 1
    record = probe_utils._read_json(next(queue.pending.glob("*.json")))
    assert record["producer_id"].startswith("training:")
    assert record["producer_sequence"] == 1


def test_fail_open_queue_setup_failure_does_not_abort_training(tmp_path, monkeypatch):
    class BrokenQueue:
        def __init__(self, root):
            raise OSError("shared PVC is read-only")

    monkeypatch.setattr(probe_utils, "DurableMetricQueue", BrokenQueue)
    args = _args(tmp_path)
    tracker = probe_utils.ProbeTracker()

    tracker.init(args, primary=True)
    tracker.log({"train/loss": 9.0}, step=1)
    tracker.finish()

    assert args.probe_capture_status == "unavailable"


def test_tracking_manager_keeps_no_argument_custom_finish_contract():
    class CustomBackend(TrackingBackend):
        finished = False

        def init(self, args, *, primary=True, **kwargs):
            return

        def log(self, metrics, step=None, **kwargs):
            return

        def finish(self):
            type(self).finished = True

    manager = TrackingManager({"custom": (CustomBackend, "use_custom")})
    manager.init(Namespace(use_custom=True))
    manager.finish(status="failed")

    assert CustomBackend.finished


def test_metric_enqueue_failure_is_named_in_terminal_capture_gaps(tmp_path, monkeypatch):
    def missing_sdk():
        raise ModuleNotFoundError("No module named 'probe'")

    monkeypatch.setattr(probe_utils, "_load_sdk", missing_sdk)
    args = _args(tmp_path, probe_snapshot=False)
    tracker = probe_utils.ProbeTracker()
    tracker.init(args, primary=True)

    def fail_enqueue(*args, **kwargs):
        raise OSError("PVC full")

    monkeypatch.setattr(tracker._queue, "enqueue_metrics", fail_enqueue)
    tracker.log({"train/loss": 3.0}, step=1)
    tracker.finish()

    finish_record = next(
        probe_utils._read_json(path)
        for path in tracker._queue.pending.glob("*.json")
        if probe_utils._read_json(path).get("type") == "finish"
    )
    assert "metric_enqueue_failure" in finish_record["summary"]["capture_missing"]


def test_repair_retries_partial_existing_run_and_publishes_new_gaps(tmp_path, monkeypatch):
    class FailingRepairRun(FakeRun):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        def snapshot(self, **kwargs):
            raise OSError("snapshot source gone")

        def link(self, **kwargs):
            raise ConnectionError("link unavailable")

    monkeypatch.setattr(probe_utils, "_load_sdk", lambda: (FakeClient, FailingRepairRun))
    queue = probe_utils.DurableMetricQueue(tmp_path / "queue")
    queue.write_intent(
        run_id="existing-run",
        state="pending_run",
        snapshot_state="missing",
        links_state="missing",
        run_spec={
            "snapshot": {"enabled": True, "cwd": "/gone"},
            "links": {"miles_run_id": "miles-1"},
        },
    )
    queue.enqueue_finish(
        run_id="existing-run",
        external_id="miles-1",
        status="failed",
        summary={"capture_completeness": "unknown", "capture_missing": []},
    )

    report = probe_utils.drain_metric_queue(queue.root, timeout=2)

    assert report["unconfirmed"] == 0
    terminal_summary = FailingRepairRun.instances[0].finishes[0][1]["summary"]
    assert terminal_summary["capture_missing"] == ["launch_snapshot", "native_run_links"]
    status = probe_utils._read_json(queue.root / "export-status.json")
    assert status["repair_missing"] == ["launch_snapshot", "native_run_links"]


def test_exporter_lease_and_run_target_prevent_competing_or_wrong_drain(tmp_path):
    queue = probe_utils.DurableMetricQueue(tmp_path / "queue")
    queue.enqueue_metrics(
        {"train/loss": 1.0},
        run_id="correct-run",
        external_id="miles-1",
        step=None,
        kind="model",
    )
    client = FakeClient()
    correct_run = FakeRun(
        client,
        data={"id": "correct-run", "experiment_id": "experiment-456", "name": "correct"},
        fail_logs=True,
    )
    live = probe_utils._MetricExporter(queue, client, correct_run, interval=1)
    with pytest.raises(RuntimeError, match="another Probe exporter"):
        competing_client = FakeClient()
        probe_utils._MetricExporter(queue, competing_client, correct_run, interval=1)
    live.drain_and_close(0)

    wrong_client = FakeClient()
    with pytest.raises(ValueError, match="queue contains records"):
        probe_utils._MetricExporter(queue, wrong_client, FakeRun(wrong_client), interval=1)


def test_initialization_outage_persists_complete_intent_and_can_resolve_later(tmp_path, monkeypatch):
    def missing_sdk():
        raise ModuleNotFoundError("No module named 'probe'")

    monkeypatch.setattr(probe_utils, "_load_sdk", missing_sdk)
    args = _args(tmp_path, probe_snapshot=False)
    tracker = probe_utils.ProbeTracker()

    tracker.init(args, primary=True)
    tracker.log({"train/loss": 2.0}, step=1, step_key="train/step")
    tracker.finish()

    queue = probe_utils.DurableMetricQueue(args.probe_queue_dir)
    report = queue.report()
    assert report["unconfirmed"] == 2  # metric + deferred run finish
    assert args.probe_capture_status == "partial"
    assert args.probe_run_id is None

    intent = probe_utils._read_json(queue.root / "intent.json")
    assert intent["run_spec"]["project"] == "customer-project"
    assert intent["run_spec"]["links"]["data_mix"] == "swebench"

    monkeypatch.setattr(probe_utils, "_load_sdk", lambda: (FakeClient, FakeRun))
    drained = probe_utils.drain_metric_queue(queue.root, timeout=2)

    assert drained["unconfirmed"] == 0
    assert drained["run_id"] == "probe-run-123"
    repaired_run = FakeClient.instances[-1].created_run
    assert repaired_run.logs[0][0] == {"train/loss": 2.0}
    assert repaired_run.finishes[0][0] == "completed"
