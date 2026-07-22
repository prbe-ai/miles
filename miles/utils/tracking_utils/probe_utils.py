"""Durable Research OS / Probe tracking for distributed Miles runs.

Every Miles process atomically queues metric batches on a shared filesystem.
Only the primary process owns a background network exporter. This keeps HTTP
latency out of training actors and keeps unconfirmed telemetry off ephemeral
worker-local spools.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .base import TrackingBackend

logger = logging.getLogger(__name__)

QUEUE_SCHEMA_VERSION = "miles.probe.metrics/v1"
_CREDENTIAL_URI = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "aws_access_key_id",
    "authorization",
    "auth_token",
    "access_token",
    "bearer_token",
    "cookie",
    "credential",
    "credentials",
    "id_token",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
    "wandb_key",
}
_MODEL_TOKEN_KEYS = {
    "bos_token",
    "cls_token",
    "eos_token",
    "mask_token",
    "pad_token",
    "sep_token",
    "stop_token",
    "unk_token",
}
_OPAQUE_CONFIG_KEYS = {"env", "env_report", "env_vars", "environment_variables"}
_JOB_ENV_LINKS = {
    "SLURM_JOB_ID": "slurm_job_id",
    "RAY_JOB_ID": "ray_job_id",
    "NEBIUS_CLUSTER_ID": "nebius_cluster_id",
    "NEBIUS_NODE_GROUP_ID": "nebius_node_group_id",
    "NEBIUS_GPU_CLUSTER_ID": "nebius_gpu_cluster_id",
    "KUBERNETES_POD_NAME": "kubernetes_pod_name",
    "KUBERNETES_NAMESPACE": "kubernetes_namespace",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some network filesystems reject directory fsync. File fsync plus atomic
        # replacement still provides the strongest contract they expose.
        pass


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"queue record {path} must be a JSON object")
    return value


def _safe_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._-")[:64] or "run"
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    parts = set(normalized.split("_"))
    token_secret = normalized.endswith("_token") and normalized not in _MODEL_TOKEN_KEYS
    signed_secret = normalized.endswith(("_signature", "_sig"))
    return (
        normalized in _SENSITIVE_KEYS
        or token_secret
        or signed_secret
        or bool(parts & {"password", "secret", "credential", "credentials"})
    )


def _scrub_string(value: str) -> str:
    scrubbed = _CREDENTIAL_URI.sub(r"\g<scheme><redacted>@", value)
    try:
        parsed = urlsplit(scrubbed)
    except ValueError:
        return scrubbed
    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return scrubbed
    query = [
        (key, "<redacted>" if _is_sensitive_key(key) else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _json_safe(value: Any, *, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, dict):
        return {str(item_key): _json_safe(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, key=key) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _capture_config(args) -> dict[str, Any]:
    """Serialize Miles arguments while excluding opaque environment payloads."""
    return {
        key: _json_safe(value, key=key) for key, value in vars(args).items() if key.lower() not in _OPAQUE_CONFIG_KEYS
    }


def _scalar(value: Any) -> float | None:
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError):
            return None
    if isinstance(value, (str, bytes, dict, list, tuple, set)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _parse_links(values: list[str] | None) -> dict[str, str]:
    links: dict[str, str] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid Probe link {raw!r}; expected KEY=VALUE")
        links[key.strip()] = value.strip()
    return links


def _default_external_id(args) -> str:
    configured = getattr(args, "probe_external_id", None) or os.environ.get("MILES_RUN_ID")
    if configured:
        return str(configured)
    for env_name in ("RAY_JOB_ID", "SLURM_JOB_ID"):
        if value := os.environ.get(env_name):
            return f"miles:{env_name.lower()}:{value}"
    return f"miles:{uuid.uuid4()}"


def _default_run_name(args, external_id: str) -> str:
    configured = getattr(args, "probe_run_name", None) or getattr(args, "wandb_group", None)
    if configured:
        return str(configured)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"miles-{timestamp}-{external_id.rsplit(':', 1)[-1][:8]}"


def _run_spec(args, external_id: str) -> dict[str, Any]:
    links = {"miles_run_id": external_id}
    for attr in ("wandb_run_id", "mlflow_run_id"):
        if value := getattr(args, attr, None):
            links[attr] = str(value)
    for env_name, link_name in _JOB_ENV_LINKS.items():
        if value := os.environ.get(env_name):
            links[link_name] = value
    links.update(_parse_links(getattr(args, "probe_links", None)))
    return {
        "project": getattr(args, "probe_project", "miles"),
        "experiment": getattr(args, "probe_experiment", "miles"),
        "hypothesis": getattr(
            args,
            "probe_hypothesis",
            "Capture Miles training, rollout, and evaluation telemetry.",
        ),
        "name": _default_run_name(args, external_id),
        "source": "miles",
        "external_id": external_id,
        "tags": list(dict.fromkeys(["miles", *(getattr(args, "probe_tags", None) or [])])),
        "config": _capture_config(args),
        "metadata": {
            "integration": "miles",
            "miles_run_id": external_id,
            "capture": {"metrics": "durable_queue", "completeness": "unknown"},
        },
        "links": links,
        "snapshot": {
            "enabled": bool(getattr(args, "probe_snapshot", True)),
            "cwd": getattr(args, "probe_snapshot_cwd", None),
        },
    }


def _resolve_queue_dir(args, external_id: str, *, primary: bool) -> Path:
    if not primary and getattr(args, "probe_queue_resolved", False):
        return Path(args.probe_queue_dir).expanduser()
    configured = getattr(args, "probe_queue_dir", None)
    if configured:
        base = Path(configured).expanduser()
    elif getattr(args, "save", None):
        base = Path(args.save).expanduser() / "probe" / "metrics"
    else:
        base = Path.cwd() / ".miles-state" / "probe" / "metrics"
    queue_dir = base / _safe_component(external_id)
    args.probe_queue_dir = str(queue_dir)
    args.probe_queue_resolved = True
    return queue_dir


class DurableMetricQueue:
    """Atomic, multi-writer queue; one JSON file is one metric batch."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.pending = self.root / "pending"
        self.inflight = self.root / "inflight"
        self.producers = self.root / "producers"
        self.pending.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.inflight.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.producers.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._metadata_lock = threading.Lock()

    def _enqueue(self, payload: dict[str, Any]) -> Path:
        record_id = f"{time.time_ns():020d}-{uuid.uuid4().hex}"
        record = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "record_id": record_id,
            "created_at": _now(),
            **payload,
        }
        destination = self.pending / f"{record_id}.json"
        _write_json_atomic(destination, record)
        return destination

    def enqueue_metrics(
        self,
        metrics: dict[str, float],
        *,
        run_id: str | None,
        external_id: str,
        step: int | None,
        kind: str,
        producer_id: str | None = None,
        producer_sequence: int | None = None,
    ) -> Path:
        return self._enqueue(
            {
                "type": "metrics",
                "run_id": run_id,
                "external_id": external_id,
                "step": step,
                "kind": kind,
                "metrics": metrics,
                "producer_id": producer_id,
                "producer_sequence": producer_sequence,
            }
        )

    def enqueue_finish(
        self,
        *,
        run_id: str | None,
        external_id: str,
        status: str,
        summary: dict[str, Any],
    ) -> Path:
        return self._enqueue(
            {
                "type": "finish",
                "run_id": run_id,
                "external_id": external_id,
                "status": status,
                "summary": summary,
            }
        )

    def register_producer(self, producer_id: str, *, role: str, primary: bool) -> None:
        _write_json_atomic(
            self.producers / f"{_safe_component(producer_id)}.json",
            {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "producer_id": producer_id,
                "role": role,
                "primary": primary,
                "state": "open",
                "last_sequence": 0,
                "registered_at": _now(),
                "updated_at": _now(),
            },
        )

    def update_producer(self, producer_id: str, *, sequence: int, state: str = "open") -> None:
        path = self.producers / f"{_safe_component(producer_id)}.json"
        existing = _read_json(path) if path.is_file() else {"producer_id": producer_id}
        _write_json_atomic(
            path,
            {
                "schema_version": QUEUE_SCHEMA_VERSION,
                **existing,
                "state": state,
                "last_sequence": sequence,
                "updated_at": _now(),
            },
        )

    def producer_report(self) -> dict[str, Any]:
        records = [_read_json(path) for path in sorted(self.producers.glob("*.json"))]
        open_ids = [str(item.get("producer_id")) for item in records if item.get("state") != "closed"]
        return {
            "observed": len(records),
            "closed": len(records) - len(open_ids),
            "open": open_ids,
            # Miles does not currently expose a reliable expected-writer set or
            # call finish on every Ray actor. Never turn an observed subset into
            # a false completeness guarantee.
            "completeness": "unknown",
            "missing": ["expected_producer_set", "distributed_close_barrier"],
        }

    def add_capture_gaps_to_terminal_records(self, gaps: list[str]) -> None:
        """Make repair-discovered gaps visible in the server terminal summary."""
        if not gaps:
            return
        for path in (*sorted(self.pending.glob("*.json")), *sorted(self.inflight.glob("*.json"))):
            record = _read_json(path)
            if record.get("type") != "finish":
                continue
            summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
            existing = summary.get("capture_missing")
            missing = list(existing) if isinstance(existing, list) else []
            summary["capture_missing"] = list(dict.fromkeys([*missing, *gaps]))
            summary["capture_completeness"] = "unknown"
            record["summary"] = summary
            _write_json_atomic(path, record)

    def write_intent(self, **values: Any) -> None:
        with self._metadata_lock:
            existing = _read_json(self.root / "intent.json") if (self.root / "intent.json").is_file() else {}
            _write_json_atomic(
                self.root / "intent.json",
                {"schema_version": QUEUE_SCHEMA_VERSION, **existing, **values, "updated_at": _now()},
            )

    def write_status(self, **values: Any) -> None:
        with self._metadata_lock:
            existing = (
                _read_json(self.root / "export-status.json") if (self.root / "export-status.json").is_file() else {}
            )
            _write_json_atomic(
                self.root / "export-status.json",
                {"schema_version": QUEUE_SCHEMA_VERSION, **existing, **values, "updated_at": _now()},
            )

    def recover_inflight(self) -> None:
        for path in sorted(self.inflight.glob("*.json")):
            destination = self.pending / path.name
            if destination.exists():
                destination = self.pending / f"{path.stem}-{uuid.uuid4().hex}.json"
            os.replace(path, destination)
        _fsync_directory(self.pending)
        _fsync_directory(self.inflight)

    def claim_next(self) -> Path | None:
        for path in sorted(self.pending.glob("*.json")):
            destination = self.inflight / path.name
            try:
                os.replace(path, destination)
            except FileNotFoundError:
                continue
            _fsync_directory(self.pending)
            _fsync_directory(self.inflight)
            return destination
        return None

    def acknowledge(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        _fsync_directory(self.inflight)

    def retry(self, path: Path) -> None:
        if not path.exists():
            return
        os.replace(path, self.pending / path.name)
        _fsync_directory(self.pending)
        _fsync_directory(self.inflight)

    def report(self) -> dict[str, Any]:
        pending = len(list(self.pending.glob("*.json")))
        inflight = len(list(self.inflight.glob("*.json")))
        status_path = self.root / "export-status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        producer_report = self.producer_report()
        return {
            "queue_dir": str(self.root),
            "pending": pending,
            "inflight": inflight,
            "unconfirmed": pending + inflight,
            "last_error": status.get("last_error"),
            "last_confirmed_at": status.get("last_confirmed_at"),
            "producers": producer_report,
        }


class _ExporterLease:
    """Exclusive lease preventing a live exporter and repair drain from racing."""

    def __init__(self, root: Path) -> None:
        self.path = root / "exporter.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            raise RuntimeError(f"another Probe exporter owns {self.path}") from exc

    def close(self) -> None:
        if self._handle.closed:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()


class _MetricExporter:
    def __init__(
        self,
        queue: DurableMetricQueue,
        client: Any,
        run: Any,
        *,
        interval: float,
        lease: _ExporterLease | None = None,
    ) -> None:
        self.queue = queue
        self.client = client
        self.run = run
        self.interval = max(float(interval), 0.05)
        self._lease = lease or _ExporterLease(queue.root)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"probe-metric-export-{run.id}",
            daemon=True,
        )
        try:
            self.queue.recover_inflight()
            target_ids = {
                str(record["run_id"])
                for path in (*sorted(self.queue.pending.glob("*.json")), *sorted(self.queue.inflight.glob("*.json")))
                if (record := _read_json(path)).get("run_id")
            }
            if target_ids - {str(run.id)}:
                raise ValueError(f"queue contains records for {sorted(target_ids)}, not exporter run {run.id}")
            self._thread.start()
        except Exception:
            self._lease.close()
            raise

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                path = self.queue.claim_next()
                if path is None:
                    self._wake.wait(self.interval)
                    self._wake.clear()
                    continue
                try:
                    record = _read_json(path)
                    if record.get("schema_version") != QUEUE_SCHEMA_VERSION:
                        raise ValueError(f"unsupported metric queue record {record.get('schema_version')!r}")
                    record_run_id = record.get("run_id")
                    if record_run_id and str(record_run_id) != str(self.run.id):
                        raise ValueError(f"queued record targets run {record_run_id}, not exporter run {self.run.id}")
                    if record.get("type") == "metrics":
                        self.run.log(
                            record.get("metrics") or {},
                            step=record.get("step"),
                            kind=record.get("kind") or "model",
                            wall_clock=record.get("created_at"),
                            strict=True,
                        )
                    elif record.get("type") == "finish":
                        self.run.set_status(
                            record.get("status") or "completed",
                            ended_at=record.get("created_at"),
                            summary=record.get("summary") or {},
                        )
                    else:
                        raise ValueError(f"unknown metric queue record type {record.get('type')!r}")
                    self.queue.acknowledge(path)
                    self.queue.write_status(last_error=None, last_confirmed_at=_now())
                except Exception as exc:  # noqa: BLE001 - preserve every unconfirmed record
                    self.queue.retry(path)
                    self.queue.write_status(last_error=f"{type(exc).__name__}: {exc}")
                    logger.warning("Probe metric export failed; durable records remain queued: %s", exc)
                    self._wake.wait(self.interval)
                    self._wake.clear()
        finally:
            try:
                self.client.close()
            finally:
                self._lease.close()

    def drain_and_close(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        self.wake()
        while time.monotonic() < deadline:
            report = self.queue.report()
            if report["unconfirmed"] == 0:
                break
            time.sleep(0.05)
            self.wake()
        self._stop.set()
        self.wake()
        self._thread.join(timeout=0.5)
        report = self.queue.report()
        report["exporter_alive"] = self._thread.is_alive()
        return report


def _load_sdk():
    from probe.sdk.client import Client
    from probe.sdk.run import Run

    return Client, Run


class ProbeTracker:
    def __init__(self) -> None:
        self._args = None
        self._queue: DurableMetricQueue | None = None
        self._exporter: _MetricExporter | None = None
        self._primary = False
        self._fail_open = True
        self._external_id: str | None = None
        self._run_id: str | None = None
        self._finished = False
        self._producer_id: str | None = None
        self._producer_sequence = 0
        self._capture_gaps: list[str] = []

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        self._args = args
        self._primary = primary
        self._fail_open = bool(getattr(args, "probe_fail_open", True))
        self._external_id = str(getattr(args, "probe_external_id", None) or _default_external_id(args))
        args.probe_external_id = self._external_id
        args.miles_run_id = self._external_id
        role = "primary" if primary else ("rollout" if kwargs.get("router_addr") else "training")
        self._producer_id = f"{role}:{os.uname().nodename}:{os.getpid()}:{uuid.uuid4().hex}"
        try:
            queue_dir = _resolve_queue_dir(args, self._external_id, primary=primary)
            self._queue = DurableMetricQueue(queue_dir)
            self._queue.register_producer(self._producer_id, role=role, primary=primary)
        except Exception as exc:
            self._queue = None
            self._handle_error("open durable metric queue", exc, unavailable=True)
            return
        self._run_id = getattr(args, "probe_run_id", None) or getattr(args, "research_os_run_id", None)
        if not primary:
            self._publish_identity(self._run_id, experiment_id=getattr(args, "probe_experiment_id", None))
            return

        try:
            run_spec = _run_spec(args, self._external_id)
            self._queue.write_intent(
                source="miles",
                external_id=self._external_id,
                run_id=self._run_id,
                state="attaching" if self._run_id else "creating",
                run_spec=run_spec,
            )
        except Exception as exc:
            self._handle_error("persist deferred run intent", exc)
            return

        client = None
        try:
            Client, Run = _load_sdk()
            client = Client(
                base_url=getattr(args, "probe_base_url", None),
                token=os.environ.get("PROBE_TOKEN"),
                fail_open=False,
            )
            if self._run_id:
                run = Run(client, client.get_run(str(self._run_id)))
            else:
                try:
                    run = client.run(
                        **{key: value for key, value in run_spec.items() if key not in {"links", "snapshot"}}
                    )
                except Exception as exc:
                    existing_id = getattr(exc, "existing_id", None)
                    if not existing_id or getattr(exc, "deleted", False):
                        raise
                    run = Run(client, client.get_run(existing_id))
            self._run_id = run.id
            self._publish_identity(run.id, experiment_id=run.experiment_id)
            self._queue.write_intent(run_id=run.id, state="exporting")
            snapshot_state = self._capture_snapshot(run)
            links_state = self._link_native_runs(run)
            self._queue.write_intent(snapshot_state=snapshot_state, links_state=links_state)
            self._exporter = _MetricExporter(
                self._queue,
                client,
                run,
                interval=getattr(args, "probe_export_interval_sec", 2.0),
            )
            client = None  # exporter owns it
        except Exception as exc:
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - retain the initialization error
                    logger.exception("Probe client cleanup failed")
            try:
                self._queue.write_status(last_error=f"{type(exc).__name__}: {exc}")
                self._queue.write_intent(state="pending_run")
            except Exception:  # noqa: BLE001 - fail-open must not be defeated by status I/O
                logger.exception("Probe could not persist initialization failure status")
            self._handle_error("initialize exporter", exc)

    def _publish_identity(self, run_id: str | None, *, experiment_id: str | None) -> None:
        self._args.probe_run_id = run_id
        self._args.research_os_run_id = run_id
        self._args.probe_experiment_id = experiment_id
        self._args.probe_capture_status = "queueing" if run_id else "pending_run"
        os.environ["MILES_RUN_ID"] = self._external_id
        if run_id:
            os.environ["PROBE_RUN_ID"] = str(run_id)
            os.environ["RESEARCH_OS_RUN_ID"] = str(run_id)
        logger.info("Probe durable metric queue: %s (run=%s)", self._queue.root, run_id)

    def _capture_snapshot(self, run: Any) -> str:
        if not getattr(self._args, "probe_snapshot", True):
            return "skipped"
        try:
            run.snapshot(
                cwd=getattr(self._args, "probe_snapshot_cwd", None),
                include_env=True,
                include_gpu=True,
                strict=True,
            )
            return "complete"
        except Exception as exc:
            self._capture_gaps.append("launch_snapshot")
            self._handle_error("capture launch snapshot", exc)
            return "missing"

    def _link_native_runs(self, run: Any) -> str:
        links = {"miles_run_id": self._external_id}
        for attr in ("wandb_run_id", "mlflow_run_id"):
            if value := getattr(self._args, attr, None):
                links[attr] = str(value)
        for env_name, link_name in _JOB_ENV_LINKS.items():
            if value := os.environ.get(env_name):
                links[link_name] = value
        try:
            links.update(_parse_links(getattr(self._args, "probe_links", None)))
            run.link(strict=True, **links)
            return "complete"
        except Exception as exc:
            self._capture_gaps.append("native_run_links")
            self._handle_error("link native run identities", exc)
            return "missing"

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        if self._queue is None:
            return
        scalar_metrics = {key: number for key, value in metrics.items() if (number := _scalar(value)) is not None}
        if not scalar_metrics:
            return
        kind = "validation" if kwargs.get("step_key") == "eval/step" else "model"
        self._producer_sequence += 1
        try:
            self._queue.enqueue_metrics(
                scalar_metrics,
                run_id=self._run_id,
                external_id=self._external_id,
                step=int(step) if step is not None else None,
                kind=kind,
                producer_id=self._producer_id,
                producer_sequence=self._producer_sequence,
            )
        except Exception as exc:
            self._capture_gaps.append("metric_enqueue_failure")
            self._handle_error("durably queue metrics", exc)
            return
        try:
            self._queue.update_producer(self._producer_id, sequence=self._producer_sequence)
        except Exception as exc:
            self._capture_gaps.append("producer_sequence_ledger")
            self._handle_error("update metric producer ledger", exc)
        if self._exporter is not None:
            self._exporter.wake()

    def finish(self, status: str = "completed") -> None:
        if self._finished or self._queue is None:
            return
        self._finished = True
        try:
            self._queue.update_producer(
                self._producer_id,
                sequence=self._producer_sequence,
                state="closed",
            )
        except Exception as exc:
            self._capture_gaps.append("producer_close_record")
            self._handle_error("close metric producer ledger", exc)
        if not self._primary:
            return
        try:
            producer_report = self._queue.producer_report()
            missing = list(dict.fromkeys([*producer_report["missing"], *self._capture_gaps]))
            self._queue.enqueue_finish(
                run_id=self._run_id,
                external_id=self._external_id,
                status=status,
                summary={
                    "integration": "miles",
                    "metric_publication": "terminal_record_confirmed",
                    "capture_completeness": "unknown",
                    "capture_missing": missing,
                    "observed_producers": producer_report["observed"],
                },
            )
            if self._exporter is not None:
                report = self._exporter.drain_and_close(getattr(self._args, "probe_finish_timeout_sec", 20.0))
            else:
                report = self._queue.report()
            drained = report["unconfirmed"] == 0 and not report.get("exporter_alive", False)
            self._args.probe_capture_status = "queue_drained" if drained else "partial"
            self._queue.write_status(
                state=self._args.probe_capture_status,
                publication_state="drained" if drained else "pending",
                capture_completeness="unknown",
                missing=missing,
                unconfirmed=report["unconfirmed"],
                producers=report["producers"],
                exporter_alive=report.get("exporter_alive", False),
            )
            if not drained:
                message = f"Probe has {report['unconfirmed']} unconfirmed record(s) on {self._queue.root}; they are retained for retry"
                if not self._fail_open:
                    raise RuntimeError(message)
                logger.warning(message)
        except Exception as exc:
            if self._exporter is not None and self._exporter._thread.is_alive():
                try:
                    self._exporter.drain_and_close(0)
                except Exception:  # noqa: BLE001 - preserve the finalization error
                    logger.exception("Probe exporter cleanup failed")
            self._handle_error("finalize durable metric export", exc)

    def _handle_error(self, operation: str, exc: Exception, *, unavailable: bool = False) -> None:
        if self._args is not None:
            self._args.probe_capture_status = "unavailable" if unavailable else "partial"
        if not self._fail_open:
            raise exc
        logger.warning("Probe could not %s; Miles will continue: %s", operation, exc, exc_info=True)


class ProbeBackend(TrackingBackend):
    """Adapter registered with Miles' existing tracking manager."""

    def __init__(self) -> None:
        self._tracker = ProbeTracker()
        self._terminal_status = "completed"

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        self._tracker.init(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        self._tracker.log(metrics, step=step, **kwargs)

    def set_terminal_status(self, status: str) -> None:
        self._terminal_status = status

    def finish(self) -> None:
        self._tracker.finish(status=self._terminal_status)


def drain_metric_queue(
    queue_dir: str | Path,
    run_id: str | None = None,
    *,
    base_url: str | None = None,
    token: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Resolve a retained run intent if needed, then drain its durable queue."""
    Client, Run = _load_sdk()
    queue = DurableMetricQueue(queue_dir)
    lease = _ExporterLease(queue.root)
    client = None
    try:
        client = Client(
            base_url=base_url,
            token=token or os.environ.get("PROBE_TOKEN"),
            fail_open=False,
        )
        intent_path = queue.root / "intent.json"
        intent = _read_json(intent_path) if intent_path.is_file() else {}
        spec = intent.get("run_spec")
        resolved_run_id = run_id or intent.get("run_id")
        if resolved_run_id:
            run = Run(client, client.get_run(str(resolved_run_id)))
        else:
            if not isinstance(spec, dict):
                raise ValueError("queue has no run ID or complete deferred run intent")
            create_values = {key: value for key, value in spec.items() if key not in {"links", "snapshot"}}
            try:
                run = client.run(**create_values)
            except Exception as exc:
                existing_id = getattr(exc, "existing_id", None)
                if not existing_id or getattr(exc, "deleted", False):
                    raise
                run = Run(client, client.get_run(existing_id))
        repair_gaps: list[str] = []
        snapshot_state = intent.get("snapshot_state")
        links_state = intent.get("links_state")
        if isinstance(spec, dict):
            snapshot = spec.get("snapshot") or {}
            if snapshot.get("enabled") and snapshot_state not in {"complete", "skipped"}:
                try:
                    run.snapshot(
                        cwd=snapshot.get("cwd"),
                        include_env=True,
                        include_gpu=True,
                        strict=True,
                    )
                    snapshot_state = "complete"
                except Exception as exc:  # noqa: BLE001 - metrics remain repairable
                    repair_gaps.append("launch_snapshot")
                    snapshot_state = "missing"
                    logger.warning("Deferred Probe snapshot failed: %s", exc)
            elif not snapshot.get("enabled"):
                snapshot_state = "skipped"
            if (links := spec.get("links")) and links_state != "complete":
                try:
                    run.link(strict=True, **links)
                    links_state = "complete"
                except Exception as exc:  # noqa: BLE001 - metrics remain repairable
                    repair_gaps.append("native_run_links")
                    links_state = "missing"
                    logger.warning("Deferred Probe links failed: %s", exc)
        queue.write_intent(
            run_id=run.id,
            state="exporting",
            snapshot_state=snapshot_state,
            links_state=links_state,
        )
        if repair_gaps:
            queue.add_capture_gaps_to_terminal_records(repair_gaps)
            queue.write_status(repair_missing=repair_gaps)
        exporter = _MetricExporter(queue, client, run, interval=0.25, lease=lease)
        lease = None  # exporter owns it
        client = None  # exporter owns it
        report = exporter.drain_and_close(timeout)
        report["run_id"] = run.id
        return report
    finally:
        if lease is not None:
            lease.close()
        if client is not None:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain a retained Miles Probe metric queue")
    parser.add_argument("queue_dir")
    parser.add_argument(
        "--run",
        dest="run_id",
        help="Existing Research OS run. Omit to resolve/create it from intent.json.",
    )
    parser.add_argument("--base-url", default=os.environ.get("PROBE_BASE_URL"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    report = drain_metric_queue(
        args.queue_dir,
        args.run_id,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["unconfirmed"]:
        raise SystemExit(2)


__all__ = [
    "DurableMetricQueue",
    "ProbeBackend",
    "ProbeTracker",
    "QUEUE_SCHEMA_VERSION",
    "drain_metric_queue",
]


if __name__ == "__main__":
    main()
