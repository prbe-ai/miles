"""Thin Miles adapter for the Probe SDK's durable tracking integration.

The queue, exporter lease, offline repair, redaction, and run lifecycle live in
``probe.integrations.miles``.  Keeping imports lazy preserves Miles' optional
``probe`` extra: users who do not enable ``--use-probe`` do not need the SDK.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .base import TrackingBackend


class ProbeBackend(TrackingBackend):
    """Delegate Miles' tracking contract to ``probe-research``."""

    def __init__(self) -> None:
        self._backend: Any | None = None
        self._terminal_status = "completed"

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from probe.integrations.miles import MilesMetricBackend

        self._backend = MilesMetricBackend()
        self._backend.init(args, primary=primary, **kwargs)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        *,
        step_key: str | None = None,
        **kwargs,
    ) -> None:
        if self._backend is not None:
            self._backend.log(metrics, step=step, step_key=step_key, **kwargs)

    def define_step_key_metric_group(self, prefix: str, step_key: str) -> None:
        if self._backend is None:
            return
        # Step-group declarations are useful for dynamic metric namespaces
        # such as multi-LoRA. Keep compatibility with older SDK releases,
        # which classify each log from its per-call ``step_key`` only.
        define_group = getattr(self._backend, "define_step_key_metric_group", None)
        if callable(define_group):
            define_group(prefix, step_key)

    def set_terminal_status(self, status: str) -> None:
        self._terminal_status = status

    def finish(self) -> None:
        if self._backend is None:
            return
        # Older compatible SDKs may not expose set_terminal_status. In that
        # case their normal finish behavior remains intact.
        setter = getattr(self._backend, "set_terminal_status", None)
        if callable(setter):
            setter(self._terminal_status)
        self._backend.finish()


def drain_metric_queue(
    queue_dir: str | Path,
    run_id: str | None = None,
    *,
    base_url: str | None = None,
    token: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Compatibility wrapper for retained Miles queues."""
    from probe.integrations.miles import drain_miles_metric_queue

    return drain_miles_metric_queue(
        queue_dir,
        run_id,
        base_url=base_url,
        token=token,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain a retained Miles Probe metric queue")
    parser.add_argument("queue_dir")
    parser.add_argument("--run", dest="run_id")
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


__all__ = ["ProbeBackend", "drain_metric_queue"]


if __name__ == "__main__":
    main()
