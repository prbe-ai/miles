"""Miles-Harbor capture fixture tests.

Tier 1 (parity logic) runs anywhere: the request/metric shapes match what Miles
sends and logs, validated against the real bridge model.

Tier 2 (``@pytest.mark.harbor``) runs a real, model-free oracle trial per sandbox
on the agent-env host and asserts the probe.sandbox-state/1 capture reflects the
sandbox's actual filesystem changes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import oracle_capture  # noqa: E402
import rollout_parity  # noqa: E402

SANDBOXES = _HERE / "sandboxes"
SANDBOX_NAMES = sorted(p.name for p in SANDBOXES.iterdir() if (p / "task.toml").is_file())


def _load_check_sandbox_bundle():
    path = _HERE.parent / "check_sandbox_bundle.py"
    spec = importlib.util.spec_from_file_location("check_sandbox_bundle_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Tier 1: rollout + training parity (no Docker, no model) ------------------


def test_oracle_run_request_matches_miles_shape():
    req = rollout_parity.build_oracle_run_request(
        "miles/capture-fixture-hello",
        step_index=600,
        rollout_id=3,
        sample_id=4,
        group_id=5,
        session_id="sess-1",
    )
    assert req["agent_name"] == "oracle"
    assert req["model"] == "oracle/none"
    assert "/sessions/sess-1/v1" in req["base_url"]
    for field in ("rollout_id", "sample_id", "group_id", "step_index", "capture_context"):
        assert field in req


def test_run_request_validates_against_the_real_bridge_model():
    bridge = oracle_capture.load_bridge()
    req = rollout_parity.build_oracle_run_request(
        "t",
        step_index=1,
        rollout_id=0,
        sample_id=0,
        group_id=0,
        session_id="s",
    )
    model = bridge.RunRequest(**req)  # pydantic validation against the real model
    assert model.agent_name == "oracle"
    assert bridge.extract_session_id(model.base_url) == "s"


def test_agent_aggregation_yields_the_25_metric_keys():
    agg = rollout_parity.aggregate_agent_metrics([rollout_parity.synthetic_agent_metrics()] * 2)
    assert len(agg) == 25
    assert all(k.startswith("agent/") for k in agg)
    assert agg["agent/turns_sum"] == 28
    assert agg["agent/n_input_tokens_sum"] == 96420


def test_vendored_aggregation_matches_generate_when_available():
    """Guard against drift from generate.aggregate_agent_metrics (needs miles/torch)."""
    try:
        spec = importlib.util.spec_from_file_location("generate_real", _HERE.parent / "generate.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
    except Exception:
        pytest.skip("generate.py needs miles/torch — parity checked on the agent-env host")

    class S:
        def __init__(self, m):
            self.metadata = {"agent_metrics": m}

    payload = rollout_parity.synthetic_agent_metrics()
    real = gen.aggregate_agent_metrics([S(payload), S(payload)])
    vendored = rollout_parity.aggregate_agent_metrics([payload, payload])
    assert real == vendored


def test_step_metrics_are_all_scalar_so_probe_forwards_them():
    reward = 1.0
    rollout = rollout_parity.rollout_step_metrics(reward, [rollout_parity.synthetic_agent_metrics()], step=600)
    train = rollout_parity.train_step_metrics(600)
    assert rollout["reward"] == reward and rollout["rollout/step"] == 600
    assert train["train/step"] == 600 and "train/loss" in train
    for name, metrics in (("rollout", rollout), ("train", train)):
        for key, value in metrics.items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}:{key} non-scalar"


# --- Tier 2: real oracle sandbox trial + capture (Docker + harbor) ------------


@pytest.mark.harbor
@pytest.mark.parametrize("sandbox", SANDBOX_NAMES)
def test_oracle_capture_reflects_sandbox_changes(tmp_path, sandbox):
    bridge = oracle_capture.load_bridge()
    settings = oracle_capture.make_oracle_settings(
        bridge,
        tasks_dir=SANDBOXES,
        trials_dir=tmp_path / "trials",
        capture_dir=tmp_path / "captures",
    )
    resp = asyncio.run(oracle_capture.run_oracle_capture(bridge, settings, sandbox, step_index=600, session_id="fx"))

    # The oracle applied the solution -> verifier passed -> reward 1.0.
    assert resp.reward == 1.0, f"{sandbox}: exit={resp.exit_status}"
    assert resp.capture is not None and resp.capture.sandbox_state is not None
    assert resp.capture.sandbox_state["status"] == {"begin": "ok", "end": "ok"}

    # The staged probe.sandbox-state/1 bundle is complete + integrity-verified.
    csb = _load_check_sandbox_bundle()
    assert _run_checker(csb, settings.capture_dir) == 0

    # The delta manifest names the files the solution actually wrote.
    bundle = _latest_bundle(settings.capture_dir)
    end_manifest = _read_manifest(bundle / "end-manifest.jsonl.gz")
    for expected in oracle_capture.solution_writes(SANDBOXES / sandbox):
        assert expected in end_manifest, f"{sandbox}: {expected} not in end manifest"

    # Deletions are derived, not stored (probe.sandbox-state/1 keeps no
    # tombstones): a deleted path is in the begin manifest and gone from the
    # end manifest.
    begin_manifest = _read_manifest(bundle / "begin-manifest.jsonl.gz")
    for expected in oracle_capture.solution_deletes(SANDBOXES / sandbox):
        assert expected in begin_manifest, f"{sandbox}: {expected} not in begin manifest"
        assert expected not in end_manifest, f"{sandbox}: deleted {expected} still in end manifest"


def _run_checker(csb, capture_dir) -> int:
    argv = sys.argv
    sys.argv = ["check_sandbox_bundle.py", str(capture_dir), "--latest", "--require-integrity"]
    try:
        return csb.main()
    finally:
        sys.argv = argv


def _latest_bundle(capture_dir: Path) -> Path:
    bundles = sorted(capture_dir.glob("**/probe-sandbox-state"))
    assert bundles, f"no sandbox-state bundle under {capture_dir}"
    return bundles[-1]


def _read_manifest(path: Path) -> set[str]:
    import gzip

    with gzip.open(path, "rb") as handle:
        return {json.loads(line)["p"] for line in handle}
