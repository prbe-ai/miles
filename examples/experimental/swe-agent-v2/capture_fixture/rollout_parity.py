"""Miles-Harbor rollout + training parity data, without the RL stack.

This is the customer-specific fixture core: it reproduces the *shape* of the
data a real Miles rollout + GRPO training step emits — the ``/run`` request Miles
sends to the Harbor bridge, the per-trial agent metrics Harbor returns, and the
``rollout/step`` + ``train/step`` scalar groups Miles logs — so the whole Probe
capture/streaming pipeline can be exercised against realistic data with no
SGLang, no Megatron, and no GPU.

Pure stdlib, no torch — importable and testable anywhere. The agent-metric
aggregation is vendored verbatim from ``generate.py``; ``test_capture_pipeline``
asserts it stays in parity with the real function when miles/torch are present.
"""

from __future__ import annotations

from typing import Any

# Step-key groups Miles logs under (miles/backends/training_utils/log_utils.py).
STEP_KEY_ROLLOUT = "rollout/step"
STEP_KEY_TRAIN = "train/step"

# The oracle agent applies the task's solution as the agent — model-free and
# deterministic, so AGENT_START/AGENT_END fire and the sandbox capture runs.
ORACLE_AGENT = "oracle"


# --- Miles rollout request (what Miles POSTs to the Harbor bridge) -----------


def build_oracle_run_request(
    task_name: str,
    *,
    step_index: int,
    rollout_id: int,
    sample_id: int,
    group_id: int,
    session_id: str,
    run_id: str | None = None,
    miles_run_id: str | None = None,
    callback_host: str = "localhost",
) -> dict[str, Any]:
    """A ``RunRequest`` payload matching a real Miles rollout, but agent=oracle.

    The base_url carries a ``/sessions/<id>/v1`` path (Miles's session URL shape)
    so the bridge extracts the session id for correlation; the oracle agent never
    calls it, so any allowed host works.
    """
    return {
        "base_url": f"http://{callback_host}/sessions/{session_id}/v1",
        "model": "oracle/none",
        "agent_name": ORACLE_AGENT,
        "instance_id": task_name,
        "run_id": run_id,
        "miles_run_id": miles_run_id,
        "rollout_id": rollout_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "step_index": step_index,
        "capture_context": {"fixture": "miles-harbor-capture", "data_mix_id": "smoke"},
    }


# --- Per-trial agent metrics (the shape Harbor's AgentResult carries) --------


def synthetic_agent_metrics(
    *,
    turns: int = 14,
    tool_calls: int = 13,
    n_input_tokens: int = 48210,
    n_cache_tokens: int = 31000,
    n_output_tokens: int = 5120,
    cost_usd: float = 0.0,
) -> dict[str, Any]:
    """A realistic per-trial ``agent_metrics`` payload.

    The oracle agent itself reports no tokens/turns, so we inject this to test
    the metric pipeline against a *real agent's* shape (parity), not the oracle's
    empty one.
    """
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "n_input_tokens": n_input_tokens,
        "n_cache_tokens": n_cache_tokens,
        "n_output_tokens": n_output_tokens,
        "cost_usd": cost_usd,
        "model_query_time_sum": 41.2,
        "env_execution_time_sum": 8.7,
        "eval_time": 22.0,
        "agent_run_time": 58.1,
        "time_per_turn": 4.1,
        "model_query_time_avg": 2.9,
        "env_execution_time_avg": 0.6,
        "model_time_ratio": 0.71,
        "env_time_ratio": 0.15,
        "eval_time_ratio": 0.14,
        "total_time": 80.1,
    }


# --- agent/* aggregation: vendored verbatim from generate.py -----------------
# Kept in sync by test_agent_aggregation_matches_generate (torch-gated).


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m.get(key, 0) for m in all_metrics]


def _collect_present_numbers(all_metrics: list[dict], key: str) -> list[float]:
    return [float(value) for metric in all_metrics if key in metric and (value := metric[key]) is not None and isinstance(value, int | float) and not isinstance(value, bool)]


def _agg_mean(metrics, all_metrics, keys, prefix="agent/", suffix="_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(all_agent_metrics: list[dict]) -> dict:
    """Mirror of generate.aggregate_agent_metrics (operates on the raw dicts)."""
    all_metrics = [m for m in all_agent_metrics if m]
    if not all_metrics:
        return {}
    metrics: dict[str, Any] = {}
    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)
    for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
        values = _collect_present_numbers(all_metrics, key)
        if values:
            metrics[f"agent/{key}_sum"] = sum(values)
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")
    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)
    return metrics


# --- Step metrics (the scalar groups Miles logs) -----------------------------


def rollout_step_metrics(
    reward: float,
    agent_metrics_per_sample: list[dict],
    *,
    step: int,
    response_length_mean: float = 812.0,
) -> dict[str, Any]:
    """A ``rollout/step`` scalar dict matching Miles's rollout logging shape.

    Reward + response-length + perf + the aggregated agent/* metrics. Non-scalar
    per-token arrays (log_probs/values) are intentionally omitted — the Probe
    backend drops them anyway.
    """
    metrics: dict[str, Any] = {
        "rollout/step": step,
        "reward": reward,
        "rollout/entropy": 0.83,
        "raw_response_length/response_length_mean": response_length_mean,
        "raw_response_length/response_length_max": response_length_mean * 1.8,
        "raw_response_length/response_length_min": response_length_mean * 0.4,
        "raw_response_length/response_length_clip_ratio": 0.02,
        "perf/step_time": 41.7,
        "perf/actor_train_tok_per_s": 5120.0,
    }
    metrics.update(aggregate_agent_metrics(agent_metrics_per_sample))
    return metrics


def train_step_metrics(step: int, *, reward_mean: float = 0.5) -> dict[str, Any]:
    """A synthetic ``train/step`` scalar dict (GRPO optimizer metrics)."""
    return {
        "train/step": step,
        "train/loss": max(0.01, 1.2 - 0.03 * step),
        "train/kl": 0.012 + 0.0005 * step,
        "train/entropy": 0.83,
        "train/grad_norm": 0.7,
        "train/lr": 1e-6,
        "train/reward_mean": reward_mean,
    }
