from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "examples/experimental/swe-agent-v2/generate.py"


def _load_generate_module():
    stubs: dict[str, ModuleType] = {}

    base_types = ModuleType("miles.rollout.base_types")
    base_types.RolloutFnTrainInput = type("RolloutFnTrainInput", (), {})
    base_types.RolloutFnTrainOutput = type("RolloutFnTrainOutput", (), {})
    stubs[base_types.__name__] = base_types

    inference = ModuleType("miles.rollout.inference_rollout.inference_rollout_common")
    inference.InferenceRolloutFn = type("InferenceRolloutFn", (), {})
    stubs[inference.__name__] = inference

    types_module = ModuleType("miles.utils.types")
    types_module.Sample = type("Sample", (), {})
    stubs[types_module.__name__] = types_module

    original = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("swe_agent_generate", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in original.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_harbor_token_and_cost_metrics_ignore_absent_values_in_mixed_data() -> None:
    module = _load_generate_module()
    samples = [
        SimpleNamespace(
            metadata={
                "agent_metrics": {
                    "n_input_tokens": 100,
                    "n_cache_tokens": 20,
                    "n_output_tokens": 30,
                    "cost_usd": 0.0,
                }
            }
        ),
        SimpleNamespace(metadata={"agent_metrics": {"n_input_tokens": 50, "n_output_tokens": 10}}),
        SimpleNamespace(metadata={"agent_metrics": {"turns": 2}}),
    ]

    metrics = module.aggregate_agent_metrics(samples)

    assert metrics["agent/n_input_tokens_sum"] == 150
    assert metrics["agent/n_input_tokens_mean"] == 75
    assert metrics["agent/n_cache_tokens_sum"] == 20
    assert metrics["agent/n_output_tokens_sum"] == 40
    assert metrics["agent/n_output_tokens_mean"] == 20
    assert metrics["agent/cost_usd_sum"] == 0
    assert metrics["agent/cost_usd_mean"] == 0
