# Miles-Harbor capture fixture

A self-contained fixture for exercising the **Probe data-capture pipeline**
(sandbox-state capture + metric streaming) the way our customer's RL framework
(Miles + Harbor) drives it — **without SGLang, Megatron, a model, or a GPU**.

It runs a *real* Harbor sandbox trial using Harbor's `oracle` agent, which
applies the task's `solution/solve.sh` **as the agent**. Because it runs in the
agent phase, `AGENT_START`/`AGENT_END` fire and the sandbox-state capture runs;
because it's the oracle, no model is ever called and the result is deterministic.
Point it at any sandbox (Harbor task dir) to validate capture on that sandbox.

## Layout

```
capture_fixture/
├── rollout_parity.py       # Miles /run request shape + rollout/step + train/step + agent/* metric parity (pure stdlib)
├── oracle_capture.py       # run a model-free oracle trial through the bridge, with sandbox capture
├── sandboxes/              # parametrizable Harbor task dirs (add your own here)
│   ├── hello-world/        # minimal: writes one file
│   └── filetree/           # stress: add/modify/delete + binary + nested + symlink
├── test_capture_pipeline.py
├── conftest.py
└── README.md
```

## Two tiers

- **Tier 1 — parity logic (runs anywhere).** Asserts the fixture's request and
  metric shapes match what Miles sends and logs, validated against the real
  bridge `RunRequest` model and the real `generate.aggregate_agent_metrics`
  (when miles/torch are importable). No Docker.

  ```bash
  pytest examples/experimental/swe-agent-v2/capture_fixture -m "not harbor"
  ```

- **Tier 2 — real oracle sandbox trial (`@pytest.mark.harbor`).** For each
  sandbox, runs a real Harbor oracle trial and asserts the `probe.sandbox-state/1`
  bundle reflects the sandbox's actual filesystem changes (via
  `check_sandbox_bundle.py`), plus reward and capture status. Auto-skips unless
  Docker + `harbor` are present, so it runs on the agent-env host:

  ```bash
  pip install -r ../requirements-public-harbor-capture.txt   # probe-research >= 0.23.0 + harbor
  pytest examples/experimental/swe-agent-v2/capture_fixture -m harbor
  ```

## Add a sandbox

Drop a Harbor task dir under `sandboxes/<name>/` (`task.toml`, `instruction.md`,
`environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh`). Tier 2 is
parametrized over every dir with a `task.toml`, so a new sandbox is picked up
automatically. Have `solve.sh` write to absolute paths so the fixture can assert
they land in the delta.

## Stream the parity metrics to Probe (optional)

`rollout_parity` produces the exact scalar groups Miles logs — feed them through
the same path a training run uses to see the charts render:

```python
from rollout_parity import rollout_step_metrics, train_step_metrics, synthetic_agent_metrics
# then log each dict at its step via `probe log <run> k=v --step N`, or the SDK.
```

The oracle capture stages a real trial + sandbox bundle under the capture dir;
`probe trial watch <capture-dir>` uploads it exactly as in production.

## Why oracle mode

`oracle` is a Harbor built-in agent (`AgentName.ORACLE`) whose `model_name` is
optional — it copies + runs the task solution instead of querying a model. This
gives a full, deterministic trial lifecycle (agent phase → verifier → reward)
with zero inference cost, which is what makes sandbox-by-sandbox capture testing
cheap.
