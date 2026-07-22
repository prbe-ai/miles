---
title: Monitoring & Logging
description: wandb, structured logs, profiling, and what to look at when something looks off.
---
Miles emits per-rollout metrics to stdout and (optionally) Weights & Biases. SGLang and
Ray write their own logs to their default directories.

## What gets logged by default

Each rollout iteration emits a structured row to stdout (illustrative shape — exact
fields depend on backend and config):

```text
[trainer] iter 12/3000 | loss=0.412 reward=0.61 kl=0.018
                      | rollout=18.4s train=22.1s p2p=2.1s  (total 42.6s)
                      | grad_norm=0.93 lr=1.0e-06
```

When `--use-wandb` is set, metrics also go to wandb under the `train/`, `rollout/`,
and `perf/` namespaces (see `miles/utils/wandb_utils.py`).

## Enabling wandb

```bash
ray job submit --address=auto -- \
  python3 train.py ... \
    --use-wandb \
    --wandb-project miles \
    --wandb-group qwen3-30b-grpo
```

Available flags: `--use-wandb`, `--wandb-project`, `--wandb-group`. `WANDB_API_KEY`
should be supplied via Ray's `env_vars` rather than baked into the launch script.

## Research OS / Probe

Miles can queue every scalar emitted by its existing tracking manager and export
it to Research OS without adding SDK calls to training code:

```bash
pip install 'miles[probe]'  # Python 3.11+
export PROBE_TOKEN='<write token>'
export MILES_USE_PROBE=1
export PROBE_PROJECT='miles'
export PROBE_EXPERIMENT='my-experiment'
export PROBE_EXTERNAL_ID='<stable-native-job-id>'
```

The primary process creates or resumes one Research OS run and publishes its ID
to the Ray actors. Each initialized producer writes scalar metric batches, its
producer ID, local sequence, step, and event time to an atomic queue under
`--probe-queue-dir` (by default, under `--save`) before returning to training. A
single leased background exporter confirms and removes them. Set the queue to a
shared PVC for distributed or ephemeral workers. API outages leave inspectable
pending records plus the complete run-creation intent for later retry instead of
blocking a training actor or relying on an ephemeral SDK spool.

The export status deliberately separates `publication_state` from
`capture_completeness`. `drained` proves that this queue had no unconfirmed
records when the exporter stopped. Miles does not currently expose a reliable
expected-producer set or close every secondary tracker, so completeness remains
`unknown` and those two missing guarantees are named in `export-status.json`.
This prevents a missing/mis-mounted worker PVC from being reported as complete.
`--probe-fail-open` is enabled by default; an inaccessible queue is reported as
`unavailable` and does not abort training.

If a job exits with unconfirmed records, the status file in that directory
contains the last export error. Once connectivity returns, drain it into the
same run without restarting training:

```bash
python -m miles.utils.tracking_utils.probe_utils \
  /shared/probe/metrics/<run-queue>
```

The repair command creates or resolves the intended Research OS run from
`intent.json`, restores its snapshot and native links, validates any run IDs
already present in queue records, and exclusively leases the exporter. Pass
`--run <research-os-run-id>` only to override an intent that has not resolved a
run yet. No ATIF input is required.

## What to watch

| Signal | Healthy pattern | Red flag |
|---|---|---|
| `loss` | Slow decay over hundreds of iterations | Spike → crash within an iteration |
| `raw_reward` | Trending up, with healthy variance | Saturates near a single value (collapse) |
| `kl_loss` | Bounded, drifts up over time | Sudden jump (policy diverged from ref) — only logged when `--use-kl-loss` |
| `train_rollout_logprob_abs_diff` | Stable and small (≪ 1.0) | Climbing without bound → train/inference precision drift |
| `entropy_loss` | Slowly decreasing | Falls to ~0 too fast (mode collapse) |
| `grad_norm` | < `clip_grad` (1.0 by default) | Repeatedly hitting clip threshold |
| `rollout_time` / `train_time` | Roughly balanced | One ≫ other → resource imbalance |
| `train/pg_clipfrac` | < 0.2 | > 0.5 means policy is moving fast → drop LR |

Panel names follow what `loss.py` and the rollout logger emit; Miles's wandb metrics
live under `train/`, `rollout/`, `perf/`, `multi_turn/`, `passrate/` namespaces.

## Custom loggers

Replace the default rollout logger with your own to push to internal systems:

```python
def my_log(rollout_id, args, samples, extra, rollout_time) -> bool:
    statsd.gauge("miles.reward", mean([s.reward for s in samples]))
    return False   # also keep default logging
```

```bash
--custom-rollout-log-function-path my_pkg.logging.my_log
```

## Profiling

| Tool | When |
|---|---|
| `nvidia-smi dmon -s u` | Quick sanity check on GPU utilization |
| `nsys profile` | Deep CUDA-level profiling |
| `py-spy dump --pid <ray worker>` | Find Python-side stalls |
| `ray timeline` | Inspect Ray task scheduling |

### Built-in PyTorch profiler

The PyTorch profiler is wired into Miles via `miles/utils/profile_utils.py`. Flags
differ by backend:

**Megatron** — choose which sub-loop to profile:

```bash
--profile-target train_overall    # or train_actor, train_log_probs (multi-arg)
```

**FSDP** — additionally exposes the standard FSDPArgs window:

```bash
--use-pytorch-profiler
--profile-step-start 10
--profile-step-end 12
--memory-snapshot-path snapshot.pickle
--tensorboard-dir /data/tb-run-42
```

Open the trace in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

## Where the log files live

| Source | Path |
|---|---|
| Trainer stdout | wherever you redirected `ray job submit` (or Ray dashboard) |
| Ray workers | `~/.ray/session_latest/logs/` |
| wandb local cache | `wandb/run-<id>/files/` |
| FSDP profiler / memory snapshot | `--tensorboard-dir`, `--memory-snapshot-path` |

## Router endpoints

The router exposes a small FastAPI surface used internally by Miles:

| Endpoint | Method | What |
|---|---|---|
| `/add_worker` | POST | Register an SGLang engine |
| `/list_workers` | GET | List registered workers |
| any other path | GET / POST / PUT / DELETE | Proxied to a selected SGLang worker — e.g. `/generate`, `/v1/chat/completions`, `/health`. |
