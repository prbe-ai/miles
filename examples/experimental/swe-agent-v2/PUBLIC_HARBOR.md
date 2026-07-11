# Public Harbor bridge for Miles agent RL

This is the current public-code path for the SWE-Agent V2 architecture. It
replaces the deleted prototype `server.py` with
`public_harbor_server.py`, built against public Harbor 0.18.0.

## Why the old general example disappeared

Two separate removals are easy to conflate:

1. [Miles PR #952](https://github.com/radixark/miles/pull/952) removed the old
   FastAPI `server.py` on April 8, 2026, saying Harbor CLI integration had
   superseded it. The old code had also drifted from Harbor's API: it directly
   constructed `Trial(config)`, while public Harbor 0.18.0 requires
   `await Trial.create(config)` before `trial.run()`.
2. [Miles PR #982](https://github.com/radixark/miles/pull/982) removed the
   generic `prepare_harbor_tasks.py` six days later because dataset-specific
   harbor-private adapters produced more accurate task images and Dockerfiles.

The first explanation was too broad for RL training. The CLI superseded
manual download/preparation and one-shot evaluation commands, but it did not
provide Miles' long-running, per-rollout HTTP protocol. The later
[private-server training example](https://github.com/radixark/miles/pull/1236)
again depends on an agent server. So the general architecture was not shown to
be invalid; its public implementation was removed while stale and the working
production path moved into harbor-private. This bridge restores only the
missing integration layer against Harbor's current public API.

## Why a bridge is still necessary

Harbor's CLI handles human-initiated, finite operations:

```bash
harbor download terminal-bench/terminal-bench-2 --export
harbor trial start -p <task> -a <agent> -m <model>
harbor run -d terminal-bench/terminal-bench-2 -a <agent> -m <model>
harbor view <jobs-or-trials-dir>
```

Those commands replaced the original example's custom task preparation and
one-shot evaluation scripts. They do not replace Miles' runtime protocol:

- each rollout creates a different session-scoped model URL;
- Miles sends one `POST /run` per prompt/sample;
- the request must block until Harbor returns that trial's verifier reward;
- Miles needs stable error statuses and token-limit cancellation;
- a training process reuses the service for thousands of trials.

The bridge is therefore an adapter around Harbor's programmatic `Trial` API,
not a replacement for Harbor itself.

## What the bridge owns

- an unauthenticated `GET /health` endpoint and authenticated `POST /run`;
- safe `instance_id` to task-directory resolution;
- per-request model URL, credentials, and supported sampling settings;
- current `TrialConfig` plus `await Trial.create(...)` integration;
- concurrency and request timeout limits;
- Miles session identity and accumulated-token monitoring;
- cancellation on `max_seq_len`;
- Harbor result to Miles reward/status/metrics conversion.

Harbor continues to own agents, task environments, verification, artifacts,
Docker/cloud sandbox execution, and cleanup.

## Repeatable macOS smoke test

Requirements: Python 3.12+, `uv`, and a running Docker Desktop.

```bash
cd /path/to/miles

uv venv .venv-harbor --python 3.12
uv pip install --python .venv-harbor/bin/python \
  -r examples/experimental/swe-agent-v2/requirements-public-harbor-server.txt

export HARBOR_TASKS_DIR="$PWD/examples/experimental/swe-agent-v2/smoke_tasks"
export HARBOR_TRIALS_DIR="$PWD/.harbor-smoke-trials"
export MILES_HARBOR_AUTH_TOKEN="local-smoke-token"
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="localhost,127.0.0.1"
export AGENT_MAX_CONCURRENT=1

.venv-harbor/bin/python \
  examples/experimental/swe-agent-v2/public_harbor_server.py \
  --host 127.0.0.1 --port 18080
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:18080/health

curl -fsS \
  -H 'Authorization: Bearer local-smoke-token' \
  -H 'Content-Type: application/json' \
  -d '{
    "base_url": "http://localhost:30000/sessions/local-smoke/v1",
    "model": "openai/model",
    "instance_id": "hello-world",
    "agent_name": "oracle"
  }' \
  http://127.0.0.1:18080/run
```

The oracle does not call `base_url`; it proves the full bridge → Harbor →
Docker → verifier → reward path. Expected response:

```json
{
  "reward": 1.0,
  "exit_status": "Submitted",
  "agent_metrics": {
    "agent_run_time": 0.1,
    "eval_time": 0.1
  },
  "eval_report": {"reward": 1.0}
}
```

Run the contract tests without loading Miles' GPU-only test fixtures:

```bash
python -m pytest -q \
  tests/fast/experimental/test_public_harbor_server.py \
  --confcutdir=tests/fast/experimental \
  --override-ini='addopts='
```

## Public Terminal-Bench tasks

Download public TB2 with the same pinned Harbor environment:

```bash
.venv-harbor/bin/harbor download \
  terminal-bench/terminal-bench-2 \
  --output-dir /data/harbor-tasks \
  --export
```

Export mode creates a dataset directory containing one directory per task. Set
`HARBOR_TASKS_DIR` to the directory that directly contains those task
directories.

## Production configuration

At minimum, set:

```bash
export HARBOR_TASKS_DIR=/data/harbor-tasks/terminal-bench-2
export HARBOR_TRIALS_DIR=/data/harbor-trials
export HARBOR_ENVIRONMENT_TYPE=docker
export HARBOR_DELETE_ENVIRONMENTS=true
export AGENT_MAX_CONCURRENT=8
export MILES_HARBOR_AUTH_TOKEN='<random-secret>'
export AGENT_SERVER_AUTH_TOKEN="$MILES_HARBOR_AUTH_TOKEN"
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS='<Miles callback hostname>'
```

The same `AGENT_SERVER_AUTH_TOKEN` must be present in the Miles Ray runtime.
The checked-in synchronous and GLM-4.7-Flash async launchers now propagate it.
They also default `AGENT_SERVER_TIMEOUT_SEC` to four hours so a valid long
Terminal-Bench trial is not cut off by the client after exactly one hour.

Public Harbor 0.18.0's Mini-SWE-Agent adapter accepts `max_tokens` and
`reasoning_effort`, but not a per-trial `temperature`; the bridge therefore
does not pretend to forward that setting for Mini-SWE-Agent. Terminus supports
the additional sampling fields mapped by the bridge.

For a cloud sandbox provider, install the corresponding Harbor extra and set
`HARBOR_ENVIRONMENT_TYPE` plus `MILES_HARBOR_ENVIRONMENT_KWARGS_JSON`. Start
with Docker on an ordinary VM because it is closest to the validated TB2
execution path.

## Remaining validation before GPU training

1. Run one Mini-SWE-Agent task against a standalone OpenAI-compatible server.
2. Run it against a real Miles session server and confirm TITO records multiple turns.
3. Force a small `max_seq_len` and confirm bridge cancellation/cleanup.
4. Run 4-8 concurrent tasks and inspect Docker cleanup and Harbor artifacts.
5. Run Miles `debug_rollout_only` on one GPU node.
6. Only then start the two-node cloud training recipe.

For the exact GPU-node preflight, networking gates, one-rollout test, and
training launch commands, follow the provider guide:

- [Runpod](RUNPOD_E2E.md)
- [Crusoe](CRUSOE_E2E.md)
