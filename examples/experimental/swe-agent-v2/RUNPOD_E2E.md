# Runpod E2E and training runbook: Miles + public Harbor

This checklist is for a coding agent operating on the actual Runpod GPU
nodes. It targets GLM-4.7-Flash and Terminal-Bench 2 using the public-Harbor
bridge in this directory.

Follow the gates in order. Do **not** start normal-mode training until one real
Mini-SWE-Agent rollout completes through Miles, Harbor, a task container, and
the verifier.

If a gate fails, stop and report the exact command, exit code, relevant log
tail, and current topology. Do not reset the checkout, delete user data, kill
unrelated containers, or improvise a materially different deployment.

## Expected topology

The minimum recommended training topology is:

- one 8-GPU Runpod node for Megatron training;
- one 8-GPU Runpod node for SGLang rollout;
- one public-Harbor bridge with access to a Docker daemon;
- persistent storage mounted at `/workspace` on both GPU nodes;
- bidirectional private connectivity between the bridge and Miles session
  server.

The bridge can run on the head GPU node only if `docker info` and the nested
container callback test below pass. Otherwise, run it on a separate
Docker-capable CPU host. Do not spend GPU time trying to repair a Pod that
cannot run Docker.

Runpod network volumes must be selected when the Pod or Instant Cluster is
created and mount at `/workspace`; they cannot be attached to an existing Pod.
They can be populated before compute starts through Runpod's S3-compatible
API. See [network volumes][runpod-volume]. Enable [global networking][runpod-networking]
for multiple Runpod Pods.

Do not put the long-running bridge `POST /run` endpoint behind Runpod's HTTP
proxy: the proxy has a 100-second request limit. Use private networking or a
direct TCP route. See [Runpod port behavior][runpod-ports].

## 0. Record deployment facts

Record these values in the run notes, excluding all secrets:

```text
Miles git revision:
Runpod Pod IDs:
GPU type/count per node:
Head and worker private IP/DNS:
Persistent-volume path:
HF checkpoint path:
Megatron torch_dist path:
Harbor bridge host and AGENT_SERVER_URL:
Miles callback hostname/IP and port:
W&B run name:
```

Never paste the Harbor bearer token, Hugging Face token, W&B key, or Runpod API
key into the notes, logs, or repository.

## 1. Verify checkout and runtimes

The checkout must contain this public bridge. Do not silently fall back to an
older `origin/main` containing only the `harbor-private` instructions.

On the head:

```bash
set -euo pipefail
export MILES_ROOT=/workspace/miles
cd "$MILES_ROOT"

git status --short
git rev-parse HEAD
test -f examples/experimental/swe-agent-v2/public_harbor_server.py
test -f examples/experimental/swe-agent-v2/RUNPOD_E2E.md
nvidia-smi
python --version
ray --version
uv --version
df -h /workspace
```

Require all expected GPUs and confirm `/workspace` is persistent. Repeat
`nvidia-smi`, `ray --version`, and `df -h /workspace` on the worker.

Install Harbor in an isolated venv so it cannot alter Miles' dependencies:

```bash
export HARBOR_VENV=/workspace/venvs/harbor-0.18
uv venv "$HARBOR_VENV" --python 3.12
uv pip install --python "$HARBOR_VENV/bin/python" \
  -r examples/experimental/swe-agent-v2/requirements-public-harbor-server.txt
uv pip install --python "$HARBOR_VENV/bin/python" pytest pytest-asyncio ruff

"$HARBOR_VENV/bin/harbor" --version
"$HARBOR_VENV/bin/python" -m pytest -q \
  tests/fast/experimental/test_public_harbor_server.py \
  --confcutdir=tests/fast/experimental \
  --override-ini='addopts='
```

Required result: Harbor `0.18.0` and all bridge contract tests passing.

## 2. Choose and prove the Harbor host

### Path A: bridge on the Runpod head

```bash
docker info
docker run --rm hello-world
```

Both must succeed. Find an address on the head that a new Docker container can
reach:

```bash
export CALLBACK_HOST="$(hostname -I | awk '{print $1}')"
python -m http.server 39090 --bind 0.0.0.0 >/tmp/callback-probe.log 2>&1 &
PROBE_PID=$!
docker run --rm curlimages/curl:8.12.1 -fsS \
  "http://${CALLBACK_HOST}:39090/" >/dev/null
kill "$PROBE_PID"
echo "Docker-to-Miles callback works: $CALLBACK_HOST"
```

Do not use `127.0.0.1`: inside a Harbor task container it means that task
container, not Miles.

### Path B: separate Docker host

Use this if either Docker command fails. On the separate host:

1. Check out the exact same Miles revision.
2. Create the Harbor venv using the commands above.
3. Download the same Harbor tasks there.
4. Establish a private route using Runpod global networking (another Runpod
   Pod), Tailscale, WireGuard, or an equivalent direct route.
5. Prove the GPU head can reach bridge port `18080`, and the bridge can reach
   the Miles callback address and session port.

On the GPU head set:

```bash
export AGENT_SERVER_URL='http://<private-bridge-host>:18080'
export CALLBACK_HOST='<GPU-address-reachable-from-bridge-and-task-containers>'
```

For a public-port fallback, require a direct symmetrical TCP mapping, then set
the launcher's now-configurable listening port:

```bash
export CALLBACK_HOST="$RUNPOD_PUBLIC_IP"
export MILES_SESSION_SERVER_PORT='<symmetrically-mapped-port>'
```

Restrict inbound traffic to the bridge host if possible. The Miles session
endpoint is not intended as a general public API.

## 3. Download TB2 and create Miles JSONL

Run task download on the bridge host:

```bash
export HARBOR_DATA_ROOT=/workspace/harbor
mkdir -p "$HARBOR_DATA_ROOT/tasks" "$HARBOR_DATA_ROOT/trials"
"$HARBOR_VENV/bin/harbor" download \
  terminal-bench/terminal-bench-2 \
  --output-dir "$HARBOR_DATA_ROOT/tasks" \
  --export

export HARBOR_TASKS_DIR="$HARBOR_DATA_ROOT/tasks/terminal-bench-2"
test -d "$HARBOR_TASKS_DIR"
find "$HARBOR_TASKS_DIR" -mindepth 2 -maxdepth 2 \
  \( -name task.toml -o -name instruction.md \) | head
```

If the export uses a versioned directory, set `HARBOR_TASKS_DIR` to the
directory that directly contains one subdirectory per task.

Create full and one-task datasets. If the bridge is remote, run this wherever
the task export lives and copy both JSONL files to the GPU volume afterward:

```bash
export TB2_JSONL=/workspace/data/tb2_all.jsonl
export TB2_SMOKE_JSONL=/workspace/data/tb2_smoke.jsonl
mkdir -p /workspace/data
export HARBOR_TASKS_DIR TB2_JSONL TB2_SMOKE_JSONL

python - <<'PY'
import json
import os
from pathlib import Path

tasks = Path(os.environ["HARBOR_TASKS_DIR"])
rows = []
for task in sorted(path for path in tasks.iterdir() if path.is_dir()):
    instruction = task / "instruction.md"
    if instruction.is_file() and (task / "task.toml").is_file():
        rows.append({
            "prompt": instruction.read_text(),
            "metadata": {
                "instance_id": task.name,
                "agent_name": "mini-swe-agent",
                "split": "train",
            },
        })
if not rows:
    raise SystemExit(f"No Harbor tasks found under {tasks}")

Path(os.environ["TB2_JSONL"]).write_text(
    "".join(json.dumps(row) + "\n" for row in rows)
)
Path(os.environ["TB2_SMOKE_JSONL"]).write_text(json.dumps(rows[0]) + "\n")
print(f"wrote {len(rows)} tasks; smoke={rows[0]['metadata']['instance_id']}")
PY

wc -l "$TB2_JSONL" "$TB2_SMOKE_JSONL"
```

Verify every JSONL `instance_id` exists on the bridge. Prompts and tasks must
come from the same exported dataset version.

## 4. Start and oracle-smoke the bridge

Generate one secret, transfer it securely to the GPU head, and export the same
value as both variables without committing it:

```bash
umask 077
export MILES_HARBOR_AUTH_TOKEN="$(openssl rand -hex 32)"
export AGENT_SERVER_AUTH_TOKEN="$MILES_HARBOR_AUTH_TOKEN"
```

On the bridge host:

```bash
export MILES_ROOT="${MILES_ROOT:-/workspace/miles}"
export HARBOR_VENV="${HARBOR_VENV:-/workspace/venvs/harbor-0.18}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/workspace/harbor/tasks/terminal-bench-2}"
export HARBOR_TRIALS_DIR=/workspace/harbor/trials
export HARBOR_ENVIRONMENT_TYPE=docker
export HARBOR_DELETE_ENVIRONMENTS=true
export AGENT_MAX_CONCURRENT=2
export MILES_HARBOR_REQUEST_TIMEOUT_SEC=14400
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="${CALLBACK_HOST},localhost,127.0.0.1"

mkdir -p /workspace/logs
nohup "$HARBOR_VENV/bin/python" \
  "$MILES_ROOT/examples/experimental/swe-agent-v2/public_harbor_server.py" \
  --host 0.0.0.0 --port 18080 \
  >/workspace/logs/public-harbor-server.log 2>&1 &
echo $! >/workspace/logs/public-harbor-server.pid
```

Set `AGENT_SERVER_URL` to an address every Ray node can resolve; for a head
bridge, use its private/global-network DNS rather than `127.0.0.1`:

```bash
export AGENT_SERVER_URL='http://<head-private-DNS-or-IP>:18080'
curl -fsS "$AGENT_SERVER_URL/health"
```

Run that health check from the GPU head and worker, not only from the bridge
host.

Run an authenticated oracle trial. The oracle proves Harbor → Docker →
verifier without needing Miles inference yet:

```bash
export SMOKE_INSTANCE="$(python -c 'import json,os; print(json.loads(open(os.environ["TB2_SMOKE_JSONL"]).readline())["metadata"]["instance_id"])')"
export MILES_SESSION_SERVER_PORT="${MILES_SESSION_SERVER_PORT:-30000}"

test "$(curl -sS -o /tmp/no-auth.json -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d "{\"base_url\":\"http://${CALLBACK_HOST}:${MILES_SESSION_SERVER_PORT}/sessions/oracle-smoke/v1\",\"model\":\"openai/model\",\"instance_id\":\"${SMOKE_INSTANCE}\",\"agent_name\":\"oracle\"}" \
  "$AGENT_SERVER_URL/run")" = 401

curl -fsS \
  -H "Authorization: Bearer ${AGENT_SERVER_AUTH_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"base_url\":\"http://${CALLBACK_HOST}:${MILES_SESSION_SERVER_PORT}/sessions/oracle-smoke/v1\",\"model\":\"openai/model\",\"instance_id\":\"${SMOKE_INSTANCE}\",\"agent_name\":\"oracle\"}" \
  "$AGENT_SERVER_URL/run"

```

Require HTTP 200, `Submitted`, a verifier report, and no leftover Harbor task
container. The oracle reward itself may be task-specific. On the bridge host,
confirm cleanup and inspect logs:

```bash
docker ps --filter 'name=hb__'
tail -n 100 /workspace/logs/public-harbor-server.log
```

## 5. Verify model checkpoints

Use the actual persistent paths:

```bash
export HF_CHECKPOINT=/workspace/models/zai-org/GLM-4.7-Flash
export REF_LOAD=/workspace/models/zai-org/GLM-4.7-Flash_torch_dist
export MEGATRON_PATH=/root/Megatron-LM

test -d "$HF_CHECKPOINT"
test -d "$MEGATRON_PATH"
```

If `REF_LOAD` is absent, convert it once on the head:

```bash
cd "$MILES_ROOT"
source scripts/models/glm4.7-flash.sh
PYTHONPATH="$MEGATRON_PATH" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --save "$REF_LOAD"
```

Do not run two conversions against the same output directory.

## 6. Start Ray and prove one real rollout

Start only the head initially. This keeps the first callback test simple:

```bash
export HEAD_IP="$(hostname -I | awk '{print $1}')"
export MASTER_ADDR="$HEAD_IP"

ray stop --force || true
ray start --head \
  --node-ip-address "$HEAD_IP" \
  --num-gpus 8 \
  --dashboard-host 0.0.0.0 \
  --disable-usage-stats
ray status
```

Export values that must enter Ray's runtime environment:

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=http://127.0.0.1:8265
export AGENT_SERVER_TIMEOUT_SEC=14400
export AGENT_MODEL_NAME=model
export MILES_ROUTER_EXTERNAL_HOST="$CALLBACK_HOST"
export MILES_HOST_IP="$HEAD_IP"
export MILES_SESSION_SERVER_PORT="${MILES_SESSION_SERVER_PORT:-30000}"

: "${AGENT_SERVER_URL:?set the bridge URL}"
: "${AGENT_SERVER_AUTH_TOKEN:?set the shared bearer token}"
: "${CALLBACK_HOST:?set the tested callback address}"
: "${TB2_SMOKE_JSONL:?set the one-task dataset path}"
```

Launch exactly one Mini-SWE-Agent rollout without a training update:

```bash
cd "$MILES_ROOT"
python examples/experimental/swe-agent-v2/run.py \
  --mode debug_rollout_only \
  --num-nodes 1 --num-gpus-per-node 8 \
  --skip-prepare \
  --megatron-path "$MEGATRON_PATH" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --prompt-data "$TB2_SMOKE_JSONL" \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --global-batch-size 1 \
  --max-seq-len 16384 \
  --rollout-max-response-len 8192 \
  --agent-server-url "$AGENT_SERVER_URL" \
  --agent-server-auth-token "$AGENT_SERVER_AUTH_TOKEN" \
  --agent-server-timeout-sec 14400 \
  --session-server-port "$MILES_SESSION_SERVER_PORT" \
  --router-external-host "$CALLBACK_HOST" \
  --miles-host-ip "$HEAD_IP"
```

Capture the submitted Ray job ID and monitor it:

```bash
ray job list --address "$RAY_ADDRESS"
ray job status '<job-id>' --address "$RAY_ADDRESS"
ray job logs '<job-id>' --address "$RAY_ADDRESS" --follow
```

This gate passes only when:

- Harbor starts `mini-swe-agent`, not `oracle`;
- the task calls `/sessions/<id>/v1/chat/completions` successfully;
- Miles records at least one model turn;
- Harbor runs the verifier and returns `Submitted` (reward may be 0 or 1);
- the Ray job exits successfully after one rollout;
- no 401, session 404, callback, or cleanup failure occurs.

A zero reward is not automatically an integration failure. `AgentError`, a
missing session, no recorded turn, or a leaked task container is.

## 7. Join the worker and repeat on the real topology

On the worker:

```bash
export HEAD_IP='<head-private-IP>'
export WORKER_IP="$(hostname -I | awk '{print $1}')"
ray stop --force || true
ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address "$WORKER_IP" \
  --num-gpus 8 \
  --disable-usage-stats
```

On the head require two alive nodes and sixteen GPUs; from the worker also
require that the bridge is reachable:

```bash
ray status
ray list nodes --address "$RAY_ADDRESS"
curl -fsS "$AGENT_SERVER_URL/health"
```

Then submit one fully-async rollout on the disaggregated topology:

```bash
export DEBUG_SAVE=/workspace/runs/runpod-public-harbor-async-debug

python examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py \
  --mode debug_rollout_only \
  --num-nodes 2 --train-num-nodes 1 --num-gpus-per-node 8 \
  --skip-prepare \
  --megatron-path "$MEGATRON_PATH" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --prompt-data "$TB2_SMOKE_JSONL" \
  --save-dir "$DEBUG_SAVE/checkpoints" \
  --save-traces-dir "$DEBUG_SAVE/traces" \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --over-sampling-batch-size 1 \
  --global-batch-size 1 \
  --max-seq-len 16384 \
  --rollout-max-response-len 8192 \
  --agent-server-url "$AGENT_SERVER_URL" \
  --agent-server-auth-token "$AGENT_SERVER_AUTH_TOKEN" \
  --agent-server-timeout-sec 14400 \
  --session-server-port "$MILES_SESSION_SERVER_PORT" \
  --router-external-host "$CALLBACK_HOST" \
  --miles-host-ip "$HEAD_IP"
```

Apply the same pass criteria. This second debug run is decisive because SGLang
and the agent callback now cross nodes.

## 8. Launch two-node training

Only proceed after the two-node debug job succeeds:

```bash
export RUN_TAG="$(date -u +%Y%m%d-%H%M)-public-harbor-tb2"
export RUN_ROOT="/workspace/runs/$RUN_TAG"
mkdir -p "$RUN_ROOT"

python examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py \
  --mode normal \
  --num-nodes 2 --train-num-nodes 1 --num-gpus-per-node 8 \
  --skip-prepare \
  --megatron-path "$MEGATRON_PATH" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --prompt-data "$TB2_JSONL" \
  --save-dir "$RUN_ROOT/checkpoints" \
  --save-traces-dir "$RUN_ROOT/traces" \
  --num-rollout 3000 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --over-sampling-batch-size 64 \
  --global-batch-size 32 \
  --max-seq-len 65536 \
  --rollout-max-response-len 8192 \
  --save-interval 5 \
  --agent-server-url "$AGENT_SERVER_URL" \
  --agent-server-auth-token "$AGENT_SERVER_AUTH_TOKEN" \
  --agent-server-timeout-sec 14400 \
  --session-server-port "$MILES_SESSION_SERVER_PORT" \
  --router-external-host "$CALLBACK_HOST" \
  --miles-host-ip "$HEAD_IP" \
  --wandb-project glm47-flash-agentic-async \
  --wandb-run-name "$RUN_TAG" \
  2>&1 | tee "$RUN_ROOT/launcher.log"
```

Public Harbor 0.18.0's Mini-SWE-Agent adapter does not accept per-trial
temperature. The bridge forwards its supported `max_tokens` and
`reasoning_effort` settings.

Do not leave the first iteration unattended. Require:

1. Ray is `RUNNING` and both nodes remain alive.
2. SGLang becomes healthy without OOM.
3. The bridge receives expected TB2 Mini-SWE-Agent trials.
4. Trials contain Miles model calls and verifier results.
5. The first rollout and GRPO `step 0` complete.
6. Checkpoint and trace files appear under `$RUN_ROOT`.
7. Harbor's container count returns toward zero between batches.

Increase `AGENT_MAX_CONCURRENT` gradually only when bridge CPU, RAM, disk, and
Docker capacity allow it. Thirty-two trials per iteration does not mean thirty-two
task containers should immediately run at once.

## 9. Monitor and stop safely

On the head:

```bash
ray job list --address "$RAY_ADDRESS"
ray job status '<job-id>' --address "$RAY_ADDRESS"
ray job logs '<job-id>' --address "$RAY_ADDRESS" --follow
watch -n 5 nvidia-smi
```

On the bridge:

```bash
tail -F /workspace/logs/public-harbor-server.log
watch -n 5 "docker ps --filter 'name=hb__' --format '{{.Names}} {{.Status}}'"
df -h /workspace
```

Stop without deleting persistent data:

```bash
ray job stop '<job-id>' --address "$RAY_ADDRESS"
ray job status '<job-id>' --address "$RAY_ADDRESS"
```

Wait for Harbor trials to finish or cancel, confirm containers are gone, then
stop the bridge or Ray. Back up checkpoints before deleting the cluster.

## Failure decisions

| Symptom | Action |
| --- | --- |
| `docker info` fails on the GPU Pod | Move the bridge to a Docker-capable host; do not train. |
| Oracle `/run` returns 524 after about 100 seconds | Remove Runpod HTTP proxy from the bridge path. |
| Bridge returns callback-host 422 | Add the exact host to `MILES_HARBOR_ALLOWED_CALLBACK_HOSTS` and restart. |
| Agent gets callback connection refused | Re-run the container probe; correct host/port. |
| Agent gets `/sessions/...` 404 | Advertised host reaches the wrong session server; stop. |
| Ray sees fewer than 16 GPUs | Fix cluster membership before the two-node command. |
| SGLang OOMs | Clean GPUs and lower memory/batch settings before changing parallelism. |
| Harbor containers accumulate | Stop submissions and inspect/cancel confirmed orphan trials. |
| Reward is zero with `Submitted` | Inspect verifier output; integration may still be healthy. |
| `AgentError` or `TimeLimitExceeded` | Inspect Harbor artifacts/callback logs; this is not reward quality. |

[runpod-volume]: https://docs.runpod.io/storage/network-volumes
[runpod-networking]: https://docs.runpod.io/pods/networking
[runpod-ports]: https://docs.runpod.io/pods/configuration/expose-ports
