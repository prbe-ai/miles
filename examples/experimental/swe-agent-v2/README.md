# Agent V2: Generalized Agent-Environment RL Training with Harbor

A unified pipeline for training agents on **mixed datasets** — SWE-bench, Terminal-Bench, custom tasks, etc. — through a single endpoint. Uses **TITO (Token In Token Out)** through SGLang's `/v1/chat/completions` for exact token-level training signals.

**Agent orchestration and grading** are handled by [Harbor](https://github.com/harbor-framework/harbor). Harbor provides unified rollout + grading in a single `Trial.run()` call. The server is **task-type agnostic** — all differentiation (environment, grading harness) is encoded in each task's 4 files (instruction.md, Dockerfile, test.sh, task.toml).

## Architecture

```
Docker Network (swe-net)
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│ Miles container (GPU)              │       │ Harbor container (agent-env)       │
│                                    │       │                                    │
│  Ray job → train.py                │       │  server.py (port 11000)            │
│    ├─ MegatronTrainRayActor (×N)  │       │    Wraps Harbor Trial API          │
│    ├─ SGLangEngine (×N, 1/GPU)    │       │    Task-type agnostic              │
│    └─ RolloutManager               │       │                                    │
│                                    │       │                                    │
│  agentic_tool_call.generate        │       │                                    │
│  (miles/rollout/generate_hub/)     │       │                                    │
│    1. Create session on Router     │       │                                    │
│    2. Call agent_function.run ─────────────►│  3. Harbor Trial.run():            │
│    5. Collect records              │       │     a. Start Docker (task image)   │
│    6. Build training samples       │       │     b. Install agent               │
│    7. Merge agent metadata         │       │     c. Run agent loop ────────┐   │
│                                    │       │                               │   │
│  Miles Router (port 30000)         │       │                               │   │
│    /sessions/{id}/v1/chat/         │       │                               │   │
│      completions                   │       │                               │   │
│    - Proxies to SGLang engines     │◄──────────── LLM via OPENAI_API_BASE │   │
│    - Records request + response    │       │     d. Run verifier (test.sh)     │
│      with TITO data                │       │     e. Return TrialResult         │
│                                    │       │                                    │
│  SGLang engines (1 per GPU)        │       │  4. Returns:                       │
│    /v1/chat/completions            │       │     reward, exit_status,           │
│    - Applies chat template         │       │     agent_metrics, eval_report     │
│    - Returns input_token_ids       │       │                                    │
│    - Returns token_id per logprob  │       │  Harbor spawns Docker containers   │
│                                    │       │  from task images (on swe-net)     │
│  Megatron (training, GRPO)         │       │                                    │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

## Files

| File | Description |
| --- | --- |
| `run.sh` | Training launcher — handles Ray lifecycle, model loading, and job submission |
| `public_harbor_server.py` | FastAPI server wrapping Harbor Trial API and staging native trial captures |
| `swe_agent_function.py` | Custom agent function — dispatches to Harbor server, returns env metadata |
| `generate.py` | Reward function, agent metrics aggregation, `RolloutFn` |
| `download_and_process_data.py` | Download from HuggingFace or local JSONL, convert to Miles format |

## Step-by-Step Setup

### Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- Model weights downloaded (e.g. `zai-org/GLM-4.7-Flash`)
- `transformers>=5` (`pip install "transformers>=5"` — GLM-4.7-Flash's `glm4_moe_lite` model type is not in transformers 4.x)
- Harbor task directories prepared under a shared path

### Step 1: Create Docker network

All containers must be on the same network so Harbor-spawned agent containers can reach the Miles Router.

```bash
docker network create swe-net
```

### Step 2: Start the Miles container (GPU, training + inference)

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --network swe-net \
  --name miles \
  radixark/miles:latest \
  /bin/bash
```

### Step 3: Start the agent-env container (Harbor server + Docker-in-Docker)

This container runs the Harbor server and spawns agent Docker containers via the host's Docker socket.

```bash
docker run -itd \
  --name agent_env \
  --shm-size 16g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /data:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network swe-net \
  python:3.12-slim \
  /bin/bash
```

Inside agent_env, install Harbor and set up the server:

```bash
pip install harbor
# Copy server.py into the container, or mount the miles repo
```

### Step 4: Prepare data and Harbor task directories

Harbor task directories are prepared on the agent server side using **harbor adapters**. Each adapter converts a specific dataset into Harbor's 4-file task format. For example, to prepare SWE-bench tasks:

```bash
# On the agent server (CPU machine), inside the harbor repo:
cd $CWD/harbor/adapters/swebench && uv sync

# Generate Harbor task directories for all SWE-bench Verified instances
uv run run_adapter.py --task-dir $HARBOR_TASKS_DIR --all
```

This uses the `swebench` Python package to produce correct Docker image names and Dockerfiles for each instance. Other adapters (e.g. `adapters/swe-gym`) follow the same pattern.

To prepare training data on the Miles side:

```bash
# Inside miles container:

# Download and convert to Miles format
python download_and_process_data.py --input SWE-Gym/SWE-Gym --output swe.jsonl
python download_and_process_data.py --input /data/tb.jsonl --output tb.jsonl \
  --agent-name terminus-2 --prompt-key instruction

# Merge into one mixed JSONL
cat swe.jsonl tb.jsonl > mixed.jsonl
```

Each Harbor task directory contains 4 files:
- `instruction.md` — agent prompt
- `task.toml` — config (timeout, resources)
- `environment/Dockerfile` (or `docker-compose.yaml`) — container image
- `tests/test.sh` — grading logic

### Step 5: Start the Harbor server (in agent_env)

```bash
docker exec -d agent_env bash -c \
  'OPENAI_API_KEY=dummy python server.py --port 11000 --max-concurrent 8 \
   > /tmp/server.log 2>&1'
```

Verify it's running:

```bash
docker exec agent_env curl -s http://localhost:11000/health
# {"status":"ok"}
```

### Step 6: Convert model weights to Megatron format

The reference checkpoint must be in Megatron distributed format for training:

```bash
# Inside miles container — one-time conversion
cd /root/miles
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-4.7-Flash \
    --save /root/GLM-4.7-Flash_torch_dist
```

### Step 7: Launch training

```bash
# Inside miles container:
cd /root/miles/examples/experimental/swe-agent-v2

# Debug mode (small batch, quick pipeline verification)
bash run.sh --mode debug --num-gpus 8 --ep 8 \
  --prompt-data /root/mixed.jsonl \
  --sglang-tool-call-parser glm47 \
  --sglang-reasoning-parser glm45 \
  --no-wait

# Full training
bash run.sh --num-gpus 8 --ep 8 \
  --prompt-data /root/mixed.jsonl \
  --sglang-tool-call-parser glm47 \
  --sglang-reasoning-parser glm45 \
  --no-wait
```

What `run.sh` does:
1. Kills stale sglang/ray processes
2. Starts a Ray head node with `--num-gpus` GPUs
3. Sources the model architecture script (e.g. `glm4.7-flash.sh`)
4. Submits a Ray job running `train.py` with all configured arguments

Monitor the job:

```bash
ray job logs <JOB_ID> --follow
```

### Step 8 (Optional): Start the trace-viewer

[radixark/trace-viewer](https://github.com/radixark/trace-viewer) visualizes agent trajectories. Run it inside agent_env where Harbor saves trial artifacts:

```bash
docker exec -d agent_env bash -c \
  'python3 /root/trace-viewer/server.py --dir /root/trials \
   --host 0.0.0.0 --port 8081 > /tmp/trace_viewer.log 2>&1'
```

If the container isn't directly reachable, set up a port forwarder on the host:

```bash
# On the Docker host — forward host:8081 to agent_env:8081
# (replace 172.18.0.4 with agent_env's IP from: docker inspect agent_env -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
socat TCP-LISTEN:8081,fork,reuseaddr TCP:172.18.0.4:8081
```

Then open `http://<host>:8081` in a browser.

## Key Configuration

### run.sh defaults vs overrides

| Parameter | Default | Debug mode override |
| --- | --- | --- |
| `--num-rollout` | 3000 | 50 |
| `--rollout-batch-size` | 8 | 4 |
| `--n-samples-per-prompt` | 8 | 4 |
| `--rollout-max-response-len` | 8192 | 4096 |
| `--global-batch-size` | 64 | 16 |
| `--max-tokens-per-gpu` | 2048 | 1024 |

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_SERVER_URL` | `http://agent_env:11000` | Harbor server URL |
| `AGENT_MODEL_NAME` | `model` | Model name passed to agents |
| `AGENT_MAX_CONCURRENT` | `8` | Max concurrent Harbor trials |
| `HARBOR_TASKS_DIR` | `/root/harbor_tasks` | Root directory containing task subdirectories |
| `HARBOR_TRIALS_DIR` | `./trials` | Harbor's native host-side trial output directory |
| `MILES_HARBOR_CAPTURE_MODE` | `off` | `off`, fail-open local `shadow`, or fail-closed `required` capture |
| `MILES_HARBOR_CAPTURE_DIR` | sibling `<trials>-captures` | Durable staging directory; set this to a shared PVC in production |
| `MILES_SANDBOX_STATE` | `false` | Ephemeral begin/end sandbox filesystem snapshots (`probe.sandbox-state/1`); requires a non-`off` capture mode |
| `MILES_SANDBOX_STATE_TIMEOUT_BEGIN` | `120` | Seconds allowed for the begin snapshot (upload+scan+download+cleanup) |
| `MILES_SANDBOX_STATE_TIMEOUT_END` | `300` | Seconds allowed for the end snapshot and delta tar |
| `MILES_SANDBOX_STATE_EXCLUDE` | unset | Colon-separated extra container path prefixes to exclude from scans |
| `MILES_SANDBOX_STATE_HASH` | `false` | sha256 every file in the manifests (slow; catches mtime-preserving edits) |
| `HARBOR_DELETE_ENVIRONMENTS` | `true` | Whether Harbor deletes the sandbox during `Trial.run()` cleanup |
| `HARBOR_ADMIN_SECRET` | unset | Optional bearer secret for `/flush`; falls back to the bridge auth token |
| `MILES_ROUTER_EXTERNAL_HOST` | `$(hostname)` | Hostname for agent containers to reach Miles Router |
| `MILES_HOST_IP` | `$(hostname)` | IP/hostname for inter-container communication |

### Model-specific arguments

These are passed as CLI args to `run.sh` (not defaults, since they vary per model):

| Model | Tool call parser | Reasoning parser |
| --- | --- | --- |
| GLM-4.7-Flash | `--sglang-tool-call-parser glm47` | `--sglang-reasoning-parser glm45` |
| Qwen3 | `--sglang-tool-call-parser qwen25` | (none) |

## How It Works

1. **Session creation**: `agentic_tool_call.generate` creates a session on Miles Router
2. **Dispatch to agent**: calls `swe_agent_function.run()` which POSTs to the Harbor server
3. **Harbor Trial**: `public_harbor_server.py` creates a `TrialConfig` and runs `Trial.run()`:
   - Starts a Docker container from the task's Dockerfile
   - Installs and runs the agent (determined by `agent_name` in metadata)
   - Agent calls back to Miles Router at `OPENAI_API_BASE` for model inference
   - Runs the verifier (`test.sh`) and returns `TrialResult` with reward
4. **TITO recording**: Miles Router proxies each `/v1/chat/completions` to SGLang and records exact token IDs and logprobs
5. **Native capture (opt-in)**: in `shadow` or `required` mode, after `Trial.run()` returns, the bridge calls the Probe SDK's producer adapter, which atomically copies and archives Harbor's complete host trial tree, hashes every regular file, and writes its versioned manifest plus a pending export request
6. **Sample building**: Records are converted to training `Sample`s with token IDs, logprobs, loss masks
7. **Training**: GRPO policy update using Megatron, then weights synced back to SGLang engines

### Native Harbor capture boundary

Capture is off by default, with no Probe import, capture-directory creation, or
capture/correlation fields added to the `/run` response. Install the optional
capture dependencies and select a mode:

```bash
pip install -r requirements-public-harbor-capture.txt
export MILES_HARBOR_CAPTURE_MODE=shadow
export MILES_HARBOR_CAPTURE_DIR=/durable/trial-captures
```

`shadow` stages locally but never changes a successful Harbor rollout into a
failure when staging fails; the response reports `capture.status=failed`.
`required` performs the same local staging but changes `exit_status` to
`Error: CaptureRequiredError` when the capture is failed or incomplete. Neither
mode uploads on the request path. The separate watcher performs network
publication.

Set `MILES_HARBOR_CAPTURE_DIR` to a shared PVC. Each completed capture contains
`trial/` (directly consumable by `probe.connectors.harbor.capture_trial`),
`trial.tar.gz`, `capture-manifest.json`, and `export-request.json`. The descriptor
includes the Probe run, Miles rollout/step, sample/group, model session, Harbor
trial, and sandbox correlation IDs when available. Export requests set
`expand: false`: native trajectories are retained as ordinary raw files today,
without requiring ATIF or creating turn/tool-call spans.

Run the Probe exporter against the same shared capture directory. It validates
the producer manifest and every file hash before uploading, then records its
remote publication state back into the descriptor:

```bash
export PROBE_TOKEN='<write token>'
probe trial watch "$MILES_HARBOR_CAPTURE_DIR" --interval 5
```

The bridge uses the SDK only for durable local staging and does not block a
rollout on network uploads. A stopped watcher or API outage leaves the descriptor and staged bytes
on the PVC; `probe trial drain "$MILES_HARBOR_CAPTURE_DIR"` resumes them. The
run ID is supplied automatically by Miles when Probe tracking is enabled, or
may be supplied as `PROBE_RUN_ID` for a pre-created run.

Completeness is deliberately scoped to the **host Harbor trial tree**. Harbor
0.18 stops/deletes its environment inside `Trial.run()` before the bridge gets
control back. Harbor downloads its configured agent logs and artifacts before
that cleanup, but arbitrary sandbox state outside those configured Harbor
outputs is unknowable after deletion. The bridge does not claim that staging
precedes sandbox teardown.

### Sandbox begin/end state snapshots (opt-in)

`MILES_SANDBOX_STATE=1` narrows the boundary above: the bridge registers
Harbor `Trial` lifecycle hooks (`AGENT_START` / `AGENT_END`) and, at each
instant, uploads a static, dependency-free snapshot binary into a random
`/tmp` workdir inside the still-running container, execs it as root,
downloads the outputs to the host, verifies them against the sha256 trailer
the binary prints to stdout, and deletes the workdir — so the container is
probe-free for the entire agent phase, and nothing of the capture survives
in the sandbox. The resulting `probe.sandbox-state/1` bundle (begin/end
JSONL manifests + a delta tar of files the agent added or modified +
host-authored `meta.json`) is written into `<trial_dir>/artifacts/` before
staging, so it rides the existing capture and exporter unchanged, and the
`/run` response reports it under `capture.sandbox_state`.

Snapshots are fail-open at every layer: a python-less/shell-less image, an
unsupported arch, a timeout, or a dead environment degrade to a partial or
absent bundle with the reason recorded — never a failed trial. The agent
phase itself is never concurrent with a scan; the cost is bounded seconds of
trial wall-clock at the two hook instants. `AGENT_END` fires before
verification in both verifier modes, so the end state is exactly "the state
the agent left," excluding `test.sh` side effects. The begin/end state of
files Harbor never materialized in-container (in-memory state, deleted-
before-end temporaries) remains out of scope.

### Multi-turn merge (TITO mode)

When TITO is working, `merge_samples()` concatenates multi-turn conversations:

```
Turn 1: [prompt_tokens] [response_tokens_1]                                    -> Sample 0
Turn 2: [prompt_tokens] [response_tokens_1] [observation_tokens] [resp_tokens_2] -> Sample 1

merge_samples() ->
  tokens:    [prompt] [resp1] [obs] [resp2]
  loss_mask: -------- [1]*r1  [0]*o [1]*r2
  logprobs:  -------- [real]  [0.0] [real]
```

Without TITO, use `--generate-multi-samples` to skip merge and train on per-turn samples instead (current default in `run.sh`).

## Troubleshooting

### Harbor containers can't reach Miles Router

Agent containers need to resolve the Miles container's hostname. Ensure:
- All containers are on the same Docker network (`swe-net`)
- `MILES_ROUTER_EXTERNAL_HOST` is set to the Miles container name (e.g. `miles-maocheng`)
- Task directories were created with `--docker-network swe-net`

### `TaskNotFound` error

The task directory for the given `instance_id` doesn't exist under `HARBOR_TASKS_DIR`. Run the appropriate harbor adapter first (e.g. `adapters/swebench/run_adapter.py` for SWE-bench tasks).

### SGLang engines OOM (`Not enough memory`)

- Ensure all GPUs are clean before starting (`nvidia-smi` shows 0 MiB)
- Kill stale processes: `pkill -9 sglang; ray stop --force; pkill -9 ray`
- `run.sh` does this automatically, but leftover processes from crashed jobs may persist

### `b.tokens must start with a.tokens` assertion error

Multi-turn merge fails due to BPE re-tokenization inconsistency. Use `--generate-multi-samples` (already default in `run.sh`) to skip merge and train on per-turn samples.

### Trace-viewer shows no trajectories

- Native trial artifacts are saved on the Harbor host at `HARBOR_TRIALS_DIR`; durable copies are published under `MILES_HARBOR_CAPTURE_DIR`
- Point the trace-viewer at the correct directory inside agent_env
- Restart the trace-viewer after clearing old data (it caches in memory)
