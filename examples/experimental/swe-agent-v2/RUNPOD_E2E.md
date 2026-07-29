# Runpod Instant Cluster E2E runbook: Miles + public Harbor

This is the operator and coding-agent handoff for running the
`examples/experimental/swe-agent-v2` experiment on a two-node Runpod Instant
Cluster. It assumes that Docker is unavailable inside the GPU Pods and uses a
public Harbor cloud environment, with Daytona as the default.

It follows Runpod's [Instant Cluster overview][runpod-clusters] and
[cluster-network configuration][runpod-cluster-config].

Read the gates in order. Do not start normal-mode training until a real
Mini-SWE-Agent rollout has completed through Miles, a Daytona task sandbox,
the Miles session server, and the Harbor verifier.

If a gate fails, stop and record:

- the command and exit code;
- the last relevant log lines;
- `NODE_RANK`, `NODE_ADDR`, and `PRIMARY_ADDR` from the affected Pod;
- the Ray node/resource summary;
- whether the callback is direct TCP or a relay URL.

Do not reset the checkout, delete `/workspace`, terminate the cluster, or kill
unrelated processes while diagnosing a gate.

> **Non-negotiable before clicking Deploy:** select the same Runpod network
> volume for the Instant Cluster and mount it at `/workspace`. If the volume is
> not selected during deployment, cloning into one Pod will not populate the
> other Pod, and normal distributed checkpointing has no shared destination.
> Do not rent the cluster first and plan to attach storage afterward.

## 1. Understand the topology

The recommended topology is:

```text
Runpod Instant Cluster
  primary Pod (NODE_RANK=0)
    - Ray head
    - Miles launcher and RolloutManager
    - Miles session server
    - public_harbor_server.py
    - outbound Daytona API calls

  worker Pod (NODE_RANK=1)
    - Ray worker
    - Megatron/SGLang GPU processes

Daytona
  - one short-lived Harbor task sandbox per trial
  - agent and verifier run here
  - agent calls the externally advertised Miles session URL
```

Harbor does **not** need Docker on Runpod when
`HARBOR_ENVIRONMENT_TYPE=daytona`. The Runpod primary controls Daytona through
outbound API calls. Miles does not need to SSH into Daytona sandboxes.

The difficult connection is the reverse direction: a Daytona sandbox must be
able to call the Miles session server on the Runpod primary.

Use one of these callback paths:

1. **Direct symmetrical Runpod TCP** for the first end-to-end smoke test. It is
   simple, but it is plain HTTP, publicly reachable, and its mapping changes
   after a Pod reset.
2. **Stable relay VM with TLS and authentication** for a real training run.
   The Runpod primary opens an outbound reverse tunnel to the relay, so Daytona
   receives one stable HTTPS URL.

Do not use Runpod's `*.proxy.runpod.net` HTTP proxy for the session server or
the bridge's long-running `POST /run`: the proxy has a 100-second request
limit. See [Runpod port behavior][runpod-ports].

## 2. Prerequisites before renting GPUs

Have these ready:

- a Runpod account permitted to create a two-node Instant Cluster;
- a Runpod template built from an image with CUDA, Python, `git`, `uv`, Ray,
  `openssh-client`, `rsync`, `curl`, `tmux`, and build tools;
- a network volume already created in the **same Runpod data center** as the
  cluster and available for selection in the Instant Cluster form;
- GitHub access to the Miles repository and branch containing the public
  Harbor bridge;
- a Daytona account and `DAYTONA_API_KEY`;
- Hugging Face access to the model and `HF_TOKEN`, if required;
- a W&B key if the run will report to W&B;
- enough persistent space for the HF checkpoint, converted Megatron
  checkpoint, traces, and training checkpoints;
- for the production callback path, a small public CPU VM, a DNS name, and an
  SSH key dedicated to the reverse tunnel.

Before renting GPUs, run the repository-only preflight from this checkout:

```bash
python examples/experimental/swe-agent-v2/runpod_preflight.py --phase repo
```

Require zero failures. A dirty-worktree warning is expected only while
developing; deploy a recorded commit so both Pods run identical code.

### Pre-populate the volume without renting the GPU cluster

When the target data center has no S3-compatible endpoint, use a temporary,
inexpensive CPU Pod in the **same data center**:

1. Attach the future cluster's network volume at `/workspace` while creating
   the CPU Pod.
2. Clone the recorded Miles branch/commit into `/workspace/miles`.
3. Download the HF checkpoint into `/workspace/models`.
4. Use a temporary Harbor 0.18/Daytona-capable venv outside `/workspace` to
   download/export Terminal-Bench tasks into `/workspace/harbor`, then create
   the Miles JSONL using section 9.
5. Record checksums or at least file counts and sizes.
6. Terminate the CPU Pod without deleting the network volume.
7. Select that same volume when creating the Instant Cluster.

Do not build the Megatron `torch_dist` checkpoint twice. If CPU conversion is
too slow or needs the Miles GPU image, leave only that conversion for the
primary GPU Pod. Also avoid creating the Harbor venv on an unrelated CPU
image; create it on the primary so its interpreter and native dependencies
match the Runpod template image.

Harbor documents Daytona as a supported cloud-sandbox path in its
[getting-started guide][harbor-getting-started].

The two GPU Pods do not need separate working conventions. Treat
`NODE_RANK=0` as the control plane: launch, monitor, and stop jobs there. Use
the worker terminal for node-local setup, health checks, and logs.

### Secrets

Store secrets as Runpod secrets or inject them interactively. Do not place
literal secrets in the template, repository, shell history, screenshots, or
run notes.

Typical secret names are:

```text
DAYTONA_API_KEY
HF_TOKEN
WANDB_API_KEY
GITHUB_TOKEN                 # only if the checkout is private
AGENT_SERVER_AUTH_TOKEN
MILES_SESSION_API_KEY        # required for direct and relay callbacks
```

The Daytona key is only required on the host running
`public_harbor_server.py`, normally the primary Pod. A template applies to all
cluster Pods, so adding it to the template also exposes it to the worker. For
least privilege, inject it only into the primary terminal or retrieve it from
a secret manager at bridge startup.

## 3. Create the Runpod template

Create or edit a private template in the Runpod console before deploying the
Instant Cluster. See Runpod's [template documentation][runpod-templates] and
[template environment-variable reference][runpod-env].

### Image and storage

- Use `radixark/miles:latest` as the Runpod template image. It already contains
  the patched SGLang, Megatron-LM, Ray, CUDA kernels, and Miles dependency tree
  that are difficult to reproduce correctly from a generic PyTorch image.
- Allocate at least 100 GB of container disk for the image, Python packages,
  caches, and temporary compilation. The node preflight requires at least
  20 GiB still free before launch.
- Select the network volume in the Instant Cluster creation form and mount it
  at `/workspace` on every node.
- Size the network volume for the HF model, converted checkpoint, datasets,
  traces, and multiple training checkpoints. The checked-in preflight defaults
  to requiring 300 GiB free; 500 GB or more is a safer starting allocation for
  this experiment.
- Keep repositories, datasets, models, checkpoints, traces, and durable logs
  under `/workspace`.
- Do not rely on files elsewhere in the container surviving a Pod edit or
  reset.

Runpod's [network-volume documentation][runpod-volume] describes data-center
placement, persistence, and transfer options.

If the India data center does not expose the S3-compatible volume API, that is
not fatal. Attach the volume first, then populate `/workspace` from a running
Pod using `git`, Hugging Face download tools, `scp`, or resumable `rsync`.
If the Instant Cluster form does not offer the volume, do not deploy: the
volume and GPUs are not in a compatible data center or the selected capacity
cannot use that storage.

The volume is shared, so perform repository updates, package installation,
model downloads, and checkpoint conversion from the primary only. Do not run
two `git pull`, `uv pip install`, or conversion processes against the same
paths concurrently. Distributed checkpoint writers may use the shared
destination only through Miles/Megatron's coordinated rank-aware save path.

The image contains `/root/miles`, but the experiment must use the branch cloned
at `/workspace/miles`. After cloning, install that checkout editable with
`--no-deps`; do not reinstall the full Miles requirements over the image.

### Ports

Configure:

```text
22/tcp       SSH, if desired
70000/tcp    request one symmetrical direct-TCP callback port
```

`70000` is a Runpod allocation signal, not the actual listening port. After
startup Runpod places the assigned, matching internal/external port in
`RUNPOD_TCP_PORT_70000`.

Do not expose Ray port `6379`, the Ray dashboard, or the Harbor bridge to the
public internet. They only need the Instant Cluster's internal network. The
production relay also needs no inbound Runpod port because the primary starts
the tunnel outbound.

### Template environment variables

Safe non-secret defaults for the template are:

```text
NCCL_SOCKET_IFNAME=ens1
GLOO_SOCKET_IFNAME=ens1
NCCL_DEBUG=WARN
HARBOR_ENVIRONMENT_TYPE=daytona
MILES_HARBOR_ENVIRONMENT_KWARGS_JSON={}
HARBOR_DELETE_ENVIRONMENTS=true
AGENT_MAX_CONCURRENT=2
AGENT_SERVER_TIMEOUT_SEC=14400
MILES_HARBOR_REQUEST_TIMEOUT_SEC=14400
MILES_SCRIPT_EXTERNAL_RAY=1
MILES_SESSION_SERVER_BIND_IP=0.0.0.0
MILES_ROOT=/workspace/miles
HARBOR_DATA_ROOT=/workspace/harbor
```

The same copy-ready values are checked in as
`runpod-template.env.example`. The file deliberately cannot select storage;
the network volume must still be chosen in the Instant Cluster form.

Do **not** set these in the template; Runpod generates them separately for
each cluster deployment or Pod:

```text
NODE_RANK
NODE_ADDR
PRIMARY_ADDR / MASTER_ADDR
PRIMARY_PORT / MASTER_PORT
NUM_NODES
NUM_TRAINERS
WORLD_SIZE
RUNPOD_POD_ID
RUNPOD_PUBLIC_IP
RUNPOD_TCP_PORT_70000
RUNPOD_VOLUME_ID
```

Do not bake `MILES_ROUTER_EXTERNAL_HOST`, `MILES_ROUTER_EXTERNAL_BASE_URL`,
`MILES_SESSION_SERVER_PORT`, or a relay URL into the template. Derive callback
values after every deployment.

### Startup command

Prefer an idempotent bootstrap that installs OS-level prerequisites and then
keeps the Pod available. Do not automatically start Ray or training from the
template while the deployment is still being validated. A human or coding
agent must first identify ranks, verify storage, and confirm both Pods are
healthy.

Changing a template does not repair already-running Pods. Editing a Pod can
reset it and erase non-volume data. If a port or mount is missing, stop the
job, preserve durable outputs, then deliberately redeploy.

## 4. Deploy and identify the primary Pod

Deploy the two-node Instant Cluster using the same template and selected
volume. Wait for both Pod cards to report telemetry before assuming they are
ready.

Open one terminal for each Pod. Run this in both:

```bash
set -euo pipefail

env | sort | grep -E \
  '^(NODE_RANK|NODE_ADDR|PRIMARY_ADDR|MASTER_ADDR|PRIMARY_PORT|MASTER_PORT|NUM_NODES|NUM_TRAINERS|WORLD_SIZE|RUNPOD_POD_ID|RUNPOD_PUBLIC_IP|RUNPOD_TCP_PORT_70000|RUNPOD_VOLUME_ID)='

if [[ "${NODE_RANK:?Runpod did not set NODE_RANK}" == "0" ]]; then
  echo "THIS IS THE PRIMARY POD / FUTURE RAY HEAD"
else
  echo "THIS IS A WORKER POD: rank=${NODE_RANK}"
fi

ip -brief address show ens1
ip -brief address show eth0
```

Runpod's `NODE_RANK=0` designation is authoritative. Do not infer the primary
from the first Pod card, Pod name, browser-tab order, `hostname -I`, or the
public IP.

For fewer human mistakes, label the terminal prompt in each shell:

```bash
export PS1='[runpod rank='"$NODE_RANK"' node='"$NODE_ADDR"'] \w # '
```

Record these non-secret facts:

```text
Miles git revision:
Runpod data center:
Primary Pod ID / NODE_ADDR / public IP:
Worker Pod ID / NODE_ADDR:
NUM_NODES / NUM_TRAINERS / WORLD_SIZE:
RUNPOD_VOLUME_ID on each Pod:
Callback mode: direct TCP or relay
Direct callback port or relay URL:
HF checkpoint path:
Megatron torch_dist path:
Harbor task export path:
W&B run name:
```

## 5. Prove storage is actually shared

Seeing `/workspace` on both Pods does not prove that they are the same volume.
Run on both Pods:

```bash
echo "pod=$RUNPOD_POD_ID rank=$NODE_RANK volume=${RUNPOD_VOLUME_ID:-unset}"
findmnt -T /workspace -o SOURCE,FSTYPE,SIZE,AVAIL,TARGET,OPTIONS
df -h /workspace
```

On the primary:

```bash
export VOLUME_PROBE="runpod-volume-probe-$(date -u +%Y%m%dT%H%M%SZ)"
printf 'created_by=%s\nnode=%s\n' "$RUNPOD_POD_ID" "$NODE_ADDR" \
  > "/workspace/$VOLUME_PROBE"
echo "$VOLUME_PROBE"
```

Copy only the printed probe filename to the worker terminal, then run:

```bash
export VOLUME_PROBE='<filename printed by primary>'
cat "/workspace/$VOLUME_PROBE"
```

If the worker cannot read it, the storage is not shared. Check that the same
volume was attached to both Pods and that work happened under its actual mount
path. Do not assume that cloning on one Pod will appear on the other.

For a one-rollout diagnostic, code and model files can be copied independently
to both nodes at identical absolute paths. For normal distributed training,
stop and establish genuinely shared checkpoint/output storage before
continuing; local directories with the same name are not a substitute for a
shared distributed-checkpoint destination.

When no S3 endpoint is available, populate the mounted volume from the primary:

```bash
cd /workspace
export MILES_GIT_REF="${MILES_GIT_REF:-public-harbor-runpod}"
git clone --branch "$MILES_GIT_REF" --single-branch \
  https://github.com/prbe-ai/miles.git miles
```

If the handoff specifies a commit rather than a branch, clone first and check
out that exact commit. Use `git pull --ff-only` for later branch updates. Use
`rsync --partial --inplace` for large interrupted transfers rather than
restarting them.

## 6. Verify the checkout and install isolated Harbor dependencies

Run the Git inspection on the primary. Run the runtime checks and editable
install on both Pods, one Pod at a time. Although the source checkout is
shared, each Pod has its own Python site-packages:

```bash
set -euo pipefail
export MILES_ROOT="${MILES_ROOT:-/workspace/miles}"
cd "$MILES_ROOT"

git status --short
git branch --show-current
git rev-parse HEAD
test -f examples/experimental/swe-agent-v2/public_harbor_server.py
test -f examples/experimental/swe-agent-v2/RUNPOD_E2E.md

nvidia-smi
python --version
ray --version
uv --version

pip install -e "$MILES_ROOT" --no-deps
```

The checkout must contain the public bridge. Do not silently switch to an
older `main` checkout containing only `harbor-private` instructions.

Install Harbor in a separate Python 3.12 venv using the Runpod requirements,
which include the Daytona SDK.

```bash
export HARBOR_VENV=/workspace/venvs/harbor-0.18-daytona
uv venv "$HARBOR_VENV" --python 3.12
uv pip install --python "$HARBOR_VENV/bin/python" \
  -r examples/experimental/swe-agent-v2/requirements-runpod.txt \
  pytest pytest-asyncio ruff

"$HARBOR_VENV/bin/harbor" --version
"$HARBOR_VENV/bin/python" -m pytest -q \
  tests/fast/experimental \
  --confcutdir=tests/fast/experimental \
  --override-ini='addopts='
```

Require Harbor `0.18.0` and all bridge contract tests to pass.

Run the node preflight on **both** Pods after setting any available paths:

```bash
python examples/experimental/swe-agent-v2/runpod_preflight.py \
  --phase node \
  --callback-mode direct
```

This proves that each Pod has a network volume attached. It cannot prove that
both Pods see the same volume contents; the sentinel-file test in section 5
remains mandatory.

## 7. Repository compatibility gate

The cloud callback must always terminate on the primary. This branch now
provides all of the following:

1. Both SWE-agent launchers pass `--pin-rollout-manager-to-head` to Miles.
2. The session server can listen on `0.0.0.0` without replacing the internal
   address Miles uses for local health, session creation, and record
   collection.
3. The agent function can advertise a complete external base URL, including
   `https`, hostname, and port, rather than rewriting only the hostname.
4. For the production relay, a bearer token reaches both the Daytona agent's
   OpenAI client and the bridge's session monitor.

Relevant files are:

- `run.py`
- `run-glm47-flash-agentic-async.py`
- `swe_agent_function.py`
- `public_harbor_server.py`
- `miles/ray/placement_group.py`
- `miles/ray/rollout/router_manager.py`
- `miles/rollout/session/server.py`

Repository state:

| Capability | Current state | Verification |
| --- | --- | --- |
| Pin RolloutManager to Ray head | Both launchers pass `--pin-rollout-manager-to-head` | Repository preflight inspects both launchers |
| Listen publicly while retaining a private session address | `session_server_bind_ip` is separate from `session_server_ip` | Router-manager test plus external callback probe |
| Advertise relay HTTPS URL | `MILES_ROUTER_EXTERNAL_BASE_URL` replaces the external origin while preserving the session path | Agent-function contract tests |
| Authenticate the bridge | `AGENT_SERVER_AUTH_TOKEN` protects Miles-to-bridge calls | Public-Harbor HTTP contract test |
| Authenticate the public session callback | `MILES_SESSION_API_KEY` reaches Miles internal clients, Daytona, and the Harbor monitor; Miles enforces it | Session-auth and agent-function contract tests |

Implementation acceptance criteria:

- URL-rewrite tests cover direct `http://IP:port` and relay
  `https://hostname` origins while preserving the session path.
- Bridge tests prove the configured session bearer is passed to the agent and
  used by `/health` and `/sessions/<id>` monitor requests.
- A session-server test proves it can listen on `0.0.0.0` while Miles' internal
  tracer continues to use a routable private address.
- Launcher tests or command inspection prove both launchers include
  `--pin-rollout-manager-to-head`.
- The existing fast tests still pass before any GPU smoke test.

Run the repository preflight and dependency-light contract suite before
deployment. The Ray-specific router-manager test additionally runs in the
Runpod/Miles environment, which contains Ray.

Do not claim the production path is complete merely because an oracle trial
passes: the oracle does not call the Miles model endpoint.

## 8. Start Ray using Runpod's cluster addresses

Use Runpod's `NODE_ADDR` for each Ray node and `PRIMARY_ADDR` for the head.
`PRIMARY_ADDR` corresponds to the high-speed internal interface. Do not use
the `172.*` management address on `eth0` for Ray, NCCL, or Gloo.

Choose a Ray port independent of Runpod's supplied `MASTER_PORT`:

```bash
export RAY_PORT=6379
export NCCL_SOCKET_IFNAME=ens1
export GLOO_SOCKET_IFNAME=ens1
```

On the primary only:

```bash
test "$NODE_RANK" = "0"
ray stop --force || true
ray start --head \
  --node-ip-address="$NODE_ADDR" \
  --port="$RAY_PORT" \
  --num-gpus="$NUM_TRAINERS" \
  --dashboard-host=0.0.0.0 \
  --disable-usage-stats
```

On the worker only:

```bash
test "$NODE_RANK" != "0"
ray stop --force || true
ray start \
  --address="$PRIMARY_ADDR:6379" \
  --node-ip-address="$NODE_ADDR" \
  --num-gpus="$NUM_TRAINERS" \
  --disable-usage-stats
```

Back on the primary:

```bash
export RAY_ADDRESS=http://127.0.0.1:8265
ray status
ray list nodes --address "$RAY_ADDRESS"

python - <<'PY'
import os
import ray

ray.init(address="auto")
resources = ray.cluster_resources()
print(resources)
expected = int(os.environ["WORLD_SIZE"])
actual = int(resources.get("GPU", 0))
if actual != expected:
    raise SystemExit(f"Ray sees {actual} GPUs; Runpod WORLD_SIZE is {expected}")
PY
```

Do not launch Miles until Ray reports both alive nodes and the full GPU count.

## 9. Download Terminal-Bench 2 and create Miles JSONL

Run on the primary, which also runs the Harbor bridge:

```bash
export MILES_ROOT="${MILES_ROOT:-/workspace/miles}"
export HARBOR_VENV="${HARBOR_VENV:-/workspace/venvs/harbor-0.18-daytona}"
export HARBOR_DATA_ROOT="${HARBOR_DATA_ROOT:-/workspace/harbor}"
mkdir -p "$HARBOR_DATA_ROOT/tasks" "$HARBOR_DATA_ROOT/trials" /workspace/data

"$HARBOR_VENV/bin/harbor" download \
  terminal-bench/terminal-bench-2 \
  --output-dir "$HARBOR_DATA_ROOT/tasks" \
  --export

export HARBOR_TASKS_DIR="$HARBOR_DATA_ROOT/tasks/terminal-bench-2"
test -d "$HARBOR_TASKS_DIR"
find "$HARBOR_TASKS_DIR" -mindepth 2 -maxdepth 2 \
  \( -name task.toml -o -name instruction.md \) | head
```

If Harbor creates a versioned export directory, set `HARBOR_TASKS_DIR` to the
directory that directly contains one subdirectory per task.

Create full and one-task datasets:

```bash
export TB2_JSONL=/workspace/data/tb2_all.jsonl
export TB2_SMOKE_JSONL=/workspace/data/tb2_smoke.jsonl
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

Every JSONL `instance_id` must have a corresponding task directory in the
same export. The bridge needs task directories; the GPU worker only needs the
JSONL and model/checkpoint paths used by Miles.

## 10. Establish the callback path

### Option A: direct symmetrical TCP for the first smoke

The template must contain `70000/tcp`. On the primary:

```bash
test "$NODE_RANK" = "0"
: "${RUNPOD_PUBLIC_IP:?the primary has no public IP}"
: "${RUNPOD_TCP_PORT_70000:?redeploy with 70000/tcp in the template}"

export CALLBACK_HOST="$RUNPOD_PUBLIC_IP"
export MILES_SESSION_SERVER_PORT="$RUNPOD_TCP_PORT_70000"
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0
export MILES_ROUTER_EXTERNAL_HOST="$CALLBACK_HOST"
export MILES_SESSION_API_KEY="${MILES_SESSION_API_KEY:-$(openssl rand -hex 32)}"

echo "Direct callback: http://${CALLBACK_HOST}:${MILES_SESSION_SERVER_PORT}"
```

Before launching Miles, prove the public route with a disposable server:

```bash
python -m http.server "$MILES_SESSION_SERVER_PORT" --bind 0.0.0.0 \
  >/tmp/runpod-callback-probe.log 2>&1 &
PROBE_PID=$!
echo "$PROBE_PID"
```

From the human operator's MacBook or any machine outside Runpod:

```bash
curl -v --max-time 10 \
  "http://<RUNPOD_PUBLIC_IP>:<RUNPOD_TCP_PORT_70000>/"
```

Then stop the disposable server on the primary:

```bash
kill "$PROBE_PID"
wait "$PROBE_PID" 2>/dev/null || true
```

The actual Miles session process must later listen on all interfaces at that
same assigned port. A successful MacBook curl proves Runpod's public mapping,
not the Miles session behavior; the real rollout is still required.

Runpod may change the external IP on some infrastructure, and it changes TCP
port mappings whenever a Pod resets. Re-read both Runpod variables and rerun
the probe after every reset. Never reuse a callback copied from an earlier
cluster.

This option is for integration testing. Miles requires the configured bearer,
but traffic remains plain HTTP and can be observed in transit. Do not use it
for sensitive or long-running training.

### Option B: stable authenticated relay for training

Use a small CPU VM with a stable public DNS name, TLS termination, SSH, and no
100-second proxy timeout. The relay should accept only HTTPS and reverse proxy
to a loopback port populated by the tunnel.

The primary initiates and maintains a reverse tunnel similar to:

```bash
export INTERNAL_SESSION_PORT=30000
export MILES_SESSION_SERVER_PORT="$INTERNAL_SESSION_PORT"
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0
export RELAY_HOST=miles-relay.example.com
export RELAY_USER=miles-relay

autossh -M 0 -NT \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -R "127.0.0.1:${INTERNAL_SESSION_PORT}:${PRIMARY_ADDR}:${INTERNAL_SESSION_PORT}" \
  "${RELAY_USER}@${RELAY_HOST}"
```

Run `autossh` under a supervisor on the primary and restrict the relay SSH key
to port forwarding. Configure Caddy, Nginx, or an equivalent relay service to:

- serve a stable URL such as `https://miles-model.example.com`;
- proxy to `127.0.0.1:30000` without a short request timeout;
- require `Authorization: Bearer <MILES_SESSION_API_KEY>`;
- limit request size/rate and log failures without logging the bearer token;
- expose no relay admin interface publicly.

Set the bridge callback allowlist to the relay hostname and advertise:

```bash
export MILES_ROUTER_EXTERNAL_BASE_URL=https://miles-model.example.com
export MILES_SESSION_API_KEY='<injected secret>'
```

Keep the Miles session server on the internal fixed port. The relay URL, not
the Runpod public IP, is sent to Daytona.

Do not put Cloudflare's ordinary proxied HTTP path or Runpod's HTTP proxy in
front of long synchronous model requests unless their timeout behavior has
been explicitly removed or the protocol has been redesigned around async job
polling.

## 11. Start and oracle-smoke the Daytona Harbor bridge

Run the bridge on the primary. It does not require Docker.

Inject `DAYTONA_API_KEY`, then generate or inject a separate bearer token for
Miles-to-bridge requests:

```bash
: "${DAYTONA_API_KEY:?inject DAYTONA_API_KEY on the primary}"
umask 077
export MILES_HARBOR_AUTH_TOKEN="${MILES_HARBOR_AUTH_TOKEN:-$(openssl rand -hex 32)}"
export AGENT_SERVER_AUTH_TOKEN="$MILES_HARBOR_AUTH_TOKEN"
```

Configure and start the bridge:

```bash
export MILES_ROOT="${MILES_ROOT:-/workspace/miles}"
export HARBOR_VENV="${HARBOR_VENV:-/workspace/venvs/harbor-0.18-daytona}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/workspace/harbor/tasks/terminal-bench-2}"
export HARBOR_TRIALS_DIR=/workspace/harbor/trials
export HARBOR_ENVIRONMENT_TYPE=daytona
export MILES_HARBOR_ENVIRONMENT_KWARGS_JSON='{}'
export HARBOR_DELETE_ENVIRONMENTS=true
export AGENT_MAX_CONCURRENT=2
export MILES_HARBOR_REQUEST_TIMEOUT_SEC=14400
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="${CALLBACK_HOST:-miles-model.example.com},localhost,127.0.0.1"
export AGENT_SERVER_URL="http://${PRIMARY_ADDR}:18080"

mkdir -p /workspace/logs "$HARBOR_TRIALS_DIR"
nohup "$HARBOR_VENV/bin/python" \
  "$MILES_ROOT/examples/experimental/swe-agent-v2/public_harbor_server.py" \
  --host 0.0.0.0 --port 18080 \
  >/workspace/logs/public-harbor-server.log 2>&1 &
echo $! >/workspace/logs/public-harbor-server.pid

curl -fsS "$AGENT_SERVER_URL/health"
```

From the worker, independently run:

```bash
curl -fsS "http://${PRIMARY_ADDR}:18080/health"
```

Run an authenticated oracle trial from the primary. The oracle proves task
upload, Daytona sandbox creation, verifier execution, result normalization,
and cleanup. It does not prove the Miles model callback.

```bash
export SMOKE_INSTANCE="$(python -c 'import json,os; print(json.loads(open(os.environ["TB2_SMOKE_JSONL"]).readline())["metadata"]["instance_id"])')"
export CALLBACK_BASE_URL="${MILES_ROUTER_EXTERNAL_BASE_URL:-http://${CALLBACK_HOST}:${MILES_SESSION_SERVER_PORT}}"

test "$(curl -sS -o /tmp/no-auth.json -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d "{\"base_url\":\"${CALLBACK_BASE_URL}/sessions/oracle-smoke/v1\",\"model\":\"openai/model\",\"instance_id\":\"${SMOKE_INSTANCE}\",\"agent_name\":\"oracle\"}" \
  "$AGENT_SERVER_URL/run")" = 401

curl -fsS \
  -H "Authorization: Bearer ${AGENT_SERVER_AUTH_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"base_url\":\"${CALLBACK_BASE_URL}/sessions/oracle-smoke/v1\",\"model\":\"openai/model\",\"instance_id\":\"${SMOKE_INSTANCE}\",\"agent_name\":\"oracle\"}" \
  "$AGENT_SERVER_URL/run"
```

Require HTTP 200, `Submitted`, a verifier report, and no live Daytona sandbox
left for the trial when `HARBOR_DELETE_ENVIRONMENTS=true`. Inspect:

```bash
tail -n 200 /workspace/logs/public-harbor-server.log
find "$HARBOR_TRIALS_DIR" -maxdepth 3 -type f | tail -n 50
```

Use the Daytona dashboard for provider-side creation or cleanup failures.

## 12. Verify model checkpoints

Set real persistent paths on the primary:

```bash
export HF_CHECKPOINT=/workspace/models/zai-org/GLM-4.7-Flash
export REF_LOAD=/workspace/models/zai-org/GLM-4.7-Flash_torch_dist
export MEGATRON_PATH=/root/Megatron-LM

test -d "$HF_CHECKPOINT"
test -d "$MEGATRON_PATH"
```

Verify the worker can read the same absolute paths. If `REF_LOAD` is missing,
convert it once from the primary:

```bash
cd "$MILES_ROOT"
source scripts/models/glm4.7-flash.sh
PYTHONPATH="$MEGATRON_PATH" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --save "$REF_LOAD"
```

Do not run two conversions against the same destination. Confirm the completed
output is visible from the worker before starting Ray workloads that use it.

## 13. Run one real Mini-SWE-Agent rollout

Use the primary terminal. Keep concurrency at one and use direct TCP for the
first proof unless the relay is already implemented and externally tested.

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=http://127.0.0.1:8265
export AGENT_SERVER_TIMEOUT_SEC=14400
export AGENT_MODEL_NAME=model
export MILES_HOST_IP="$PRIMARY_ADDR"
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0

: "${AGENT_SERVER_URL:?start the bridge}"
: "${AGENT_SERVER_AUTH_TOKEN:?set bridge auth}"
: "${TB2_SMOKE_JSONL:?create the smoke JSONL}"
: "${MILES_SESSION_SERVER_PORT:?set callback/internal session port}"

if [[ -n "${MILES_ROUTER_EXTERNAL_BASE_URL:-}" ]]; then
  CALLBACK_ARGS=(--router-external-base-url "$MILES_ROUTER_EXTERNAL_BASE_URL")
else
  CALLBACK_ARGS=(--router-external-host "$CALLBACK_HOST")
fi
```

After the repository preflight and contract tests pass, launch exactly one
rollout without a training update:

```bash
cd "$MILES_ROOT"
python examples/experimental/swe-agent-v2/run.py \
  --mode debug_rollout_only \
  --num-nodes 1 --num-gpus-per-node "$NUM_TRAINERS" \
  --skip-prepare \
  --megatron-path "$MEGATRON_PATH" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --prompt-data "$TB2_SMOKE_JSONL" \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --global-batch-size 2 \
  --max-seq-len 16384 \
  --rollout-max-response-len 8192 \
  --agent-server-url "$AGENT_SERVER_URL" \
  --agent-server-auth-token "$AGENT_SERVER_AUTH_TOKEN" \
  --agent-server-timeout-sec 14400 \
  --session-server-port "$MILES_SESSION_SERVER_PORT" \
  --session-server-bind-ip "$MILES_SESSION_SERVER_BIND_IP" \
  "${CALLBACK_ARGS[@]}" \
  --miles-host-ip "$PRIMARY_ADDR"
```

The launcher exposes `--num-rollout` and `--rollout-max-response-len`; keep
both explicit for this bounded gate. With TP=4 on eight H100s, Megatron derives
DP=2, so global batch size 2 is the smallest valid value.

The callback argument array selects the full relay origin when set and
otherwise uses the direct callback host. Do not set both callback variables
manually.

Monitor from a second primary terminal:

```bash
ray job list --address "$RAY_ADDRESS"
ray job status '<job-id>' --address "$RAY_ADDRESS"
ray job logs '<job-id>' --address "$RAY_ADDRESS" --follow
tail -F /workspace/logs/public-harbor-server.log
```

This gate passes only when:

- Ray places the RolloutManager/session server on the primary;
- Harbor creates a Daytona sandbox running `mini-swe-agent`, not `oracle`;
- the sandbox successfully calls `/sessions/<id>/v1/chat/completions`;
- Miles records at least one model turn and collects the session records;
- Harbor runs the verifier and returns `Submitted`;
- the Ray job exits successfully;
- the Daytona sandbox is deleted; and
- there is no 401, 404, callback, identity, or cleanup failure.

A reward of zero with `Submitted` can be a valid model outcome. `AgentError`,
no recorded turn, a missing session, or a leaked sandbox is an integration
failure.

## 14. Prove the two-node topology

The worker should already be in Ray. Reconfirm before the async launcher:

```bash
ray status
ray list nodes --address "$RAY_ADDRESS"
```

Run one fully async rollout using one training node and one rollout node:

```bash
export DEBUG_SAVE=/workspace/runs/runpod-public-harbor-async-debug

if [[ -n "${MILES_ROUTER_EXTERNAL_BASE_URL:-}" ]]; then
  CALLBACK_ARGS=(--router-external-base-url "$MILES_ROUTER_EXTERNAL_BASE_URL")
else
  CALLBACK_ARGS=(--router-external-host "$CALLBACK_HOST")
fi

python examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py \
  --mode debug_rollout_only \
  --num-nodes 2 --train-num-nodes 1 --num-gpus-per-node "$NUM_TRAINERS" \
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
  --session-server-bind-ip "$MILES_SESSION_SERVER_BIND_IP" \
  "${CALLBACK_ARGS[@]}" \
  --miles-host-ip "$PRIMARY_ADDR"
```

Apply the same pass criteria. This run is decisive because training,
inference, Harbor control, and the external agent callback now exercise the
actual two-Pod topology.

## 15. Launch training

Do not run this with the plain-HTTP direct-TCP callback. Establish the
TLS/authenticated relay and rerun the one- and two-node smoke gates through the
relay first.

```bash
export RUN_TAG="$(date -u +%Y%m%d-%H%M)-public-harbor-tb2"
export RUN_ROOT="/workspace/runs/$RUN_TAG"
mkdir -p "$RUN_ROOT"

python examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py \
  --mode normal \
  --num-nodes 2 --train-num-nodes 1 --num-gpus-per-node "$NUM_TRAINERS" \
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
  --session-server-bind-ip "$MILES_SESSION_SERVER_BIND_IP" \
  --router-external-base-url "$MILES_ROUTER_EXTERNAL_BASE_URL" \
  --miles-host-ip "$PRIMARY_ADDR" \
  --wandb-project glm47-flash-agentic-async \
  --wandb-run-name "$RUN_TAG" \
  2>&1 | tee "$RUN_ROOT/launcher.log"
```

Do not leave the first iteration unattended. Require:

1. Both Ray nodes remain alive.
2. SGLang becomes healthy without OOM.
3. Daytona trials call the expected relay/session URLs.
4. Miles collects model calls and Harbor returns verifier results.
5. The first rollout and GRPO `step 0` complete.
6. Checkpoint and trace files appear under `$RUN_ROOT` and are visible from
   both Pods.
7. Daytona environments return toward zero between batches.

Increase `AGENT_MAX_CONCURRENT` gradually: `1`, then `2`, then `4`. Provider
quota, bridge file descriptors, API rate limits, model capacity, and cleanup
behavior all matter. A global batch size of 32 does not require starting 32
sandboxes simultaneously.

## 16. Human operating guide

Use `tmux` on the primary so a browser disconnect does not kill monitoring:

```bash
tmux new -s miles
```

Suggested windows are:

```text
0 launcher / Ray job
1 Ray and GPU monitoring
2 Harbor bridge logs
3 callback tunnel or direct-port probe
4 checkpoint and disk monitoring
```

Useful commands:

```bash
ray job list --address "$RAY_ADDRESS"
ray job logs '<job-id>' --address "$RAY_ADDRESS" --follow
watch -n 5 nvidia-smi
watch -n 10 'ray status'
tail -F /workspace/logs/public-harbor-server.log
df -h /workspace
```

Runpod-specific cautions:

- A green Pod card means the Pod is running, not that Ray, Miles, or the model
  is ready. Check telemetry and application health.
- Stopping or editing a Pod may change its direct TCP mapping. Recompute the
  callback after every reset.
- Files outside the mounted persistent path may disappear after edits or
  resets.
- Do not terminate the cluster to fix a shell variable or a failed Ray job.
- Stop the Ray job first, let Harbor cancel/delete confirmed trials, and verify
  durable checkpoint files before stopping compute.
- Keep one source of truth for run variables in the primary `tmux` session.
  Do not copy a stale environment block from an earlier deployment.

Stop a job without deleting persistent data:

```bash
ray job stop '<job-id>' --address "$RAY_ADDRESS"
ray job status '<job-id>' --address "$RAY_ADDRESS"
```

After trials finish or cancel, stop the bridge and tunnel deliberately. Back
up or verify checkpoints before terminating the Instant Cluster.

## 17. Failure decisions

| Symptom | Action |
| --- | --- |
| The two terminals both claim the same rank | Record the environment and contact Runpod; do not create Ray twice. |
| `hostname -I` and `NODE_ADDR` differ | Expected; use `NODE_ADDR`/`PRIMARY_ADDR`, not the first hostname address. |
| A file created under `/workspace` is absent on the worker | The volume is not shared or the mount differs; fix storage before normal training. |
| India volume has no S3 endpoint | Populate it from an attached Pod with Git/HF/scp/rsync. |
| `RUNPOD_TCP_PORT_70000` is missing | Add `70000/tcp` to the template and deliberately redeploy/reset. |
| External callback probe times out | Check the primary public IP, assigned port, template mapping, service bind address, and provider firewall. |
| `*.proxy.runpod.net` returns 524 | Remove the Runpod HTTP proxy from the synchronous path. |
| `docker info` fails | Expected for this topology; use Daytona rather than repairing Docker. |
| Harbor cannot import Daytona | Install `harbor[daytona]==0.18.0` in the isolated Harbor venv. |
| Daytona cannot build a task | Inspect the task Dockerfile and Daytona trial logs; most cloud providers require Dockerfile-defined environments. |
| Bridge returns callback-host 422 | Add the exact direct IP or relay hostname to `MILES_HARBOR_ALLOWED_CALLBACK_HOSTS`. |
| Agent gets connection refused | Re-run the external callback probe and confirm the session server is on the primary and listening correctly. |
| Agent gets `/sessions/...` 404 | The advertised URL reached the wrong or restarted session server; stop the run. |
| Session-server identity changes | The server restarted or traffic hit a different node; stop rather than mixing records. |
| RolloutManager appears on the worker | Stop: the wrong revision ran or Ray head identification is inconsistent; capture the generated command and Ray node state. |
| Ray sees fewer GPUs than `WORLD_SIZE` | Fix Ray membership or node addresses before launching Miles. |
| NCCL connects over `eth0` or times out | Set `NCCL_SOCKET_IFNAME=ens1` and restart the affected job. |
| SGLang OOMs | Clean stale GPU processes and lower memory/batch settings before changing topology. |
| Daytona sandboxes accumulate | Stop submissions, inspect provider/Harbor cancellation logs, and delete only confirmed orphan environments. |
| Reward is zero with `Submitted` | Inspect verifier output; integration may still be healthy. |
| `AgentError` or `TimeLimitExceeded` | Inspect Harbor artifacts, session records, and callback logs; this is not reward quality. |

## 18. Completion report for the next person or agent

At handoff, provide:

```text
Git revision and branch:
Primary/worker Pod IDs and NODE_ADDR values:
Ray node and GPU summary:
Shared-volume probe result:
HF and converted checkpoint paths:
Harbor version and cloud environment:
Callback mode and non-secret URL:
Oracle smoke result:
One-rollout result and Ray job ID:
Two-node smoke result and Ray job ID:
Training job ID, RUN_ROOT, and W&B run URL:
Known warnings or deviations:
```

Never include access tokens, private keys, or bearer secrets in the handoff.

[runpod-clusters]: https://docs.runpod.io/instant-clusters
[runpod-cluster-config]: https://docs.runpod.io/instant-clusters/configuration
[runpod-volume]: https://docs.runpod.io/storage/network-volumes
[runpod-ports]: https://docs.runpod.io/pods/configuration/expose-ports
[runpod-env]: https://docs.runpod.io/pods/templates/environment-variables
[runpod-templates]: https://docs.runpod.io/pods/templates/overview
[harbor-getting-started]: https://www.harborframework.com/docs/getting-started
