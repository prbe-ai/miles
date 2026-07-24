# Crusoe E2E deployment: Miles + public Harbor

This is the Crusoe-specific companion to [RUNPOD_E2E.md](RUNPOD_E2E.md).
The Miles, Harbor, dataset, checkpoint, one-rollout, and training commands are
the same. This guide replaces the provider-specific provisioning, storage, and
networking steps.

Do not start normal training until the authenticated oracle trial, one-node
Mini-SWE-Agent rollout, and two-node fully-async rollout all pass.

## Recommended topology

Use three VMs in one Crusoe location, VPC subnet, and (for supported GPU
types) InfiniBand partition:

```text
Crusoe VPC
├── miles-head:    8 GPUs, Ray head, Megatron training
├── miles-rollout: 8 GPUs, Ray worker, SGLang inference
└── miles-harbor:  CPU VM, public-Harbor bridge and task Docker containers

Crusoe Shared Disk mounted at /workspace on all three VMs
```

Running Harbor on a CPU VM keeps Docker builds, task containers, and verifier
load away from paid GPU resources. Running it on `miles-head` is acceptable for
an initial smoke test if `docker info` passes, but a separate host is preferred
for sustained training.

Crusoe publishes a GPU image named like
`ubuntu22.04-nvidia-sxm-docker`, with NVIDIA drivers, InfiniBand support, and
Docker runtime. Confirm the current exact image name with the CLI rather than
assuming it is unchanged.

## 1. Select one location and network

Install/configure the Crusoe CLI, then inspect resources before provisioning:

```bash
crusoe locations list
crusoe compute vms types
crusoe compute images list
crusoe storage disks list
```

Record:

```bash
export CRUSOE_LOCATION='<location>'
export CRUSOE_SUBNET_ID='<vpc-subnet-id>'
export CRUSOE_IB_PARTITION_ID='<ib-partition-id>'
export CRUSOE_GPU_TYPE='<available-8-gpu-vm-type>'
export CRUSOE_GPU_IMAGE='<current-ubuntu22.04-nvidia-sxm-docker-image>'
export CRUSOE_CPU_TYPE='<cpu-vm-type-with-enough-cores-and-ram>'
export CRUSOE_KEYFILE="$HOME/.ssh/id_ed25519.pub"
```

All VMs and the Shared Disk must be in the same location. Do not provision the
two GPU nodes from unrelated availability pools merely because the GPU model
matches; the Ray/NCCL path requires direct private connectivity.

## 2. Create persistent shared storage

Create at least a 1 TiB Shared Disk for source, datasets, model checkpoints,
training outputs, and copied Harbor artifacts:

```bash
crusoe storage disks create \
  --name miles-shared \
  --type shared-volume \
  --size 1TiB \
  --location "$CRUSOE_LOCATION"
```

Crusoe GPU local NVMe is ephemeral. Use it only for regenerable caches,
temporary checkpoint conversion data, and Docker build scratch. Anything that
must survive a VM stop/start or hardware failure belongs on the Shared Disk or
an external backup.

For a long run, also attach a persistent SSD to `miles-harbor` and place
Docker's data root and active Harbor trial directory there. TB2 images and
concurrent containers can exhaust the CPU VM's default OS disk. Copy completed
trial artifacts to `/workspace/harbor/trials-archive` if they must be retained.

## 3. Create the VMs

The current CLI accepts a VPC subnet, image, optional IB partition, SSH key,
and disks at VM creation. These commands are templates; substitute values
reported by `vms types` and `images list`:

```bash
crusoe compute vms create \
  --name miles-head \
  --type "$CRUSOE_GPU_TYPE" \
  --location "$CRUSOE_LOCATION" \
  --image "$CRUSOE_GPU_IMAGE" \
  --keyfile "$CRUSOE_KEYFILE" \
  --vpc-subnet-id "$CRUSOE_SUBNET_ID" \
  --ib-partition-id "$CRUSOE_IB_PARTITION_ID" \
  --disk name=miles-shared,mode=read-write

crusoe compute vms create \
  --name miles-rollout \
  --type "$CRUSOE_GPU_TYPE" \
  --location "$CRUSOE_LOCATION" \
  --image "$CRUSOE_GPU_IMAGE" \
  --keyfile "$CRUSOE_KEYFILE" \
  --vpc-subnet-id "$CRUSOE_SUBNET_ID" \
  --ib-partition-id "$CRUSOE_IB_PARTITION_ID" \
  --disk name=miles-shared,mode=read-write

crusoe compute vms create \
  --name miles-harbor \
  --type "$CRUSOE_CPU_TYPE" \
  --location "$CRUSOE_LOCATION" \
  --image ubuntu22.04:latest \
  --keyfile "$CRUSOE_KEYFILE" \
  --vpc-subnet-id "$CRUSOE_SUBNET_ID" \
  --disk name=miles-shared,mode=read-write

crusoe compute vms start miles-head
crusoe compute vms start miles-rollout
crusoe compute vms start miles-harbor
```

If the selected GPU type does not use InfiniBand, omit
`--ib-partition-id`. Do not invent a partition ID. For repeated deployments,
prefer an instance template plus `crusoe compute vms bulk-create` so the two
GPU nodes cannot drift in image, disk, subnet, or startup configuration.

## 4. Mount and verify `/workspace`

On every VM, verify Crusoe's VAST NFS driver:

```bash
vastnfs-ctl status
```

If it is absent, install the driver using Crusoe's current NFS-driver
instructions. Obtain the exact Shared Disk mount command from the Crusoe
Console. Mount the same volume at `/workspace` on all three VMs and add the
documented `_netdev,nofail,x-systemd.automount` form to `/etc/fstab` only after
a manual mount succeeds.

Verify shared read/write behavior before installing anything:

```bash
# miles-head
sudo mkdir -p /workspace
printf 'head=%s\n' "$(hostname)" | sudo tee /workspace/.miles-shared-check
sync

# miles-rollout and miles-harbor
cat /workspace/.miles-shared-check
findmnt /workspace
df -h /workspace
```

The same marker must be visible from all VMs. Do not continue with separate
local directories that merely share the name `/workspace`.

Shared Disk traffic uses NFS and is not encrypted in transit. Keep it on the
private VPC and do not store plaintext cloud/API secrets on it.

## 5. Configure private networking and firewall rules

Record the VPC/private addresses:

```bash
export HEAD_PRIVATE_IP='<miles-head-private-ip>'
export ROLLOUT_PRIVATE_IP='<miles-rollout-private-ip>'
export HARBOR_PRIVATE_IP='<miles-harbor-private-ip>'
```

Prefer private addresses for every runtime route. No Miles, Ray, SGLang, or
Harbor service needs a public ingress.

Allow internal traffic among the three VMs for:

- Harbor bridge TCP `18080` from both GPU VMs;
- Miles session TCP `30000` from `miles-harbor` and its task containers;
- Ray head/GCS/dashboard and worker traffic between GPU VMs;
- NCCL/Gloo and the selected InfiniBand fabric between GPU VMs;
- SSH only from trusted administration addresses.

Ray uses dynamically selected worker ports unless explicitly constrained. The
simplest safe rule is unrestricted east-west traffic only within the dedicated
training subnet/security group, with public ingress denied except SSH. Do not
publish ports `18080`, `30000`, `6379`, `8265`, or SGLang endpoints to the
internet.

Validate both directions before starting GPU software:

```bash
# miles-head: temporary callback probe
python -m http.server 39090 --bind 0.0.0.0 >/tmp/callback-probe.log 2>&1 &
echo $! >/tmp/callback-probe.pid

# miles-harbor: ordinary host and Docker-container probes
curl -fsS "http://${HEAD_PRIVATE_IP}:39090/" >/dev/null
docker run --rm curlimages/curl:8.12.1 -fsS \
  "http://${HEAD_PRIVATE_IP}:39090/" >/dev/null

# miles-head
kill "$(cat /tmp/callback-probe.pid)"
```

The Docker-container probe is mandatory: the Mini-SWE-Agent runs inside that
network context, not directly on `miles-harbor`.

## 6. Install the shared checkout and Harbor bridge

Clone the published branch onto the Shared Disk from one VM:

```bash
cd /workspace
git clone --branch public-harbor-runpod --single-branch \
  https://github.com/mahitoburrito/miles.git
cd /workspace/miles
git rev-parse HEAD
```

Use local virtual environments on each VM when practical; thousands of Python
metadata reads over NFS make startup slower and simultaneous writes to one
shared venv can corrupt it. Keep source and durable output shared.

On `miles-harbor`, install/verify Docker, then follow these sections of
[RUNPOD_E2E.md](RUNPOD_E2E.md):

1. **Verify checkout and runtimes**.
2. **Download TB2 and create Miles JSONL**.
3. **Start and oracle-smoke the bridge**.

Use Crusoe private addresses:

```bash
# miles-harbor
export CALLBACK_HOST="$HEAD_PRIVATE_IP"
export HARBOR_TASKS_DIR=/workspace/harbor/tasks/terminal-bench-2
export HARBOR_TRIALS_DIR=/var/lib/miles-harbor/trials
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="$HEAD_PRIVATE_IP"

# miles-head and miles-rollout
export AGENT_SERVER_URL="http://${HARBOR_PRIVATE_IP}:18080"
export MILES_ROUTER_EXTERNAL_HOST="$HEAD_PRIVATE_IP"
export MILES_HOST_IP="$HEAD_PRIVATE_IP"
export MILES_SESSION_SERVER_PORT=30000
```

Generate one Harbor bearer token, transfer it through a secret manager or SSH,
and export the same value as `MILES_HARBOR_AUTH_TOKEN` on the bridge and
`AGENT_SERVER_AUTH_TOKEN` on the GPU head. Do not put it on the Shared Disk or
in a startup script committed to Git.

From both GPU VMs require:

```bash
curl -fsS "http://${HARBOR_PRIVATE_IP}:18080/health"
```

## 7. Validate GPU fabric before Ray

On both GPU VMs:

```bash
nvidia-smi
nvidia-smi topo -m
ibstat || true
ip -br address
```

Require eight healthy GPUs per node. If using an IB-capable type, require the
expected HCA/ports to be active. Identify the actual VPC and IB interface names
instead of assuming `eth0`, `ens5`, or a particular `mlx5_*` device.

Set the socket interface consistently on both nodes:

```bash
export NCCL_SOCKET_IFNAME='<private-vpc-interface>'
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_DEBUG=INFO
```

`NCCL_SOCKET_IFNAME` is the IP/bootstrap interface, not the InfiniBand HCA.
If the cluster requires explicit IB variables such as `NCCL_IB_HCA`, pass them
through the launcher's `--extra-env-vars` JSON so Ray workers receive them. Do
not copy HCA names from another Crusoe location without inspecting this
deployment.

## 8. Run the GPU gates and training

Continue with [RUNPOD_E2E.md](RUNPOD_E2E.md), using these substitutions:

| Runpod runbook concept | Crusoe value |
| --- | --- |
| `/workspace` network volume | Crusoe Shared Disk mounted at `/workspace` |
| Runpod global-network DNS | Crusoe VPC/private IP |
| Same-Pod Docker bridge | `miles-harbor` CPU VM |
| `CALLBACK_HOST` | `HEAD_PRIVATE_IP` |
| `AGENT_SERVER_URL` | `http://HARBOR_PRIVATE_IP:18080` |
| Head/worker addresses | Crusoe VPC private IPs |
| Public symmetrical port | Not needed; keep port `30000` private |

Execute the gates in this order:

1. Verify or convert the GLM-4.7-Flash checkpoint on persistent/shared storage.
2. Start Ray on `miles-head` only and run exactly one `debug_rollout_only`
   Mini-SWE-Agent trial.
3. Join `miles-rollout` to Ray and require 2 alive nodes / 16 GPUs.
4. Run exactly one fully-async two-node rollout.
5. Inspect Harbor artifacts, Miles session/TITO records, and container cleanup.
6. Only then launch the two-node normal training command.

The debug rollout must show a real Mini-SWE-Agent model call, not only the
oracle smoke. Reward zero is acceptable for integration; missing model turns,
session 404s, `AgentError`, or leaked task containers are not.

## Crusoe-specific operational considerations

- **Ephemeral GPU disks:** stopping/restarting a VM or host failure can erase
  local GPU NVMe. Checkpoints and source must be on Shared/Persistent Disk.
- **Docker capacity:** monitor `/var/lib/docker`, active containers, and image
  cache on `miles-harbor`; TB2 builds can consume substantial space.
- **NFS semantics:** write each checkpoint to a temporary directory and rely on
  the training code's completed-checkpoint marker before treating it as valid.
  Avoid multiple processes converting into the same model directory.
- **Shared Disk confidentiality:** data is encrypted at rest but not in transit;
  use only private VPC routes and keep secrets elsewhere.
- **IB and NCCL:** colocated GPU model names do not prove the nodes share the
  intended fabric. Verify partition membership and run the two-node debug gate.
- **Firewall:** Ray's dynamic workers make narrowly guessed port rules brittle.
  Prefer a dedicated private subnet with unrestricted internal east-west traffic.
- **Harbor concurrency:** start `AGENT_MAX_CONCURRENT=2`, observe CPU/RAM/disk,
  then increase gradually. Trial batch size is not the desired Docker concurrency.
- **Backups:** a Shared Disk is persistent but not a checkpoint backup. Copy
  important checkpoints to external object storage before deleting resources.

## Shutdown order

1. Stop the Ray job and wait for its status to settle.
2. Wait for or cancel Harbor trials; confirm no `hb__` containers remain.
3. Stop the bridge.
4. Sync checkpoints/artifacts to Shared Disk and external backup.
5. Unmount Shared Disk cleanly before detaching or deleting it.
6. Stop GPU VMs, then the CPU VM.

Do not delete `miles-shared` as part of ordinary compute teardown.

## Official Crusoe references

- [VM creation and CLI flags](https://docs.crusoecloud.com/reference/cli/crusoe_compute_vms_create)
- [GPU/CPU VM specifications](https://docs.crusoecloud.com/compute/virtual-machines/overview/)
- [Docker-enabled VM images](https://docs.crusoecloud.com/compute/images/overview/index.html)
- [VPC networking](https://docs.crusoecloud.com/networking/vpc-networks/overview/index.html)
- [Shared Disk overview](https://docs.crusoecloud.com/storage/disks/overview/)
- [Creating and mounting Shared Disks](https://docs.crusoecloud.com/storage/disks/managing-shared-disks/)
- [VAST NFS driver setup](https://docs.crusoecloud.com/storage/disks/setup-nfs-driver/index.html)
