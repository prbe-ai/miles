# Nebius standalone-VM E2E: Miles + public Harbor on two InfiniBand nodes

This is the recommended first-run path for
`examples/experimental/swe-agent-v2` on Nebius. It uses two ordinary Compute
VMs instead of Managed Kubernetes:

- two `gpu-h100-sxm` / `8gpu-128vcpu-1600gb` VMs;
- both VMs assigned to the same Nebius GPU cluster and InfiniBand fabric;
- one shared SSD filesystem mounted at `/mnt/data` on both hosts and exposed
  to the Miles containers as `/workspace`;
- one Ray head/Miles control container and one Ray worker container;
- Daytona for Harbor task sandboxes; and
- a separately authenticated external callback to the head.

Use [NEBIUS_E2E.md](NEBIUS_E2E.md) instead if Managed Kubernetes is a firm
requirement. MK8S adds scheduling, Services, CSI/PVCs, and automated node
replacement, but Miles does not require Kubernetes.

Do not start normal training until these pass in order:

1. quota and capacity check;
2. both hosts show eight healthy GPUs and InfiniBand devices;
3. cross-host shared-filesystem sentinel;
4. Nebius's two-host NCCL test;
5. Ray reports 16 GPUs across two nodes;
6. Harbor/Daytona oracle trial;
7. one real model rollout;
8. one fully async two-node rollout; and
9. authenticated TLS callback test.

## 1. Architecture

```text
Nebius project (eu-north1)
  one GPU cluster / one H100-compatible InfiniBand fabric

  miles-head
    8x H100
    private IP + one controlled management/public path
    Miles container
      Ray head
      RolloutManager and session server
      Harbor/Daytona bridge

  miles-worker
    8x H100
    private IP only
    Miles container
      Ray worker
      training and rollout actors

  one shared network_ssd filesystem
    mounted on both hosts at /mnt/data
    bind-mounted in both containers at /workspace
```

The Nebius GPU cluster is only the InfiniBand fabric grouping. Ray is the
distributed application runtime. The shared filesystem provides the common
repository, model, dataset, trace, and checkpoint paths.

Standalone VMs do not use Kubernetes, Pods, Services, CSI, or PVCs. Hostnames
and private IP addresses replace Kubernetes DNS.

## 2. Record required inputs before provisioning

Do not record secret values in this file or in shared storage.

```text
Nebius tenant ID:
Nebius project ID:
Project region:
Selected subnet ID:
Security-group ID:
Selected H100 InfiniBand fabric:
GPU-cluster ID:
Shared-filesystem ID, type, size, block size, and mount tag:
Head VM name, ID, private IP, and public allocation:
Worker VM name, ID, and private IP:
SSH username and key fingerprint:
Miles image tag or immutable digest:
Miles git branch and commit:
Callback relay/DNS name:
Daytona concurrency quota:
HF checkpoint and converted-checkpoint paths:
Budget owner and teardown deadline:
```

The project supplied for this deployment is:

```bash
export NB_PROJECT_ID=project-e00k7pwtpr00cv6jdxkrxb
```

Confirm it is an `eu-north1` project. H100 SXM is available only there. The
project, both VMs, GPU cluster, subnet, security group, and filesystem must be
compatible and the filesystem/VMs must belong to the same project.

## 3. Capacity, storage, and image gates

### GPU capacity

Two H100 SXM nodes consume 16 H100 GPUs, 256 vCPUs, and 3200 GiB of RAM. Check
project quota and current fabric capacity before creating chargeable VMs:

```bash
export NB_TENANT_ID='<tenant ID>'

nebius capacity resource-advice list \
  --parent-id "$NB_TENANT_ID" \
  --format table
```

Select a compatible `eu-north1` fabric with capacity for two regular
`8gpu-128vcpu-1600gb` instances. Capacity advice is point-in-time, not a
reservation. Use regular rather than preemptible VMs for the first run.

### Shared filesystem

Use the existing shared filesystem with:

- type `network_ssd`;
- block size `4096` bytes / 4 KiB;
- deletion protection enabled;
- `READ_WRITE` attachment to both VMs;
- the same mount tag on both, for example `miles-shared-fs`;
- Auto mount enabled at `/mnt/data` on both.

The Nebius web form does not expose a filesystem-type selector because the
normal shared filesystem is SSD-backed. Verify the created object:

```bash
export NB_FS_NAME='<filesystem name>'

nebius compute filesystem get-by-name \
  --name "$NB_FS_NAME" \
  --format json |
  jq '.spec | {type, size_bytes, block_size_bytes, forbid_deletion}'
```

Expect `network_ssd` and `4096`. Fifty GiB is not enough: the configured
GLM-4.7-Flash Hugging Face files alone are about 62.5 GB, and conversion keeps
another model-sized representation. Start with at least 300 GiB for limited
smoke testing; 500 GiB or more is safer for training. One TiB is conservative,
not mandatory. Nebius permits growth but not shrinkage.

### Boot image and disk

Use Nebius's current Ubuntu 24.04 NVIDIA GPU image with CUDA 13.0. It matches
the current `radixark/miles:latest` CUDA generation. Allocate at least 200 GiB;
256-300 GiB leaves safer room for Docker layers, caches, and logs.

Pin an immutable Miles image digest before a long run. `latest` is acceptable
only for the first smoke test.

## 4. Networking and security before VM creation

Both VMs must use the same private network and subnet. Ray, Gloo, NCCL control
traffic, Harbor, and Miles host-to-host traffic use private addresses.

Do not leave a public VM on Nebius's default security group: the default group
allows all ingress and egress. Create a custom security group with:

1. inbound TCP 22 from the human operator's current public `/32` address;
2. inbound traffic from the same security group to itself, allowing the two
   private VM addresses to communicate on Ray/NCCL dynamic ports;
3. outbound internet access for GitHub, Docker Hub, Hugging Face, Daytona,
   package indexes, W&B, and the TLS relay; and
4. no other public inbound traffic.

The worker has no public IP. Give the head either:

- a persistent public allocation for direct SSH; or
- no public IP when a WireGuard/bastion/private access path already exists.

Do not expose Ray `6379`, Ray dashboard `8265`, Harbor `18080`, or Miles
session port `30000` directly to the public internet. A later smoke test may
use a temporary authenticated callback route, but normal training must use a
reviewed TLS path.

## 5. Create the two standalone VMs

Create two separate Compute VMs and assign both to the same GPU-cluster object
at creation time. A VM cannot be added to a GPU cluster afterward.

Use these settings:

| Setting | `miles-head` | `miles-worker` |
| --- | --- | --- |
| Platform/preset | H100 SXM / `8gpu-128vcpu-1600gb` | Same |
| VM type | Regular | Regular |
| GPU cluster | Same reviewed GPU-cluster ID | Same |
| Boot image | Ubuntu 24.04 for NVIDIA GPUs, CUDA 13.0 | Same |
| Boot disk | 256-300 GiB SSD | Same |
| Subnet | Same private subnet | Same |
| Public IP | Persistent allocation or controlled static IP | None |
| Hostname | `miles-head` | `miles-worker` |
| Shared filesystem | Existing filesystem, RW, Auto mount | Same |
| Mount tag/path | `miles-shared-fs` / `/mnt/data` | Same |
| SSH credentials | Same non-root user/public key | Same |
| Service account | None unless private Nebius services require it | Same |

Use a username such as `miles`, not `root` or `admin`. Paste only the public
Ed25519 key:

```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```

Leave custom cloud-init disabled when the console's Auto mount option is
enabled. If custom cloud-init is required, ensure the final rendered config
still includes the SSH user/key, sudo access, the `virtiofs` mount, and an
`/etc/fstab` entry containing `nofail`. Never put Daytona, HF, W&B, bridge, or
session secrets in cloud-init.

A Nebius service account is unnecessary for public Docker Hub/GitHub/Hugging
Face access. Attach a least-privilege account only when the VM must pull from a
private Nebius registry, read Nebius Object Storage, or call Nebius APIs.

## 6. Connect and record stable roles

From the operator machine:

```bash
export NEBIUS_SSH_USER=miles
export HEAD_PUBLIC_IP='<head public IP>'
export HEAD_PRIVATE_IP='<head private IP>'
export WORKER_PRIVATE_IP='<worker private IP>'

ssh "$NEBIUS_SSH_USER@$HEAD_PUBLIC_IP"
ssh -J "$NEBIUS_SSH_USER@$HEAD_PUBLIC_IP" \
  "$NEBIUS_SSH_USER@$WORKER_PRIVATE_IP"
```

Optional local SSH config:

```sshconfig
Host miles-head
  HostName <head-public-ip>
  User miles
  IdentityFile ~/.ssh/id_ed25519

Host miles-worker
  HostName <worker-private-ip>
  User miles
  IdentityFile ~/.ssh/id_ed25519
  ProxyJump miles-head
```

On each VM, compare the recorded address with the host state:

```bash
hostname
hostname -I
ip -brief address
ip route
```

Do not use the public IP for Ray or NCCL. Write non-secret cluster coordinates
once on the head's shared filesystem. Re-export the recorded addresses after
SSHing into the head because local shell variables do not cross SSH sessions:

```bash
export HEAD_PRIVATE_IP='<head private IP>'
export WORKER_PRIVATE_IP='<worker private IP>'

sudo mkdir -p /mnt/data/miles-workspace/nebius
sudo chown -R "$USER:$USER" /mnt/data/miles-workspace

cat >/mnt/data/miles-workspace/nebius/cluster.env <<EOF
HEAD_PRIVATE_IP=$HEAD_PRIVATE_IP
WORKER_PRIVATE_IP=$WORKER_PRIVATE_IP
NUM_NODES=2
NUM_TRAINERS=8
EOF
chmod 0644 /mnt/data/miles-workspace/nebius/cluster.env
```

## 7. Host and shared-filesystem preflight

Run on both VMs:

```bash
set -euo pipefail

nvidia-smi -L
test "$(nvidia-smi -L | wc -l)" -eq 8
ls -l /dev/infiniband
findmnt -t virtiofs
findmnt /mnt/data
df -h / /mnt/data
test -w /mnt/data/miles-workspace
```

Require eight GPUs, visible InfiniBand devices, the same backing filesystem,
and adequate free space on both hosts.

Prove that storage is genuinely shared. On the head:

```bash
export NB_SENTINEL="nebius-shared-$(date -u +%Y%m%dT%H%M%SZ)"
printf 'head=%s time=%s\n' "$(hostname)" "$(date -u +%FT%TZ)" \
  | tee "/mnt/data/miles-workspace/$NB_SENTINEL"
echo "$NB_SENTINEL"
```

On the worker, use the printed filename:

```bash
cat "/mnt/data/miles-workspace/<sentinel filename>"
```

Stop if it is absent. Only the head should clone/update the repository,
download data/models, create the Harbor venv, or convert checkpoints.

## 8. Install or verify the container runtime

Nebius's NVIDIA image supplies the GPU driver. Check whether Docker and the
NVIDIA Container Toolkit are already installed:

```bash
docker --version
nvidia-ctk --version
```

If either is missing, install Docker and `nvidia-container-toolkit` using the
current Nebius/NVIDIA Ubuntu instructions, then configure the runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info
```

The operator may instead create **Containers over VMs** with the custom public
image `radixark/miles:latest`; the required runtime arguments below remain the
same.

Pull and record the image on both hosts:

```bash
export MILES_IMAGE=radixark/miles:latest
sudo docker pull "$MILES_IMAGE"
sudo docker image inspect "$MILES_IMAGE" \
  --format '{{index .RepoDigests 0}}'
```

Launch one long-lived container on each host:

```bash
source /mnt/data/miles-workspace/nebius/cluster.env

sudo docker rm -f miles 2>/dev/null || true
sudo docker run -d \
  --name miles \
  --restart unless-stopped \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband \
  -v /mnt/data/miles-workspace:/workspace \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=mlx5 \
  -e UCX_NET_DEVICES=eth0 \
  -e SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1 \
  -e NCCL_COLLNET_ENABLE=0 \
  -e NCCL_DEBUG=INFO \
  --entrypoint /bin/bash \
  "$MILES_IMAGE" -lc 'exec sleep infinity'
```

If Docker rejects both `--privileged` and `--device`, keep `--privileged` and
remove only the redundant device flag. Do not remove InfiniBand access.

Verify inside both containers:

```bash
sudo docker exec miles nvidia-smi -L
sudo docker exec miles bash -lc \
  'ls -l /dev/infiniband; df -h /workspace /dev/shm; python -c "import miles,ray,sglang,torch; assert torch.cuda.device_count() == 8; print(torch.__version__)"'
```

## 9. Prove InfiniBand/NCCL before Miles

Follow Nebius's official **Running parallel jobs with MPIrun** NCCL procedure
for two Compute VMs. It covers Open MPI installation, host-to-host SSH,
`nccl-tests`, and the multi-host launch.

The SSH key entered in the VM form is for the human operator. Do not copy the
operator laptop's private key onto the head. If MPIrun needs passwordless
head-to-worker SSH, generate a separate temporary key on the head, add only
that public key to the worker through the existing operator connection, and
remove it after the NCCL gate. Ray itself does not require SSH between hosts.

Record:

```text
NCCL test image/build:
NCCL command:
Average bus bandwidth:
NCCL transport and HCA:
Warnings/fallbacks:
```

Stop if NCCL uses an unexpected socket fallback, cannot open an `mlx5` HCA,
reports errors, or hangs. Do not debug Miles on an unproven fabric.

## 10. Clone the pinned source and install Harbor

Run repository mutations on the head only:

```bash
sudo docker exec -it miles bash

cd /workspace
export MILES_COMMIT='<recorded commit SHA>'
git clone --branch public-harbor-runpod --single-branch \
  https://github.com/prbe-ai/miles.git miles
cd /workspace/miles
git checkout --detach "$MILES_COMMIT"
test "$(git rev-parse HEAD)" = "$MILES_COMMIT"
pip install -e /workspace/miles --no-deps
```

On the worker, install the same shared checkout into its container-local
Python environment:

```bash
sudo docker exec miles \
  pip install -e /workspace/miles --no-deps
```

On the head container, create the isolated Harbor/Daytona environment:

```bash
cd /workspace/miles
export HARBOR_VENV=/workspace/venvs/harbor-0.18-daytona

uv venv "$HARBOR_VENV" --python 3.12
uv pip install --python "$HARBOR_VENV/bin/python" \
  -r examples/experimental/swe-agent-v2/requirements-runpod.txt \
  pytest pytest-asyncio ruff

python examples/experimental/swe-agent-v2/runpod_preflight.py --phase repo
"$HARBOR_VENV/bin/python" -m pytest -q \
  tests/fast/experimental \
  --confcutdir=tests/fast/experimental \
  --override-ini='addopts='
```

The repository phase is provider-neutral. Do not use the Runpod-specific
`--phase node` check on Nebius.

## 11. Start Ray over private addresses

On the head host:

```bash
source /mnt/data/miles-workspace/nebius/cluster.env

sudo docker exec miles bash -lc \
  "ray stop --force || true; ray start --head --node-ip-address='$HEAD_PRIVATE_IP' --port=6379 --num-gpus=8 --dashboard-host=127.0.0.1 --disable-usage-stats"
```

On the worker host:

```bash
source /mnt/data/miles-workspace/nebius/cluster.env

sudo docker exec miles bash -lc \
  "ray stop --force || true; ray start --address='$HEAD_PRIVATE_IP:6379' --node-ip-address='$WORKER_PRIVATE_IP' --num-gpus=8 --disable-usage-stats"
```

Verify on the head:

```bash
sudo docker exec miles ray status
sudo docker exec -i miles python - <<'PY'
import ray

ray.init(address="auto")
resources = ray.cluster_resources()
print(resources)
assert int(resources.get("GPU", 0)) == 16, resources
PY
```

For a dashboard tunnel from the operator machine:

```bash
ssh -L 8265:127.0.0.1:8265 \
  "$NEBIUS_SSH_USER@$HEAD_PUBLIC_IP"
```

## 12. Data, model, Harbor, callback, and training gates

The remaining application flow is shared with the Runpod guide. Use
[RUNPOD_E2E.md](RUNPOD_E2E.md) with these substitutions:

| Runpod term | Nebius standalone value |
| --- | --- |
| primary Pod | `miles-head` VM/container |
| worker Pod | `miles-worker` VM/container |
| `/workspace` network volume | `/mnt/data/miles-workspace` bind-mounted as `/workspace` |
| `PRIMARY_ADDR` | `HEAD_PRIVATE_IP` |
| worker `NODE_ADDR` | `WORKER_PRIVATE_IP` |
| `ens1` | `eth0` for the Nebius VPC control path; `mlx5` for InfiniBand |
| Runpod exposed TCP port | head public allocation only for temporary smoke, otherwise TLS relay |

Perform these sections from the head container:

1. Runpod section 9: download/export Terminal-Bench 2 and build JSONL.
2. Runpod section 11: create head-only secret environment and oracle-smoke
   the Daytona Harbor bridge.
3. Runpod section 12: download GLM-4.7-Flash and convert it once to Megatron
   `torch_dist`; verify the worker reads both paths.
4. Runpod section 13: exactly one real Mini-SWE-Agent rollout.
5. Runpod section 14: fully async two-node rollout.
6. Runpod section 15: normal training only after the TLS callback passes.

Set the standalone-specific variables on the head container:

```bash
source /workspace/nebius/cluster.env

export PRIMARY_ADDR="$HEAD_PRIVATE_IP"
export NODE_ADDR="$HEAD_PRIVATE_IP"
export NODE_RANK=0
export MASTER_ADDR="$HEAD_PRIVATE_IP"
export NUM_NODES=2
export NUM_TRAINERS=8
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export MILES_HOST_IP="$HEAD_PRIVATE_IP"
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0
export MILES_SESSION_SERVER_PORT=30000
export AGENT_SERVER_URL="http://${HEAD_PRIVATE_IP}:18080"
export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=http://127.0.0.1:8265
```

Do not copy secret environment files onto `/workspace`; the worker and any
process with shared-filesystem access can read it. Keep the Daytona, HF, W&B,
Harbor bearer, and Miles session keys in a mode-`0600` file on the head's
local boot disk or inject them through the operator shell.

For the direct callback smoke, advertise the head's controlled public address
and temporarily permit only the required authenticated session port. For
normal training, use the outbound TLS relay from `RUNPOD_E2E.md` and set
`MILES_ROUTER_EXTERNAL_BASE_URL`. Never expose Ray or Harbor publicly.

## 13. Monitoring and failure decisions

Monitor from the two hosts:

```bash
sudo docker logs --tail 200 -f miles
sudo docker exec miles nvidia-smi
sudo docker exec miles df -h /workspace
sudo docker exec miles ray status  # head only
```

| Symptom | Action |
| --- | --- |
| Second VM cannot select the GPU cluster | Stop; GPU-cluster assignment is creation-time and platform/fabric must match. |
| Worker cannot read the sentinel | Stop; the filesystem is not the same mount/backing object. |
| Docker cannot see eight GPUs | Fix the host driver/Container Toolkit before continuing. |
| Container lacks `/dev/infiniband` | Fix device/privileged configuration before NCCL or Miles. |
| NCCL uses an unexpected transport | Fix interface/HCA/host setup before application debugging. |
| Ray worker cannot join | Check private IPs and same-security-group private ingress. |
| Ray reports fewer than 16 GPUs | Fix membership/GPU visibility before launch. |
| Head public IP changes | Update SSH/callback state or use a persistent allocation/TLS relay. |
| Daytona cannot reach the session URL | Check the relay/public route, callback allowlist, bind address, and session bearer. |
| Shared storage fills | Stop cleanly, grow it, and implement checkpoint retention/offload. |
| Either container/VM restarts | Treat the active Ray/training job as interrupted and verify checkpoint consistency. |

## 14. Teardown

1. Stop the Ray job and allow Harbor/Daytona environment cleanup.
2. Verify and copy required checkpoints.
3. Stop/remove both Miles containers.
4. Delete worker and head VMs to stop GPU charges.
5. Delete public allocations/security resources only if no longer needed.
6. Delete the GPU cluster after all attached VMs are gone.
7. Delete the shared filesystem last, only after backup and disabling deletion
   protection.

Stopping a VM stops compute charges but not storage charges. Record resource
IDs so partial deployments can be cleaned up without guessing.

## Official references

- [Create an eight-GPU VM with InfiniBand and shared storage](https://docs.nebius.com/compute/quickstart)
- [InfiniBand GPU clusters for Compute VMs](https://docs.nebius.com/compute/clusters/gpu)
- [Containers over Compute VMs](https://docs.nebius.com/compute/virtual-machines/containers)
- [Run distributed NCCL tests with MPIrun](https://docs.nebius.com/3p-integrations/mpirun)
- [Attach and mount shared filesystems](https://docs.nebius.com/compute/storage/use)
- [Shared-filesystem types and performance](https://docs.nebius.com/compute/storage/types)
- [VM public/private IP behavior](https://docs.nebius.com/compute/virtual-machines/network)
- [Security groups and permissive default group](https://docs.nebius.com/vpc/security-groups/overview)
- [Capacity advisor](https://docs.nebius.com/compute/virtual-machines/capacity-advisor)
