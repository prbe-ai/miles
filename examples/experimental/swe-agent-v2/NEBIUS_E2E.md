# Nebius MK8S E2E: Miles + public Harbor on two InfiniBand GPU nodes

This is the operator and coding-agent handoff for running
`examples/experimental/swe-agent-v2` on Nebius Managed Service for
Kubernetes (MK8S). It adapts the Runpod workflow to:

- two Kubernetes nodes with 8 GPUs each;
- a Nebius GPU cluster providing InfiniBand/GPUDirect RDMA;
- a Nebius shared filesystem mounted into both Miles Pods through CSI;
- one Ray head/Miles control Pod and one Ray worker Pod;
- Daytona for Harbor task sandboxes; and
- either a public TCP load balancer for the first smoke test or an
  authenticated TLS relay for training.

For the simpler and recommended first deployment on ordinary Compute VMs, use
[NEBIUS_STANDALONE_E2E.md](NEBIUS_STANDALONE_E2E.md). Miles does not require
Kubernetes; this MK8S path is for operators who explicitly want Kubernetes
scheduling, Services, CSI/PVCs, and managed node replacement.

The target configuration is `gpu-h100-sxm` / `8gpu-128vcpu-1600gb` in
`eu-north1`. H100 is only offered in that region. Nebius also supports H200 in
more regions, but changing GPU platform must be treated as a separate capacity
and performance decision.

Do not start normal training until all of these pass in order:

1. quota and capacity check;
2. two healthy GPU nodes;
3. official two-node NCCL/InfiniBand test;
4. shared-filesystem cross-node sentinel test;
5. Harbor/Daytona oracle trial;
6. one real Mini-SWE-Agent model rollout;
7. one fully async two-node rollout; and
8. the authenticated TLS callback test.

## 1. Architecture and important differences from Runpod

```text
Nebius project (eu-north1)
  Managed Kubernetes control plane

  H100 node group attached to one Nebius GPU cluster
    node A: 8x H100
      miles-head Pod
      Ray head + RolloutManager + session server + Harbor bridge

    node B: 8x H100
      miles-worker Pod
      Ray worker + training/rollout actors

  Nebius shared filesystem
    attached to both nodes as virtiofs at /mnt/data
    exposed through Nebius CSI as one ReadWriteMany PVC
    mounted inside both Miles Pods at /workspace

Daytona
  short-lived Harbor task sandboxes
  agent calls the externally advertised Miles session URL
```

The Nebius **GPU cluster object is the InfiniBand fabric grouping**. It is not
the Kubernetes control plane or the Ray cluster. MK8S schedules the Pods; Ray
coordinates Miles processes inside those Pods.

Mounting the shared filesystem at `/mnt/data` on each Kubernetes node does not
automatically expose it inside workload containers. Install the Nebius CSI
driver, create a `ReadWriteMany` PVC, and mount that PVC at `/workspace` in
both Miles Pods.

The Runpod `NODE_RANK`, `NODE_ADDR`, `PRIMARY_ADDR`, `ens1`, and symmetrical
TCP-port instructions do not apply. Kubernetes DNS, Pod IPs, Services, and
Nebius's `eth0`/InfiniBand configuration replace them.

## 2. Information required before creating chargeable resources

Record these values before running any create command. Do not record secrets.

```text
Nebius tenant ID:
Nebius project ID:
Project region:
Selected subnet ID:
Node-group service-account ID:
Selected H100 InfiniBand fabric:
H100 quota available:
Current capacity-advisor result and timestamp:
Kubernetes version:
Shared-filesystem type and size:
Miles image tag or immutable digest:
Miles git branch and commit:
Callback mode for smoke: public LoadBalancer or relay:
Production callback DNS name / relay host:
Daytona concurrency quota:
HF checkpoint path:
Converted Megatron checkpoint path:
W&B project and run name:
Budget owner and teardown deadline:
```

The project from the initial request is:

```bash
export NB_PROJECT_ID=project-e00k7pwtpr00cv6jdxkrxb
```

Verify that this project is actually in `eu-north1`; Nebius projects are
region-specific. We still need the tenant ID, a fabric with capacity, the
subnet ID, the node-group service-account ID, and confirmation that this
project's quota permits at least 16 regular H100 GPUs, two GPU VMs, one GPU
cluster, the MK8S resources, and the requested shared filesystem.

### Capacity is a hard gate

Nebius currently documents several H100 fabrics in `eu-north1` (`fabric-2`,
`fabric-3`, `fabric-4`, and `fabric-6`). Do not hardcode `fabric-3` merely
because it appears in the NCCL tutorial. Use the capacity dashboard or:

```bash
nebius capacity resource-advice list \
  --parent-id "$NB_TENANT_ID" \
  --format table
```

Select a fabric that reports enough current capacity for two
`8gpu-128vcpu-1600gb` regular nodes. Capacity-advisor data is point-in-time,
not a guarantee. For reliable repeated availability, ask Nebius about a
capacity reservation.

Check project quotas in **Administration → Limits → Quotas**. Default quotas
do not prove the project's current quota or free capacity.

## 3. Local operator prerequisites and CLI profile

Install and verify:

```bash
nebius version
kubectl version --client
helm version
jq --version
```

Create a project-specific profile if needed:

```bash
export NB_PROFILE_NAME=miles-eu-north1
export NB_PROJECT_ID=project-e00k7pwtpr00cv6jdxkrxb

nebius profile create \
  --profile "$NB_PROFILE_NAME" \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id "$NB_PROJECT_ID"

nebius profile list
```

If the profile already exists:

```bash
nebius profile update \
  --profile "$NB_PROFILE_NAME" \
  --parent-id "$NB_PROJECT_ID"
```

Confirm the current CLI release supports the flags used below:

```bash
nebius mk8s cluster create --help
nebius mk8s node-group create --help
nebius compute filesystem create --help
nebius compute gpu-cluster create --help
```

List supported Kubernetes versions instead of assuming a version forever:

```bash
nebius mk8s cluster list-control-plane-versions
export NB_K8S_VERSION=1.33
```

At the time this runbook was written, `1.33` is Nebius's recommended version.

## 4. Resolve subnet and node service account

Do not blindly use the first subnet until a human confirms it belongs to the
intended network and has enough free address space:

```bash
nebius vpc subnet list --format table
export NB_SUBNET_ID='<confirmed eu-north1 subnet ID>'
```

MK8S needs address space for the control plane, Services, and each node's Pod
CIDR. The default subnet is normally appropriate; a custom subnet must have
the documented free CIDRs.

Resolve the standard node-group service account. Nebius projects normally
come with `k8s-node-group-sa` in the default `viewers` group; confirm it exists
instead of creating a duplicate:

```bash
nebius iam service-account get-by-name \
  --name k8s-node-group-sa \
  --format json

export NB_NODE_SA_ID="$(nebius iam service-account get-by-name \
  --name k8s-node-group-sa \
  --format json | jq -r '.metadata.id')"

test -n "$NB_NODE_SA_ID"
```

The supplied draft omitted this service account, but Nebius's official NCCL
node-group command includes `--template-service-account-id`.

## 5. Create the InfiniBand GPU cluster and MK8S control plane

Set reviewed names and the fabric selected from current capacity data:

```bash
export NB_GPU_FABRIC='<fabric-2, fabric-3, fabric-4, or fabric-6>'
export NB_GPU_CLUSTER_NAME=miles-h100-fabric
export NB_MK8S_NAME=miles-h100-mk8s

: "${NB_PROJECT_ID:?}"
: "${NB_SUBNET_ID:?}"
: "${NB_NODE_SA_ID:?}"
: "${NB_GPU_FABRIC:?}"
```

Create the GPU cluster object:

```bash
export NB_GPU_CLUSTER_ID="$(nebius compute gpu-cluster create \
  --name "$NB_GPU_CLUSTER_NAME" \
  --infiniband-fabric "$NB_GPU_FABRIC" \
  --format json | jq -r '.metadata.id')"

test -n "$NB_GPU_CLUSTER_ID"
```

Create the MK8S control plane:

```bash
export NB_MK8S_CLUSTER_ID="$(nebius mk8s cluster create \
  --name "$NB_MK8S_NAME" \
  --control-plane-version "$NB_K8S_VERSION" \
  --control-plane-endpoints-public-endpoint=true \
  --control-plane-subnet-id "$NB_SUBNET_ID" \
  --format json | jq -r '.metadata.id')"

test -n "$NB_MK8S_CLUSTER_ID"
```

A public Kubernetes API endpoint is convenient for this first deployment.
Restrict its source addresses according to your organization's policy; do not
confuse the control-plane endpoint with the Miles model callback.

## 6. Create and attach the shared filesystem

One TiB is a reasonable starting point but may be too small once multiple
checkpoints and traces accumulate. Size can affect storage throughput as well
as capacity. Decide retention and backup policy before training.

```bash
export NB_FS_NAME=miles-shared-fs
export NB_FS_SIZE_GIB=1024
export NB_MOUNT_POINT=/mnt/data
export NB_MOUNT_TAG=miles-shared-fs

export NB_FS_ID="$(nebius compute filesystem create \
  --name "$NB_FS_NAME" \
  --type network_ssd \
  --size-gibibytes "$NB_FS_SIZE_GIB" \
  --block-size-bytes 4096 \
  --forbid-deletion \
  --format json | jq -r '.metadata.id')"

test -n "$NB_FS_ID"
```

Prepare cloud-init. `nofail` is intentional: Nebius warns that omitting it can
prevent a node from booting if the filesystem is unavailable.

```bash
export NB_USER_DATA="$(jq -Rrs '.' <<EOF
runcmd:
  - sudo mkdir -p $NB_MOUNT_POINT
  - sudo mount -t virtiofs $NB_MOUNT_TAG $NB_MOUNT_POINT
  - printf "%s %s virtiofs defaults,nofail 0 2\n" "$NB_MOUNT_TAG" "$NB_MOUNT_POINT" | sudo tee -a /etc/fstab
EOF
)"
```

## 7. Create the two-node H100 node group

Use regular rather than preemptible nodes for the first integration and
training run. The 256 GiB boot disk below gives the Miles image, container
layers, packages, and compilation caches more room than the 128 GiB NCCL
tutorial default.

In the web console, leave **Assign public IPv4 addresses** disabled. This
removes public addresses only; every node still receives a private IPv4
address for Kubernetes, Ray, and NCCL control traffic. Normal administration
uses the public MK8S control-plane endpoint and `kubectl`, not SSH to each
node.

The **Username and SSH key** field is optional. Leave it empty unless your
operating policy requires emergency node-level access. If credentials are
added while node public IPs remain disabled, SSH still requires a WireGuard,
bastion, or other route into the private subnet. Never use `root` or `admin`,
and paste only the public key.

Select `k8s-node-group-sa` under **Service account**. Keep GPU settings enabled,
select CUDA 13.0 with Ubuntu 24.04, attach the existing filesystem as
read/write, and keep its Auto mount option enabled. Do not add a separate
custom cloud-init configuration in the console when Auto mount already
generates the filesystem mount.

```bash
export NB_NODE_GROUP_NAME=miles-h100-nodes

nebius mk8s node-group create \
  --parent-id "$NB_MK8S_CLUSTER_ID" \
  --name "$NB_NODE_GROUP_NAME" \
  --fixed-node-count 2 \
  --version "$NB_K8S_VERSION" \
  --template-service-account-id "$NB_NODE_SA_ID" \
  --template-resources-platform gpu-h100-sxm \
  --template-resources-preset 8gpu-128vcpu-1600gb \
  --template-gpu-settings-drivers-preset cuda13.0 \
  --template-os ubuntu24.04 \
  --template-boot-disk-type network_ssd \
  --template-boot-disk-size-bytes 274877906944 \
  --template-gpu-cluster-id "$NB_GPU_CLUSTER_ID" \
  --template-network-interfaces "[{\"subnet_id\":\"$NB_SUBNET_ID\"}]" \
  --template-filesystems "[{\"existing_filesystem\":{\"id\":\"$NB_FS_ID\"},\"attach_mode\":\"READ_WRITE\",\"mount_tag\":\"$NB_MOUNT_TAG\"}]" \
  --template-cloud-init-user-data "$NB_USER_DATA"
```

If this command is rejected, do not simplify away the GPU cluster,
filesystem, service account, or driver preset. Compare `--help` with the
installed CLI version and use the equivalent JSON spec.

Get credentials and wait for both nodes:

```bash
nebius mk8s cluster get-credentials \
  --id "$NB_MK8S_CLUSTER_ID" \
  --external

kubectl cluster-info
kubectl get nodes -o wide --watch
```

Require two `Ready` GPU nodes before continuing.

## 8. Prove GPU and InfiniBand health first

The `cuda13.0` driver preset is intended to install the supported NVIDIA
drivers/components. Do not install a second GPU operator blindly on top of
working components.

Verify Kubernetes advertises all 16 GPUs:

```bash
kubectl get nodes \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'
```

Require `8` GPUs on each node.

Run Nebius's official two-node NCCL tutorial before deploying Miles. It uses a
Kubeflow `MPIJob`, two workers, eight GPUs per worker, privileged containers,
`NCCL_IB_HCA=mlx5`, and Nebius's recommended NCCL settings:

```bash
kubectl apply --server-side -k \
  'github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.9.3'
```

Then apply the H100 version of Nebius's documented `nccl-test.yaml`. Record:

```text
NCCL test image:
NCCL test command:
Average bus bandwidth:
Any transport fallback or warning:
```

Stop if NCCL falls back unexpectedly, `/dev/infiniband` is absent, GPUs are
missing, or the all-reduce test fails. Do not debug Miles on an unproven
fabric.

If the driver preset did not install working components, follow Nebius's
current **With InfiniBand: GPU and network operators** procedure in the exact
documented order. At the time of writing it uses the Nebius NVIDIA Network
Operator chart followed by the GPU Operator; GPUDirect RDMA is enabled by
default. Provider chart versions change, so copy current commands from the
linked official page rather than freezing old versions into automation.

## 9. Install CSI and expose the filesystem to Pods

The filesystem is attached to the nodes but is not yet available inside Miles
containers. Install Nebius's CSI mounted-filesystem-path driver:

```bash
helm pull \
  oci://cr.eu-north1.nebius.cloud/mk8s/helm/csi-mounted-fs-path \
  --version 0.1.5

helm upgrade csi-mounted-fs-path \
  ./csi-mounted-fs-path-0.1.5.tgz \
  --install \
  --set dataDir="$NB_MOUNT_POINT/csi-mounted-fs-path-data/"
```

Confirm the CSI Pods are healthy:

```bash
kubectl get pods -A | grep csi-mounted-fs-path
kubectl get storageclass csi-mounted-fs-path-sc
```

Create the namespace and RWX PVC:

```bash
kubectl create namespace miles --dry-run=client -o yaml | kubectl apply -f -

cat > /tmp/miles-pvc.yaml <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: miles-workspace
  namespace: miles
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: csi-mounted-fs-path-sc
  resources:
    requests:
      storage: 1Ti
YAML

kubectl apply -f /tmp/miles-pvc.yaml
kubectl get pvc -n miles miles-workspace --watch
```

Require `Bound`. The PVC is the object mounted by the Miles Pods; do not use a
node `hostPath` as a substitute.

## 10. Create Kubernetes secrets without committing them

Generate independent bridge and session secrets. Create a local file with
mode `0600`; do not commit or paste it into run notes:

```bash
umask 077
export AGENT_SERVER_AUTH_TOKEN="$(openssl rand -hex 32)"
export MILES_SESSION_API_KEY="$(openssl rand -hex 32)"

cat > /tmp/miles-secrets.env <<EOF
DAYTONA_API_KEY=<daytona-key>
HF_TOKEN=<hugging-face-token>
WANDB_API_KEY=<wandb-key-or-empty>
MILES_HARBOR_AUTH_TOKEN=$AGENT_SERVER_AUTH_TOKEN
AGENT_SERVER_AUTH_TOKEN=$AGENT_SERVER_AUTH_TOKEN
MILES_SESSION_API_KEY=$MILES_SESSION_API_KEY
EOF

if grep -Eq '<[^>]+>' /tmp/miles-secrets.env; then
  echo 'Replace every placeholder in /tmp/miles-secrets.env first' >&2
  exit 1
fi

kubectl create secret generic miles-secrets \
  --namespace miles \
  --from-env-file=/tmp/miles-secrets.env \
  --dry-run=client -o yaml | kubectl apply -f -

rm /tmp/miles-secrets.env
```

Kubernetes Secrets are not a substitute for organization-wide secret
management. Apply the project's encryption, RBAC, and rotation policy.

## 11. Deploy the Ray head and worker Pods

Use the official Miles image initially:

```text
radixark/miles:latest
```

For repeatability, resolve and record an immutable image digest or mirror the
image into Nebius Container Registry. The source branch is still cloned into
the shared filesystem and installed editable with `--no-deps`; do not rebuild
the patched CUDA/SGLang/Megatron dependency stack from a generic image.

Create `/tmp/miles-nebius.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: miles-head
  namespace: miles
spec:
  selector:
    app: miles
    role: head
  ports:
    - name: ray
      port: 6379
      targetPort: 6379
    - name: dashboard
      port: 8265
      targetPort: 8265
    - name: harbor
      port: 18080
      targetPort: 18080
    - name: session
      port: 30000
      targetPort: 30000
---
apiVersion: v1
kind: Pod
metadata:
  name: miles-head
  namespace: miles
  labels:
    app: miles
    role: head
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 120
  containers:
    - name: miles
      image: radixark/miles:latest
      imagePullPolicy: Always
      command: ["/bin/bash", "-lc"]
      args: ["ulimit -l unlimited; exec sleep infinity"]
      securityContext:
        privileged: true
      envFrom:
        - secretRef:
            name: miles-secrets
      env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: NCCL_IB_HCA
          value: mlx5
        - name: UCX_NET_DEVICES
          value: eth0
        - name: SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING
          value: "1"
        - name: NCCL_COLLNET_ENABLE
          value: "0"
        - name: NCCL_DEBUG
          value: INFO
        - name: MILES_ROOT
          value: /workspace/miles
        - name: MEGATRON_PATH
          value: /root/Megatron-LM
        - name: MILES_SCRIPT_EXTERNAL_RAY
          value: "1"
        - name: MILES_SESSION_SERVER_BIND_IP
          value: 0.0.0.0
        - name: HARBOR_ENVIRONMENT_TYPE
          value: daytona
        - name: MILES_HARBOR_ENVIRONMENT_KWARGS_JSON
          value: "{}"
        - name: HARBOR_DELETE_ENVIRONMENTS
          value: "true"
        - name: AGENT_MAX_CONCURRENT
          value: "1"
        - name: AGENT_SERVER_TIMEOUT_SEC
          value: "14400"
        - name: MILES_HARBOR_REQUEST_TIMEOUT_SEC
          value: "14400"
      resources:
        requests:
          nvidia.com/gpu: "8"
        limits:
          nvidia.com/gpu: "8"
      volumeMounts:
        - name: workspace
          mountPath: /workspace
        - name: dshm
          mountPath: /dev/shm
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: miles-workspace
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 256Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: miles-worker
  namespace: miles
  labels:
    app: miles
    role: worker
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 120
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: miles
              role: head
          topologyKey: kubernetes.io/hostname
  containers:
    - name: miles
      image: radixark/miles:latest
      imagePullPolicy: Always
      command: ["/bin/bash", "-lc"]
      args: ["ulimit -l unlimited; exec sleep infinity"]
      securityContext:
        privileged: true
      env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: NCCL_IB_HCA
          value: mlx5
        - name: UCX_NET_DEVICES
          value: eth0
        - name: SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING
          value: "1"
        - name: NCCL_COLLNET_ENABLE
          value: "0"
        - name: NCCL_DEBUG
          value: INFO
        - name: MILES_ROOT
          value: /workspace/miles
        - name: MEGATRON_PATH
          value: /root/Megatron-LM
      resources:
        requests:
          nvidia.com/gpu: "8"
        limits:
          nvidia.com/gpu: "8"
      volumeMounts:
        - name: workspace
          mountPath: /workspace
        - name: dshm
          mountPath: /dev/shm
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: miles-workspace
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 256Gi
```

Apply and wait:

```bash
kubectl apply -f /tmp/miles-nebius.yaml
kubectl get pods -n miles -o wide --watch
```

The eight-GPU request plus required anti-affinity ensures the Pods occupy
different GPU nodes. Stop if either is not `Running`, has fewer than eight
visible GPUs, or both report the same Kubernetes node.

## 12. GPU, image, and shared-filesystem preflight

```bash
kubectl exec -n miles miles-head -- nvidia-smi -L
kubectl exec -n miles miles-worker -- nvidia-smi -L

kubectl exec -n miles miles-head -- \
  python -c 'import miles,ray,sglang,torch; assert torch.cuda.device_count() == 8; print(torch.__version__)'

kubectl exec -n miles miles-worker -- \
  python -c 'import miles,ray,sglang,torch; assert torch.cuda.device_count() == 8; print(torch.__version__)'

kubectl exec -n miles miles-head -- bash -lc \
  'ls -l /dev/infiniband; df -h /workspace /dev/shm'

kubectl exec -n miles miles-worker -- bash -lc \
  'ls -l /dev/infiniband; df -h /workspace /dev/shm'
```

Prove the PVC is genuinely shared across nodes:

```bash
export NB_SENTINEL="nebius-shared-$(date -u +%Y%m%dT%H%M%SZ)"

kubectl exec -n miles miles-head -- bash -lc \
  "printf 'written-by=head\\n' > /workspace/$NB_SENTINEL"

kubectl exec -n miles miles-worker -- \
  cat "/workspace/$NB_SENTINEL"
```

Only the primary should mutate the repository, download models/tasks, install
the Harbor venv, or convert checkpoints. The filesystem is shared; concurrent
uncoordinated writes to the same paths can corrupt state.

## 13. Populate the shared workspace

On the head Pod:

```bash
kubectl exec -it -n miles miles-head -- bash

cd /workspace
export MILES_COMMIT='<recorded commit SHA>'
git clone --branch public-harbor-runpod --single-branch \
  https://github.com/prbe-ai/miles.git miles
cd /workspace/miles
git checkout --detach "$MILES_COMMIT"
git rev-parse HEAD
test "$(git rev-parse HEAD)" = "$MILES_COMMIT"
```

Use the commit recorded before provisioning. Install the shared source
editable into each Pod's local Python environment, sequentially:

```bash
kubectl exec -n miles miles-head -- \
  pip install -e /workspace/miles --no-deps

kubectl exec -n miles miles-worker -- \
  pip install -e /workspace/miles --no-deps
```

Install the isolated Harbor/Daytona venv on the head only:

```bash
kubectl exec -it -n miles miles-head -- bash

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

The repository preflight is provider-neutral; its `--phase node` mode is
Runpod-specific and must not be used on Nebius.

Download/export TB2 and create JSONL using section 9 of `RUNPOD_E2E.md`, with
all paths under `/workspace`. Download the HF checkpoint and convert it once
to the Megatron `torch_dist` format. Verify all outputs from the worker before
starting Ray.

## 14. Start Ray over Kubernetes networking

Start the Ray head:

```bash
kubectl exec -n miles miles-head -- bash -lc '
  ray stop --force || true
  ray start --head \
    --node-ip-address="$POD_IP" \
    --port=6379 \
    --num-gpus=8 \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats
'
```

Start the worker using the head Service DNS:

```bash
kubectl exec -n miles miles-worker -- bash -lc '
  ray stop --force || true
  ray start \
    --address=miles-head.miles.svc.cluster.local:6379 \
    --node-ip-address="$POD_IP" \
    --num-gpus=8 \
    --disable-usage-stats
'
```

Verify from the head:

```bash
kubectl exec -n miles miles-head -- ray status
kubectl exec -i -n miles miles-head -- python - <<'PY'
import ray
ray.init(address="auto")
resources = ray.cluster_resources()
print(resources)
assert int(resources.get("GPU", 0)) == 16, resources
PY
```

For a local dashboard tunnel:

```bash
kubectl port-forward -n miles service/miles-head 8265:8265
```

Do not expose Ray port `6379` or dashboard `8265` publicly.

## 15. Start and oracle-smoke the Daytona Harbor bridge

Inside the head Pod:

```bash
export MILES_ROOT=/workspace/miles
export HARBOR_VENV=/workspace/venvs/harbor-0.18-daytona
export HARBOR_DATA_ROOT=/workspace/harbor
export HARBOR_TASKS_DIR="$HARBOR_DATA_ROOT/tasks/terminal-bench-2"
export HARBOR_TRIALS_DIR="$HARBOR_DATA_ROOT/trials"
export HARBOR_ENVIRONMENT_TYPE=daytona
export MILES_HARBOR_ENVIRONMENT_KWARGS_JSON='{}'
export HARBOR_DELETE_ENVIRONMENTS=true
export AGENT_MAX_CONCURRENT=1
export MILES_HARBOR_REQUEST_TIMEOUT_SEC=14400
export AGENT_SERVER_URL=http://miles-head.miles.svc.cluster.local:18080

mkdir -p /workspace/logs "$HARBOR_TRIALS_DIR"
nohup "$HARBOR_VENV/bin/python" \
  "$MILES_ROOT/examples/experimental/swe-agent-v2/public_harbor_server.py" \
  --host 0.0.0.0 --port 18080 \
  >/workspace/logs/public-harbor-server.log 2>&1 &

curl -fsS "$AGENT_SERVER_URL/health"
```

Run the authenticated oracle payload from `RUNPOD_E2E.md`. Require HTTP 200,
`Submitted`, verifier output, and Daytona cleanup. This still does not prove
the model callback.

## 16. Expose the first model callback

### Direct public TCP/HTTP smoke

Create a public LoadBalancer Service that selects only the head Pod:

```bash
cat > /tmp/miles-session-public.yaml <<'YAML'
apiVersion: v1
kind: Service
metadata:
  name: miles-session-public
  namespace: miles
spec:
  type: LoadBalancer
  selector:
    app: miles
    role: head
  ports:
    - name: session
      protocol: TCP
      port: 30000
      targetPort: 30000
YAML

kubectl apply -f /tmp/miles-session-public.yaml
kubectl get service -n miles miles-session-public --watch
```

Record the external IP:

```bash
export MILES_PUBLIC_IP="$(kubectl get service -n miles miles-session-public \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

test -n "$MILES_PUBLIC_IP"
```

Use this only for the initial smoke:

```bash
export MILES_SESSION_SERVER_PORT=30000
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0
export MILES_ROUTER_EXTERNAL_HOST="$MILES_PUBLIC_IP"
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="$MILES_PUBLIC_IP,localhost,127.0.0.1"
```

The Miles session server enforces `MILES_SESSION_API_KEY`, but direct traffic
is still plain HTTP. Restart the bridge after changing its callback allowlist.
Do not use this path for sensitive or long-running training.

### Production TLS callback

Use either:

- the outbound reverse-tunnel/TLS relay described in `RUNPOD_E2E.md`; or
- a reviewed Kubernetes TLS proxy with a stable DNS name and no short request
  timeout in front of the head-only session Service.

Set:

```bash
export MILES_ROUTER_EXTERNAL_BASE_URL=https://miles-model.example.com
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS=miles-model.example.com,localhost,127.0.0.1
```

If using a Nebius LoadBalancer directly, convert its dynamic public IP to a
reusable allocation and add `nebius.com/load-balancer-allocation-id` to the
Service. A stable IP does not provide TLS by itself.

Avoid ordinary HTTP proxies/CDNs with fixed short synchronous request
timeouts. Agent model calls can exceed 100 seconds.

## 17. Run the one-rollout and two-node gates

Open a shell in the head Pod and export the same model/task variables used by
the Runpod guide:

```bash
kubectl exec -it -n miles miles-head -- bash

cd /workspace/miles
export RAY_ADDRESS=http://127.0.0.1:8265
export MILES_SCRIPT_EXTERNAL_RAY=1
export AGENT_SERVER_URL=http://miles-head.miles.svc.cluster.local:18080
export AGENT_SERVER_TIMEOUT_SEC=14400
export AGENT_MODEL_NAME=model
export MILES_SESSION_SERVER_PORT=30000
export MILES_SESSION_SERVER_BIND_IP=0.0.0.0
```

For direct smoke:

```bash
export MILES_ROUTER_EXTERNAL_HOST='<LoadBalancer external IP>'
unset MILES_ROUTER_EXTERNAL_BASE_URL
```

For relay/TLS:

```bash
export MILES_ROUTER_EXTERNAL_BASE_URL=https://miles-model.example.com
unset MILES_ROUTER_EXTERNAL_HOST
```

Run section 13 of `RUNPOD_E2E.md` for exactly one colocated
`debug_rollout_only` trial, with these changes:

- use `--session-server-port 30000`;
- keep `--session-server-bind-ip 0.0.0.0`;
- do not pass `--miles-host-ip`; Kubernetes/Pod networking should determine
  each process's routable address;
- use the external host for direct smoke or full external base URL for relay.

Then run section 14's fully async two-node debug command with the same changes.
The gate succeeds only when:

- RolloutManager/session server is in `miles-head`;
- Ray reports 16 GPUs across two nodes;
- Daytona calls the advertised `/sessions/<id>/v1/chat/completions` URL;
- Miles records and collects at least one model turn;
- Harbor returns `Submitted` and verifier output;
- no session identity, auth, NCCL, storage, or cleanup failure occurs.

## 18. Launch and monitor training

Only launch normal training after rerunning both smoke gates through the TLS
callback. Use section 15 of `RUNPOD_E2E.md`, with:

```text
--num-nodes 2
--train-num-nodes 1
--num-gpus-per-node 8
--session-server-port 30000
--session-server-bind-ip 0.0.0.0
--router-external-base-url <TLS callback origin>
```

Do not pass Runpod-specific node variables or `--miles-host-ip`.

Monitor:

```bash
kubectl get pods -n miles -o wide
kubectl exec -n miles miles-head -- ray status
kubectl exec -n miles miles-head -- nvidia-smi
kubectl exec -n miles miles-worker -- nvidia-smi
kubectl exec -n miles miles-head -- tail -F /workspace/logs/public-harbor-server.log
kubectl exec -n miles miles-head -- df -h /workspace
```

Do not leave the first iteration unattended. Require the first rollout, GRPO
step 0, trace write, and checkpoint write before increasing Daytona
concurrency above one.

## 19. Failure decisions

| Symptom | Action |
| --- | --- |
| Node group creation says capacity unavailable | Re-check capacity advisor and choose another compatible H100 fabric; do not remove InfiniBand. |
| Quota denies two nodes or 16 H100s | Request a project quota increase or reservation before retrying. |
| Nodes are not `Ready` | Inspect node-group status, cloud-init, subnet capacity, and driver preset before deploying Miles. |
| Kubernetes shows no `nvidia.com/gpu` | Fix the Nebius GPU components/operator installation. |
| `/dev/infiniband` is absent | Fix Network Operator/driver configuration; do not run Miles. |
| NCCL test fails or uses sockets unexpectedly | Fix the fabric and NCCL settings before application debugging. |
| PVC stays `Pending` | Verify node filesystem attachment, CSI chart, storage class, and `/mnt/data` mount. |
| Head sentinel is absent on worker | Stop: Pods do not share the same PVC/filesystem. |
| Docker Hub pull is throttled | Mirror the Miles image into Nebius Container Registry and pin its digest. |
| Both Miles Pods land on one node | Verify eight-GPU requests and required pod anti-affinity. |
| Ray sees fewer than 16 GPUs | Fix Pod GPU visibility or Ray membership before launch. |
| Daytona cannot reach the callback | Check public Service/relay, allowlist, bearer, DNS, and session bind address. |
| Callback returns 401 | Confirm the same `MILES_SESSION_API_KEY` reaches Miles, Harbor monitor, and Daytona agent. |
| Callback returns a session 404/identity mismatch | Stop: traffic reached the wrong or restarted session server. |
| Shared filesystem is slow | Benchmark it and revisit filesystem size/type; do not assume capacity alone implies required bandwidth. |
| Head Pod restarts | Treat the training job/session state as interrupted; do not silently continue mixed rollouts. |

## 20. Stop and tear down safely

Stop the Ray job first and allow Harbor/Daytona cleanup:

```bash
kubectl exec -n miles miles-head -- \
  ray job stop '<job-id>' --address http://127.0.0.1:8265
```

Verify durable checkpoints, then remove the public callback and workload Pods:

```bash
kubectl delete service -n miles miles-session-public --ignore-not-found
kubectl delete -f /tmp/miles-nebius.yaml --ignore-not-found
```

Delete infrastructure only after deciding what must be retained. The safe
order is:

1. Ray/Miles workloads and public LoadBalancer;
2. H100 node group;
3. MK8S cluster;
4. GPU cluster object; and
5. shared filesystem last, only after backup and removal of deletion
   protection.

Record all resource IDs at creation time so a failed partial deployment can be
cleaned up without guessing.

## 21. Information still needed from the operator

Before creating the Nebius cluster, provide or confirm:

1. `NB_TENANT_ID` for capacity-advisor queries.
2. That `project-e00k7pwtpr00cv6jdxkrxb` is the intended `eu-north1` project.
3. The actual H100 quota in that project and whether regular or reserved
   capacity will be used.
4. Which H100 fabric currently has capacity for two 8-GPU nodes.
5. The subnet ID and confirmation it has enough MK8S Pod/control-plane CIDRs.
6. The `k8s-node-group-sa` service-account ID and required IAM roles.
7. Desired shared-filesystem size/type and checkpoint retention policy.
8. Whether `radixark/miles:latest` may be pulled from Docker Hub or must be
   mirrored into Nebius Container Registry; ideally provide a pinned digest.
9. The exact Miles commit to run.
10. Daytona account/API key and expected maximum concurrency.
11. HF model access and whether the model is already staged anywhere.
12. Production callback choice: existing relay VM/domain or a new Nebius TLS
    endpoint, including DNS ownership.
13. Budget, maximum cluster lifetime, and who is authorized to delete
    resources.
14. Whether H200 is an acceptable fallback if H100 capacity is unavailable;
    changing platform/region may require a different project, fabric, and
    performance validation.
15. Whether to add a small CPU node group for cluster add-ons and operator
    tooling. It is optional for the first smoke test, but avoids using costly
    GPU-node CPU/RAM for non-GPU services.

Also reserve operational headroom for node-group maintenance. Nebius may use
a surge node during an update; for this preset that can temporarily require a
third 8-GPU H100 node. If quota or budget cannot allow that, review and test a
non-surge update strategy before the first node-group upgrade.

Do not create the H100 node group until items 1-8 are known. Items 9-13 must be
resolved before the real rollout/training gates.

## Official Nebius references

- [GPU node groups with InfiniBand](https://docs.nebius.com/kubernetes/gpu/clusters)
- [Official two-node NCCL tutorial](https://docs.nebius.com/kubernetes/gpu/nccl-test)
- [GPU drivers, Network Operator, GPU Operator, and GPUDirect RDMA](https://docs.nebius.com/kubernetes/gpu/set-up)
- [Shared filesystem through CSI and RWX PVCs](https://docs.nebius.com/kubernetes/storage/filesystem-over-csi)
- [Shared filesystem types and limits](https://docs.nebius.com/compute/storage/types)
- [GPU fabrics and region compatibility](https://docs.nebius.com/compute/clusters/gpu)
- [MK8S networking address requirements](https://docs.nebius.com/kubernetes/networking/requirements)
- [Kubernetes LoadBalancer Services](https://docs.nebius.com/kubernetes/clusters/load-balancer)
- [Reusable public IP allocations](https://docs.nebius.com/kubernetes/networking/dynamic-to-static)
- [Capacity advisor](https://docs.nebius.com/compute/virtual-machines/capacity-advisor)
- [Compute/GPU/storage/InfiniBand quotas](https://docs.nebius.com/compute/resources/quotas-limits)
- [Nebius CLI profile configuration](https://docs.nebius.com/cli/configure)
- [Supported Kubernetes versions](https://docs.nebius.com/kubernetes/versions)
