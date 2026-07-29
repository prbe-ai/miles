# Nebius MK8S E2E: Miles + public Harbor on two InfiniBand GPU nodes

This is the operator and coding-agent handoff for running
`examples/experimental/swe-agent-v2` on Nebius Managed Service for
Kubernetes (MK8S). It adapts the Runpod workflow to:

- two Kubernetes nodes with 8 GPUs each;
- a Nebius GPU cluster providing InfiniBand/GPUDirect RDMA;
- a Nebius shared filesystem mounted into both Miles Pods through CSI;
- one Ray head/Miles control Pod and one Ray worker Pod;
- Modal for Harbor task sandboxes, with Daytona retained as a fallback; and
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

Complete the experiment gates in this order:

1. quota and capacity check;
2. two healthy GPU nodes;
3. official two-node NCCL/InfiniBand test;
4. shared-filesystem cross-node sentinel test;
5. Harbor/Modal oracle trial;
6. one real Mini-SWE-Agent model rollout;
7. one fully async two-node rollout;
8. the authenticated TLS callback test; and
9. one bounded normal-mode training step that consumes a Harbor rollout,
   performs an optimizer/policy update, transfers the updated weights, and
   writes its trace and checkpoint evidence.

The initial scope focused on cluster, distributed-runtime, Harbor, and rollout
validation. The current experiment scope also requires gate 9 so the run
exercises the training side of Miles rather than stopping at generation.
`--debug-rollout-only` is an intermediate diagnostic and never satisfies the
experiment by itself: it does not prove that the training model, gradient
buffers, optimizer state, and inference model fit in the selected GPU
partition.

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

Modal
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
Modal account, token pair, and intended concurrency:
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

### Operate the cluster with a coding agent

Kubernetes does not provide a general login shell for the cluster. The
recommended operating model is to run the coding agent on the operator
machine, in the Miles checkout, and let it control the remote GPU workloads
through the local `kubectl` context:

```text
coding agent on operator machine
  -> local Nebius profile and kubeconfig
  -> Kubernetes API
     -> miles-head Pod on GPU node A
     -> miles-worker Pod on GPU node B
```

This keeps the Nebius login and powerful Kubernetes credentials outside the
workload Pods. It also lets the agent apply manifests, inspect both Pods, copy
files, collect logs, and execute GPU commands without SSH access to either
node. Before starting the agent, verify the active target and permissions:

```bash
kubectl config current-context
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods --all-namespaces
kubectl auth can-i get pods --all-namespaces
```

The local `kubectl` client must be within one minor version of the MK8S
control-plane version. Do not continue with an unsupported client/server
version combination.

Start the coding agent from the checked-out branch containing this runbook and
give it a bounded instruction such as:

```text
Follow examples/experimental/swe-agent-v2/NEBIUS_E2E.md starting at
section 8. Operate the Nebius cluster through kubectl. Stop and report if the
node, GPU, InfiniBand/NCCL, shared-filesystem, Harbor, or rollout gate does not
match the documented expectation. Do not create, rotate, print, or delete
credentials, and do not tear down infrastructure without explicit approval.
```

Typical remote operations performed by that local agent are:

```bash
kubectl exec -n miles miles-head -- nvidia-smi
kubectl exec -n miles miles-worker -- nvidia-smi
kubectl logs -n miles miles-head
kubectl get events -n miles --sort-by=.lastTimestamp
```

The `miles-head` and `miles-worker` Pods do not exist until section 11. After
they are deployed, an interactive coding-agent CLI may instead run directly
inside the head Pod:

```bash
kubectl wait --for=condition=Ready pod/miles-head \
  -n miles --timeout=15m
kubectl exec -it -n miles miles-head -- bash

cd /workspace/miles
<launch the selected coding-agent CLI>
```

Running the agent in `miles-head` gives it direct access to the GPU runtime
and the shared `/workspace`. Source edits under `/workspace` are visible to
the worker, but each Pod still has its own Python environment, so section 13
installs the shared checkout editable in both Pods.

An in-Pod agent does **not** automatically have permission to apply Kubernetes
resources or execute into the worker. Do not copy an administrator kubeconfig
into the Pod or grant it `cluster-admin`. If the in-Pod agent truly needs to
control Kubernetes, create and review a dedicated service account with only
the required verbs and resources in the `miles` namespace. Otherwise, keep
cluster-wide orchestration in the local agent and use the in-Pod agent only
for repository, runtime, Harbor, Ray, and training work.

Do not rely on a foreground `kubectl exec` session for a long training run.
Use the Ray job submission flow in section 18 so the job continues if the
operator terminal or coding-agent session disconnects.

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
read -rsp 'Probe write token: ' PROBE_TOKEN
echo

cat > /tmp/miles-secrets.env <<EOF
HF_TOKEN=<hugging-face-token>
WANDB_API_KEY=<wandb-key-or-empty>
PROBE_TOKEN=$PROBE_TOKEN
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
unset PROBE_TOKEN
```

Create a separate Modal credential Secret. This preserves `miles-secrets`
unchanged during future provider credential rotations and never puts either
Modal token in a command argument or local file:

1. In the Modal workspace settings, create an API token or, preferably, a
   dedicated service user for this run.
2. If Modal RBAC is enabled, grant that service user `Contributor` access to
   the intended Modal Environment (`main` unless deliberately changed).
3. Copy the API token ID (`ak-...`) and secret (`as-...`) when shown. Do not use
   Modal proxy tokens (`wk-...` / `ws-...`); those authenticate Web Functions,
   not the SDK that creates Sandboxes.

See [Modal service users](https://modal.com/docs/guide/service-users) and
[Modal client configuration](https://modal.com/docs/sdk/py/latest/config).

```bash
read -rsp 'Modal token ID: ' MODAL_TOKEN_ID
echo
read -rsp 'Modal token secret: ' MODAL_TOKEN_SECRET
echo

printf 'MODAL_TOKEN_ID=%s\nMODAL_TOKEN_SECRET=%s\n' \
  "$MODAL_TOKEN_ID" "$MODAL_TOKEN_SECRET" |
  kubectl create secret generic modal-credentials \
    --namespace miles \
    --from-env-file=/dev/stdin \
    --dry-run=client -o yaml |
  kubectl apply -f -

unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET

kubectl get secret modal-credentials -n miles \
  -o go-template='{{if and (index .data "MODAL_TOKEN_ID") (index .data "MODAL_TOKEN_SECRET")}}modal_credentials=present{{else}}modal_credentials=absent{{end}}{{"\n"}}'
```

For the durable cluster variant that reuses the existing `miles-dev`
Deployment as the Ray head, wire the Secret references into its Pod template
**while the GPU node group is still down**. Updating the template creates a
new ReplicaSet; do not run these commands from a live in-Pod session:

```bash
kubectl --context nebius-miles patch deployment miles-dev \
  --namespace miles \
  --type merge \
  --patch \
  '{"spec":{"template":{"metadata":{"labels":{"miles.prbe.ai/ray-role":"head"}}}}}'

kubectl --context nebius-miles set env deployment/miles-dev \
  --namespace miles \
  --from=secret/miles-secrets \
  --keys=PROBE_TOKEN

kubectl --context nebius-miles set env deployment/miles-dev \
  --namespace miles \
  --from=secret/modal-credentials \
  --keys=MODAL_TOKEN_ID,MODAL_TOKEN_SECRET

kubectl --context nebius-miles set env deployment/miles-dev \
  --namespace miles \
  HARBOR_ENVIRONMENT_TYPE=modal \
  'MILES_HARBOR_ENVIRONMENT_KWARGS_JSON={"sandbox_timeout_secs":14400}' \
  HARBOR_DELETE_ENVIRONMENTS=true

kubectl --context nebius-miles set env deployment/miles-dev \
  --namespace miles --list |
  grep -E '^(PROBE_TOKEN|MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|HARBOR_ENVIRONMENT_TYPE|MILES_HARBOR_ENVIRONMENT_KWARGS_JSON)='

test "$(
  kubectl --context nebius-miles get deployment miles-dev \
    --namespace miles \
    -o jsonpath='{.spec.template.metadata.labels.miles\.prbe\.ai/ray-role}'
)" = "head"
```

The listing prints Secret references rather than Secret values. After the node
group returns, require the replacement `miles-dev` Pod to be Ready and the
`miles-head` EndpointSlice to resolve to its Pod IP before starting Ray or
Harbor. Putting the head label on the Deployment template prevents the Service
endpoint from disappearing on every rollout.

For that durable Deployment, use this macOS-compatible discovery gate after
the node group becomes Ready:

```bash
kubectl --context nebius-miles rollout status deployment/miles-dev \
  --namespace miles --timeout=15m

export HEAD_POD="$(
  kubectl --context nebius-miles get pods --namespace miles \
    -l app=miles-dev \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
  awk 'NF{print; exit}'
)"
test -n "$HEAD_POD"

kubectl --context nebius-miles wait --namespace miles \
  --for=condition=Ready "pod/$HEAD_POD" --timeout=10m

export HEAD_IP="$(
  kubectl --context nebius-miles get pod --namespace miles "$HEAD_POD" \
    -o jsonpath='{.status.podIP}'
)"
export SERVICE_ENDPOINT="$(
  kubectl --context nebius-miles get endpointslice --namespace miles \
    -l kubernetes.io/service-name=miles-head \
    -o jsonpath='{range .items[*].endpoints[*].addresses[*]}{.}{"\n"}{end}' |
  awk 'NF{print; exit}'
)"
test -n "$HEAD_IP"
test "$SERVICE_ENDPOINT" = "$HEAD_IP"

kubectl --context nebius-miles exec --namespace miles "$HEAD_POD" -- \
  sh -lc '
    test -n "${PROBE_TOKEN:-}"
    test -n "${MODAL_TOKEN_ID:-}"
    test -n "${MODAL_TOKEN_SECRET:-}"
    test "${HARBOR_ENVIRONMENT_TYPE:-}" = modal
    printf "modal_and_probe_credentials=present\n"
  '
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
        - secretRef:
            name: modal-credentials
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
          value: modal
        - name: MILES_HARBOR_ENVIRONMENT_KWARGS_JSON
          value: '{"sandbox_timeout_secs":14400}'
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
export HEAD_POD="${HEAD_POD:-miles-head}"

kubectl exec -n miles "$HEAD_POD" -- \
  pip install -e /workspace/miles --no-deps

kubectl exec -n miles miles-worker -- \
  pip install -e /workspace/miles --no-deps
```

Probe metric tracking (`MILES_USE_PROBE=1`, section 18/23) needs the Probe
SDK in **each training environment** — `ProbeBackend` imports
`probe.integrations.miles`, so without this the training run raises at
tracking init. The floor is **probe-research >= 0.22.0**: 0.22.0 added the
per-sample rollout rail (section 18/23) and fixed a trajectory-capture bug
(0.21.0 and earlier sent `coords: null` in expanded trajectory span batches,
which the 0059+ server rejects with a 422 — any capture of a recognized
trajectory failed mid-flight). The `@main` git pin below satisfies this; do
not substitute a cached older wheel. Install on both Pods (binaries
committed, no Go toolchain):

```bash
kubectl exec -n miles "$HEAD_POD" -- \
  pip install "probe-research @ git+https://github.com/prbe-ai/research-os-agent.git@main"

kubectl exec -n miles miles-worker -- \
  pip install "probe-research @ git+https://github.com/prbe-ai/research-os-agent.git@main"
```

Install the isolated Harbor/Modal venv on the head only. Use a new path rather
than modifying the validated Daytona environment; this keeps rollback
possible and lets Modal's dependency set remain independently pinned:

```bash
export HEAD_POD="${HEAD_POD:-miles-head}"
kubectl exec -it -n miles "$HEAD_POD" -- bash

cd /workspace/miles
export HARBOR_VENV=/workspace/venvs/harbor-0.18-modal
uv venv "$HARBOR_VENV" --python 3.12
# requirements-public-harbor-capture.txt pulls probe-research (>= 0.23.0 — the
# harbor_capture facade — with the packaged sandbox-snapshot binaries) from git. WITHOUT it the bridge raises at
# startup under MILES_HARBOR_CAPTURE_MODE!=off or MILES_SANDBOX_STATE=1, and the
# `probe` CLI (watcher, below) is absent.
uv pip install --python "$HARBOR_VENV/bin/python" \
  -r examples/experimental/swe-agent-v2/requirements-nebius-modal.txt \
  -r examples/experimental/swe-agent-v2/requirements-public-harbor-capture.txt \
  pytest pytest-asyncio ruff

test -n "${MODAL_TOKEN_ID:-}"
test -n "${MODAL_TOKEN_SECRET:-}"
"$HARBOR_VENV/bin/python" -c \
  'from importlib import metadata; import modal; print("harbor", metadata.version("harbor")); print("modal", metadata.version("modal")); print("modal_module", modal.__file__)'

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

## 15. Start and oracle-smoke the Modal Harbor bridge

Inside the head Pod:

```bash
export MILES_ROOT=/workspace/miles
export HARBOR_VENV=/workspace/venvs/harbor-0.18-modal
export HARBOR_DATA_ROOT=/workspace/harbor
export HARBOR_TASKS_DIR="$HARBOR_DATA_ROOT/tasks/terminal-bench-2"
export HARBOR_TRIALS_DIR="$HARBOR_DATA_ROOT/trials"
export MILES_HARBOR_CAPTURE_DIR="$HARBOR_DATA_ROOT/captures"
# Capture is off by default since the additive capture modes shipped; without
# this the bridge stages nothing and the watcher below has nothing to export.
export MILES_HARBOR_CAPTURE_MODE=shadow
# Ephemeral begin/end sandbox filesystem snapshots (probe.sandbox-state/1).
# Requires probe-research >= 0.23.0 (harbor_capture facade + packaged
# probe-sandbox-snapshot binaries — install from git, see the capture reqs).
export MILES_SANDBOX_STATE=1
export HARBOR_ENVIRONMENT_TYPE=modal
# Keep the provider lifetime aligned with the four-hour bridge/client timeout.
# Modal supports up to 24 hours, but a bounded value limits leaked compute.
export MILES_HARBOR_ENVIRONMENT_KWARGS_JSON='{"sandbox_timeout_secs":14400}'
export HARBOR_DELETE_ENVIRONMENTS=true
export AGENT_MAX_CONCURRENT=1
export MILES_HARBOR_REQUEST_TIMEOUT_SEC=14400
export AGENT_SERVER_URL=http://miles-head.miles.svc.cluster.local:18080

: "${MODAL_TOKEN_ID:?MODAL_TOKEN_ID is absent from the head Pod}"
: "${MODAL_TOKEN_SECRET:?MODAL_TOKEN_SECRET is absent from the head Pod}"
: "${PROBE_TOKEN:?PROBE_TOKEN is absent from the head Pod}"

mkdir -p /workspace/logs "$HARBOR_TRIALS_DIR" "$MILES_HARBOR_CAPTURE_DIR"
nohup "$HARBOR_VENV/bin/python" \
  "$MILES_ROOT/examples/experimental/swe-agent-v2/public_harbor_server.py" \
  --host 0.0.0.0 --port 18080 \
  >/workspace/logs/public-harbor-server.log 2>&1 &

curl -fsS "$AGENT_SERVER_URL/health"
```

Start the independent Probe consumer against the same PVC. The Harbor bridge
calls the Probe SDK only to stage native files and an SDK-owned descriptor;
this process validates and uploads them without adding network latency to
`Trial.run()`:

```bash
# Use the Harbor venv's probe CLI (that is where probe-research was installed).
nohup "$HARBOR_VENV/bin/probe" trial watch "$MILES_HARBOR_CAPTURE_DIR" --interval 5 \
  >/workspace/logs/probe-harbor-export.log 2>&1 &
```

Do not point the watcher at `HARBOR_TRIALS_DIR`. The capture directory is the
atomic handoff boundary and survives Harbor sandbox teardown, bridge restarts,
and Research OS outages. `probe trial drain "$MILES_HARBOR_CAPTURE_DIR"`
performs a one-shot repair after an outage.

With `MILES_SANDBOX_STATE=1` each staged trial additionally carries
`trial/artifacts/probe-sandbox-state/` — begin/end filesystem manifests and
the agent's delta tarball, captured inside the sandbox at `AGENT_START` /
`AGENT_END` via an uploaded static binary and removed from the container in
the same instant (the sandbox is probe-free during the whole agent phase).

Run the authenticated oracle payload from `RUNPOD_E2E.md`. Require HTTP 200,
`Submitted`, verifier output, a nonempty `provider_sandbox_id`, and Modal
cleanup. This still does not prove the model callback.

The bridge retains the provider ID during Harbor `AGENT_START`/`AGENT_END`,
before Harbor clears the Modal SDK handle. For Modal the response ID is
`Sandbox.object_id` and normally begins with `sb-`. Save the oracle response as
`$ORACLE_JSON`, then prove that exact Sandbox disappears:

```bash
export PROVIDER_SANDBOX_ID="$(
  python -c 'import json,os; print(json.load(open(os.environ["ORACLE_JSON"]))["provider_sandbox_id"])'
)"
test -n "$PROVIDER_SANDBOX_ID"

"$HARBOR_VENV/bin/python" - "$PROVIDER_SANDBOX_ID" <<'PY'
import sys
import time

import modal

provider_id = sys.argv[1]
for _ in range(24):
    live_ids = {sandbox.object_id for sandbox in modal.Sandbox.list()}
    if provider_id not in live_ids:
        print("modal_cleanup=passed", "provider_sandbox_id", provider_id)
        break
    time.sleep(5)
else:
    raise AssertionError(
        f"Modal Sandbox {provider_id} still exists after 120 seconds"
    )
PY
```

**Sandbox-capture self-check (couple this to the oracle smoke).** Once the
oracle trial has staged, validate its bundle with the shipped checker
(stdlib-only, no probe install needed):

```bash
python3 examples/experimental/swe-agent-v2/check_sandbox_bundle.py \
  "$MILES_HARBOR_CAPTURE_DIR" --latest --require-integrity
```

Exit 0 means the latest trial's bundle is present, both phases are `ok`,
sha256 integrity holds, and the manifests are non-empty; it prints the
`+added/~modified/-deleted` summary and file count so you can sanity-check
the capture scope against the task image. Exit 1 flags an incomplete or
integrity-failed bundle (a missing `meta.json` = capture died midway; the
reason is in `/run`'s `capture.sandbox_state` and the bridge log). Exit 2
means no bundle was found — check that `MILES_SANDBOX_STATE=1` and a
non-`off` capture mode were both set. Snapshot failures never fail the
trial itself, so this check is how you confirm the sandbox half is live
before committing to the full run.

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
    miles.prbe.ai/ray-role: head
  ports:
    - name: http-session
      protocol: TCP
      port: 80
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
export MILES_ROUTER_EXTERNAL_BASE_URL="http://$MILES_PUBLIC_IP"
unset MILES_ROUTER_EXTERNAL_HOST
export MILES_HARBOR_ALLOWED_CALLBACK_HOSTS="$MILES_PUBLIC_IP,localhost,127.0.0.1"
```

The Miles session server enforces `MILES_SESSION_API_KEY`, but direct traffic
is still plain HTTP. Restart the bridge after changing its callback allowlist.
Do not use this path for sensitive or long-running training.

Modal Sandboxes allow outbound access to public IPs by default, so this path
does not require a provider-tier egress upgrade. The callback must still be
publicly routable, included in `MILES_HARBOR_ALLOWED_CALLBACK_HOSTS`, and
authenticated with `MILES_SESSION_API_KEY`. Do not confuse Modal's outbound
access with an inbound route to the Kubernetes ClusterIP Service; a remote
Sandbox cannot resolve `*.svc.cluster.local`.

If Daytona is deliberately selected as the fallback provider, its
organization must still be Tier 3 or higher for this callback path. See
[Daytona network limits](https://www.daytona.io/docs/en/network-limits/).

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
export HEAD_POD="$(
  kubectl get pods -n miles \
    -l miles.prbe.ai/ray-role=head \
    -o jsonpath='{.items[0].metadata.name}'
)"
test -n "$HEAD_POD"
kubectl exec -it -n miles "$HEAD_POD" -- bash

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
export MILES_ROUTER_EXTERNAL_BASE_URL='http://<LoadBalancer external IP>'
unset MILES_ROUTER_EXTERNAL_HOST
```

For relay/TLS:

```bash
export MILES_ROUTER_EXTERNAL_BASE_URL=https://miles-model.example.com
unset MILES_ROUTER_EXTERNAL_HOST
```

Run section 13 of `RUNPOD_E2E.md` for exactly two colocated
`debug_rollout_only` trials, with these changes:

- use `--session-server-port 30000`;
- keep `--session-server-bind-ip 0.0.0.0`;
- use `--rollout-batch-size 1` and `--n-samples-per-prompt 2`;
- use `--global-batch-size 2` (TP=4 on eight GPUs gives DP=2);
- do not pass `--miles-host-ip`; Kubernetes/Pod networking should determine
  each process's routable address;
- use the external host for direct smoke or full external base URL for relay.

Then run section 14's fully async two-node debug command with the same changes,
plus `--over-sampling-batch-size 2`. The async launcher's debug mode retains
separate eight-GPU actor and rollout placement pools so Ray must place one pool
on each node; stop if its placement log puts all bundles on one node.
The gate succeeds only when:

- RolloutManager/session server is in the head Pod selected by
  `miles.prbe.ai/ray-role=head`;
- Ray reports 16 GPUs across two nodes;
- Modal calls the advertised `/sessions/<id>/v1/chat/completions` URL;
- Miles records and collects at least one model turn;
- Harbor returns `Submitted` and verifier output;
- no session identity, auth, NCCL, storage, or cleanup failure occurs.

## 18. Launch and monitor training

Only launch normal training after rerunning both smoke gates through the TLS
callback.

**Enable Probe metric tracking BEFORE launching** (these are read at
argument-parse time via `env_flag`, so they must be in the training
environment — the Ray driver *and* both node workers — before the launch, not
set afterward in section 23). `MILES_USE_PROBE=1` flips on `--use-probe`; the
`PROBE_*` vars name the run and its durable queue:

```bash
export MILES_USE_PROBE=1
export PROBE_PROJECT=miles-nebius
export PROBE_EXPERIMENT=swe-agent-v2-nebius
export PROBE_EXTERNAL_ID='<stable Nebius/Ray job ID>'
export PROBE_QUEUE_DIR=/workspace/probe/metrics
```

If the launcher submits a Ray job with an isolated `runtime_env`, pass these
through it so the workers inherit them; a bare shell export on the head alone
will not reach the actors. Section 23 covers the queue/exporter mechanics and
recovery.

**Per-sample metric rail (probe-research >= 0.22.0, optional but wanted for
sample-level visibility):** miles' own `--custom-rollout-log-function-path`
hook hands the raw per-sample list to a function inside the RolloutManager
process; Probe ships a drop-in for it. Add ONE launch argument:

```text
--custom-rollout-log-function-path probe.connectors.miles.per_sample_rollout_log
```

Every rollout sample then streams `rollout/reward` and
`rollout/response_length` through the same durable queue as label-identified
points (`labels={"sample": <data-source sample index>, "group":
<prompt-group index>}` — miles' global integer counters, point identity only,
never a series axis). The hook is fail-open (a failure logs one warning and
never reaches the rollout loop) and always returns False, so miles' default
aggregate logging runs unchanged. Unconfigured (no `--use-probe`, no
`PROBE_TOKEN`) it is a silent no-op — safe to leave in the launch template.

Then use section 15 of `RUNPOD_E2E.md`, with:

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
step 0, trace write, and checkpoint write before increasing Modal
concurrency above one.

The experiment is complete only when the logs and durable outputs prove all of
the following for the same run:

- Harbor launched the selected task in Modal and returned a non-aborted
  rollout to Miles;
- Miles consumed that rollout in normal mode and completed one optimizer step;
- updated policy weights were transferred to or acknowledged by the rollout
  engine after the step;
- the step's metrics, rollout/trajectory, trace, and checkpoint or checkpoint
  manifest were stored durably; and
- Research OS recorded the run configuration, `env_ref`, metrics, notes, and
  portable Harbor artifacts without treating an optional artifact failure as
  the training outcome.

Set the run length and save interval so the job exits after this evidence is
written. Do not scale concurrency or continue into an open-ended training run
as part of this validation.

### Training-memory constraint observed on H100 80 GB

The later one-update smoke test split the two-node cluster into one 8-GPU
training node and one 8-GPU inference node. With GLM-4.7-Flash training set to
tensor parallel 4, expert parallel 2, and data parallel 2, each training GPU
already held approximately 57.83 GiB when Megatron tried to allocate its
parameter/gradient buffer. Only 21.34 GiB remained, while the next allocation
required 26.44 GiB, so initialization failed before generation or an optimizer
step. CPU optimizer offload did not remove this GPU-resident gradient-buffer
requirement.

This was not a Nebius node-size, Kubernetes cgroup, quota, or credit failure:
the Pods had the intended 120 CPUs, 1400 GiB RAM, and eight H100s. It was a
model-parallel layout that did not fit in 80 GiB per training GPU. Before
retrying a policy update, validate one of these changes in isolation:

- increase training tensor parallelism from 4 to 8 and adjust expert/data
  parallelism so the world-size and batch divisibility constraints still hold;
- reduce model, sequence, microbatch, or retained activation requirements;
- use a memory-reducing distributed strategy whose checkpoint conversion is
  compatible with this model; or
- add training GPUs/nodes and recompute the parallelism topology.

Megatron also required the global batch size to be divisible by microbatch size
times data-parallel size. In the observed DP=2 layout, `global_batch_size=1`
failed validation; changing it to 2 exposed the subsequent gradient-buffer OOM.

Before paying for another online sandbox batch, replay a verified two-sample
debug rollout through the real training stack:

```text
--mode debug_train_only
--load-debug-rollout-data <section-17-or-18-traces>/rollout_data/{rollout_id}.pt
--num-rollout 1
--global-batch-size 2
```

The async launcher retains `--grad-reduce-in-bf16` for this gate. Require
optimizer step 0, a trace, a checkpoint, idle GPUs afterward, and a completed
Probe run with readable production metrics. This isolates training-memory and
checkpoint correctness from callback/TLS and sandbox cost; it does not replace
the final online normal-mode gate.

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
| Modal cannot reach the callback | Check the public Service/relay, bridge callback allowlist, bearer, DNS, and session bind address; a Modal Sandbox cannot use Kubernetes ClusterIP DNS. |
| Modal authentication fails | Confirm both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are present in the head Pod and were issued for the intended workspace. |
| Modal Sandbox remains after a trial | Stop before training; require `HARBOR_DELETE_ENVIRONMENTS=true`, inspect the bridge log, and terminate the leaked `provider_sandbox_id`. |
| Daytona fallback reports Internet is restricted on Tier 1 or Tier 2 | Upgrade the Daytona organization to Tier 3 or higher; sandbox-level allowlists cannot override the organization policy. |
| Callback returns 401 | Confirm the same `MILES_SESSION_API_KEY` reaches Miles, Harbor monitor, and the Modal agent. |
| Callback returns a session 404/identity mismatch | Stop: traffic reached the wrong or restarted session server. |
| Shared filesystem is slow | Benchmark it and revisit filesystem size/type; do not assume capacity alone implies required bandwidth. |
| Head Pod restarts | Treat the training job/session state as interrupted; do not silently continue mixed rollouts. |
| Megatron rejects global batch divisibility | Make global batch size divisible by microbatch size times data-parallel size; do not change GPU resources to fix an arithmetic constraint. |
| Training initialization OOMs while inference fits | Recalculate TP/EP/DP and gradient/optimizer memory independently from rollout memory; the observed TP=4, EP=2, DP=2 layout needed a 26.44 GiB allocation with only 21.34 GiB free per H100. |

## 20. Stop and tear down safely

Stop the Ray job first and allow Harbor/Modal cleanup:

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
10. Modal workspace token ID/secret and expected maximum concurrency.
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

## 22. Known-good Nebius validation snapshot (2026-07-13)

The following state was validated end to end and is a useful baseline for a
new coding agent. IDs are project-specific; resolve them again rather than
copying them into a new project.

```text
project:       project-e00k7pwtpr00cv6jdxkrxb
region:        eu-north1
MK8S cluster:  mk8scluster-e00hqgby0p1dtsvyzp
GPU cluster:   computegpucluster-e00kga31xmnqr04909 (fabric-3)
node group:    mk8snodegroup-e00phqhzyz337r0scj (2 x 8gpu-128vcpu-1600gb)
filesystem:    computefilesystem-e00kmzw35erg3cz659 (1 TiB, filesystem-i8)
PVC:           miles/miles-workspace (RWX, mounted at /workspace)
```

The durable development Deployment is `miles/miles-dev`, with `Recreate`
strategy and these effective resources on its sole Pod:

```yaml
requests:
  cpu: "120"
  memory: 1400Gi
  nvidia.com/gpu: "8"
limits:
  cpu: "120"
  memory: 1400Gi
  nvidia.com/gpu: "8"
```

It is privileged only because this controlled training Pod needs the exposed
RDMA devices, and it mounts a 256 GiB memory-backed `/dev/shm`. The same
resource shape was used for the temporary `miles-worker-validation` Pod on the
second node. Keep the second Pod under a reviewed Deployment/Job for a real
run; the validation Pod is intentionally disposable.

Reconfirm the state before a run:

```bash
kubectl --context nebius-miles get nodes -o wide
kubectl --context nebius-miles get pods -n miles -o wide
kubectl --context nebius-miles get pvc -n miles
kubectl --context nebius-miles exec -n miles deployment/miles-dev -- nvidia-smi -L
kubectl --context nebius-miles exec -n miles deployment/miles-dev -- df -h /workspace /dev/shm
```

The successful gates were: repository preflight and 18 experimental contract
tests; eight-GPU CUDA and single-node NCCL; a two-node `/workspace` sentinel;
two-node NCCL with `WORLD_SIZE=16` and approximately 415 GB/s bus bandwidth;
two-node Ray with eight GPU actors on each Pod; and one authenticated Daytona
Harbor oracle returning `Submitted` with reward `1.0`. The GLM-4.7-Flash HF
checkpoint and `torch_dist` conversion were staged under `/workspace/models`.

A later `debug_rollout_only` attempt loaded GLM-4.7-Flash in SGLang across
eight H100s, launched the Miles router and session server, selected two
Terminal-Bench 2 tasks (`cobol-modernization` and `path-tracing-reverse`), and
created both Daytona sandboxes. This verified the real callback/task path
through sandbox agent launch, but the run was stopped before either complete
trajectory returned. A separate normal-mode attempt reached Megatron model
initialization but did not complete a generation or policy update because of
the training-memory constraint documented in section 18. Therefore gate 9 is
part of the current scope and remains unverified; the existing validation
snapshot is a reproducible baseline, not a completed experiment.

If a node returns `NotEnoughResources`, inspect the capacity advisor before
changing Kubernetes resources. The error occurs before the node joins MK8S;
do not lower the Pod limits or mix fabrics for a distributed NCCL run. A new
GPU cluster and node group on another fabric is a migration option only when
that fabric reports capacity for both nodes.

## 23. Research OS instrumentation and Harbor artifact capture

Research OS writes use the Probe SDK/CLI. Keep its write token in the
Kubernetes secret only; never put it in this file, shell history, logs, run
configuration, or capture manifests.

The SDK gives an MLflow/W&B-like lifecycle:

```python
from probe import Client

with Client() as probe:
    run = probe.run(
        project="miles-nebius",
        experiment="swe-agent-v2-nebius",
        hypothesis="A full-resource two-node H100 MK8S deployment can pass the Harbor oracle and distributed communication gates.",
        name="nebius-e2e-<utc-timestamp>",
        external_id="nebius-e2e-<stable-id>",
        tags=["nebius", "mk8s", "h100", "harbor", "modal"],
    )
    run.snapshot(cwd="/workspace/miles", include_env=True, include_gpu=True)
    run.log({"nccl_bus_gbps": 414.57, "harbor_reward": 1.0}, step=0,
            kind="validation")
    run.log({"gpu_count": 8, "node_count": 2}, kind="hardware")
    run.link(nebius_cluster="<cluster-id>", node_group="<node-group-id>",
             gpu_cluster="<gpu-cluster-id>")
    run.finish("completed", summary={"callback_rollout": "not_run"})
```

For a training run, prefer the additive Miles backend instead of inserting
SDK calls into rollout code:

```bash
export MILES_USE_PROBE=1
export PROBE_PROJECT=miles-nebius
export PROBE_EXPERIMENT=swe-agent-v2-nebius
export PROBE_EXTERNAL_ID='<stable Nebius/Ray job ID>'
export PROBE_QUEUE_DIR=/workspace/probe/metrics
```

The Probe SDK integration queues every Miles scalar with its existing step,
event time, producer ID, and producer-local sequence before returning to
training. The primary process creates or resumes the run, captures the launch
snapshot and native IDs, and exports from the PVC in the background.

With the per-sample rail enabled (section 18), the RolloutManager process
also enqueues one labeled point per sample per step on the same queue under
its own producer identity; the single-lease exporter drains both rails in
order. On the dashboard (research-os >= v0.35.2.0) these appear in a series'
samples drawer, and a point whose sample id matches a captured Harbor trial
carries a "trial" pill opening that trial's trajectory and sandbox-state view
in place. For the pill to bind, the bridge/glue must stamp the SAME ids on
the trial capture — pass `labels={"sample": <sample.index>, "group":
<sample.group_index>}` (via `run.unit(...)` or the export correlation's
`sample_id`); the reward point ↔ rollout span binding itself is automatic
(the SDK logs the trial reward with the span's id as exemplar pointer). API
initialization failures retain a complete run-creation intent; repair with
`python -m miles.utils.tracking_utils.probe_utils <queue-directory>`.
That command prints the resolved `run_id`. If bridge descriptors were created
before the run existed, bind and drain them with
`probe trial drain "$MILES_HARBOR_CAPTURE_DIR" --run <resolved-run-id>`; the
consumer rejects conflicting identities and persists the repair into each
descriptor.

The Harbor bridge receives that run ID automatically through Miles rollout
metadata and passes only the trial path plus native correlation to the SDK.
The SDK stages `trial/`, `trial.tar.gz`,
`capture-manifest.json`, and `export-request.json` under
`MILES_HARBOR_CAPTURE_DIR`. `probe trial watch` validates hashes and sizes,
creates the default Harbor trial span, uploads every regular file with its
rollout step/correlation metadata, and updates the local publication ledger.
No client-specific `manifest.json` edit and no ATIF conversion are required.

Harbor's `HARBOR_DELETE_ENVIRONMENTS=true` correctly cleaned the historical
Daytona oracle sandbox. The Modal gate in section 15 now requires the same
cleanup proof for the exact retained `provider_sandbox_id`. The durable upload
is the collected trial bundle (`agent/oracle.txt`, `result.json`, verifier
output, config/lock, and manifest), not the original sandbox filesystem. If a
future run requires reproducible sandbox inspection, set deletion off only for
a bounded debug trial, collect the sandbox directory explicitly, and delete it
after the upload. Uploading a directory requires archiving it first;
`log_artifact` uploads bytes when given a file path and otherwise records only
a reference.

Verified Research OS behavior after the 2026-07-13 agent and service fixes:

- a Harbor execution log uploaded through the presign/PUT/confirm flow as a
  complete, non-reference artifact and downloaded byte-for-byte with the same
  SHA-256;
- `run.snapshot()` created a content-addressed execution record containing
  code, dependencies, hardware, paths, and settings;
- the run's top-level `env_ref` persisted through `RunPatch` and resolved back
  to that execution record; and
- research notes and final run status/summary persisted normally.

Research OS agent PR #13 fixed forwarding server-provided artifact upload
headers and added `env_ref` persistence verification. The service-side
presigning and `RunPatch.env_ref` fixes must also be deployed; an updated agent
alone is insufficient against an older service.

Remaining instrumentation considerations:

1. The model callback has no stable HTTPS origin in the base MK8S deployment;
   Harbor can pass its oracle while a real agent turn still cannot call Miles.
2. A single high-level `snapshot` captures code, dependencies, GPU metadata,
   and an execution record, but not Kubernetes YAML, Pod events, NCCL logs, or
   cloud sandbox contents. Add those as explicit files and link the cluster,
   node-group, GPU-cluster, Pod, and filesystem IDs.
3. The hosted MCP health endpoint is reachable, but a direct streamable-HTTP
   probe returned `Session not found` on `tools/list`; restart Claude Code so
   the plugin loads its `.mcp.json`, then verify the `research_*` tools from
   inside Claude. Continue using the SDK/CLI for writes.
4. Metric points are append-only; dimensions are the bounded grouping axes
   (<=8 keys) and labels the unbounded per-sample ids. Log per-rank/per-node
   metrics with dimensions rather than emitting thousands of unique keys;
   per-sample values go in labels (the per-sample rail does this for you) and
   count against the run's labeled-point budget (`labeled_point_budget`,
   default 2M — size it as num_rollout x rollout_batch_size x
   n_samples_per_prompt x keys). Flush/finish the run even after a failed
   rollout.
5. Research OS artifact upload is content-addressed and remote, but large
   checkpoints and full filesystem trees should remain on the durable PVC;
   upload manifests, logs, trial bundles, and checksums rather than copying
   hundreds of gigabytes into the tracker.
6. A drained metric queue proves publication only for records observed in that
   queue. Miles does not yet expose a reliable expected-Ray-producer set or a
   close barrier for every secondary tracker, so the ledger reports capture
   completeness as `unknown` and names those missing guarantees instead of
   claiming completeness.

## Official references

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
- [Connect to MK8S with kubectl](https://docs.nebius.com/kubernetes/connect)
- [Supported Kubernetes versions](https://docs.nebius.com/kubernetes/versions)
- [Kubernetes kubectl version-skew policy](https://kubernetes.io/releases/version-skew-policy/#kubectl)
