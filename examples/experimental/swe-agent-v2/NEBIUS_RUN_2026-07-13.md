# Nebius RL iteration run log — 2026-07-13

This log records the attempt to run one iteration of the public-Harbor
SWE-Agent V2 example from commit
`2439d401b43d250da1cbaa8e2bfc72f45bfa10a7` inside the supplied Nebius
Kubernetes workload. Secret values are intentionally omitted.

## Intended gate

Start with one `debug_rollout_only` Mini-SWE-Agent trial, then run a normal
training iteration only after the Nebius and public-Harbor gates pass. The
operator runbook explicitly prohibits normal training before those gates.

## Observed environment

- Current Pod: `miles-dev-5c4db8f886-x6qwx`, namespace `miles`.
- Repository: detached at
  `2439d401b43d250da1cbaa8e2bfc72f45bfa10a7`; initially clean.
- Accelerator: 8 NVIDIA H100 80 GB GPUs visible to `nvidia-smi`.
- Host-visible CPU/RAM: `nproc`/`free` report 128 CPUs and approximately 1.5
  TiB RAM, but the Pod cgroup actually limits it to 8 CPUs and 32 GiB RAM with
  no swap. Ray correctly discovers 8 CPUs and approximately 22.5 GB schedulable
  memory after reserving its object store.
- Shared path: `/workspace` is a writable 1 TiB filesystem with approximately
  1 TiB free.
- Container shared memory: `/dev/shm` is only 16 GiB, versus the runbook's
  256 GiB `emptyDir` target.
- InfiniBand: `/dev/infiniband` and `/sys/class/infiniband` are absent in this
  Pod.
- Python stack: Miles 0.2.1, Ray 2.56.0, SGLang
  0.5.15.dev24+g2fdb655, PyTorch 2.11.0+cu130, Transformers 5.8.1, and
  Megatron-Core 0.16.0rc0.
- Megatron-LM exists at `/root/Megatron-LM`.
- No model checkpoint, converted Megatron checkpoint, Harbor task export,
  Harbor virtual environment, or prior run artifacts were staged initially.
- `kubectl`, `helm`, `nebius`, and `jq` are absent. Direct read-only Kubernetes
  API calls using the mounted service-account token return HTTP 403 for the
  current Pod, Nodes, Pods, Services, PVCs, and StorageClasses.
- No usable `DAYTONA_API_KEY`, Harbor/session bearer, W&B credential, callback
  URL, or Ray cluster variables are injected. A Daytona credential was later
  supplied through an unsafe channel and must be rotated before use. Only
  `MILES_ROOT` and an unrelated `NCCL_VERSION` are present among the relevant
  environment names.
- The documented `miles-head.miles.svc.cluster.local` and `miles-worker` DNS
  names do not currently resolve from this Pod.

## Checks completed

- `runpod_preflight.py --phase repo`: 14 passes, 0 failures, 0 warnings.
- The worktree was clean before this log was added.
- Pinned Harbor/Daytona environment created at
  `/workspace/venvs/harbor-0.18-daytona`.
- Experimental contract suite: 18 tests passed in 2.75 seconds.
- Public Terminal-Bench 2 export: 89 tasks under
  `/workspace/harbor/tasks/terminal-bench-2`.
- Miles datasets: 89-row `/workspace/data/tb2_all.jsonl` and one-row
  `/workspace/data/tb2_smoke.jsonl`; smoke task is
  `adaptive-rejection-sampler`.
- GLM-4.7-Flash HF snapshot: 58 files / 62.5 GB advertised, with all 48
  safetensor shards downloaded to
  `/workspace/models/zai-org/GLM-4.7-Flash`. `AutoConfig` and `AutoTokenizer`
  load successfully; model type is `glm4_moe_lite` and vocabulary size is
  154856 (padded Megatron vocabulary is 154880).
- CUDA: a matrix multiplication passed independently on all eight GPUs, and
  CUDA peer access is enabled between GPU 0 and each of the other seven.
- NVLink/NCCL: `nvidia-smi topo -m` reports NV18 links between every GPU pair;
  an eight-process, eight-GPU single-node NCCL all-reduce passed with the
  expected sum of 36.
- Harbor bridge process: `/health` returned `status=ok`, environment
  `daytona`, and concurrency 1 using the real TB2 export. An unauthenticated
  `/run` request was rejected with HTTP 401. This did not create a Daytona
  sandbox and does not replace the authenticated oracle gate.
- HF-to-Megatron conversion: the final low-memory eight-rank conversion
  succeeded. `/workspace/models/zai-org/GLM-4.7-Flash_torch_dist` is 57 GB,
  its tracker contains `release`, Megatron recognizes it as a
  `TorchDistLoadShardedStrategy` checkpoint, and its metadata contains 723
  tensors. A single-process PyTorch distributed-checkpoint read reproduced
  five sampled embedding rows exactly from the source safetensor, including
  the first and final rows.
- Ray: a one-node Ray 2.56.0 head started on the Pod IP, reported exactly one
  node and eight H100 GPUs, and passed the resource assertion. The command
  runner tears down descendant daemons at cell exit, so Ray must be started in
  the same long-lived process/session used to submit the real job.

The successful constrained-Pod conversion command was:

```bash
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint /workspace/models/zai-org/GLM-4.7-Flash \
  --save /workspace/models/zai-org/GLM-4.7-Flash_torch_dist \
  --low-memory-checkpoint-save
```

## Nebius and Kubernetes resource findings

The Nebius H100 MK8S preset is `8gpu-128vcpu-1600gb`: eight H100 GPUs, 128
vCPUs, and 1600 GiB RAM. The observed 8-CPU/32-GiB ceiling is therefore a Pod
or container resource limit (possibly a namespace `LimitRange` default), not
the physical capacity of that preset. Kubernetes enforces container CPU and
memory limits with cgroups; crossing the memory limit can invoke an OOM kill.

The durable fix is to update the owning controller's Pod template and recreate
the Pod. For the eight-GPU head and worker manifests, the runbook now proposes:

```yaml
resources:
  requests:
    cpu: "96"
    memory: 1Ti
    nvidia.com/gpu: "8"
  limits:
    cpu: "120"
    memory: 1400Gi
    nvidia.com/gpu: "8"
```

It also mounts a 256 GiB memory-backed `emptyDir` at `/dev/shm`. That tmpfs
usage counts toward the container memory limit, so it does not add 256 GiB on
top of the 1400 GiB limit. Before applying, an operator with cluster RBAC must
verify Node `allocatable`, namespace `LimitRange`/`ResourceQuota`, and the
controller that owns the current Pod. On the runbook's Kubernetes 1.33
baseline, plan for Pod replacement after the template change rather than
assuming newer in-place resize support.

An eight-GPU Pod occupies an entire H100 node. A rolling controller update with
`maxSurge: 1` can require a temporary third eight-GPU node. If quota and budget
do not allow that, use a reviewed maintenance strategy such as `Recreate` or
`maxSurge: 0`, `maxUnavailable: 1`, after stopping Ray and Harbor and confirming
that `/workspace` is durable.

Useful operator checks are now included in `NEBIUS_E2E.md`; the minimum set is:

```bash
kubectl get nodes \
  -o custom-columns='NAME:.metadata.name,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl get limitrange,resourcequota -n miles -o yaml
kubectl get pod -n miles miles-head -o yaml
kubectl exec -n miles miles-head -- cat /sys/fs/cgroup/memory.max
kubectl exec -n miles miles-head -- cat /sys/fs/cgroup/cpu.max
kubectl exec -n miles miles-head -- df -h /dev/shm
```

Official references:

- [Nebius GPU node groups](https://docs.nebius.com/kubernetes/gpu/set-up)
- [Nebius GPU VM presets](https://docs.nebius.com/compute/virtual-machines/types)
- [Nebius node-group updates](https://docs.nebius.com/kubernetes/node-groups/manage)
- [Kubernetes container resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Kubernetes volumes and memory-backed `emptyDir`](https://kubernetes.io/docs/concepts/storage/volumes/)

## Docker and VM options

Docker is supported on Nebius Compute VMs. It is not exposed by the current
MK8S Pod: there is no Docker CLI or socket, and the Pod lacks the privileges
needed to run a nested Docker daemon. Docker-in-Docker would require a
privileged Pod, daemon storage, more memory/PIDs, and a security review; it
would not remove the current Pod's 32 GiB cgroup limit.

Two workable Docker topologies are documented in `NEBIUS_E2E.md`:

1. Run the standalone Nebius Compute VM path from
   `NEBIUS_STANDALONE_E2E.md`, with Miles and Docker on ordinary VMs. This is
   the simplest Docker-native topology when MK8S scheduling is unnecessary.
2. Keep Miles on MK8S and put Harbor plus its Docker task containers on a
   separate CPU Compute VM in the same region/subnet. Set
   `HARBOR_ENVIRONMENT_TYPE=docker`, expose only the Miles session callback
   through an internal Nebius LoadBalancer, and protect it with the existing
   session key, security groups, and private TLS/DNS as required. This design
   replaces Daytona rather than running alongside it.

For the current MK8S proof, a rotated Daytona credential remains the least
invasive path because it does not require privileged Pods or another VM. It
still needs an externally reachable authenticated session callback.

Official references:

- [Nebius containers over Compute VMs](https://docs.nebius.com/compute/virtual-machines/containers)
- [Nebius container applications on VMs](https://docs.nebius.com/compute/virtual-machines/applications-containers)
- [Nebius internal Kubernetes LoadBalancers](https://docs.nebius.com/kubernetes/clusters/load-balancer)

## Issues and unknowns

1. **The supplied workload is not the runbook topology.** It is an existing
   single `miles-dev` Pod, not the paired `miles-head` and `miles-worker` Pods.
   Cluster topology and the cross-node shared-filesystem sentinel cannot be
   verified with the current RBAC.
2. **No InfiniBand device is exposed.** A one-node smoke may still be possible,
   but the required two-node NCCL/GPUDirect gate cannot pass from this Pod.
3. **The session callback is unknown.** A Daytona sandbox needs a stable,
   externally reachable callback to the Miles session server. No LoadBalancer
   address or TLS relay is configured or discoverable.
4. **The Daytona credential must be rotated and injected.** A value was pasted
   into chat, so it must be treated as exposed and revoked. The replacement
   must be delivered through the operator's secret manager or a Kubernetes
   Secret, never committed or pasted into this log. This service account is
   forbidden from reading or creating Secrets, so an operator with appropriate
   RBAC must perform that step.
5. **Model and data staging is complete, but does not prove training fits.**
   The public TB2 tasks, HF snapshot, and converted `torch_dist` checkpoint are
   durable under `/workspace`. The model conversions fit only after the
   low-memory workaround; the RL optimizer remains blocked by the cgroup cap.
6. **`/dev/shm` is undersized relative to the documented deployment.** Confirm
   whether this Pod can be replaced or patched with a 256 GiB memory-backed
   volume before loading the model.
7. **Exact cluster evidence is missing.** Node count, per-node GPU allocation,
   prior Nebius NCCL results, CSI/PVC configuration, GPU fabric, quota, and
   teardown ownership are not visible from this service account.
8. **The image checkout and supplied checkout initially conflicted.** The
   image had Miles 0.2.1 installed editable from `/root/miles`. The first
   conversion attempt therefore failed with `ImportError` for
   `configure_logger_raw`. The documented editable install of
   `/workspace/miles` initially hit Debian PEP 668; repeating it with
    `--break-system-packages` replaced only the existing Miles editable install
    and made imports resolve to the supplied commit. Conversion was then
    restarted. The runbook should call out the possible PEP 668 flag for this
    image or make the launcher prepend the source checkout to `PYTHONPATH`.
9. **Single-node NIC evidence is inconsistent but explainable.**
    `nvidia-smi topo -m` names eight `mlx5` NICs, while device and sysfs
    visibility are absent inside the Pod. This indicates PCI topology alone
    is not proof that RDMA devices were passed into the workload.
10. **Host resource tools are misleading inside this Pod.** `free` and `nproc`
    show node totals, while cgroup v2 reports `memory.max=34359738368`,
    `memory.swap.max=0`, and `cpu.max=800000 100000`. Use cgroup limits or Ray
    discovery when deciding whether a run fits.
11. **The documented conversion path exceeds this Pod's memory limit.** A
    one-rank conversion loaded the model on GPU 0 but was OOM-killed during
    checkpoint save. The launcher's eight-rank pipeline conversion also hit
    the 32 GiB ceiling in Megatron's `filesystem_async.preload_tensors`; ranks
    0 and 7 were SIGKILLed and `memory.events:oom_kill` increased from 1 to 3.
    The incomplete outputs had no `release` tracker and were removed before
    retrying.
12. **A low-memory conversion workaround was needed.** A new optional
    `--low-memory-checkpoint-save` path in
    `tools/convert_hf_to_torch_dist.py` uses PyTorch's synchronous filesystem
    writer with a 10 MB copy-ahead window instead of staging an entire local
    shard in RAM. The eight-rank retry completed and produced loadable
    `torch_dist` metadata. This is an operational workaround, not evidence
    that 32 GiB is sufficient for RL training. It relies on private Megatron
    conversion helpers and therefore needs retesting when Megatron or PyTorch
    is upgraded.
13. **The RL optimizer step cannot be treated as safe at 32 GiB.** This example
    enables optimizer CPU offload for a roughly 30.6-billion-parameter model.
    Removing or materially raising the Pod memory limit is required before a
    normal iteration; the target Nebius Pod spec relies on the 1.6 TiB node and
    does not set this development Pod's 32 GiB cap.

## Safety decisions

- No chargeable Nebius infrastructure was created or deleted.
- No normal training job will be started while the documented fabric,
  callback, Harbor, model, and one-rollout gates remain unproven.
- Credentials will be checked only by presence and never printed.

## Work in progress

- Rotate the exposed Daytona credential, then inject the replacement as
  `DAYTONA_API_KEY` without recording it here.
- Obtain an externally reachable authenticated TLS callback origin for the
  Miles session server, or sufficient Kubernetes RBAC to inspect and configure
  the callback Service.
- Remove or materially raise the Pod's 32 GiB memory limit and expand
  `/dev/shm` to the runbook's 256 GiB target.
- Obtain visibility of the intended second GPU Pod/node and prior NCCL result,
  then pass the shared-filesystem sentinel and two-node Ray gates.
- Run the authenticated oracle, one real model rollout, and only then the first
  monitored RL optimizer iteration.

## Resume inputs required from the operator

1. Update the owning controller and replace the workload so each training
   container requests 96 CPUs/1 TiB, is limited to 120 CPUs/1400 GiB, and
   mounts a 256 GiB memory-backed `/dev/shm`, after confirming Node allocatable
   and namespace quota. All downloaded data and checkpoints are already
   durable under `/workspace`.
2. Rotate the exposed Daytona credential and inject only its replacement into
   the head Pod as `DAYTONA_API_KEY`, without putting it in Git, this log, or
   chat.
3. Provide the reviewed HTTPS callback origin for the Miles session server, or
   RBAC and the intended LoadBalancer/relay design needed to configure it.
4. Identify the second eight-GPU Pod/node and provide the official Nebius
   two-node NCCL result (including average bus bandwidth and any warnings), or
   grant read access needed to verify those gates directly.

After those inputs are present, resume at sections 14–18 of `NEBIUS_E2E.md`:
start the two-node Ray cluster; start the Harbor bridge at concurrency 1; pass
the authenticated oracle; pass one `debug_rollout_only` Mini-SWE-Agent trial;
pass the two-node fully async trial through TLS; then launch normal mode and
watch the first rollout, GRPO step 0, trace, checkpoint, and sandbox cleanup.

## Follow-up after the full-node rollout — 2026-07-13

This section supersedes the old Pod-resource, credential, and InfiniBand
observations above. The historical failures remain recorded because they
explain why the Deployment was replaced and why the low-memory conversion path
was added.

### Live Kubernetes state

- Deployment `miles-dev` uses strategy `Recreate` and has one healthy Pod,
  `miles-dev-68cd5f597d-wvqhx`, on the only Ready GPU node.
- Container requests and limits are identical: 120 CPUs, 1400 GiB memory, and
  eight `nvidia.com/gpu` devices. Kubernetes therefore assigns Guaranteed QoS.
- The Pod is privileged, mounts the shared `miles-workspace` PVC at
  `/workspace`, and mounts a 256 GiB memory-backed `emptyDir` at `/dev/shm`.
- Cgroup v2 is visible as a host-wide hierarchy in the privileged container.
  The effective container path must be resolved from `/proc/self/cgroup`; the
  root `/sys/fs/cgroup/{cpu,memory}.max` paths are not valid in this layout.
  At the delegated path, `cpu.max=12000000 100000`,
  `memory.max=1503238553600`, `memory.swap.max=0`, and all OOM counters were
  zero.
- `/workspace` passed a write/read/delete probe and had approximately 908 GiB
  free. The second-node cross-PVC sentinel remains blocked until another node
  is Ready.
- `DAYTONA_API_KEY`, `MILES_HARBOR_AUTH_TOKEN`,
  `AGENT_SERVER_AUTH_TOKEN`, and `MILES_SESSION_API_KEY` are all injected from
  Kubernetes Secrets. Only presence was checked; values were not printed.

### Single-node gates passed

- Eight NVIDIA H100 80 GB GPUs are visible. PyTorch 2.11.0+cu130 reports CUDA
  13.0; Ray 2.56.0 and SGLang 0.5.15.dev24+g2fdb655 import successfully.
- `/dev/infiniband` exposes eight `uverbs`, `umad`, and `issm` devices plus
  `rdma_cm`. `mlx5_0` through `mlx5_7` all reported `ACTIVE` at 400 Gb/s
  (4X NDR).
- `nvidia-smi topo -m` reports NV18 between every GPU pair and maps eight
  Mellanox NICs.
- Fresh matrix multiplication on every GPU and an eight-rank NCCL all-reduce
  passed with the expected sum of 36. This proves single-node CUDA/NVLink/NCCL,
  not cross-node GPUDirect RDMA.
- A temporary one-node Ray head reported exactly 120 CPUs, eight GPUs,
  approximately 1.36 TiB schedulable memory, and a 200 GB object store. Eight
  simultaneous one-GPU Ray actors were assigned unique devices 0 through 7
  and completed CUDA work. Ray was stopped after the gate.
- `runpod_preflight.py --phase repo` passed 14 checks with zero failures or
  warnings. The relevant experimental contract suite passed all 18 tests, and
  Ruff passed the bridge, agent adapter, and tests (apart from an existing
  top-level-settings deprecation warning).
- The staged data still contains 89 Terminal-Bench 2 tasks and a one-row smoke
  dataset for `adaptive-rejection-sampler`. The GLM-4.7-Flash HF checkpoint
  loads as `glm4_moe_lite` with vocabulary size 154856, and the converted
  57 GB `torch_dist` checkpoint has its `release` tracker and metadata.

### Authenticated Daytona oracle passed

The bridge was bound only to `127.0.0.1:18081` at concurrency one for this
gate; no public Service or load balancer was created.

- `/health` returned the expected Daytona configuration.
- An unauthenticated `/run` request returned HTTP 401.
- The authenticated `adaptive-rejection-sampler` oracle created a real Daytona
  sandbox, executed the task and verifier, returned HTTP 200 with
  `exit_status=Submitted` and reward `1.0`, and wrote complete Harbor trial
  artifacts.
- The bridge process was stopped afterward. A Daytona API listing returned
  zero live sandboxes, confirming provider cleanup.
- Non-secret artifacts are under
  `/workspace/harbor/trials/adaptive-rejection-sampler__Dfnf3b3` and
  `/workspace/logs/public-harbor-oracle-20260713T185127Z.json`.

The oracle deliberately does not call the Miles model endpoint. A real
Mini-SWE-Agent rollout still needs a reviewed externally reachable callback
origin for the session server.

### Capacity finding and remaining gates

The node group still targets two regular `8gpu-128vcpu-1600gb` VMs on
`fabric-3`, but only `computeinstance-e00z8dmjvrx3svrpx2` is Ready. The second
managed instance repeatedly failed Compute placement with
`NotEnoughResources` before entering another `STARTING` reconciliation.

At the 2026-07-13 18:44 UTC capacity check, the project quota allowed four
regular eight-GPU VMs, while the advisor exposed no immediately available
regular eight-H100 placement on fabrics 2, 3, 4, or 6. Fabric 2 and fabric 4
reported high preemptible availability; fabric 3 did not. This is provider
physical capacity, not a Miles Pod limit and not evidence of insufficient
account credits.

Do not remove the current fabric assignment or delete the working node. For a
stable two-node job, obtain regular capacity or a reservation. If an
interruption-tolerant preemptible run is acceptable, create a separate GPU
cluster and two-node group on a fabric with capacity so both nodes share one
InfiniBand topology.

The remaining hard gates are:

1. two Ready GPU nodes in one GPU cluster/fabric;
2. the official two-node NCCL/InfiniBand test;
3. a cross-node shared-filesystem sentinel;
4. 16-GPU/two-node Ray and the fully asynchronous rollout path;
5. an authenticated externally reachable TLS session callback;
6. one real Mini-SWE-Agent rollout through that callback; and
7. only then, a monitored normal-mode optimizer iteration.
