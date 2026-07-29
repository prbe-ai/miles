"""GLM-4.7-Flash fully-async agentic training with SWE-bench data.

Disaggregated fully-async variant for agentic tasks: training and rollout run
on separate nodes concurrently. Uses train_async.py and the fully_async_rollout
module so that weight updates do not block generation. Agent tasks are dispatched
to a Harbor-based agent server.

GLM-4.7-Flash architecture: 47 layers, 20 attention heads, 64 routed experts,
hidden_size=2048, first_k_dense_replace=1. TP must divide 20 (valid: 1,2,4,5).
Default split: 1 node training + 7 nodes inference (configurable via
--train-num-nodes), sized for an 8-node job.

Data preparation (run separately before training):
    python download_and_process_data.py \\
        --input SWE-bench/SWE-bench_Verified \\
        --output /root/swe_train.jsonl \\
        --agent-name mini-swe-agent --split test

Usage:
    python run-glm47-flash-agentic-async.py --num-nodes 8
    python run-glm47-flash-agentic-async.py --num-nodes 8 --train-num-nodes 1
    python run-glm47-flash-agentic-async.py --num-nodes 8 \\
        --agent-server-url http://ts-egress-aws-agent-server:8080
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent
FULLY_ASYNC_DIR = (Path(__file__).resolve().parent.parent.parent / "fully_async").resolve()

# Cluster-wide GPU-node ceiling for the ckpt-conversion job. Kept below the
# raw node count so ckpt conversion doesn't starve the rest of the cluster.
MAX_CONVERT_GPUS = 92


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only", "debug_train_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = "/models/zai-org/GLM-4.7-Flash"
    ref_load: str = "/models/zai-org/GLM-4.7-Flash_torch_dist"
    save_dir: str = "/root/GLM-4.7-Flash_agentic_async/"
    # Directory to dump rollout + training traces (per-rollout .pt files). Empty
    # means default to ``<save_dir>/traces``; set to ``"disabled"`` to skip.
    save_traces_dir: str = ""
    load_debug_rollout_data: str = ""
    prompt_data: str = "/root/swe_train.jsonl"
    max_seq_len: int = 16384
    rollout_max_response_len: int = 8192
    save_interval: int = 5

    # Rollout / training batch sizing (overridable for smoke tests)
    num_rollout: int = 3000
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 4
    global_batch_size: int = 32
    over_sampling_batch_size: int = 64
    custom_rollout_log_function_path: str = os.environ.get(
        "MILES_CUSTOM_ROLLOUT_LOG_FUNCTION_PATH", ""
    )

    # Rollout precision
    rollout_fp8: bool = False
    rollout_health_check_first_wait: int = 1800

    # Agent settings
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://ts-egress-aws-agent-server:8080")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    agent_server_auth_token: str = field(
        default=os.environ.get(
            "AGENT_SERVER_AUTH_TOKEN", os.environ.get("MILES_HARBOR_AUTH_TOKEN", "")
        ),
        metadata={"sensitive": True},
    )
    agent_server_timeout_sec: float = float(os.environ.get("AGENT_SERVER_TIMEOUT_SEC", "14400"))
    session_server_port: int = int(os.environ.get("MILES_SESSION_SERVER_PORT", "30000"))
    session_server_bind_ip: str = os.environ.get("MILES_SESSION_SERVER_BIND_IP", "")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    router_external_base_url: str = os.environ.get("MILES_ROUTER_EXTERNAL_BASE_URL", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # Disaggregated fully-async settings
    train_num_nodes: int = 1
    pause_generation_mode: Literal["in_place", "retract"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast", "p2p"] = "broadcast"
    accumulate_allreduce_grads_in_fp32: bool = False
    max_tokens_per_gpu: int = 8192
    optimizer_cpu_offload: bool = True
    use_precision_aware_optimizer: bool = True
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 1
    decoder_last_pipeline_num_layers: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int = 1

    # W&B settings
    wandb_key: str = field(
        default=os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", "")),
        metadata={"sensitive": True},
    )
    wandb_project: str = os.environ.get("WANDB_PROJECT", "glm47-flash-agentic")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-flash-swe-async"
    disable_wandb_random_suffix: bool = True

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "glm47-flash-swe-async"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "train_async.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        # Bracket-wrap the first char so the pgrep pattern doesn't match its
        # own shell/subprocess command line (which literally contains the
        # bracketed pattern and thus fails the regex).
        pattern = f"[{t[0]}]{t[1:]}"
        subprocess.run(
            f"pgrep -f '{pattern}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format."""
    max_convert_nodes = MAX_CONVERT_GPUS // args.num_gpus_per_node
    convert_nodes = min(args.num_nodes, max_convert_nodes)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        num_nodes=convert_nodes,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    if args.pause_generation_mode == "in_place" and args.update_weight_transfer_mode == "p2p":
        raise ValueError(
            "in_place + p2p is not supported: P2P transfer engine conflicts with "
            "active NCCL inference. Use broadcast with in_place, or retract with p2p."
        )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path fully_async_rollout.generate_rollout_fully_async "
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--pause-generation-mode {args.pause_generation_mode} "
    )
    if args.load_debug_rollout_data:
        rollout_args += f"--load-debug-rollout-data {args.load_debug_rollout_data} "
    if args.custom_rollout_log_function_path:
        rollout_args += (
            "--custom-rollout-log-function-path "
            f"{args.custom_rollout_log_function_path} "
        )

    eval_args = ""

    # Disaggregated split: training on train_num_nodes, inference on the rest.
    rollout_num_nodes = args.num_nodes - args.train_num_nodes
    assert rollout_num_nodes > 0, (
        f"train_num_nodes ({args.train_num_nodes}) must be less than "
        f"num_nodes ({args.num_nodes}) to leave room for inference"
    )
    train_gpus = args.train_num_nodes * args.num_gpus_per_node
    rollout_gpus = rollout_num_nodes * args.num_gpus_per_node
    print(
        f"Disagg split: {args.train_num_nodes} nodes ({train_gpus} GPUs) training, "
        f"{rollout_num_nodes} nodes ({rollout_gpus} GPUs) inference"
    )

    # Flash has 20 attention heads, so TP must divide 20. Keep the legacy
    # TP=4/PP=1 defaults, but expose the full training topology so 80-GiB H100
    # runs can use Miles' checked-in TP=2/PP=2/CP=2/EP=4 layout.
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    cp = args.context_parallel_size
    etp = args.expert_tensor_parallel_size
    model_parallel_size = tp * pp * cp
    assert 20 % tp == 0, f"GLM-4.7-Flash attention heads (20) must be divisible by TP ({tp})"
    assert train_gpus % model_parallel_size == 0, (
        f"train GPUs ({train_gpus}) must be divisible by TP*PP*CP "
        f"({tp}*{pp}*{cp}={model_parallel_size})"
    )
    dp = train_gpus // model_parallel_size
    num_experts = 64
    ep = args.expert_model_parallel_size
    if ep is None:
        ep = max(d for d in range(1, dp + 1) if num_experts % d == 0 and dp % d == 0)
    assert num_experts % ep == 0, f"experts ({num_experts}) must be divisible by EP ({ep})"
    assert train_gpus % (etp * ep * pp) == 0, (
        f"train GPUs ({train_gpus}) must be divisible by ETP*EP*PP "
        f"({etp}*{ep}*{pp}={etp * ep * pp})"
    )

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        f"--context-parallel-size {cp} "
        f"--expert-model-parallel-size {ep} "
        f"--expert-tensor-parallel-size {etp} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    if args.decoder_last_pipeline_num_layers is not None:
        perf_args += (
            f"--decoder-last-pipeline-num-layers "
            f"{args.decoder_last_pipeline_num_layers} "
        )
    if args.optimizer_cpu_offload:
        perf_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    if args.use_precision_aware_optimizer:
        perf_args += "--use-precision-aware-optimizer "

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # SGLang: single-node engines with DP-attention. Flash has 20 attention
    # heads so TP=8 crashes (20 % 8 != 0); we use attn_tp=4, attn_dp=2 while
    # MoE stays 8-way TP/EP across the 8 GPUs in each rollout node.
    sglang_nodes_per_engine = 1
    sglang_world_size = sglang_nodes_per_engine * args.num_gpus_per_node
    num_engines = rollout_num_nodes // sglang_nodes_per_engine
    assert rollout_num_nodes % sglang_nodes_per_engine == 0, (
        f"rollout nodes ({rollout_num_nodes}) must be divisible by "
        f"sglang_nodes_per_engine ({sglang_nodes_per_engine})"
    )
    print(f"Inference: {num_engines} engines x {sglang_world_size} GPUs/engine")
    sglang_decode_max_bs = 256
    sglang_attn_tp_size = 4
    assert sglang_world_size % sglang_attn_tp_size == 0, (
        f"sglang world ({sglang_world_size}) must be divisible by " f"attn_tp_size ({sglang_attn_tp_size})"
    )
    sglang_attn_dp_size = sglang_world_size // sglang_attn_tp_size

    sglang_p2p_extra = ""
    if args.update_weight_transfer_mode == "p2p":
        sglang_p2p_extra = "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.80 "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--use-miles-router "
        "--sglang-router-port 31000 "
        f"{sglang_p2p_extra}"
    )
    sglang_extra_env_vars: dict[str, str] = {}

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--tito-model glm47 "
        "--use-session-server "
        f"--session-server-port {args.session_server_port} "
        "--tito-allowed-append-roles user tool "
    )
    if args.session_server_bind_ip:
        agent_args += f"--session-server-bind-ip {args.session_server_bind_ip} "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.train_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_gpus} "
        "--grad-reduce-in-bf16 "
        "--use-fault-tolerance "
        "--pin-rollout-manager-to-head "
        f"--rollout-health-check-first-wait {args.rollout_health_check_first_wait} "
    )
    if args.accumulate_allreduce_grads_in_fp32:
        misc_args += "--accumulate-allreduce-grads-in-fp32 "

    traces_dir = args.save_traces_dir or f"{args.save_dir.rstrip('/')}/traces"
    if traces_dir != "disabled":
        misc_args += f"--dump-details {traces_dir} "

    if args.mode == "debug_rollout_only":
        debug_args = "--debug-rollout-only --debug-rollout-only-disaggregated "
    elif args.mode == "debug_train_only":
        if not args.load_debug_rollout_data:
            raise ValueError("debug_train_only requires --load-debug-rollout-data")
        debug_args = "--debug-train-only "
    else:
        debug_args = ""

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.wandb_run_name} "
            f"--wandb-key {args.wandb_key} "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "
        if args.disable_wandb_random_suffix:
            wandb_args += "--disable-wandb-random-suffix "

    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus "
            f"--prometheus-port {args.prometheus_port} "
            f"--prometheus-run-name {args.prometheus_run_name} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{eval_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{FULLY_ASYNC_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK", "0"),
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "AGENT_SERVER_AUTH_TOKEN": args.agent_server_auth_token,
        "AGENT_SERVER_TIMEOUT_SEC": str(args.agent_server_timeout_sec),
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        **U.collect_probe_runtime_env(),
        **sglang_extra_env_vars,
    }
    if args.router_external_host:
        extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    if args.router_external_base_url:
        extra_env_vars["MILES_ROUTER_EXTERNAL_BASE_URL"] = args.router_external_base_url
    if session_api_key := os.environ.get("MILES_SESSION_API_KEY"):
        extra_env_vars["MILES_SESSION_API_KEY"] = session_api_key
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
