"""Fail-fast checks for the public-Harbor Runpod deployment.

Run ``--phase repo`` before renting GPUs. Run ``--phase node`` inside every
Instant Cluster Pod before starting Ray. The script never prints secret values.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, success: str, failure: str) -> None:
        if condition:
            print(f"PASS: {success}")
        else:
            print(f"FAIL: {failure}")
            self.failures.append(failure)

    def warn(self, condition: bool, success: str, warning: str) -> None:
        if condition:
            print(f"PASS: {success}")
        else:
            print(f"WARN: {warning}")
            self.warnings.append(warning)


def _contains(path: Path, marker: str) -> bool:
    return path.is_file() and marker in path.read_text()


def check_repo(report: Report, root: Path) -> None:
    files = {
        "runpod guide": root / "examples/experimental/swe-agent-v2/RUNPOD_E2E.md",
        "public Harbor bridge": root / "examples/experimental/swe-agent-v2/public_harbor_server.py",
        "Runpod requirements": root / "examples/experimental/swe-agent-v2/requirements-runpod.txt",
        "Runpod template environment example": root
        / "examples/experimental/swe-agent-v2/runpod-template.env.example",
        "single-node launcher": root / "examples/experimental/swe-agent-v2/run.py",
        "async launcher": root / "examples/experimental/swe-agent-v2/run-glm47-flash-agentic-async.py",
    }
    for label, path in files.items():
        report.check(path.is_file(), f"found {label}", f"missing {label}: {path}")

    for launcher_name in ("run.py", "run-glm47-flash-agentic-async.py"):
        launcher = SCRIPT_DIR / launcher_name
        report.check(
            _contains(launcher, "--pin-rollout-manager-to-head"),
            f"{launcher_name} pins RolloutManager to the Ray head",
            f"{launcher_name} does not pin RolloutManager to the Ray head",
        )
        report.check(
            _contains(launcher, "--session-server-bind-ip"),
            f"{launcher_name} configures a separate session bind address",
            f"{launcher_name} lacks the separate session bind address",
        )

    report.check(
        _contains(SCRIPT_DIR / "swe_agent_function.py", "MILES_ROUTER_EXTERNAL_BASE_URL"),
        "agent callback supports a full external origin",
        "agent callback lacks full external-origin support",
    )
    report.check(
        _contains(SCRIPT_DIR / "swe_agent_function.py", "MILES_SESSION_API_KEY"),
        "agent callback propagates the session bearer",
        "agent callback does not propagate the session bearer",
    )
    report.check(
        _contains(root / "miles/rollout/session/server.py", "require_session_authorization"),
        "Miles session server enforces configured bearer authentication",
        "Miles session server lacks bearer enforcement",
    )

    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report.warnings.append("could not inspect git status")
        print("WARN: could not inspect git status")
    else:
        report.warn(not status, "git worktree is clean", "git worktree has local changes; record them before deployment")


def _required_env(report: Report, name: str) -> str:
    value = os.getenv(name, "").strip()
    report.check(bool(value), f"Runpod provided {name}", f"required Runpod variable {name} is missing")
    return value


def _check_path(report: Report, label: str, raw_path: str) -> None:
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    report.check(path.exists(), f"{label} exists: {path}", f"{label} is missing: {path}")


def _check_free_space(report: Report, path: str, minimum_gb: int, label: str) -> None:
    if not Path(path).exists():
        return
    free_gb = shutil.disk_usage(path).free // (1024**3)
    report.check(
        free_gb >= minimum_gb,
        f"{label} has {free_gb} GiB free (minimum {minimum_gb} GiB)",
        f"{label} has only {free_gb} GiB free; require at least {minimum_gb} GiB",
    )


def check_node(report: Report, args: argparse.Namespace) -> None:
    rank = _required_env(report, "NODE_RANK")
    node_addr = _required_env(report, "NODE_ADDR")
    primary_addr = _required_env(report, "PRIMARY_ADDR")
    num_nodes = _required_env(report, "NUM_NODES")
    num_trainers = _required_env(report, "NUM_TRAINERS")
    world_size = _required_env(report, "WORLD_SIZE")
    _required_env(report, "RUNPOD_VOLUME_ID")

    report.check(rank.isdigit(), "NODE_RANK is numeric", f"invalid NODE_RANK: {rank!r}")
    report.check(
        all(value.isdigit() for value in (num_nodes, num_trainers, world_size)),
        "cluster size variables are numeric",
        "NUM_NODES, NUM_TRAINERS, and WORLD_SIZE must be numeric",
    )
    if all(value.isdigit() for value in (num_nodes, num_trainers, world_size)):
        report.check(
            int(num_nodes) * int(num_trainers) == int(world_size),
            "WORLD_SIZE equals NUM_NODES * NUM_TRAINERS",
            "Runpod cluster size variables are inconsistent",
        )

    report.check(
        os.getenv("NCCL_SOCKET_IFNAME") == "ens1",
        "NCCL uses ens1",
        "set NCCL_SOCKET_IFNAME=ens1 in the template",
    )
    report.check(
        os.getenv("GLOO_SOCKET_IFNAME") == "ens1",
        "Gloo uses ens1",
        "set GLOO_SOCKET_IFNAME=ens1 in the template",
    )
    report.check(Path("/sys/class/net/ens1").exists(), "ens1 exists", "Runpod high-speed interface ens1 is missing")
    report.check(Path("/workspace").is_dir(), "/workspace exists", "/workspace is missing")
    report.check(os.access("/workspace", os.W_OK), "/workspace is writable", "/workspace is not writable")
    _check_free_space(report, "/", args.min_container_free_gb, "container disk")
    _check_free_space(report, "/workspace", args.min_workspace_free_gb, "network volume")
    for command in ("git", "uv", "ray", "nvidia-smi", "curl", "rsync"):
        report.check(shutil.which(command) is not None, f"found {command}", f"required command is missing: {command}")

    gpu_probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    report.check(
        gpu_probe is not None and gpu_probe.returncode == 0 and "GPU " in gpu_probe.stdout,
        "NVIDIA GPUs are visible",
        "nvidia-smi cannot see any GPUs",
    )

    python_probe = subprocess.run(
        [sys.executable, "-c", "import miles, ray, sglang, torch; assert torch.cuda.is_available()"],
        capture_output=True,
        text=True,
    )
    report.check(
        python_probe.returncode == 0,
        "Miles, Ray, SGLang, PyTorch, and CUDA import successfully",
        "Miles/Ray/SGLang/PyTorch import or CUDA availability check failed",
    )

    if rank == "0" and args.callback_mode == "direct":
        _required_env(report, "RUNPOD_PUBLIC_IP")
        _required_env(report, "RUNPOD_TCP_PORT_70000")

    _check_path(report, "Miles checkout", args.miles_root)
    _check_path(report, "HF checkpoint", args.hf_checkpoint)
    _check_path(report, "converted Megatron checkpoint", args.ref_load)
    _check_path(report, "Megatron-LM checkout", args.megatron_path)
    _check_path(report, "Harbor task export", args.harbor_tasks_dir)

    role = "PRIMARY / Ray head" if rank == "0" else f"WORKER rank {rank}"
    print(f"INFO: role={role} node_addr={node_addr} primary_addr={primary_addr}")
    print("INFO: network-volume attachment is proven only for this Pod.")
    print("INFO: complete the cross-Pod sentinel-file test in RUNPOD_E2E.md before starting Ray.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("repo", "node"), default="repo")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--callback-mode", choices=("direct", "relay"), default="direct")
    parser.add_argument("--miles-root", default=os.getenv("MILES_ROOT", "/workspace/miles"))
    parser.add_argument("--hf-checkpoint", default=os.getenv("HF_CHECKPOINT", ""))
    parser.add_argument("--ref-load", default=os.getenv("REF_LOAD", ""))
    parser.add_argument("--megatron-path", default=os.getenv("MEGATRON_PATH", ""))
    parser.add_argument("--harbor-tasks-dir", default=os.getenv("HARBOR_TASKS_DIR", ""))
    parser.add_argument("--min-container-free-gb", type=int, default=20)
    parser.add_argument("--min-workspace-free-gb", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    if args.phase == "repo":
        check_repo(report, Path(args.repo_root).expanduser().resolve())
    else:
        check_node(report, args)

    print(f"SUMMARY: failures={len(report.failures)} warnings={len(report.warnings)}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
