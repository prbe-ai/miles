import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_DIR = ROOT / "examples/experimental/swe-agent-v2"


def _script_args_fields(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ScriptArgs":
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
    raise AssertionError(f"ScriptArgs not found in {path}")


def test_single_node_launcher_exposes_bounded_rollout_options():
    path = LAUNCHER_DIR / "run.py"
    source = path.read_text()

    assert {"num_rollout", "rollout_max_response_len"} <= _script_args_fields(path)
    assert 'f"--num-rollout {args.num_rollout} "' in source
    assert 'f"--rollout-max-response-len {args.rollout_max_response_len} "' in source


def test_launchers_expose_probe_per_sample_hook():
    for name in ("run.py", "run-glm47-flash-agentic-async.py"):
        path = LAUNCHER_DIR / name
        source = path.read_text()
        assert "custom_rollout_log_function_path" in _script_args_fields(path)
        assert "--custom-rollout-log-function-path " in source
        assert "collect_probe_runtime_env()" in source
