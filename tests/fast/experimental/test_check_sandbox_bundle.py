"""Runbook self-check: check_sandbox_bundle.py (probe.sandbox-state/1 validator)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_MODULE = _ROOT / "examples/experimental/swe-agent-v2/check_sandbox_bundle.py"
_SPEC = importlib.util.spec_from_file_location("check_sandbox_bundle", _MODULE)
assert _SPEC and _SPEC.loader
csb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = csb
_SPEC.loader.exec_module(csb)


def _bundle(root: Path, *, meta: dict | None, manifests: bool = True) -> Path:
    d = root / "trial" / "artifacts" / "probe-sandbox-state"
    d.mkdir(parents=True)
    if manifests:
        for name in ("begin-manifest.jsonl.gz", "end-manifest.jsonl.gz"):
            with gzip.open(d / name, "wb") as h:
                h.write(b'{"p": "/a", "t": "f"}\n')
        (d / "end-delta.tar.gz").write_bytes(b"tar")
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return root


def _healthy_meta() -> dict:
    # SDK-shaped meta (probe-research >= 0.23.0 SandboxStateRecorder): phase
    # statuses are pending/ok/failed and integrity is the two booleans.
    return {
        "schema": "probe.sandbox-state/1",
        "tool": {"arch": "amd64"},
        "status": {"begin": "ok", "end": "ok"},
        "summary": {"begin_files": 92, "added": 2, "modified": 1, "deleted": 0},
        "integrity": {"begin_verified": True, "end_verified": True},
        "limits": {"truncated": False},
    }


def test_healthy_bundle_passes(tmp_path):
    root = _bundle(tmp_path, meta=_healthy_meta())
    assert _run(csb, [str(root), "--require-integrity"]) == 0


def test_incomplete_bundle_fails(tmp_path):
    # bundle dir exists but capture died before writing meta.json
    root = _bundle(tmp_path, meta=None)
    assert _run(csb, [str(root)]) == 1


def test_no_bundle_is_exit_2(tmp_path):
    assert _run(csb, [str(tmp_path / "nothing")]) == 2


def test_integrity_mismatch_warns_but_fails_when_required(tmp_path):
    meta = _healthy_meta()
    meta["integrity"]["end_verified"] = False
    root = _bundle(tmp_path, meta=meta)
    assert _run(csb, [str(root)]) == 0  # warn only
    assert _run(csb, [str(root), "--require-integrity"]) == 1  # hard fail


def test_bad_status_fails(tmp_path):
    meta = _healthy_meta()
    meta["status"]["end"] = "failed"
    root = _bundle(tmp_path, meta=meta)
    assert _run(csb, [str(root)]) == 1


def _run(mod, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = ["check_sandbox_bundle.py", *argv]
    try:
        return mod.main()
    finally:
        sys.argv = old
