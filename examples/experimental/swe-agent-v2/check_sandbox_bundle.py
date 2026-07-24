#!/usr/bin/env python3
"""Validate probe.sandbox-state/1 bundles in a staged Harbor capture.

Doubles as the sandbox-capture self-check for the oracle smoke step: point it
at MILES_HARBOR_CAPTURE_DIR (or a single trial dir) after a trial has staged,
and it asserts that each ``probe-sandbox-state`` bundle is present, complete,
and integrity-verified.

Stdlib only — runs on the Nebius head pod without probe-research installed.

    python3 check_sandbox_bundle.py <capture-dir-or-trial-dir> [--latest] [--require-integrity]

Exit codes: 0 = all found bundles healthy; 1 = a bundle is missing/invalid;
2 = no bundle found at all (nothing to check).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

SCHEMA = "probe.sandbox-state/1"
BUNDLE_DIRNAME = "probe-sandbox-state"
MANIFESTS = ("begin-manifest.jsonl.gz", "end-manifest.jsonl.gz")


def _gzip_lines(path: str) -> int:
    n = 0
    with gzip.open(path, "rb") as handle:
        for _ in handle:
            n += 1
    return n


def check_bundle(bundle_dir: str, *, require_integrity: bool) -> list[str]:
    """Return a list of problems with one bundle; empty means healthy."""
    problems: list[str] = []
    meta_path = os.path.join(bundle_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return [f"{bundle_dir}: no meta.json (capture never completed)"]
    try:
        meta = json.loads(open(meta_path).read())
    except (OSError, ValueError) as exc:
        return [f"{meta_path}: unreadable ({exc})"]

    if meta.get("schema") != SCHEMA:
        problems.append(f"{meta_path}: schema={meta.get('schema')!r} (want {SCHEMA})")

    status = meta.get("status") or {}
    for phase in ("begin", "end"):
        if status.get(phase) != "ok":
            problems.append(f"{bundle_dir}: {phase} phase status={status.get(phase)!r}")

    integrity = meta.get("integrity") or {}
    bad = [name for name, ok in integrity.items() if not ok]
    if bad and require_integrity:
        problems.append(f"{bundle_dir}: integrity failed for {bad}")
    elif bad:
        problems.append(f"{bundle_dir}: WARNING integrity unverified for {bad}")

    # Files exist and manifests are non-empty + parseable.
    for name in MANIFESTS:
        p = os.path.join(bundle_dir, name)
        if not os.path.isfile(p):
            problems.append(f"{bundle_dir}: missing {name}")
            continue
        try:
            if _gzip_lines(p) == 0:
                problems.append(f"{bundle_dir}: {name} is empty")
        except OSError as exc:
            problems.append(f"{bundle_dir}: {name} unreadable ({exc})")

    if meta.get("limits", {}).get("truncated"):
        dropped = meta["limits"].get("dropped_count") or len(meta["limits"].get("dropped", []))
        problems.append(f"{bundle_dir}: WARNING scan truncated ({dropped} dropped) — capture is partial")

    return problems


def find_bundles(root: str) -> list[str]:
    # Match the bundle DIRECTORY, not meta.json — so a bundle whose capture died
    # before writing meta.json is still found and flagged as failed (rather than
    # silently reported as "no bundle").
    if os.path.basename(root.rstrip("/")) == BUNDLE_DIRNAME:
        return [root]
    return sorted(
        p for p in glob.glob(os.path.join(root, "**", BUNDLE_DIRNAME), recursive=True)
        if os.path.isdir(p)
    )


def summarize(bundle_dir: str) -> str:
    meta_path = os.path.join(bundle_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return "(no meta.json — capture incomplete)"
    try:
        meta = json.loads(open(meta_path).read())
    except (OSError, ValueError):
        return "(meta.json unreadable)"
    s = meta.get("summary", {})
    lim = meta.get("limits", {})
    return (
        f"begin_files={s.get('begin_files')} "
        f"+{s.get('added')}/~{s.get('modified')}/-{s.get('deleted')} "
        f"arch={meta.get('tool', {}).get('arch')} "
        f"integrity={meta.get('integrity')} "
        f"truncated={lim.get('truncated')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="capture dir, trial dir, or a probe-sandbox-state dir")
    ap.add_argument("--latest", action="store_true", help="check only the most recently modified bundle")
    ap.add_argument("--require-integrity", action="store_true", help="fail (not warn) on any integrity mismatch")
    args = ap.parse_args()

    bundles = find_bundles(args.path)
    if not bundles:
        print(f"no probe.sandbox-state/1 bundle found under {args.path}", file=sys.stderr)
        return 2
    if args.latest:
        bundles = [max(bundles, key=lambda b: os.path.getmtime(os.path.join(b, "meta.json")))]

    failures = 0
    warnings = 0
    for bundle in bundles:
        problems = check_bundle(bundle, require_integrity=args.require_integrity)
        hard = [p for p in problems if "WARNING" not in p]
        warns = [p for p in problems if "WARNING" in p]
        status = "OK" if not hard else "FAIL"
        print(f"[{status}] {bundle}")
        print(f"        {summarize(bundle)}")
        for w in warns:
            print(f"        {w}")
            warnings += 1
        for p in hard:
            print(f"        {p}")
        failures += bool(hard)

    print(f"\n{len(bundles)} bundle(s): {len(bundles) - failures} ok, {failures} failed, {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
