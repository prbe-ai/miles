#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
ok=1

[ -f /app/added.txt ] || ok=0
[ "$(cat /app/existing.txt 2>/dev/null)" = "baseline v2 (edited)" ] || ok=0
[ -f /app/data/blob.bin ] || ok=0
[ -L /app/link ] || ok=0
[ ! -e /app/todelete.txt ] || ok=0

printf '%s\n' "$ok" > /logs/verifier/reward.txt
[ "$ok" = "1" ]
