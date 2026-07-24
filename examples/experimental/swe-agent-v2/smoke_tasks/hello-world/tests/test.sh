#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

if [ -f /app/hello.txt ] && [ "$(cat /app/hello.txt)" = "Hello, world!" ]; then
  printf '1\n' > /logs/verifier/reward.txt
  exit 0
fi

printf '0\n' > /logs/verifier/reward.txt
exit 1
