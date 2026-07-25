#!/usr/bin/env bash
set -euo pipefail

# added
printf 'brand new\n' > /app/added.txt
# modified
printf 'baseline v2 (edited)\n' > /app/existing.txt
# binary, nested
mkdir -p /app/data
head -c 4096 /dev/urandom > /app/data/blob.bin
# symlink
ln -sf /app/added.txt /app/link
# deleted
rm -f /app/todelete.txt
