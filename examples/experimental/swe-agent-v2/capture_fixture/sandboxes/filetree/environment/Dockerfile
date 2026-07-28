FROM ubuntu:24.04

WORKDIR /app

# Baseline files the solution modifies / deletes, so the delta exercises the
# "modified" and "deleted" classes (not just "added").
RUN printf 'baseline v1\n' > /app/existing.txt \
    && printf 'delete me\n' > /app/todelete.txt
