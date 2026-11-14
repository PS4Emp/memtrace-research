#!/usr/bin/env bash
set -Eeuo pipefail
cd "/workspace/MEMTRACE_RunPod_Target1_Rebuild"
source ./environment.sh
exec "$MEMTRACE_POLICY_PY" -u "/workspace/memtrace/target2/development/target2_server_v1.py" \
    --arm identity \
    --port 18111 \
    --output "/workspace/memtrace/target2/development/identity_server_v1" \
    >> "/workspace/memtrace/target2/development/identity_server_v1.log" 2>&1
