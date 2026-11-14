#!/usr/bin/env bash
set -Eeuo pipefail
cd "/workspace/MEMTRACE_RunPod_Target1_Rebuild"
source ./environment.sh
timeout --signal=TERM --kill-after=10s 3600s \
    "/workspace/memtrace/envs/sim/bin/python" -u "/workspace/memtrace/target2/development/single_age_episode_v1.py" \
    >> "/workspace/memtrace/target2/development/age_episode_v1.log" 2>&1
