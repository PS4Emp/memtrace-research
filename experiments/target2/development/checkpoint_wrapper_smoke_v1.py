
import os
import sys
import json
import hashlib
from pathlib import Path

import numpy as np

root = Path("/workspace/memtrace")
repo = Path(os.environ["MEMTRACE_REPO"])
packet = Path(os.environ["MEMTRACE_PACKET"])
checkpoint = Path(os.environ["MEMTRACE_CKPT"])
freeze_file = Path(os.environ["MEMTRACE_FREEZE"])
assignment = Path(os.environ["MEMTRACE_ASSIGNMENT"])

output = root / "target2/development/checkpoint_wrapper_smoke_v1"

if output.exists():
    raise RuntimeError(
        f"Existing diagnostic output preserved: {output}"
    )

for name in (
    "MME_FRAMESAMP_OVERRIDE_PATH",
    "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"
):
    if os.environ.get(name):
        raise RuntimeError(
            f"Unexpected active intervention override: {name}"
        )

sys.path.insert(0, str(packet / "code"))
sys.path.insert(0, str(root / "target2/development"))

from run_target1 import bootstrap, sha, tree
bootstrap(repo)

import jax
import run_replication_a as ra
from mme_vla_suite.policies import policy_config
from mme_vla_suite.training import config
from target2_policy_wrapper_v1 import Target2PolicyWrapper

assert jax.default_backend() == "gpu"

# Check trained model and frozen research provenance.
frozen = json.loads(freeze_file.read_text())
assert tree(checkpoint) == frozen["checkpoint_files"]

source = (
    "src/mme_vla_suite/models/integration/history_gemma.py"
)
assert sha(repo / source) == frozen["source"][source]

ra.ASSIGNMENT_SOURCE = assignment
assert sha(assignment) == ra.ASSIGNMENT_SOURCE_SHA256

print("CHECKPOINT AND SOURCE VERIFIED")

output.mkdir(parents=True, exist_ok=False)

# Reuse development inputs from a frozen episode.
inputs = ra.build_replication_a_unit_inputs(
    "VideoUnmask",
    5,
    output / "environment",
    1300
)

policy = policy_config.create_trained_policy(
    config.get_config("mme_vla_suite"),
    checkpoint,
    seed=42
)

ra.validate_replication_a_policy(policy)

assert policy._model.integration_type == "modulation"

assembly = inputs["assemblies"]["far"]

policy.reset()
policy.add_buffer(
    ra.pack_buffer(
        assembly.images,
        assembly.states,
        assembly.exec_start_idx
    )
)

assert policy.step_idx >= 31

cache_before = ra.hash_buffer_feats(
    policy.mem_buffer._history_feats
)

wrappers = {
    arm: Target2PolicyWrapper(
        policy,
        arm,
        log_path=output / f"{arm}_calls.jsonl"
    )
    for arm in ("native", "identity", "age")
}

def run(arm):
    response = wrappers[arm].infer({
        **inputs["query_obs"],
        "reset_rng": True
    })

    actions = np.asarray(response["actions"])

    assert actions.shape == (20, 8)
    assert np.isfinite(actions).all()

    return actions

native = run("native")
identity = run("identity")
age = run("age")

# Repeat to verify deterministic inference.
native_repeat = run("native")
identity_repeat = run("identity")
age_repeat = run("age")

def exact(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == b.tobytes()
    )

identity_exact = exact(native, identity)
native_replay = exact(native, native_repeat)
identity_replay = exact(identity, identity_repeat)
age_replay = exact(age, age_repeat)

age_mae = float(np.mean(np.abs(age - native)))

cache_after = ra.hash_buffer_feats(
    policy.mem_buffer._history_feats
)

cache_unchanged = cache_before == cache_after

age_logs = [
    json.loads(line)
    for line in (output / "age_calls.jsonl")
        .read_text().splitlines()
    if line.strip()
]

changed = [
    row["changed_active_tokens"]
    for row in age_logs
]

report = {
    "status": "PASS" if (
        identity_exact
        and native_replay
        and identity_replay
        and age_replay
        and cache_unchanged
        and min(changed) > 0
        and age_mae > 0
    ) else "INVESTIGATE",
    "scope": "Trained checkpoint, development episode, open-loop",
    "episode": "VideoUnmask_ep5",
    "checkpoint": str(checkpoint),
    "native_identity_exact": identity_exact,
    "native_replay_exact": native_replay,
    "identity_replay_exact": identity_replay,
    "age_replay_exact": age_replay,
    "memory_cache_unchanged": cache_unchanged,
    "age_changed_active_tokens": changed,
    "native_vs_age_action_mae": age_mae
}

with (output / "report.json").open("x") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print("TARGET 2 — TRAINED-CHECKPOINT WRAPPER TEST")
print("=" * 60)
print("STATUS:", report["status"])
print("NATIVE == IDENTITY:", identity_exact)
print("NATIVE REPLAY EXACT:", native_replay)
print("IDENTITY REPLAY EXACT:", identity_replay)
print("AGE REPLAY EXACT:", age_replay)
print("MEMORY CACHE UNCHANGED:", cache_unchanged)
print("AGE CHANGED ACTIVE TOKENS:", changed)
print("NATIVE VS AGE ACTION MAE:", age_mae)
print("REPORT:", output / "report.json")

if report["status"] != "PASS":
    raise RuntimeError(
        "Checkpoint wrapper test requires investigation."
    )

print("\nTARGET2_CHECKPOINT_WRAPPER_PASS")
