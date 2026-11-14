
import os
import sys
import json
import hashlib
from pathlib import Path

import numpy as np

root = Path("/workspace/memtrace")
packet = Path(os.environ["MEMTRACE_PACKET"])
repo = Path(os.environ["MEMTRACE_REPO"])
checkpoint = Path(os.environ["MEMTRACE_CKPT"])
assignment = Path(os.environ["MEMTRACE_ASSIGNMENT"])
freeze_path = Path(os.environ["MEMTRACE_FREEZE"])
out = root / "analysis/target1_checkpoint_hook/checkpoint_probe_v1"

if out.exists():
    raise RuntimeError(
        f"Existing diagnostic evidence must be preserved: {out}"
    )

sys.path.insert(0, str(packet / "code"))

from run_target1 import bootstrap, sha, tree
from bridge_core import token_map, reduction_order

bootstrap(repo)

import jax
import jax.numpy as jnp
import run_replication_a as ra
from mme_vla_suite.policies import policy_config
from mme_vla_suite.training import config
from mme_vla_suite.models.integration import history_gemma as hg

assert jax.default_backend() == "gpu"
assert checkpoint.is_dir()
assert freeze_path.is_file()

# Verify the actual source and checkpoint against frozen provenance.
frozen = json.loads(freeze_path.read_text())

source_rel = (
    "src/mme_vla_suite/models/integration/history_gemma.py"
)
source_file = (repo / source_rel).resolve()

assert Path(hg.__file__).resolve() == source_file
assert sha(source_file) == frozen["source"][source_rel]
assert tree(checkpoint) == frozen["checkpoint_files"]

ra.ASSIGNMENT_SOURCE = assignment
assert sha(assignment) == ra.ASSIGNMENT_SOURCE_SHA256

formal = json.loads(
    (root / "results/target1_v1/seed42/formal/complete.json").read_text()
)
assert formal["status"] == "FORMAL_COMPLETE"
assert formal["protocol_sha256"] == (
    "bc46027c5906f311920926327161b5f92a416dccc1814d0e101ef041685433e0"
)

episode_key = "VideoUnmask_ep5"
assert episode_key in formal["units"]

out.mkdir(parents=True, exist_ok=False)

# Reconstruct one previously frozen formal episode.
inputs = ra.build_replication_a_unit_inputs(
    "VideoUnmask", 5, out / "environment", 1300
)

plans = {
    c: ra.build_row_intervention_plan(
        c, inputs["assemblies"], inputs["retained_manifest"]
    )
    for c in ("far", "middle", "recent")
}

source_condition = "middle"
source_plan = plans[source_condition]
source_entries = source_plan["retained_entries"]
far_entries = plans["far"]["retained_entries"]

native_positions = np.arange(512, dtype=np.int32)[None, :]
aligned_positions = np.asarray(
    token_map(source_entries, far_entries),
    dtype=np.int32
)

assert aligned_positions.shape == native_positions.shape
assert not np.array_equal(native_positions, aligned_positions)

matched_order = np.asarray(
    reduction_order(source_entries, far_entries),
    dtype=np.int32
)

assert matched_order.shape == (1, 512)

selection = out / "selection.json"
selection.write_text(json.dumps({
    "step_idx": int(source_plan["step_idx"]),
    "indices_to_load": source_plan["selected_indices"]
}, indent=2))

os.environ["MME_FRAMESAMP_OVERRIDE_PATH"] = str(selection.resolve())

# Instrument the production implementation BEFORE model construction.
original_rope = hg._apply_rope
records = {"native": {"key": [], "query": []},
           "aligned": {"key": [], "query": []}}
current = {"name": None}

def digest(array):
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.tobytes()).hexdigest()

def traced_rope(tensor, positions):
    is_key = positions.shape[-1] == 512
    kind = "key" if is_key else "query"

    rotated = original_rope(tensor, positions=positions)

    def record_callback(p, before, after):
        name = current["name"]
        if name not in records:
            raise RuntimeError("Unexpected callback without active case")

        records[name][kind].append({
            "positions": np.asarray(p).tolist(),
            "before_sha256": digest(before),
            "after_sha256": digest(after),
            "tensor_shape": list(np.asarray(before).shape)
        })

    jax.debug.callback(
        record_callback,
        positions,
        tensor,
        rotated,
        ordered=True
    )

    return rotated

hg._apply_rope = traced_rope

try:
    policy = policy_config.create_trained_policy(
        config.get_config("mme_vla_suite"),
        checkpoint,
        seed=42
    )

    ra.validate_replication_a_policy(policy)

    assert policy._model.integration_type == "modulation"
    assert policy.config.use_pos_emb
    assert not policy.config.use_state_emb

    assembly = inputs["assemblies"][source_condition]

    policy.reset()
    policy.add_buffer(
        ra.pack_buffer(
            assembly.images,
            assembly.states,
            assembly.exec_start_idx
        )
    )

    cache_before = ra.hash_buffer_feats(
        policy.mem_buffer._history_feats
    )

    def infer(name, key_positions):
        current["name"] = name

        trace = out / f"{name}_selection_trace.jsonl"
        os.environ["MME_FRAMESAMP_TRACE_PATH"] = str(trace.resolve())

        result = policy.infer({
            **inputs["query_obs"],
            "reset_rng": True,
            "temporal_pos_override": {"enabled": False},
            "mem_frame_permutation": np.arange(32, dtype=np.int32),
            "mem_k_positions": key_positions,
            "mem_reduction_order": matched_order
        })

        actions = np.asarray(result["actions"])
        assert actions.shape == (20, 8)
        assert np.isfinite(actions).all()

        jax.effects_barrier()

        trace_rows = [
            json.loads(line)
            for line in trace.read_text().splitlines()
            if line.strip()
        ]

        assert len(trace_rows) == 1
        assert (
            trace_rows[0]["indices_to_load"]
            == source_plan["selected_indices"]
        )

        return actions

    native_actions = infer("native", native_positions)
    aligned_actions = infer("aligned", aligned_positions)

finally:
    hg._apply_rope = original_rope
    os.environ.pop("MME_FRAMESAMP_OVERRIDE_PATH", None)
    os.environ.pop("MME_FRAMESAMP_TRACE_PATH", None)

cache_after = ra.hash_buffer_feats(
    policy.mem_buffer._history_feats
)

assert cache_before == cache_after

nq = records["native"]["query"]
aq = records["aligned"]["query"]
nk = records["native"]["key"]
ak = records["aligned"]["key"]

assert nq and aq and nk and ak, "Runtime callbacks did not execute"
assert len(nq) == len(aq)
assert len(nk) == len(ak)

# Query positions are constant across all diffusion steps.
assert all(
    a["positions"] == b["positions"]
    for a, b in zip(nq, aq)
)

# First query must be identical before intervention affects dynamics.
assert nq[0]["before_sha256"] == aq[0]["before_sha256"]
assert nq[0]["after_sha256"] == aq[0]["after_sha256"]

# Memory-key inputs must remain identical across paired inference.
assert all(
    a["before_sha256"] == b["before_sha256"]
    for a, b in zip(nk, ak)
)

# Verify actual RoPE position IDs, not merely caller arguments.
assert all(
    event["positions"] == native_positions.tolist()
    for event in nk
)
assert all(
    event["positions"] == aligned_positions.tolist()
    for event in ak
)

assert any(
    a["after_sha256"] != b["after_sha256"]
    for a, b in zip(nk, ak)
)

action_mae = float(np.mean(
    np.abs(native_actions - aligned_actions)
))
assert action_mae > 0

report = {
    "status": "TRAINED_CHECKPOINT_RUNTIME_PASS",
    "scope": "One frozen episode, matched paired policy inference",
    "episode": episode_key,
    "source_condition": source_condition,
    "target_key_alignment": "far",
    "model_checkpoint": str(checkpoint),
    "checkpoint_freeze": str(freeze_path),
    "checkpoint_frozen_file_count":
        len(frozen["checkpoint_files"]),
    "attention_source_sha256": sha(source_file),
    "formal_protocol_sha256":
        formal["protocol_sha256"],
    "gpu": [str(device) for device in jax.devices()],
    "query_callbacks_per_case": len(nq),
    "key_callbacks_per_case": len(nk),
    "native_positions": native_positions.tolist(),
    "aligned_positions": aligned_positions.tolist(),
    "position_assignments_changed":
        int(np.count_nonzero(
            native_positions != aligned_positions
        )),
    "query_positions_unchanged": True,
    "initial_query_tensors_unchanged": True,
    "all_unrotated_keys_unchanged": True,
    "rotated_keys_changed": True,
    "memory_cache_unchanged": True,
    "matched_reduction_order": matched_order.tolist(),
    "paired_action_mae": action_mae,
    "captures": records
}

report_path = out / "checkpoint_runtime_report.json"

with report_path.open("x") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TARGET 1 — TRAINED-CHECKPOINT MECHANISM VALIDATION")
print("=" * 65)
print("STATUS:", report["status"])
print("EPISODE:", episode_key)
print("GPU:", report["gpu"])
print("SOURCE AND CHECKPOINT FREEZE: VERIFIED")
print("POSITION ASSIGNMENTS CHANGED:",
      report["position_assignments_changed"])
print("QUERY POSITIONS UNCHANGED: True")
print("INITIAL QUERY TENSORS UNCHANGED: True")
print("ALL UNROTATED MEMORY KEYS UNCHANGED: True")
print("ROTATED MEMORY KEYS CHANGED: True")
print("MEMORY CACHE UNCHANGED: True")
print("PAIRED ACTION MAE:", action_mae)
print("REPORT:", report_path)
