
from pathlib import Path
import json
import hashlib
import numpy as np
import jax
import jax.numpy as jnp

from mme_vla_suite.models.integration import history_gemma as hg

EXPECTED_SOURCE = Path(
    "/workspace/memtrace/source/policy/src/"
    "mme_vla_suite/models/integration/history_gemma.py"
)

actual_source = Path(hg.__file__).resolve()
assert actual_source == EXPECTED_SOURCE.resolve(), (
    f"Unexpected attention source: {actual_source}"
)

B, T, S, D = 1, 3, 8, 1024

x = jax.random.normal(jax.random.key(101), (B, T, D))
memory = jax.random.normal(jax.random.key(102), (B, S, D))
mask = jnp.ones((B, S), dtype=bool)

# Identical reduction order in both conditions.
reduction_order = jnp.arange(S)[None, :]

native_positions = np.arange(S, dtype=np.int32)[None, :]
intervened_positions = np.array(
    [[2, 3, 0, 1, 6, 7, 4, 5]],
    dtype=np.int32
)

module = hg.MemoryAttention()

# Initialize parameters once, before enabling instrumentation.
variables = module.init(
    jax.random.key(103),
    x,
    memory,
    mask,
    mem_k_positions=None,
    mem_reduction_order=reduction_order
)

original_rope = hg._apply_rope
captures = {}
current = {"case": None}

def sha256_tensor(array):
    a = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(a.tobytes()).hexdigest()

def traced_rope(tensor, positions):
    tag = "key" if tensor.shape[1] == S else "query"

    def record(stage, p, t):
        case = current["case"]
        captures[case][f"{tag}_{stage}"] = {
            "positions": np.asarray(p).tolist(),
            "tensor_sha256": sha256_tensor(t),
            "shape": list(np.asarray(t).shape)
        }

    jax.debug.callback(
        lambda p, t: record("before", p, t),
        positions,
        tensor,
        ordered=True
    )

    rotated = original_rope(tensor, positions=positions)

    jax.debug.callback(
        lambda p, t: record("after", p, t),
        positions,
        rotated,
        ordered=True
    )

    return rotated

def run_case(name, position_override):
    current["case"] = name
    captures[name] = {}

    def forward():
        return module.apply(
            variables,
            x,
            memory,
            mask,
            mem_k_positions=position_override,
            mem_reduction_order=reduction_order
        )

    output = jax.jit(forward)()
    output.block_until_ready()
    return np.asarray(output)

try:
    hg._apply_rope = traced_rope

    native_output = run_case("native", None)
    corrected_output = run_case(
        "intervened",
        jnp.asarray(intervened_positions)
    )

finally:
    hg._apply_rope = original_rope

n = captures["native"]
i = captures["intervened"]

# Confirm all expected runtime callbacks executed.
expected_keys = {
    "query_before", "query_after",
    "key_before", "key_after"
}
assert set(n) == expected_keys
assert set(i) == expected_keys

# Position assignments must be exactly as requested.
assert n["key_before"]["positions"] == native_positions.tolist()
assert i["key_before"]["positions"] == intervened_positions.tolist()

# Queries must be unaffected.
assert n["query_before"] == i["query_before"]
assert n["query_after"] == i["query_after"]

# Unrotated keys must be identical.
assert (
    n["key_before"]["tensor_sha256"]
    == i["key_before"]["tensor_sha256"]
)

# Rotated keys must change.
assert (
    n["key_after"]["tensor_sha256"]
    != i["key_after"]["tensor_sha256"]
)

output_mae = float(
    np.mean(np.abs(native_output - corrected_output))
)
assert output_mae > 0, "Key-position change had no output effect."

report = {
    "status": "ATTENTION_LAYER_RUNTIME_PASS",
    "scope": "JIT-compiled attention layer with synthetic inputs",
    "checkpoint_used": False,
    "source": str(actual_source),
    "jax_backend": jax.default_backend(),
    "native_key_positions": native_positions.tolist(),
    "intervened_key_positions": intervened_positions.tolist(),
    "queries_unchanged": True,
    "unrotated_keys_unchanged": True,
    "rotated_keys_changed": True,
    "memory_inputs_fixed": True,
    "reduction_order_fixed": True,
    "attention_output_mae": output_mae,
    "captures": captures
}

out = Path(
    "/workspace/memtrace/analysis/"
    "target1_hook_validation/layer_runtime_report.json"
)

if out.exists():
    raise FileExistsError(
        f"Report already exists; preserving prior evidence: {out}"
    )

out.write_text(json.dumps(report, indent=2))

print("=" * 60)
print("TARGET 1 — RUNTIME ATTENTION-HOOK VALIDATION")
print("=" * 60)
print("STATUS:", report["status"])
print("JAX BACKEND:", report["jax_backend"])
print("QUERIES UNCHANGED:", report["queries_unchanged"])
print("UNROTATED KEYS UNCHANGED:", report["unrotated_keys_unchanged"])
print("ROTATED KEYS CHANGED:", report["rotated_keys_changed"])
print("MEMORY INPUTS FIXED:", report["memory_inputs_fixed"])
print("REDUCTION ORDER FIXED:", report["reduction_order_fixed"])
print("OUTPUT MAE:", output_mae)
print("REPORT:", out)
