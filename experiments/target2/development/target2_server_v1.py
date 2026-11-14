
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("native", "identity", "age"), required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

root = Path("/workspace/memtrace")
repo = Path(os.environ["MEMTRACE_REPO"])
packet = Path(os.environ["MEMTRACE_PACKET"])
checkpoint = Path(os.environ["MEMTRACE_CKPT"])
freeze_path = Path(os.environ["MEMTRACE_FREEZE"])

sys.path.insert(0, str(packet / "code"))
sys.path.insert(0, str(root / "target2/development"))

from run_target1 import bootstrap, sha, tree
bootstrap(repo)

import jax
from mme_vla_suite.policies import policy_config
from mme_vla_suite.training import config
from mme_vla_suite.serving import websocket_policy_server
from target2_policy_wrapper_v1 import Target2PolicyWrapper

assert jax.default_backend() == "gpu"

for name in (
    "MME_FRAMESAMP_OVERRIDE_PATH",
    "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"
):
    assert not os.environ.get(name), (
        f"Unexpected intervention environment variable: {name}"
    )

frozen = json.loads(freeze_path.read_text())

source_rel = "src/mme_vla_suite/models/integration/history_gemma.py"

assert sha(repo / source_rel) == frozen["source"][source_rel]
assert tree(checkpoint) == frozen["checkpoint_files"]

print("FROZEN CHECKPOINT AND ATTENTION SOURCE VERIFIED", flush=True)

policy = policy_config.create_trained_policy(
    config.get_config("mme_vla_suite"),
    checkpoint,
    seed=42
)

assert policy._model.integration_type == "modulation"

wrapped = Target2PolicyWrapper(
    policy,
    args.arm,
    log_path=args.output / "inference_calls.jsonl"
)

args.output.mkdir(parents=True, exist_ok=False)

manifest = {
    "status": "DEVELOPMENT_SERVER",
    "arm": args.arm,
    "checkpoint": str(checkpoint),
    "checkpoint_freeze": str(freeze_path),
    "checkpoint_frozen_file_count": len(frozen["checkpoint_files"]),
    "attention_source_sha256": sha(repo / source_rel),
    "host": "127.0.0.1",
    "port": args.port,
    "model_seed": 42,
    "target1_source_untouched": True
}

(args.output / "manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

server = websocket_policy_server.WebsocketPolicyServer(
    policy=wrapped,
    host="127.0.0.1",
    port=args.port,
    metadata=wrapped.metadata
)

print(
    f"TARGET2_SERVER_READY arm={args.arm} port={args.port}",
    flush=True
)

server.serve_forever()
