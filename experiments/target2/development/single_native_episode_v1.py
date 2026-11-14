
import json
import os
import sys
import time
import traceback
from pathlib import Path

root = Path("/workspace/memtrace")
repo = root / "source/policy"
output = (
    root / "target2/development/rollouts"
    / "native_VideoUnmask_ep5_v1"
)

sys.path.insert(
    0, str(repo / "examples/robomme")
)

from eval import Args, EpisodeEvaluator
from env_runner import EnvRunner
from subgoal_predictor import build_subgoal_predictor
from utils import TASK_NAME_LIST

TASK = "VideoUnmask"
EPISODE = 5
MAX_STEPS = 1300

assert TASK in TASK_NAME_LIST
assert not output.exists(), "Existing output preserved"

output.mkdir(parents=True, exist_ok=False)
videos = output / "videos"
videos.mkdir()

args = Args(
    host="127.0.0.1",
    port=18111,
    obs_horizon=16,
    max_steps=MAX_STEPS,
    save_dir=str(output),
    use_history=True,
    policy_name="target2_native",
    model_seed=42,
    controlled_recency_condition=None,
    use_oracle=False,
    use_qwenvl=False,
    use_memer=False,
    use_gemini=False,
    subgoal_type=None
)

assert not any([
    args.use_oracle,
    args.use_qwenvl,
    args.use_memer,
    args.use_gemini
])

manifest = {
    "phase": "target2_development",
    "arm": "native",
    "task": TASK,
    "episode": EPISODE,
    "model_seed": 42,
    "max_steps": MAX_STEPS,
    "obs_horizon": 16,
    "controlled_recency": False,
    "external_subgoal_predictors": False,
    "held_out": False
}

(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print("TARGET 2 — SINGLE NATIVE CLOSED-LOOP EPISODE")
print("=" * 65)
print("Task:", TASK)
print("Episode:", EPISODE)
print("Server: 127.0.0.1:18111")
print("Maximum simulator steps:", MAX_STEPS)
print("External APIs: DISABLED")
print("Development only: YES")
print(flush=True)

runner = None
start = time.monotonic()

report = {
    **manifest,
    "status": "infrastructure_error",
    "official_outcome": None,
    "wall_seconds": None,
    "error": None
}

try:
    runner = EnvRunner(
        TASK,
        str(videos),
        max_steps=MAX_STEPS
    )

    print(
        "Available episodes:",
        runner.num_episodes,
        flush=True
    )

    assert 0 <= EPISODE < runner.num_episodes

    runner.make_env(EPISODE)

    predictor = build_subgoal_predictor(
        args, output
    )

    evaluator = EpisodeEvaluator(
        args, output
    )

    print("ROLLOUT_STARTED", flush=True)

    outcome = evaluator.eval_each_episode(
        runner,
        predictor,
        videos
    )

    report["official_outcome"] = outcome

    if outcome in ("success", "fail", "timeout"):
        report["status"] = "completed"
    elif outcome in ("error", "unknown"):
        report["status"] = "infrastructure_error"
    else:
        report["status"] = "unrecognized_outcome"

    print("OFFICIAL_OUTCOME:", outcome, flush=True)

except Exception:
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)

finally:
    report["wall_seconds"] = round(
        time.monotonic() - start, 3
    )

    if runner is not None:
        try:
            runner.close_env()
        except Exception:
            report["cleanup_error"] = traceback.format_exc()

    with (output / "result.json").open("x") as f:
        json.dump(report, f, indent=2)

    print("FINAL_STATUS:", report["status"], flush=True)
    print(
        "RESULT_FILE:",
        output / "result.json",
        flush=True
    )

if report["status"] != "completed":
    raise RuntimeError(
        "Episode did not produce a valid official outcome."
    )

print("TARGET2_NATIVE_EPISODE_COMPLETE", flush=True)
