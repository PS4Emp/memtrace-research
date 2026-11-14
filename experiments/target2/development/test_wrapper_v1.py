
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from target2_policy_wrapper_v1 import Target2PolicyWrapper


class FakeBuffer:
    def get_frame_sampling_indices(
        self, step, token_budget, token_per_image
    ):
        capacity = token_budget // (token_per_image * 2)
        if step < capacity:
            return list(range(step + 1))
        return np.linspace(
            0, step, capacity, dtype=np.int32
        ).tolist()


class FakePolicy:
    def __init__(self, step):
        self.config = SimpleNamespace(
            budget=512, token_per_image=8, num_views=2
        )
        self.mem_buffer = FakeBuffer()
        self.step_idx = step
        self.metadata = {"test": True}
        self.last_request = None

    def infer(self, obs):
        self.last_request = dict(obs)
        return {
            "actions": np.zeros((20, 8), dtype=np.float32)
        }

    def reset(self):
        self.step_idx = -1

    def add_buffer(self, obs):
        self.step_idx += len(obs["images"])


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)

    for step in (0, 15, 30, 31, 32, 100, 1200):
        records = {}

        for arm in ("native", "identity", "age"):
            fake = FakePolicy(step)

            wrapper = Target2PolicyWrapper(
                fake, arm,
                log_path=root / f"{arm}_{step}.jsonl"
            )

            original_obs = {"prompt": "test task"}
            result = wrapper.infer(original_obs)

            assert result["actions"].shape == (20, 8)
            assert original_obs == {"prompt": "test task"}

            request = fake.last_request
            records[arm] = request

            if arm == "native":
                assert "mem_k_positions" not in request
            else:
                assert request["mem_k_positions"].shape == (1, 512)
                np.testing.assert_array_equal(
                    request["mem_frame_permutation"],
                    np.arange(32)
                )
                np.testing.assert_array_equal(
                    request["mem_reduction_order"],
                    np.arange(512)[None, :]
                )

        native_keys = np.arange(512)[None, :]
        np.testing.assert_array_equal(
            records["identity"]["mem_k_positions"],
            native_keys
        )

        changed = int(np.count_nonzero(
            records["age"]["mem_k_positions"]
            != native_keys
        ))

        if step < 31:
            assert changed == 0
        else:
            assert changed > 0

        for arm in ("identity", "age"):
            line = (root / f"{arm}_{step}.jsonl").read_text()
            diagnostic = json.loads(line)
            assert diagnostic["step_idx"] == step

        print(
            f"step={step:4d}  "
            f"age_changed_tokens={changed:3d}  "
            "PASS"
        )

    print("\nTARGET2_WRAPPER_UNIT_TEST_PASS")
    print("All three arms construct valid inference requests.")
    print("No trained checkpoint loaded.")
    print("No robot rollouts executed.")
