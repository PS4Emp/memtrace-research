
import json
from pathlib import Path

import numpy as np


class Target2PolicyWrapper:
    """
    Target 2 development-only policy wrapper.

    Arms:
      native   - untouched policy inference
      identity - explicit native key positions
      age      - conservative online age-position rule

    Does not modify the underlying model, weights,
    FrameSamp selector, or upstream embeddings.
    """

    ARMS = ("native", "identity", "age")

    def __init__(self, policy, arm, log_path=None):
        if arm not in self.ARMS:
            raise ValueError(f"Invalid arm: {arm}")

        self.policy = policy
        self.arm = arm
        self.log_path = Path(log_path) if log_path else None

        cfg = policy.config
        self.token_budget = int(cfg.budget)
        self.token_per_image = int(cfg.token_per_image)
        self.num_views = int(cfg.num_views)

        self.tokens_per_frame = (
            self.token_per_image * self.num_views
        )

        if self.token_budget != 512:
            raise RuntimeError("Expected 512 memory tokens")

        if self.tokens_per_frame != 16:
            raise RuntimeError("Expected 16 tokens per frame")

        self.capacity = (
            self.token_budget // self.tokens_per_frame
        )
        assert self.capacity == 32

        self.calls = 0

    @property
    def metadata(self):
        return self.policy.metadata

    def reset(self):
        self.calls = 0
        return self.policy.reset()

    def add_buffer(self, obs):
        return self.policy.add_buffer(obs)

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def build_mapping(self):
        step = int(self.policy.step_idx)
        if step < 0:
            raise RuntimeError("No history available")

        selected = np.asarray(
            self.policy.mem_buffer.get_frame_sampling_indices(
                step,
                self.token_budget,
                self.token_per_image
            ),
            dtype=np.int64
        )

        n = len(selected)
        assert 1 <= n <= self.capacity
        assert np.all(np.diff(selected) > 0)
        assert selected[0] >= 0
        assert selected[-1] <= step

        native_slots = np.arange(
            self.capacity, dtype=np.int32
        )
        slots = native_slots.copy()

        if self.arm == "age" and n == self.capacity:
            ages = step - selected
            old = np.log1p(ages[:16].astype(float))
            span = old[0] - old[-1]

            if span > 0:
                slots[:16] = np.rint(
                    15 * (old[0] - old) / span
                ).astype(np.int32)

        offsets = np.arange(
            self.tokens_per_frame, dtype=np.int32
        )

        keys = (
            slots[:, None] * self.tokens_per_frame
            + offsets[None, :]
        ).reshape(1, self.token_budget)

        assert keys.dtype == np.int32
        assert keys.min() >= 0
        assert keys.max() < self.token_budget

        active_mask = np.repeat(
            np.arange(self.capacity) < n,
            self.tokens_per_frame
        )

        original_keys = np.arange(
            self.token_budget, dtype=np.int32
        )

        changed_active = int(np.count_nonzero(
            keys[0, active_mask]
            != original_keys[active_mask]
        ))

        assert np.array_equal(
            keys[0, ~active_mask],
            original_keys[~active_mask]
        )

        if n < self.capacity:
            assert changed_active == 0

        if n == self.capacity:
            assert np.array_equal(
                slots[16:], native_slots[16:]
            )

        diagnostics = {
            "arm": self.arm,
            "step_idx": step,
            "selected_frames": selected.tolist(),
            "active_frames": n,
            "active_tokens": int(active_mask.sum()),
            "changed_active_tokens": changed_active,
            "frame_collisions": int(
                n - len(np.unique(slots[:n]))
            ),
            "native_slots": native_slots.tolist(),
            "assigned_slots": slots.tolist(),
            "min_key_position": int(keys.min()),
            "max_key_position": int(keys.max()),
        }

        return keys, diagnostics

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("Expected observation dictionary")

        # Reject accidental interference from another experiment.
        reserved = (
            "mem_k_positions",
            "mem_frame_permutation",
            "mem_reduction_order",
            "temporal_pos_override"
        )
        for key in reserved:
            if key in obs:
                raise ValueError(
                    f"Target 2 wrapper forbids external {key}"
                )

        request = dict(obs)

        if self.arm == "native":
            diagnostics = {
                "arm": "native",
                "step_idx": int(self.policy.step_idx),
                "changed_active_tokens": 0
            }
        else:
            positions, diagnostics = self.build_mapping()

            request["mem_k_positions"] = positions

            # Identical numerical execution interface for
            # the identity and age arms.
            request["mem_frame_permutation"] = np.arange(
                self.capacity, dtype=np.int32
            )

            request["mem_reduction_order"] = np.arange(
                self.token_budget, dtype=np.int32
            )[None, :]

        result = self.policy.infer(request)
        self.calls += 1

        if self.log_path is not None:
            self.log_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            diagnostics["policy_call"] = self.calls

            with self.log_path.open("a") as f:
                f.write(json.dumps(diagnostics) + "\n")

        return result
