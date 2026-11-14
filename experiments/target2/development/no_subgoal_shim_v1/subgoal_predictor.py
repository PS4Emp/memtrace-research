from typing import Optional, Tuple, Any
from pathlib import Path
from env_runner import EnvRunner
from utils import EpisodeState

class SubgoalPredictorBase:
    def __init__(
        self,
        args,
        save_dir: Path,
    ):
        self.args = args
        self.save_dir = save_dir
        self.video_buffer = []
        self.episode_dir: Optional[str] = None
        
        self.setup_api()

    def setup_api(self) -> None:
        pass

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.env_name = env_runner.env_id
        self.episode_id = env_runner.episode_id
        self.task_goal = env_runner.task_goal
        self.env_runner = env_runner

    def step(self, epstate: EpisodeState) -> None:
        pass

    def maybe_extend_video(self, images: list) -> None:
        pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # return (subgoal_str, has_api_error)
        raise NotImplementedError

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass

class NullSubgoalPredictor(SubgoalPredictorBase):
    def get_subgoal(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        return None, False

def build_subgoal_predictor(args, save_dir: Path):
    if any((
        args.use_gemini,
        args.use_qwenvl,
        args.use_memer,
        args.use_oracle,
    )):
        raise RuntimeError(
            "External subgoal predictors are forbidden "
            "in Target 2 no-subgoal evaluation."
        )
    if args.subgoal_type is not None:
        raise RuntimeError(
            "Target 2 no-subgoal configuration requires "
            "subgoal_type=None."
        )
    return NullSubgoalPredictor(args, save_dir)
