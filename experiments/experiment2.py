import typing

import gym
import tella

from experiments.agents import Dqn, make_cnn
from experiments.curriculums.minigrid import (
    MiniGridCrossing,
    LavaCrossingSteCurriculum,
    EmptyCrossingSteCurriculum,
    ObstacleCrossingSteCurriculum,
)


class DqnCnn(Dqn):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            make_cnn(),
            rng_seed,
            observation_space,
            action_space,
            num_envs,
            config_file,
        )


def train_stes():
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=LavaCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=6,
        log_dir="logs",
    )
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=EmptyCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=6,
        log_dir="logs",
    )
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=ObstacleCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=6,
        log_dir="logs",
    )


if __name__ == "__main__":
    # train_stes()
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="logs",
    )
