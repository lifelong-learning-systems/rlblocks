import typing

import gym
import tella
import sys
import os

sys.path.append(os.path.abspath(os.curdir))

from experiments.agents import Dqn, DqnEwc, make_cnn
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


class DqnEwcCnn(DqnEwc):
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
        num_parallel_envs=5,
        log_dir="logs",
    )
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=EmptyCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=5,
        log_dir="logs",
    )
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=ObstacleCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=5,
        log_dir="logs",
    )


def train_dqn():
    tella.rl_experiment(
        agent_factory=DqnCnn,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=5,
        log_dir="logs",
    )


def train_dqn_ewc():
    tella.rl_experiment(
        agent_factory=DqnEwcCnn,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=5,
        log_dir="logs",
    )


if __name__ == "__main__":
    # train_stes()
    # train_dqn()
    train_dqn_ewc()
