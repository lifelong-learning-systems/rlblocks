"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import typing
import gym
import tella
import sys
import os

sys.path.append(os.path.abspath(os.curdir))

from experiments.agents import Dqn, DqnEwc, make_cnn, DqnTaskMemory
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


class DqnTaskMemoryCnn(DqnTaskMemory):
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


def train_dqn_task_memory():
    tella.rl_experiment(
        agent_factory=DqnTaskMemoryCnn,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=5,
        log_dir="logs",
    )


if __name__ == "__main__":
    # train_stes()
    # train_dqn()
    train_dqn_ewc()
    # train_dqn_task_memory()
