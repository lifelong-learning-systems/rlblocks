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

import tella
from .dqn import Dqn, DqnEwc, DqnTaskMemory, DqnScp, DqnReservoir, DqnCoverage
from .curriculum import (
    MiniGridCrossing,
    LavaCrossingSteCurriculum,
    EmptyCrossingSteCurriculum,
    ObstacleCrossingSteCurriculum,
)


NUM_ENV = 10
AGENT_SEED = 12345
CURRICULUM_SEED = 54321


def train_stes():
    tella.rl_experiment(
        agent_factory=Dqn,
        curriculum_factory=LavaCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )
    tella.rl_experiment(
        agent_factory=Dqn,
        curriculum_factory=EmptyCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )
    tella.rl_experiment(
        agent_factory=Dqn,
        curriculum_factory=ObstacleCrossingSteCurriculum,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn():
    tella.rl_experiment(
        agent_factory=Dqn,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn_ewc():
    tella.rl_experiment(
        agent_factory=DqnEwc,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn_scp():
    tella.rl_experiment(
        agent_factory=DqnScp,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn_task_memory():
    tella.rl_experiment(
        agent_factory=DqnTaskMemory,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn_reservoir_replay():
    tella.rl_experiment(
        agent_factory=DqnReservoir,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


def train_dqn_coverage_replay():
    tella.rl_experiment(
        agent_factory=DqnCoverage,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=NUM_ENV,
        log_dir="logs",
        agent_seed=AGENT_SEED,
        curriculum_seed=CURRICULUM_SEED,
    )


if __name__ == "__main__":
    train_stes()
    train_dqn()
    train_dqn_ewc()
    train_dqn_task_memory()
    train_dqn_scp()
    train_dqn_reservoir_replay()
