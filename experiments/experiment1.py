import typing

import gym
import tella

from experiments.agents import Dqn, make_mlp


class DqnMlp(Dqn):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            make_mlp(),
            rng_seed,
            observation_space,
            action_space,
            num_envs,
            config_file,
        )


if __name__ == "__main__":
    tella.rl_experiment(
        agent_factory=DqnMlp,
        curriculum_factory=tella.curriculum.curriculum_registry["CartPole-1000"],
        num_lifetimes=1,
        num_parallel_envs=10,
        log_dir="logs",
    )
