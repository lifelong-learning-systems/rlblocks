from typing import *
import tella
from tella.curriculum import *


def default():
    env = gym.make("CartPole-v1")
    return env


def low_grav():
    env = default()
    env.gravity = 9.8 / 2
    return env


def low_force():
    env = default()
    env.force_mag = 10.0 / 2
    return env


TASKS = [default, low_grav, low_force]


class PhysicsCartPole(InterleavedEvalCurriculum):
    def learn_blocks(self) -> Iterable[AbstractLearnBlock]:
        for env_fn in TASKS:
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        env_fn,
                        rng_seed=self.rng.bit_generator.random_raw(),
                        task_label="CartPole",
                        variant_label=env_fn.__name__,
                        num_episodes=100,
                    )
                ]
            )

    def eval_block(self) -> AbstractEvalBlock:
        return simple_eval_block(
            [
                EpisodicTaskVariant(
                    env_fn,
                    rng_seed=0,
                    task_label="CartPole",
                    variant_label=env_fn.__name__,
                    num_episodes=100,
                )
                for env_fn in TASKS
            ]
        )


tella.curriculum.curriculum_registry["PhysicsCartPole"] = PhysicsCartPole
