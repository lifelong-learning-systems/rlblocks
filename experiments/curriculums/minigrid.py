import typing
import gym
import numpy as np
import tella
from tella.curriculum import (
    InterleavedEvalCurriculum,
    LearnBlock,
    EvalBlock,
    TaskVariant,
    simple_eval_block,
    simple_learn_block,
)
from gym_minigrid.envs import CrossingEnv, EmptyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from tella._curriculums.minigrid.envs import CustomDynamicObstaclesEnv

GRID_SIZE = 7


class LavaCrossing(CrossingEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE, num_crossings=1)


class Empty(EmptyEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE)


class ObstacleCrossing(CustomDynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE, n_obstacles=1)


class StandardizedMiniGridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        env.action_space = gym.spaces.Discrete(3)
        env = ImgObsWrapper(env)
        super().__init__(env)

    def step(self, action):
        assert 0 <= action < 3
        obs, reward, done, info = super().step(action)

        if not done:
            reward = 1.0 / self.env.max_steps
        elif self.env.step_count >= self.env.max_steps:
            reward = -(self.env.step_count - 1) / self.env.max_steps
        else:
            assert done
            curr_cell = self.env.grid.get(*self.env.agent_pos)
            if curr_cell is not None and curr_cell.type == "goal":
                reward = 1.0 - (self.env.step_count - 1) / self.env.max_steps
            else:
                reward = -(self.env.step_count - 1) / self.env.max_steps
        return obs, reward, done, info


def empty_crossing():
    return StandardizedMiniGridWrapper(Empty())


def lava_crossing():
    return StandardizedMiniGridWrapper(LavaCrossing())


def obstacle_crossing():
    return StandardizedMiniGridWrapper(ObstacleCrossing())


TASKS = [
    empty_crossing,
    obstacle_crossing,
    lava_crossing,
]


class MiniGridCrossing(InterleavedEvalCurriculum):
    NUM_LEARN_STEPS_PER_VARIANT = 20_000
    NUM_EVAL_EPISODES_PER_VARIANT = 10

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for cls in TASKS:
            yield simple_learn_block(
                [
                    TaskVariant(
                        cls,
                        task_label=cls.__name__,
                        variant_label="Default",
                        num_steps=self.NUM_LEARN_STEPS_PER_VARIANT,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )

    def eval_block(self) -> EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return simple_eval_block(
            TaskVariant(
                cls,
                task_label=cls.__name__,
                variant_label="Default",
                num_episodes=self.NUM_EVAL_EPISODES_PER_VARIANT,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for cls in TASKS
        )


tella.curriculum.curriculum_registry["MiniGridCrossing"] = MiniGridCrossing
