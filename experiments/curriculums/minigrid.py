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
from tella._curriculums.minigrid.m21 import MiniGridReducedActionSpaceWrapper
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
        super().__init__(size=GRID_SIZE, n_obstacles=2)


class StandardizedMiniGridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = ImgObsWrapper(self.env)
        self.env = MiniGridReducedActionSpaceWrapper(self.env, num_actions=3)
        self.env.max_steps = 4 * self.env.width * self.env.height

    def step(self, action):
        obs, reward, done, info = super().step(action)

        reward = 0.0
        if done:
            curr_cell = self.env.grid.get(*self.env.agent_pos)
            if curr_cell is not None and curr_cell.type == "goal":
                reward = 1.0

        return obs, reward, done, info


def empty_crossing():
    return StandardizedMiniGridWrapper(Empty())


def lava_crossing():
    return StandardizedMiniGridWrapper(LavaCrossing())


def obstacle_crossing():
    return StandardizedMiniGridWrapper(ObstacleCrossing())


TASKS = [
    lava_crossing,
    empty_crossing,
    obstacle_crossing,
]


class MiniGridCrossing(InterleavedEvalCurriculum):
    NUM_LEARN_STEPS_PER_VARIANT = 50_000
    NUM_EVAL_EPISODES_PER_VARIANT = 100
    NUM_LOOPS_THROUGH_TASKS = 2

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for _ in range(self.NUM_LOOPS_THROUGH_TASKS):
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


class LavaCrossingSteCurriculum(MiniGridCrossing):
    NUM_LEARN_STEPS_PER_VARIANT = 100_000
    NUM_LOOPS_THROUGH_TASKS = 2

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for _ in range(self.NUM_LOOPS_THROUGH_TASKS):
            yield simple_learn_block(
                [
                    TaskVariant(
                        lava_crossing,
                        task_label="lava_crossing",
                        variant_label="Default",
                        num_steps=self.NUM_LEARN_STEPS_PER_VARIANT,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )


class EmptyCrossingSteCurriculum(MiniGridCrossing):
    NUM_LEARN_STEPS_PER_VARIANT = 100_000
    NUM_LOOPS_THROUGH_TASKS = 2

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for _ in range(self.NUM_LOOPS_THROUGH_TASKS):
            yield simple_learn_block(
                [
                    TaskVariant(
                        empty_crossing,
                        task_label="empty_crossing",
                        variant_label="Default",
                        num_steps=self.NUM_LEARN_STEPS_PER_VARIANT,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )


class ObstacleCrossingSteCurriculum(MiniGridCrossing):
    NUM_LEARN_STEPS_PER_VARIANT = 100_000
    NUM_LOOPS_THROUGH_TASKS = 2

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for _ in range(self.NUM_LOOPS_THROUGH_TASKS):
            yield simple_learn_block(
                [
                    TaskVariant(
                        obstacle_crossing,
                        task_label="obstacle_crossing",
                        variant_label="Default",
                        num_steps=self.NUM_LEARN_STEPS_PER_VARIANT,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )
