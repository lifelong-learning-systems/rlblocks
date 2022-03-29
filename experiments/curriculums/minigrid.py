import typing
import gym
import gym_minigrid.minigrid
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


GRID_SIZE = 5


class LavaCrossing(CrossingEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE, num_crossings=1)


class Empty(EmptyEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE)


class ObstacleCrossing(CustomDynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=GRID_SIZE, n_obstacles=1)


class MiniGridCenteredFullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding, centered on agent
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * 2 - 3, self.env.height * 2 - 3, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            gym_minigrid.minigrid.OBJECT_TO_IDX['agent'],
            gym_minigrid.minigrid.COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        # Add to FullyObsWrapper: rotate and center obs on agent
        a, b, c = self.observation_space.spaces["image"].shape
        centered_grid = np.zeros((a, b, c), dtype=np.uint8)
        x = (a + 1) // 2 - env.agent_pos[0] - 1
        y = (b + 1) // 2 - env.agent_pos[1] - 1
        centered_grid[x:x+env.width, y:y+env.height] = full_grid
        centered_grid = np.rot90(centered_grid, k=-env.agent_dir, axes=(0, 1))

        # Verify that the rotation is correct:
        front_cell = env.grid.get(*env.front_pos)
        front_idx = centered_grid[(env.width + 1) // 2 + 1, (env.height + 1) // 2, 0]
        front_type = gym_minigrid.envs.IDX_TO_OBJECT[front_idx]
        assert front_type == ("empty" if front_cell is None else front_cell.type)

        return {
            'mission': obs['mission'],
            'image': centered_grid
        }


class StandardizedMiniGridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = MiniGridCenteredFullyObsWrapper(self.env)
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
    NUM_LEARN_STEPS_PER_VARIANT = 100_000
    NUM_EVAL_EPISODES_PER_VARIANT = 100
    NUM_LOOPS_THROUGH_TASKS = 3

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for cls in TASKS:
            for _ in range(self.NUM_LOOPS_THROUGH_TASKS):
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
    NUM_LOOPS_THROUGH_TASKS = 3

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
    NUM_LOOPS_THROUGH_TASKS = 3

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
    NUM_LOOPS_THROUGH_TASKS = 3

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
