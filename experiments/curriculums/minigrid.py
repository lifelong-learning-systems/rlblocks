import typing
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
from tella._curriculums.minigrid.m21 import CustomUnlockS5, DoorKeyS5


TASKS = [
    (CustomUnlockS5, "CustomUnlock", "S5"),
    (DoorKeyS5, "DoorKey", "S5"),
]


class TwoTaskMiniGrid(InterleavedEvalCurriculum):
    NUM_LEARN_STEPS_PER_VARIANT = 10_000
    NUM_EVAL_EPISODES_PER_VARIANT = 10

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for cls, task_label, variant_label in self.rng.permutation(TASKS):
            yield simple_learn_block(
                [
                    TaskVariant(
                        cls,
                        task_label=task_label,
                        variant_label=variant_label,
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
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=self.NUM_EVAL_EPISODES_PER_VARIANT,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for cls, task_label, variant_label in TASKS
        )


tella.curriculum.curriculum_registry["TwoTaskMiniGrid"] = TwoTaskMiniGrid
