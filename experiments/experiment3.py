import tella

from experiments.agents import RandAgent
from experiments.curriculums.minigrid import MiniGridCrossing


if __name__ == "__main__":
    tella.rl_experiment(
        agent_factory=RandAgent,
        curriculum_factory=MiniGridCrossing,
        num_lifetimes=1,
        num_parallel_envs=6,
        log_dir="logs",
    )
