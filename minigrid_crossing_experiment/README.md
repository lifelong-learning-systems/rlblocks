# MiniGrid Crossing - Continual RL experiment in minigrid

A continual reinforcement learning experiment in [gym-minigrid](https://github.com/maximecb/gym-minigrid).

## Running the experiment

```bash
python -m minigrid_crossing_experiment
```

This will train a number of agents specified in [__main__.py](__main__.py)

## Environments

The MiniGridCrossing curriculum contains three environments all generally
related to crossing the grid world:

1. [Lava Crossing](https://github.com/maximecb/gym-minigrid#lava-crossing-environment)
2. [Empty Crossing](https://github.com/maximecb/gym-minigrid#empty-environment)
3. [Dynamic Obstacles](https://github.com/maximecb/gym-minigrid#dynamic-obstacles-environment)

Each environment is sized as a 5x5 grid, meaning there is a 3x3 area for
the agent to walk in.

TODO describe the observation

All environments can be found in [curriculum.py](curriculum.py).

## Curriculum

The curriculum is built using [tella](https://github.com/lifelong-learning-systems/tella).
It consists of a fixed order of lava crossing, empty crossing, and dynamic obstacles; an
agent gets 30k steps in each environment to learn to solve it. Between each task the
agent is evaluated on all three tasks.

The curriculum definition can be found in [curriculum.py](curriculum.py).

## Agents

We have a number of agents built using components within rlblocks, and inserted
into the [tella](https://github.com/lifelong-learning-systems/tella) agent class.

All agents can be found in [dqn.py](dqn.py).

#### DQN Baseline

The base agent is an implementation of Deep Q Learning (DQN). It uses the following rlblocks components:

1. Replay buffer with room for 10k steps using [TransitionDatasetWithMaxCapacity]
2. Loss function using [DoubleQLoss].
3. Target model updates using [HardParameterUpdate]
4. Epsilon greedy using [Interpolate], [ChooseBetween], [NumpyToTorchConverter], and [ArgmaxAction]
5. [PeriodicCallbacks] to coordinate all of the above.

#### DQN + Elastic Weight Consolidation (EWC)

The first continual RL agent is the baseline agent
with the [EWC] component added in.

#### DQN + Task Specific Memory

The second continual RL agent is the DQN agent, but the replay buffer stores data separately for each task. This still has the same amount of data as the normal DQN agent, but split between the tasks.

This utilizes the [MetaBatchSampler] component to sample data from multiple buffers.

#### DQN + Sliced Cramer Preservation

The baseline agent combined with the [SCP] component. This is very similar to the EWC agent.

#### DQN + Reservoir Sampling

The baseline agent, but instead of the replay buffer using a FIFO queue, it drops data based on Random priority using the [RandomPriority] component.

#### DQN + Coverage

The baseline agent, but instead of the replay buffer using a FIFO queue, it drops data based on state coverage using the [CoveragePriority] component.

## Results

