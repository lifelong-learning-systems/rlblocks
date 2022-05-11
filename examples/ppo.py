from copy import deepcopy
import gym
from typing import *
from rlblocks import *
from rlblocks.replay_v3 import (
    CoverageMaximizationPriority,
    DropLowestPriorityTransition,
    RLBuffer,
    ComputeAdvantage,
)
import torch
from torch import nn, optim
from torch.distributions import Distribution, Categorical
import numpy as np


class AdvantageTransition(Transition):
    def __init__(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        next_observation: Observation,
        advantage: float,
    ) -> None:
        super().__init__(observation, action, reward, done, next_observation)
        self.advantage = advantage

    @staticmethod
    def from_transition(t: Transition) -> "AdvantageTransition":
        return AdvantageTransition(
            t.observation, t.action, t.reward, t.done, t.next_observation, None
        )


def collate_adv(ts: List[AdvantageTransition]) -> TorchBatch:
    observations = torch.from_numpy(np.array([t.observation for t in ts])).float()
    actions = torch.from_numpy(np.array([t.action for t in ts])).long()
    rewards = torch.from_numpy(np.array([t.reward for t in ts])).float()
    dones = torch.from_numpy(np.array([t.done for t in ts])).float()
    next_observations = torch.from_numpy(
        np.array([t.next_observation for t in ts])
    ).float()
    assert all(t.advantage is not None for t in ts)
    advantage = torch.from_numpy(np.array([t.advantage for t in ts])).float()

    return AdvantageTransition(
        observations, actions, rewards, dones, next_observations, advantage
    )


class CategoricalPPOModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.v = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def action_dist(self, obs: Observation) -> Distribution:
        return Categorical(logits=self.pi(obs))

    def state_value(self, obs: Observation):
        return self.v(obs)


rng = np.random.default_rng(0)
torch.manual_seed(0)

model = CategoricalPPOModel()
old_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

buffer = RLBuffer(
    AdvantageTransition,
    callbacks=[
        ComputeAdvantage(value_fn=NumpyToTorchConverter(model.state_value)),
        DropLowestPriorityTransition(5000, CoverageMaximizationPriority()),
    ],
)
sampler = UniformRandomBatchSampler(buffer)

policy_loss_fn = ClippedPolicyGradientLoss(model.action_dist, old_model.action_dist)
value_loss_fn = TDAdvantageLoss(model.state_value, old_model.state_value)


def update_model_fn():
    for _i_epoch in range(4):
        for transitions in sampler.generate_batches(batch_size=64, drop_last=True):
            batch = collate_adv(transitions)
            batch.advantage = (batch.advantage - batch.advantage.mean()) / (
                1e-8 + batch.advantage.std()
            )
            loss = policy_loss_fn(batch) + value_loss_fn(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    HardParameterUpdate(model, old_model)()


stats = RewardTracker()

callbacks = PeriodicCallbacks(
    (Every(20, Steps), lambda: print(stats)),
    (Every(512, Steps), buffer.complete_partial_episodes),
    (Every(512, Steps), update_model_fn),
    (Every(512, Steps), buffer.clear),
)

run_env_interaction(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=NumpyToTorchConverter(SampleAction(model.action_dist)),
    transition_observers=[buffer, callbacks, stats],
    duration=Duration(20_000, Steps),
)
