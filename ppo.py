from copy import deepcopy
import gym
from rlblocks import *
import torch
from torch import nn, optim
from torch.distributions import Distribution, Categorical
import numpy as np


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

model = CategoricalPPOModel()
old_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = GeneralizedAdvantageEstimatingBuffer(
    value_fn=NumpyToTorchConverter(model.state_value)
)

policy_loss_fn = ClippedPolicyGradientLoss(model.action_dist, old_model.action_dist)
value_loss_fn = TDAdvantageLoss(model.state_value, old_model.state_value)
update_old_model_fn = HardParameterUpdate(model, old_model)
get_buffer_iterator = BatchIterator(buffer, batch_size=64, rng=rng)


def update_model_fn():
    for _i_epoch in range(4):
        for batch in get_buffer_iterator():
            batch = batch_normalize_advantange(batch)
            pi_loss = policy_loss_fn(batch)
            v_loss = value_loss_fn(batch)
            optimizer.zero_grad()
            (pi_loss + v_loss).backward()
            optimizer.step()


stats = RewardTracker()

callbacks = PeriodicCallbacks(
    {
        Every(20, Steps): lambda: print(stats),
        Every(512, Steps): CallFunctions(
            buffer.complete_partial_trajectories,
            update_model_fn,
            update_old_model_fn,
            buffer.clear,
        ),
    },
)

run_env_interaction(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=NumpyToTorchConverter(SampleAction(model.action_dist)),
    transition_observers=[buffer, callbacks, stats],
    duration=Duration(20_000, Steps),
)
