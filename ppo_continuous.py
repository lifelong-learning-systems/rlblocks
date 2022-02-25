from copy import deepcopy
import gym
from typing import *
from rlblocks import *
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    TransformedDistribution,
    TanhTransform,
    AffineTransform,
    ComposeTransform,
)
import numpy as np


class SafeTanhTransform(TanhTransform):
    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        # clip action to avoid NaNs
        return super()._inverse(y.clamp(-1.0 + eps, 1.0 - eps))


class DiagGaussianPPOModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diag_gaussian = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.v = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def action_dist(self, obs: Observation) -> Distribution:
        mean, log_std = self.diag_gaussian(obs).chunk(2, dim=-1)
        dist = Normal(mean, log_std.exp())
        dist = Independent(dist, 1)
        dist = TransformedDistribution(
            dist,
            ComposeTransform([SafeTanhTransform(), AffineTransform(0.0, 2.0)]),
        )
        return dist

    def state_value(self, obs: Observation):
        return self.v(obs)


torch.manual_seed(0)
rng = np.random.default_rng(0)

model = DiagGaussianPPOModel()
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
    env_fn=lambda: gym.make("Pendulum-v1"),
    choose_action_fn=NumpyToTorchConverter(SampleAction(model.action_dist)),
    step_observers=[buffer, callbacks, stats],
    duration=Duration(20_000, Steps),
)
