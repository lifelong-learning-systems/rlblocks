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
        dist = Normal(mean, F.softplus(log_std))
        # dist = Independent(dist, 1)
        dist = TransformedDistribution(
            dist,
            ComposeTransform(
                [TanhTransform(cache_size=1), AffineTransform(0.0, 2.0, cache_size=1)],
                cache_size=1,
            ),
        )
        return dist

    def state_value(self, obs: Observation):
        return self.v(obs)


torch.manual_seed(0)
rng = np.random.default_rng(0)

model = DiagGaussianPPOModel()
old_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = GeneralizedAdvantageEstimatingBuffer(value_fn=Numpy(model.state_value))

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


callbacks = PeriodicCallbacks(
    {
        Every(512, Steps): RunFunctions(
            buffer.complete_partial_trajectories,
            update_model_fn,
            update_old_model_fn,
            buffer.clear,
        )
    },
)

run_env_interaction(
    env_fn=lambda: gym.make("Pendulum-v1"),
    choose_action_fn=Numpy(SampleAction(model.action_dist)),
    step_observers=[buffer, callbacks, StdoutReport(Every(20, Steps))],
    duration=Duration(20_000, Steps),
)
