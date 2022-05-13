"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from copy import deepcopy
import gym
from typing import *
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


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    model = CategoricalPPOModel()
    old_model = deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    buffer = TransitionDatasetWithAdvantage(NumpyToTorchConverter(model.state_value))
    sampler = UniformRandomBatchSampler(buffer)

    policy_loss_fn = ClippedSurrogatePolicyGradientLoss(
        model.action_dist, old_model.action_dist
    )
    value_loss_fn = TDAdvantageLoss(model.state_value, old_model.state_value)

    def update_model_fn():
        for _i_epoch in range(4):
            for transitions in sampler.generate_batches(batch_size=64, drop_last=True):
                batch = collate(transitions)
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
        (Every(512, Steps), buffer.complete_partial_trajectories),
        (Every(512, Steps), update_model_fn),
        (Every(512, Steps), buffer.clear),
    )

    run_env_interaction(
        env_fn=lambda: gym.make("CartPole-v1"),
        choose_action_fn=NumpyToTorchConverter(SampleAction(model.action_dist)),
        transition_observers=[buffer, callbacks, stats],
        duration=Duration(20_000, Steps),
    )


if __name__ == "__main__":
    main()
