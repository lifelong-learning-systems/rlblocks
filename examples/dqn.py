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
from rlblocks.replay import RandomPriority
import torch
from torch import nn, optim
import numpy as np


def q_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    model = q_model()
    old_model = deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    buffer = TransitionDatasetWithMaxCapacity(10_000, drop=RandomPriority(rng_seed=0))
    sampler = UniformRandomBatchSampler(buffer)

    q_loss_fn = DoubleQLoss(model, old_model)

    greedy_policy = NumpyToTorchConverter(ArgmaxAction(model))
    epsilon = Interpolate(start=1.0, end=0.02, n=3_000)
    epsilon_greedy_policy = ChooseBetween(
        lambda o: rng.choice(2, size=len(o)),
        greedy_policy,
        prob_fn=epsilon,
        rng=rng,
    )

    def update_model():
        batch = collate(sampler.sample_batch(batch_size=32))
        loss = q_loss_fn(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stats = RewardTracker()

    callbacks = PeriodicCallbacks(
        (Every(1, Steps, offset=200), update_model),
        (Every(1, Steps), SoftParameterUpdate(model, old_model, 0.5)),
        (Every(20, Steps), lambda: print(stats)),
    )

    run_env_interaction(
        env_fn=lambda: gym.make("CartPole-v1"),
        choose_action_fn=epsilon_greedy_policy,
        transition_observers=[buffer, callbacks, stats],
        duration=Duration(20_000, Steps),
    )


if __name__ == "__main__":
    main()
