from copy import deepcopy
import gym
from rlblocks import *
from rlblocks.datasets import (
    PrioritizedTransitionDataset,
    UniformRandomBatchSampler,
    collate,
)
from torch import nn, optim
import numpy as np

rng = np.random.default_rng(0)

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
old_model = deepcopy(model)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
buffer = PrioritizedTransitionDataset(50_000)
sampler = UniformRandomBatchSampler(buffer)

dqn_loss_fn = QLoss(model, old_model)


def update_model_fn():
    for _ in range(4):
        batch = collate(sampler.sample_batch(batch_size=128))
        loss = dqn_loss_fn(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


epsilon_greedy = ChooseBetween(
    lambda o: rng.choice(2, size=len(o)),
    NumpyToTorchConverter(ArgmaxAction(model)),
    prob_fn=Interpolate(start=1.0, end=0.0, n=2000),
    rng=rng,
)

stats = RewardTracker()

callbacks = PeriodicCallbacks(
    {
        Every(100, Steps): HardParameterUpdate(model, old_model),
        Every(1, Steps, offset=1000): update_model_fn,
        Every(20, Steps): lambda: print(stats),
    },
)

run_env_interaction(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=epsilon_greedy,
    transition_observers=[buffer, callbacks, stats],
    duration=Duration(20_000, Steps),
)
