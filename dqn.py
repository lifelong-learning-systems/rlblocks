from copy import deepcopy
import gym
from rlblocks import *
from rlblocks.datasets import (
    TransitionDatasetWithMaxCapacity,
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
buffer = TransitionDatasetWithMaxCapacity(10_000)
sampler = UniformRandomBatchSampler(buffer)

dqn_loss_fn = QLoss(model, old_model)
ewc_loss_fn = OnlineElasticWeightConsolidationLoss(model=model, ewc_lambda=1, update_relaxation=0.5)


def update_model_fn():
    for _ in range(4):
        batch = collate(sampler.sample_batch(batch_size=128))
        loss = dqn_loss_fn(batch)
        loss += ewc_loss_fn()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def add_ewc_anchors():
    inds = rng.choice(len(buffer), size=128)
    batch = buffer.get_batch(inds)
    optimizer.zero_grad()
    loss_value = dqn_loss_fn(batch)
    loss_value.backward()  # Gradients now accessible as an attribute of each parameter
    ewc_loss_fn.add_anchors()
    print("Added EWC anchors!")


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
        Every(1000, Steps, offset=1000): add_ewc_anchors,
    },
)

run_env_interaction(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=epsilon_greedy,
    transition_observers=[buffer, callbacks, stats],
    duration=Duration(20_000, Steps),
)
