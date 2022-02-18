from copy import deepcopy
import gym
from rlblocks import *
from torch import nn, optim
import numpy as np


model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
old_model = deepcopy(model)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
rng = np.random.default_rng(0)
buffer = FIFOBuffer(max_size=50_000)

dqn_loss = QLoss(model, old_model)
opt_step = OptimizerStep(optimizer)


def update_model():
    for _ in range(4):
        inds = rng.choice(len(buffer), size=128)
        batch = buffer.get_batch(inds)
        loss = dqn_loss(batch)
        opt_step(loss)


epsilon_greedy = ChooseBetween(
    lambda o: rng.choice(2, size=len(o)),
    Numpy(ArgmaxAction(model)),
    prob_fn=Interpolate(start=1.0, end=0.0, n=2000),
    rng=rng,
)

callbacks = PeriodicCallbacks(
    {
        Every(1, Steps, delay=1000): [update_model],
        Every(100, Steps): [HardParameterUpdate(model, old_model)],
    },
)

training_report_generator = mdp_report_generator(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=epsilon_greedy,
    transition_observers=[buffer, callbacks],
    duration=Duration(20_000, Steps),
)
for report in training_report_generator:
    print(report)
