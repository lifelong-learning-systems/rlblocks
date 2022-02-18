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


model = CategoricalPPOModel()
old_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
rng = np.random.default_rng(0)
buffer = GeneralizedAdvantageEstimatingBuffer(value_fn=Numpy(model.state_value))

opt_step_fn = OptimizerStep(optimizer)
policy_loss = ClippedSurrogatePolicyGradientLoss(
    model.action_dist, old_model.action_dist, clip_range=0.2
)
value_loss = TDAdvantageLoss(model.state_value, old_model.state_value)
update_old_model = HardParameterUpdate(model, old_model)
batch_iter = BatchIterator(buffer, batch_size=64, rng=rng)


def update_model():
    for _i_epoch in range(4):
        for batch in batch_iter():
            batch = batch_normalize_advantange(batch)
            pi_loss = policy_loss(batch)
            v_loss = value_loss(batch)
            # TODO clip gradients?
            opt_step_fn(pi_loss + v_loss)


callbacks = PeriodicCallbacks(
    {
        Every(512, Steps): [
            buffer.complete_partial_trajectories,
            update_model,
            update_old_model,
            buffer.clear,
        ],
    },
)

training_report_generator = mdp_report_generator(
    env_fn=lambda: gym.make("CartPole-v1"),
    choose_action_fn=Numpy(SampleAction(model.action_dist)),
    transition_observers=[buffer, callbacks],
    duration=Duration(20_000, Steps),
)
for report in training_report_generator:
    print(report)
