from copy import deepcopy
import typing

import gym
import numpy as np
import tella
from torch import nn, optim

from rlblocks import (
    ArgmaxAction,
    ChooseBetween,
    Interpolate,
    NumpyToTorchConverter,
    RewardTracker,
    Every,
    Steps,
    HardParameterUpdate,
    PeriodicCallbacks,
    QLoss,
    DoubleQLoss,
    ElasticWeightConsolidationLoss,
)
from rlblocks.replay.datasets import (
    TransitionDatasetWithMaxCapacity,
    UniformRandomBatchSampler,
    collate,
    TorchBatch,
)


class Dqn(tella.ContinualRLAgent):
    def __init__(
        self,
        network: nn.Module,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            rng_seed,
            observation_space,
            action_space,
            num_envs,
            config_file,
        )

        # Check that this environment is compatible with DQN
        assert isinstance(
            action_space, gym.spaces.Discrete
        ), "This DQN agent requires discrete action spaces"

        self.rng = np.random.default_rng(self.rng_seed)

        self.replay_buffer = TransitionDatasetWithMaxCapacity(max_size=10_000)
        self.replay_sampler = UniformRandomBatchSampler(self.replay_buffer)

        self.model = network
        self.target_model = deepcopy(network)
        self.dqn_loss_fn = DoubleQLoss(
            self.model, self.target_model, criterion=nn.SmoothL1Loss()
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.greedy_policy = NumpyToTorchConverter(ArgmaxAction(self.model))
        self.epsilon_greedy_policy = ChooseBetween(
            lambda o: self.rng.choice(self.action_space.n, size=len(o)),
            self.greedy_policy,
            prob_fn=Interpolate(start=1.0, end=0.02, n=10_000),
            rng=self.rng,
        )

        self.reward_signal_per_step = -0.01

        self.reward_tracker = RewardTracker()
        self.transition_observers = [
            self.replay_buffer,
            self.reward_tracker,
            PeriodicCallbacks(
                [
                    (
                        Every(1000, Steps),
                        HardParameterUpdate(self.model, self.target_model),
                    ),
                    (Every(1, Steps, offset=200), self.update_model),
                    (Every(500, Steps), lambda: print(self.reward_tracker)),
                ],
            ),
        ]

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter * self.num_envs):
            batch = collate(self.replay_sampler.sample_batch(batch_size=64))
            batch = TorchBatch(
                state=batch.state,
                action=batch.action,
                reward=batch.reward + self.reward_signal_per_step,
                done=batch.done,
                next_state=batch.next_state,
                advantage=batch.advantage,
            )
            loss = self.dqn_loss_fn(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        policy = (
            self.epsilon_greedy_policy
            if self.is_learning_allowed
            else self.greedy_policy
        )
        # Handling masked observation vectors is a bit messy. Maybe this should be provided as a function somewhere
        actions = policy(
            np.array([obs for obs in observations if obs is not None])
        ).tolist()
        return [actions.pop(0) if obs is not None else None for obs in observations]

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[tella.Transition]]
    ) -> None:
        if self.is_learning_allowed:
            masked_transitions = [t for t in transitions if t is not None]
            for observer in self.transition_observers:
                observer.receive_transitions(masked_transitions)


def dqn_with_memory(model: nn.Module):
    ...


class DqnEwc(Dqn):
    def __init__(
        self,
        network: nn.Module,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            network,
            rng_seed,
            observation_space,
            action_space,
            num_envs,
            config_file,
        )
        self.ewc_loss = ElasticWeightConsolidationLoss(
            self.model, ewc_lambda=1.0
        )  # TODO: tune

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter):
            batch = collate(self.replay_sampler.sample_batch(batch_size=64))
            batch = TorchBatch(
                state=batch.state,
                action=batch.action,
                reward=batch.reward + self.reward_signal_per_step,
                done=batch.done,
                next_state=batch.next_state,
                advantage=batch.advantage,
            )
            loss = self.dqn_loss_fn(batch) + self.ewc_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        if self.is_learning_allowed:
            print(
                f"Starting learning on {task_name} - {variant_name} (removing associated EWC weights)"
            )
            self.ewc_loss.remove_anchors(key=(task_name, variant_name))

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        if self.is_learning_allowed:
            print(
                f"Finished learning on {task_name} - {variant_name} (adding associated EWC weights)"
            )
            # batch = collate(
            #     [
            #         self.replay_buffer[len(self.replay_buffer) - 1 - ii]
            #         for ii in range(128)
            #     ]
            # )
            batch = collate(self.replay_buffer[-128:])
            self.optimizer.zero_grad()
            loss_value = self.dqn_loss_fn(batch)
            loss_value.backward()  # Gradients now accessible as an attribute of each parameter
            self.ewc_loss.add_anchors(key=(task_name, variant_name))


def dqn_with_memory_and_ewc(model: nn.Module):
    ...
