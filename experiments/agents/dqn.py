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
)
from rlblocks.replay.datasets import (
    TransitionDatasetWithMaxCapacity,
    UniformRandomBatchSampler,
    MetaBatchSampler,
    collate,
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

        self.replay_buffer = TransitionDatasetWithMaxCapacity(10_000)
        self.replay_sampler = UniformRandomBatchSampler(self.replay_buffer)

        self.model = network
        self.target_model = deepcopy(network)
        self.dqn_loss_fn = QLoss(self.model, self.target_model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)

        self.greedy_policy = NumpyToTorchConverter(ArgmaxAction(self.model))
        self.epsilon_greedy_policy = ChooseBetween(
            lambda o: self.rng.choice(self.action_space.n, size=len(o)),
            self.greedy_policy,
            prob_fn=Interpolate(start=1.0, end=0.0, n=2000),
            rng=self.rng,
        )

        self.reward_tracker = RewardTracker()
        self.transition_observers = [
            self.replay_buffer,
            self.reward_tracker,
            PeriodicCallbacks(
                {
                    Every(100, Steps): HardParameterUpdate(
                        self.model, self.target_model
                    ),
                    Every(1, Steps, offset=1000): self.update_model,
                    Every(20, Steps): lambda: print(self.reward_tracker),
                },
            ),
        ]

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter):
            batch = collate(self.replay_sampler.sample_batch(batch_size=128))
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


class DqnWithTaskSpecificBuffers(tella.ContinualRLAgent):
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

        self.task_name = None
        self.variant_name = None
        self.buffer_by_task = {}
        self.sampler_by_task = {}

        self.model = network
        self.target_model = deepcopy(network)
        self.dqn_loss_fn = QLoss(self.model, self.target_model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)

        self.greedy_policy = NumpyToTorchConverter(ArgmaxAction(self.model))
        self.epsilon_greedy_policy = ChooseBetween(
            lambda o: self.rng.choice(self.action_space.n, size=len(o)),
            self.greedy_policy,
            prob_fn=Interpolate(start=1.0, end=0.0, n=2000),
            rng=self.rng,
        )

        self.reward_tracker = RewardTracker()
        self.transition_observers = [
            self.reward_tracker,
            PeriodicCallbacks(
                {
                    Every(100, Steps): HardParameterUpdate(
                        self.model, self.target_model
                    ),
                    Every(1, Steps, offset=1000): self.update_model,
                    Every(20, Steps): lambda: print(self.reward_tracker),
                },
            ),
        ]

    def task_variant_start(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        self.task_name = task_name
        self.variant_name = variant_name

    def update_model(self, n_iter: int = 4):
        p = 1 / len(self.buffer_by_task)
        sampler = MetaBatchSampler(
            *[(self.buffer_by_task[task], p) for task in self.sampler_by_task]
        )
        for _ in range(n_iter):
            batch = collate(sampler.sample_batch(batch_size=128))
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
            if self.task_name not in self.buffer_by_task:
                self.buffer_by_task[self.task_name] = TransitionDatasetWithMaxCapacity(
                    10_000
                )
                self.sampler_by_task[self.task_name] = UniformRandomBatchSampler(
                    self.buffer_by_task[self.task_name]
                )
            self.buffer_by_task[self.task_name].receive_transitions(masked_transitions)
            for observer in self.transition_observers:
                observer.receive_transitions(masked_transitions)
