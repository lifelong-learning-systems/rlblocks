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
    SlicedCramerPreservation,
)
from rlblocks.replay.datasets import (
    MetaBatchSampler,
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
            batch.reward += self.reward_signal_per_step
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
        self.ewc_loss_fn = ElasticWeightConsolidationLoss(self.model)
        self.ewc_lambda = 1e10  # NOTE: original paper used 400

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter * self.num_envs):
            batch = collate(self.replay_sampler.sample_batch(batch_size=64))
            batch.reward += self.reward_signal_per_step
            loss = self.dqn_loss_fn(batch) + self.ewc_loss_fn() * self.ewc_lambda
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
            self.ewc_loss_fn.remove_anchors(key=(task_name, variant_name))

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        if self.is_learning_allowed:
            print(
                f"Finished learning on {task_name} - {variant_name} (adding associated EWC weights)"
            )
            key = (task_name, variant_name)
            self.ewc_loss_fn.set_anchors(key)
            # NOTE: original paper only drew 100 batches to calculate this
            for transitions in self.replay_sampler.generate_batches(128, True):
                batch = collate(transitions)
                self.optimizer.zero_grad()
                # .backward() puts gradients as an attribute of each parameter
                self.dqn_loss_fn(batch).backward()
                self.ewc_loss_fn.sample_fisher_information(key)


class DqnTaskMemory(Dqn):
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
            network, rng_seed, observation_space, action_space, num_envs, config_file
        )
        self.buffers = {}
        self.samplers = {}
        self.max_size = self.replay_buffer.max_size

    def task_variant_start(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        super().task_variant_start(task_name, variant_name)
        if not self.is_learning_allowed:
            return
        key = (task_name, variant_name)
        if key not in self.buffers:
            self.buffers[key] = TransitionDatasetWithMaxCapacity(self.max_size)
            self.samplers[key] = UniformRandomBatchSampler(self.buffers[key])
            for k, buffer in self.buffers.items():
                buffer.max_size = self.max_size // len(self.buffers)
                buffer.shrink_to_fit()

            self.replay_sampler = MetaBatchSampler(
                *[
                    (sampler, 1 / len(self.samplers))
                    for sampler in self.samplers.values()
                ]
            )

        self.transition_observers[0] = self.buffers[key]


class DqnScp(Dqn):
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
        self.scp_projections = 10
        self.scp_loss_fn = SlicedCramerPreservation(self.model, self.scp_projections)
        self.scp_lambda = 1  # NOTE: original paper used 400

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter * self.num_envs):
            batch = collate(self.replay_sampler.sample_batch(batch_size=64))
            batch.reward += self.reward_signal_per_step
            loss = self.dqn_loss_fn(batch) + self.scp_loss_fn() * self.scp_lambda
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        if self.is_learning_allowed:
            print(f"Starting learning on {task_name} - {variant_name}")
            # self.scp_loss_fn.remove_anchors(key=(task_name, variant_name))

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        if self.is_learning_allowed:
            print(
                f"Finished learning on {task_name} - {variant_name} (adding associated EWC weights)"
            )
            key = (task_name, variant_name)
            self.scp_loss_fn.set_anchors(key)
            # NOTE: original paper only drew 100 batches to calculate this
            for transitions in self.replay_sampler.generate_batches(128, True):
                batch = collate(transitions)
                self.optimizer.zero_grad()
                # .backward() puts gradients as an attribute of each parameter
                self.dqn_loss_fn(batch).backward()
                # self.scp_loss_fn.sample_fisher_information(key)
