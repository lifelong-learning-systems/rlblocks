import collections
from collections import Counter
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
    ActionTracker,
    Every,
    Steps,
    HardParameterUpdate,
    PeriodicCallbacks,
    QLoss,
)
from rlblocks.replay.datasets import (
    TransitionDatasetWithMaxCapacity,
    UniformRandomBatchSampler,
    collate,
    TorchBatch,
)


class LossTracker:
    def __init__(self, window: int = 100):
        self.n = 0
        self.total = 0
        self.hist = collections.deque(maxlen=window)

    def record(self, loss_value):
        self.n += 1
        self.total += loss_value
        self.hist.append(loss_value)

    def __str__(self) -> str:
        return (
            f"Avg. loss = {self.total / self.n:0.4f} "
            f"(over last {len(self.hist)}, avg. = {sum(self.hist) / len(self.hist):0.4f})"
        )


class ExplorationSchedule(typing.Callable[[], float]):
    def __init__(self, halflife: float, period: float, maximum: float = 1., minimum: float = 0.) -> None:
        self.halflife = halflife
        self.base = 2 ** (1 / halflife)

        self.period = period

        assert 0 <= minimum < maximum <= 1
        self.maximum = maximum
        self.minimum = minimum
        self.range = maximum - minimum

        self.i = 0
        self.val = None

    def __call__(self) -> float:
        self.i += 1
        self.val = self.minimum + self.range * self.base ** -self.i * np.cos(np.pi * self.i / self.period) ** 2
        return self.val

    def __str__(self) -> str:
        return f"{self.val:0.2f}"


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

        self.states_visited = Counter()

        self.replay_buffer = TransitionDatasetWithMaxCapacity(50_000)
        self.replay_sampler = UniformRandomBatchSampler(self.replay_buffer)

        self.model = network
        self.target_model = deepcopy(network)
        self.dqn_loss_fn = QLoss(self.model, self.target_model, discount=0.8)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        self.greedy_policy = NumpyToTorchConverter(ArgmaxAction(self.model))
        self.exploration_schedule = ExplorationSchedule(
            halflife=50_000,
            period=5_000,
            minimum=0.0,
            maximum=0.3,
        )
        self.epsilon_greedy_policy = ChooseBetween(
            lambda o: self.rng.choice(self.action_space.n, size=len(o)),
            self.greedy_policy,
            prob_fn=self.exploration_schedule,
            rng=self.rng,
        )

        self.reward_signal_per_step = 0.01
        self.loss_tracker = LossTracker()

        self.reward_tracker = RewardTracker()
        self.action_tracker = ActionTracker()
        self.transition_observers = [
            self.replay_buffer,
            self.reward_tracker,
            self.action_tracker,
            PeriodicCallbacks(
                [
                    (Every(500, Steps), HardParameterUpdate(
                        self.model, self.target_model
                    )),
                    (Every(1, Steps, offset=200), self.update_model),
                    (Every(500, Steps), lambda: print(
                        f"\n{self.reward_tracker}"
                        f"\n{self.action_tracker}"
                        f"\nCurrent epsilon: {self.exploration_schedule}"
                        f"\n{self.loss_tracker}"
                    )),
                ],
            ),
        ]

    def update_model(self, n_iter: int = 4):
        for _ in range(n_iter):
            batch = collate(self.replay_sampler.sample_batch(batch_size=128))
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

            self.loss_tracker.record(loss.item())

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


def dqn_with_ewc(model: nn.Module):
    ...


def dqn_with_memory_and_ewc(model: nn.Module):
    ...
