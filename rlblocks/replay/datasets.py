import abc
from typing import *
import torch
from torch.utils.data import Dataset
from ..base import Transition, TransitionObserver, Vectorized
import numpy as np
import scipy

PriorityFn = Callable[[Transition], float]


class TorchBatch(NamedTuple):
    state: torch.FloatTensor
    action: torch.Tensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    next_state: torch.FloatTensor
    advantage: Optional[torch.FloatTensor]


def collate(transitions: List[Transition]) -> TorchBatch:
    states, actions, rewards, dones, next_states, *advantage = zip(*transitions)
    assert len(advantage) == 0 or len(advantage) == 1

    states = torch.from_numpy(np.array(states)).float()
    actions = torch.from_numpy(np.array(actions))
    if actions.dtype == torch.int32:
        actions = actions.type(torch.int64)
    if actions.dtype == torch.int64:
        actions = actions.unsqueeze(1)
    rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
    dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1)
    next_states = torch.from_numpy(np.array(next_states)).float()

    if len(advantage) > 0:
        advantage = torch.from_numpy(np.array(advantage[0])).float().unsqueeze(1)
    else:
        advantage = None

    return TorchBatch(states, actions, rewards, dones, next_states, advantage)


class TransitionWithMaxReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return transition.reward


class TransitionWithMinReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return -transition.reward


class TransitionWithMaxAbsReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return abs(transition.reward)


class TransitionWithMinAbsReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return -abs(transition.reward)


class OldestTransition(PriorityFn):
    def __init__(self) -> None:
        self.t = 0

    def __call__(self, _transition: Transition) -> float:
        priority = -self.t
        self.t += 1
        return priority


class _TransitionDataset(Dataset[Transition]):
    def __init__(self) -> None:
        self.next_id = 0
        self.transition_ids = []
        self.transition_by_id = {}

    def add_transition(self, transition: Transition) -> int:
        transition_id = self.next_id
        self.next_id += 1

        self.transition_ids.append(transition_id)
        self.transition_by_id[transition_id] = transition

        return transition_id

    def remove_transition(self, transition_id: int) -> None:
        self.transition_ids.remove(transition_id)
        del self.transition_by_id[transition_id]

    def __getitem__(self, index: int) -> Transition:
        # TODO does using a priority queue that reorders items affect sampling algorithms?
        transition_id = self.transition_ids[index]
        return self.transition_by_id[transition_id]

    def __len__(self) -> int:
        return len(self.transition_ids)

    def clear(self) -> None:
        self.next_id = 0
        self.transition_ids.clear()
        self.transition_by_id.clear()


class TransitionDatasetWithMaxCapacity(_TransitionDataset, TransitionObserver):
    def __init__(self, max_size: int, drop: PriorityFn = OldestTransition()) -> None:
        super().__init__()
        self.max_size = max_size
        self.priority_fn = drop
        self.priority_by_id = {}

    def add_transition(self, transition: Transition) -> int:
        transition_id = super().add_transition(transition)
        self.priority_by_id[transition_id] = self.priority_fn(transition)
        return transition_id

    def remove_transition(self, transition_id: int) -> None:
        del self.priority_by_id[transition_id]
        return super().remove_transition(transition_id)

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        for transition in transitions:
            self.add_transition(transition)
            if len(self.transition_ids) >= self.max_size:
                self.remove_transition(self.highest_priority_drop())

    def highest_priority_drop(self) -> int:
        # TODO use an actual priority queue (heapq package)?
        return max(self.priority_by_id, key=self.priority_by_id.get)

    def clear(self) -> None:
        self.priority_by_id.clear()
        return super().clear()


class TransitionDatasetWithAdvantage(_TransitionDataset, TransitionObserver):
    def __init__(
        self,
        value_fn: Callable[[np.ndarray], np.ndarray],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        super().__init__()
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam
        self.episode_transitions = []
        self.episode_values = []
        self.episode_rewards = []

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        if len(self.episode_transitions) == 0:
            for _ in range(len(transitions)):
                self.episode_transitions.append([])
                self.episode_values.append([])
                self.episode_rewards.append([])
        assert len(self.episode_transitions) == len(transitions)
        for i_ep, t in enumerate(transitions):
            self.episode_transitions[i_ep].append(t)
            self.episode_rewards[i_ep].append(t.reward)
            self.episode_values[i_ep].append(self.value_fn(t.state))
            if t.done:
                self._add_transitions_from_episode(i_ep, np.array([0.0]))

    def _add_transitions_from_episode(self, i_ep: int, final_value: np.ndarray):
        self.episode_values[i_ep].append(final_value)

        values = np.concatenate(self.episode_values[i_ep])
        rewards = np.array(self.episode_rewards[i_ep])
        assert len(values) == len(rewards) + 1

        # NOTE: copied from rllib & others
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = scipy.signal.lfilter(
            [1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0
        )[::-1]

        assert len(advantages) == len(self.episode_transitions[i_ep])
        for i_step, t in enumerate(self.episode_transitions[i_ep]):
            self.add_transition([*t, advantages[i_step]])

        self.episode_transitions[i_ep].clear()
        self.episode_rewards[i_ep].clear()
        self.episode_values[i_ep].clear()

    def clear(self) -> None:
        self.episode_transitions.clear()
        self.episode_values.clear()
        self.episode_rewards.clear()
        return super().clear()

    def complete_partial_trajectories(self):
        for i_ep, partial_trajectory in enumerate(self.episode_transitions):
            if len(partial_trajectory) > 0:
                final_step = partial_trajectory[-1]
                self._add_transitions_from_episode(
                    i_ep,
                    final_value=self.value_fn(final_step.next_state),
                )


BatchGenerator = Generator[List[Transition], None, None]


class BatchSampler(abc.ABC):
    @abc.abstractmethod
    def sample_batch(self, batch_size: int) -> List[Transition]:
        ...

    @abc.abstractmethod
    def generate_batches(self, batch_size: int, drop_last: bool) -> BatchGenerator:
        ...


class UniformRandomBatchSampler(BatchSampler):
    def __init__(self, source: Dataset[Transition], seed: int = 0) -> None:
        self.source = source
        self.rng = np.random.default_rng(seed)

    def _make_batch(self, inds: List[int]) -> List[Transition]:
        return [self.source[i] for i in inds]

    def sample_batch(self, batch_size: int) -> List[Transition]:
        return self._make_batch(self.rng.choice(len(self.source), batch_size))

    def generate_batches(self, batch_size: int, drop_last: bool) -> BatchGenerator:
        N = len(self.source)
        inds = self.rng.permutation(N)
        i = 0
        while i < N:
            yield self._make_batch(inds[i : i + batch_size])
            i += batch_size
            if drop_last and i + batch_size >= N:
                break


# TODO add PrioritizedBatchSampler(BatchSampler)


class MetaBatchSampler(BatchSampler):
    # TODO what is the most clear API for this? tuples? dicts? something else?
    def __init__(self, *samplers_and_pcts: Tuple[BatchSampler, float]) -> None:
        _samplers, pcts = list(zip(*samplers_and_pcts))
        assert all(0.0 <= p <= 1.0 for p in pcts)
        assert sum(pcts) == 1.0
        self.samplers_and_pcts = samplers_and_pcts

    def sample_batch(self, batch_size: int) -> List[Transition]:
        # FIXME: round(p * batch_size) does not result in total amount of `batch_size`
        assert all(round(p * batch_size) > 0 for _, p in self.samplers_and_pcts)
        return sum(
            (s.sample_batch(round(p * batch_size)) for s, p in self.samplers_and_pcts),
            start=[],
        )

    def generate_batches(self, batch_size: int, drop_last: bool) -> BatchGenerator:
        # FIXME: round(p * batch_size) does not result in total amount of `batch_size`
        assert all(round(p * batch_size) > 0 for _, p in self.samplers_and_pcts)
        generators = [
            s.generate_batches(round(p * batch_size), drop_last)
            for s, p in self.samplers_and_pcts
        ]
        for batches in zip(*generators):
            yield sum(batches, start=[])
