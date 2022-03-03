import abc
from typing import *
import torch
from torch.utils.data import Dataset
from .base import Transition, TransitionObserver, Vectorized
import numpy as np

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


class TransitionsWithMaxReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return transition.reward


class TransitionsWithMinReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return -transition.reward


class TransitionsWithMaxAbsReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return abs(transition.reward)


class TransitionsWithMinAbsReward(PriorityFn):
    def __call__(self, transition: Transition) -> float:
        return -abs(transition.reward)


class MostRecentlyAddedTransitions(PriorityFn):
    def __init__(self) -> None:
        self.t = 0

    def __call__(self, _transition: Transition) -> float:
        priority = self.t
        self.t += 1
        return priority


class PrioritizedTransitionDataset(Dataset[Transition], TransitionObserver):
    def __init__(
        self, max_size: int, keep: PriorityFn = MostRecentlyAddedTransitions()
    ) -> None:
        self.priority_queue = []
        self.max_size = max_size
        self.priority_fn = keep

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        for transition in transitions:
            priority = self.priority_fn(transition)
            self.priority_queue.append((transition, priority))
        if len(self.priority_queue) > self.max_size:
            self._drop_by_priority()

    def _drop_by_priority(self) -> None:
        # TODO use an actual priority queue (heapq package)?
        self.priority_queue = sorted(self.priority_queue, key=lambda t_p: t_p[1])[
            : self.max_size
        ]

    def __getitem__(self, index: int) -> Transition:
        # TODO does using a priority queue that reorders items affect sampling algorithms?
        return self.priority_queue[index][0]

    def __len__(self) -> int:
        return len(self.priority_queue)


# TODO add GeneralizedAdvantageEstimationDataset

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
