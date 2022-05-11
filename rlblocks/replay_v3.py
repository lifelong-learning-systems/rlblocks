from typing import *
import abc
import numpy as np
import heapq

import scipy
import scipy.signal
from .base import Transition, TransitionObserver, Vectorized


T = TypeVar("T", bound=Transition)


class RLBufferCallbacks(abc.ABC):
    def on_item_added(self, buffer: "RLBuffer[T]", item_id: int) -> None:
        """
        Called when an item is added to the buffer.

        :param item_id: The id of the new item.
        """

    def on_item_removed(self, buffer: "RLBuffer[T]", item_id: int) -> None:
        """
        Called when an item is removed from the buffer (by another callback).

        :param item_id: The item that was just removed.
        """

    def on_item_updated(
        self,
        buffer: "RLBuffer[T]",
        item_id: int,
        attr_name: str,
        old_value: float,
        new_value: float,
    ) -> None:
        """
        Called when an item in the buffer is updated.

        :param item_id: The id of the item that was just updated
        :param attr_name: The name of the field updated on the item
        :param old_value: The previous value
        :param new_value: The new value that was just set
        """

    def on_episode_finished(self, buffer: "RLBuffer[T]", item_ids: List[int]) -> None:
        """
        Called when an episode is finished and all transitions from it are added to the buffer.

        :param item_ids: A list containing the ids of all transitions from the episode, in correct order.
        """

    def on_cleared(self):
        """
        Called when buffer is cleared
        """


class RLBuffer(TransitionObserver, Generic[T]):
    """
    Represents a buffer of items of type `T` that can be added, removed, or updated.

    Most usage will involve using objects that implement :class:`RLBufferCallbacks`.

    A vanilla replay buffer with size limit of 5000:
    ```python
    buffer = RLBuffer(Transition, DropLowestPriorityTransition(max_size=5000, OldestTransition()))
    ```

    If you want to add fields to transition, the expected way to do this is to
    subclass Transition and pass that new class as the first argument to `RLBuffer`.

    An example of this is a replay buffer with advantage for on policy algorithms:
    ```python
    class AdvantageTransition(Transition):
        def __init__(self, ...):
            super().__init__(...)
            self.advantage = None

    value_fn = ...
    buffer = RLBuffer(AdvantageTransition, ComputeAdvantage(value_fn, gamma=0.9, lam=0.95))
    ```
    """

    def __init__(
        self,
        item_cls: Type[T],
        *,
        callbacks: List[RLBufferCallbacks],
    ) -> None:
        self.item_cls = item_cls
        self.callbacks = callbacks

        self._item_ids = []
        self._item_by_id = {}
        self._episode_by_id = {}
        self._items_by_episode_id = {}
        self._episode_ids = None
        self._next_episode_id = None
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._item_by_id)

    def __getitem__(self, index: int) -> T:
        return self.get_item(self._item_ids[index])

    def get_item(self, item_id: int) -> T:
        return self._item_by_id[item_id]

    def get_items_with_indices(self, indices: List[int]) -> List[T]:
        return [self.get_item(self._item_ids[i]) for i in indices]

    def clear(self):
        self._item_ids = []
        self._item_by_id = {}
        self._episode_by_id = {}
        self._items_by_episode_id = {}
        self._episode_ids = None
        self._next_episode_id = None
        self._next_id = 0
        for cb in self.callbacks:
            cb.on_cleared()

    def remove_item(self, item_id: int) -> None:
        episode_id = self._episode_by_id[item_id]
        del self._item_by_id[item_id]
        self._item_ids.remove(item_id)
        del self._episode_by_id[item_id]
        self._items_by_episode_id[episode_id].remove(item_id)
        for cb in self.callbacks:
            cb.on_item_removed(self, item_id)

    def update_items(self, item_ids: List[int], fields: Dict[str, List[float]]):
        """
        Update a list of items with the list of values in the `fields` dictionary.

        Example:

        ```python
        >>> items = [1, 2, 3]
        >>> print([buffer.get_item(i).reward for i in items])
        [4, 5, 6]
        >>> fields = {"reward": [0.0, -1.0, -2.0]}
        >>> buffer.update_items(items, fields)
        >>> print([buffer.get_item(i).reward for i in items])
        [0.0, -1.0, 2.0]
        ```

        :param item_ids: The ids of the items to update
        :param fields: A dictionary where the keys are the attribute on the items to update,
            and the values are the new values for the items.
        """
        for k, values in fields.items():
            assert len(item_ids) == len(values)
            for item_id, new_value in zip(item_ids, values):
                old_value = getattr(self._item_by_id[item_id], k)
                setattr(self._item_by_id[item_id], k, new_value)
                for cb in self.callbacks:
                    cb.on_item_updated(self, item_id, k, old_value, new_value)

    def receive_transitions(self, transitions: Vectorized[T]) -> None:
        """
        Adds a set of transitions of type `T` to the buffer. This expects a list of transitions
        to support vectorization.
        """
        if self._episode_ids is None:
            self._episode_ids = list(range(len(transitions)))
            self._next_episode_id = len(transitions)
            for eid in self._episode_ids:
                self._items_by_episode_id[eid] = []

        for i, t in enumerate(transitions):
            # get ids for this item
            item_id = self._next_id
            episode_id = self._episode_ids[i]

            # store value in buffer
            self._item_ids.append(item_id)
            self._item_by_id[item_id] = self.item_cls.from_transition(t)
            self._episode_by_id[item_id] = episode_id
            self._items_by_episode_id[episode_id].append(item_id)

            # update _next_id & episode_id
            self._next_id += 1
            if t.done:
                self._episode_ids[i] = self._next_episode_id
                self._next_episode_id += 1
                self._items_by_episode_id[self._episode_ids[i]] = []

            # call callbacks
            for cb in self.callbacks:
                cb.on_item_added(self, item_id)
                if t.done:
                    cb.on_episode_finished(self, self._items_by_episode_id[episode_id])

        return super().receive_transitions(transitions)

    def complete_partial_episodes(self):
        for episode_id in self._episode_ids:
            items = self._items_by_episode_id[episode_id]
            if len(items) > 0:
                for cb in self.callbacks:
                    cb.on_episode_finished(self, items)


class PriorityAssigner(abc.ABC):
    @abc.abstractmethod
    def get_priority(self, buffer: "RLBuffer[T]", item_id: int) -> float:
        ...


class DropLowestPriorityTransition(RLBufferCallbacks):
    def __init__(self, max_size: int, drop_priority: PriorityAssigner) -> None:
        self.drop_priority = drop_priority
        self.is_static = drop_priority.priority_is_static
        self.max_size = max_size
        self.heap: List[Tuple[int, float]] = []

    def on_cleared(self):
        self.heap = []

    def on_item_updated(
        self,
        buffer: "RLBuffer[T]",
        item_id: int,
        attr_name: str,
        old_value: float,
        new_value: float,
    ) -> None:
        # TODO update the priority of this item?
        ...

    def on_item_added(self, buffer: RLBuffer[T], item_id: int) -> None:
        heapq.heappush(
            self.heap, (item_id, self.get_priority(buffer.get_item(item_id)))
        )
        if len(self.heap) > self.max_size:
            item_id, _ = heapq.heappop(self.heap)
            buffer.remove_item(item_id)

    def on_item_removed(self, buffer: RLBuffer[T], item_id: int) -> None:
        # TODO remove the item from the heap
        ...

    def on_episode_finished(self, buffer: RLBuffer[T], item_ids: List[int]) -> None:
        pass


class ComputeAdvantage(RLBufferCallbacks):
    def __init__(
        self,
        value_fn: Callable[[np.ndarray], float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam

    def on_episode_finished(self, buffer: RLBuffer[T], item_ids: List[int]) -> None:
        # TODO call value_fn in batch form
        values = []
        rewards = []
        for item_id in item_ids:
            item = buffer.get_item(item_id)
            values.append(self.value_fn(item.observation))
            rewards.append(item.reward)
        values.append(np.array([0.0]))

        values = np.concatenate(values)
        rewards = np.array(rewards)
        assert len(values) == len(rewards) + 1

        # NOTE: copied from rllib & others
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = scipy.signal.lfilter(
            [1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0
        )[::-1]

        assert len(advantages) == len(item_ids)

        buffer.update_items(item_ids, {"advantage": advantages})


class OldestTransition(PriorityAssigner):
    """
    This is a normal FIFO queue, it assigns priority based on age.

    This is a static priority assigner, meaning priority is calculated once
    at insertion time.
    """

    def __init__(self) -> None:
        self.t = 0

    @staticmethod
    def priority_is_static() -> bool:
        return True

    def get_priority(self, buffer: "RLBuffer[T]", item_id: int) -> float:
        priority = self.t
        self.t += 1.0
        return priority


class GlobalDistributionMatchingPriority(PriorityAssigner):
    """
    "Global Distribution Matching" from: Selective Experience Replay for Lifelong Learning
        Isele and Cosgun, 2018. https://arxiv.org/abs/1802.10269

    This is a static priority assigner, meaning priority is calculated once
    at insertion time.
    """

    def __init__(self, rng_seed: int = None) -> None:
        self.rng = np.random.default_rng(rng_seed)

    @staticmethod
    def priority_is_static() -> bool:
        return True

    def get_priority(self, buffer: "RLBuffer[T]", item_id: int) -> float:
        return self.rng.normal()


class CoverageMaximizationPriority(PriorityAssigner):
    """
    "Coverage Maximization" from: Selective Experience Replay for Lifelong Learning
        Isele and Cosgun, 2018. https://arxiv.org/abs/1802.10269

    This is a static priority assigner, meaning priority is calculated once
    at insertion time.
    """

    def __init__(
        self,
        rng_seed: int = None,
        distance_function: Callable[[np.ndarray, np.ndarray], float] = None,
        neighbor_threshold: float = None,
        sample_size: int = None,
    ) -> None:
        """

        :param rng_seed: Seed value for repeatable random number generation
        :param distance_function: Function for comparing sample distance
        :param neighbor_threshold: Optional threshold for determining which samples are neighbors
        :param sample_size: Optional limit on number of samples to compare
        """
        self.rng = np.random.default_rng(rng_seed)
        self.t = 0
        self.sample_size = sample_size
        self.neighbor_threshold = neighbor_threshold

        self.distance_function = (
            distance_function
            if distance_function is not None
            else lambda a, b: np.sqrt(np.sum(np.square(a - b))).item()
        )

    def get_priority(self, buffer: "RLBuffer[T]", item_id: int) -> float:
        self.t += 1

        if self.sample_size is None or self.sample_size >= len(buffer):
            num_samples = len(buffer)
        else:
            num_samples = self.sample_size

        item = buffer.get_item(item_id)
        sampled_items = buffer.get_items_with_indices(
            self.rng.choice(len(buffer), size=num_samples, replace=False)
        )

        distances = np.array(
            [
                self.distance_function(item.observation, sample.observation)
                for sample in sampled_items
            ]
        )

        if self.neighbor_threshold is None:
            # Record median distance
            distance_metric = np.median(distances)
        else:
            # Record number of samples that are near neighbors (negative for priority)
            distance_metric = -np.sum(distances < self.neighbor_threshold)

        return distance_metric, self.t
