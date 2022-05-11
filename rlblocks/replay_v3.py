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
        *callbacks: RLBufferCallbacks,
    ) -> None:
        self.item_cls = item_cls
        self.callbacks = callbacks

        self._item_by_id = {}
        self._episode_by_id = {}
        self._items_by_episode_id = {}
        self._episode_ids = None
        self._next_episode_id = None
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._item_by_id)

    def get_item(self, item_id: int) -> T:
        return self._item_by_id[item_id]

    def remove_item(self, item_id: int) -> None:
        episode_id = self._episode_by_id[item_id]
        del self._item_by_id[item_id]
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
            self._item_by_id[item_id] = t
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


class DropLowestPriorityTransition(RLBufferCallbacks):
    def __init__(self, max_size: int, get_priority: Callable[[T], float]) -> None:
        self.get_priority = get_priority
        self.max_size = max_size
        self.heap: List[Tuple[int, float]] = []

    def on_item_updated(
        self, buffer: RLBuffer[T], item_id: int, attr_name, attr_value
    ) -> None:
        # TODO update the priority of this item
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
        value_fn: Callable[[np.ndarray], np.ndarray],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam

    def on_item_added(self, buffer: RLBuffer[T], item_id: int) -> None:
        ...

    def on_item_updated(
        self, buffer: RLBuffer[T], item_id: int, attr_name: str, attr_value: float
    ) -> None:
        ...

    def on_item_removed(self, buffer: RLBuffer[T], item_id: int) -> None:
        ...

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
