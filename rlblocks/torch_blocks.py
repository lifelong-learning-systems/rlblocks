"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import *
import torch
from torch.distributions import Distribution
import numpy as np
from .base import Action, Observation, Transition


class TorchBatch:
    """
    A batch of transition data collated and converted to pyTorch tensors
    """
    def __init__(
        self,
        observation: torch.FloatTensor,
        action: torch.Tensor,
        reward: torch.FloatTensor,
        done: torch.FloatTensor,
        next_observation: torch.FloatTensor,
        advantage: Optional[torch.FloatTensor],
    ):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.next_observation = next_observation
        self.advantage = advantage

    def torch(self) -> "TorchBatch":
        observation = torch.from_numpy(np.array(self.observation)).float()
        action = torch.from_numpy(np.array(self.action))
        if action.dtype == torch.int32:
            action = action.type(torch.int64)
        if action.dtype == torch.int64:
            action = action.unsqueeze(1)
        reward = torch.from_numpy(np.array(self.reward)).float().unsqueeze(1)
        done = torch.from_numpy(np.array(self.done)).float().unsqueeze(1)
        next_observation = torch.from_numpy(np.array(self.next_observation)).float()
        if self.advantage is not None:
            advantage = torch.from_numpy(np.array(self.advantage)).float().unsqueeze(1)
        else:
            advantage = None
        return TorchBatch(observation, action, reward, done, next_observation, advantage)


def collate(transitions: List[Transition]) -> TorchBatch:
    """
    Create a TorchBatch instance from a list of Transition data
    """
    observations, actions, rewards, dones, next_observations, *advantage = zip(*transitions)
    assert len(advantage) == 0 or len(advantage) == 1

    observations = torch.from_numpy(np.array(observations)).float()
    actions = torch.from_numpy(np.array(actions))
    if actions.dtype == torch.int32:
        actions = actions.type(torch.int64)
    if actions.dtype == torch.int64:
        actions = actions.unsqueeze(1)
    rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
    dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1)
    next_observations = torch.from_numpy(np.array(next_observations)).float()

    if len(advantage) > 0:
        advantage = torch.from_numpy(np.array(advantage[0])).float().unsqueeze(1)
    else:
        advantage = None

    return TorchBatch(observations, actions, rewards, dones, next_observations, advantage)


class NumpyToTorchConverter(Callable[[np.ndarray], np.ndarray]):
    """
    Wraps around a Callable that takes torch tensors, and converts numpy arrays to torch tensors before and after calling the function.
    """
    def __init__(self, torch_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.torch_fn = torch_fn

    def __call__(self, x_numpy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_torch = torch.from_numpy(x_numpy).float()
            y_torch = self.torch_fn(x_torch)
            y_numpy = y_torch.numpy()
            return y_numpy


class SampleAction(Callable[[Observation], Action]):
    """
    Calls :meth:`Distribution.sample()` on a pytorch Distribution produced from the callable you pass in.
    """
    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, observation: Observation) -> Action:
        return self.distribution_fn(observation).sample()


class ArgmaxAction(Callable[[Observation], Action]):
    """
    Calls .argmax(-1) on the result of the callable you pass in. You can use this to get the highest valued action for a categorical model.
    """
    def __init__(self, score_fn: Callable[[Observation], torch.Tensor]) -> None:
        self.score_fn = score_fn

    def __call__(self, observation: Observation) -> Action:
        return self.score_fn(observation).argmax(-1)


class ChooseBetween(Callable[[Observation], Action]):
    def __init__(
        self,
        action_fn_le: Callable[[Observation], Action],
        action_fn_g: Callable[[Observation], Action],
        prob_fn: Callable[[], float],
        rng: np.random.Generator,
    ) -> None:
        self.action_fn_le = action_fn_le
        self.action_fn_g = action_fn_g
        self.prob_fn = prob_fn
        self.rng = rng

    def __call__(self, observation: Observation) -> Action:
        le = self.action_fn_le(observation)
        g = self.action_fn_g(observation)
        return np.array(
            [
                le[i] if self.rng.random() <= self.prob_fn() else g[i]
                for i in range(len(observation))
            ]
        )


class Interpolate(Callable[[], float]):
    def __init__(self, start: float, end: float, n: int) -> None:
        self.start = start
        self.end = end
        self.n = n
        self.i = 0

    def __call__(self) -> float:
        if self.i >= self.n:
            return self.end
        t = self.i / self.n
        self.i += 1
        return self.start * (1.0 - t) + self.end * t

    def reset(self):
        self.i = 0


class SoftParameterUpdate(Callable[[], None]):
    def __init__(
        self, model: torch.nn.Module, old_model: torch.nn.Module, tau: float
    ) -> None:
        self.model = model
        self.old_model = old_model
        self.tau = tau

    def __call__(self):
        with torch.no_grad():
            for curr_p, old_p in zip(
                self.model.parameters(), self.old_model.parameters()
            ):
                old_p.data.copy_(self.tau * curr_p.data + (1.0 - self.tau) * old_p.data)


PolyakParameterUpdate = SoftParameterUpdate


class HardParameterUpdate(SoftParameterUpdate):
    def __init__(self, model: torch.nn.Module, target: torch.nn.Module) -> None:
        super().__init__(model, target, 1.0)


def batch_normalize_advantange(batch: TorchBatch) -> TorchBatch:
    assert batch.advantage is not None
    return TorchBatch(
        batch.observation,
        batch.action,
        batch.reward,
        batch.done,
        batch.next_observation,
        (batch.advantage - batch.advantage.mean()) / (1e-6 + batch.advantage.std()),
    )
