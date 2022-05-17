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
from torch import nn
from torch.distributions import Distribution
import numpy as np
from .base import Observation
from .torch_blocks import TorchBatch


LogProbs = torch.Tensor
Value = torch.Tensor
QValues = torch.Tensor


class QLoss(Callable[[TorchBatch], torch.Tensor]):
    # used in DQN
    def __init__(
        self,
        q_fn: Callable[[Observation], QValues],
        old_q_fn: Callable[[Observation], QValues],
        discount: float = 0.99,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        self.q_fn = q_fn
        self.old_q_fn = old_q_fn
        self.discount = discount
        self.criterion = criterion

    def _get_max_next_q(self, batch: TorchBatch) -> torch.Tensor:
        return self.old_q_fn(batch.next_observation).max(-1)[0].unsqueeze(1)

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            max_next_q = self._get_max_next_q(batch)
            target_q = batch.reward + self.discount * max_next_q * (1.0 - batch.done)
        current_q = self.q_fn(batch.observation).gather(dim=1, index=batch.action)
        return self.criterion(current_q, target_q)


class DoubleQLoss(QLoss):
    def _get_max_next_q(self, batch: TorchBatch) -> torch.Tensor:
        next_action = self.q_fn(batch.next_observation).max(-1)[1].unsqueeze(1)
        return self.old_q_fn(batch.next_observation).gather(dim=-1, index=next_action)


class EntropyLoss(Callable[[TorchBatch], torch.Tensor]):
    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        # TODO this potentially invokes distribution_fn twice, is that different from reusing?
        dist = self.distribution_fn(batch.observation)
        return -dist.entropy().mean()


class PolicyGradientLoss(Callable[[TorchBatch], torch.Tensor]):
    """
    A2C from [1].

    [1] Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning."
        International conference on machine learning. PMLR, 2016.
    """

    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        dist = self.distribution_fn(batch.observation)
        log_prob_a = dist.log_prob(batch.action.reshape(dist.batch_space))
        adv = batch.advantage.reshape(log_prob_a.shape)
        return ((0.0 - log_prob_a) * adv).mean()


class ClippedSurrogatePolicyGradientLoss:
    """
    PPO Policy loss from [1].

    [1] Schulman, John, et al. "Proximal policy optimization algorithms."
        arXiv preprint arXiv:1707.06347 (2017).
    """

    def __init__(
        self,
        distribution_fn: Callable[[Observation], Distribution],
        old_distribution_fn: Callable[[Observation], Distribution],
        clip_range: float = 0.2,
    ) -> None:
        self.distribution_fn = distribution_fn
        self.old_distribution_fn = old_distribution_fn
        self.clip_range = clip_range

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            old_dist = self.old_distribution_fn(batch.observation)
            old_log_prob_a = old_dist.log_prob(
                batch.action.reshape(old_dist.batch_shape)
            )

        dist = self.distribution_fn(batch.observation)
        log_prob_a = dist.log_prob(batch.action.reshape(dist.batch_shape))

        ratio = torch.exp(log_prob_a - old_log_prob_a)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        adv = batch.advantage.reshape(ratio.shape)

        return (0.0 - torch.min(ratio * adv, clipped_ratio * adv)).mean()


class TDAdvantageLoss(Callable[[TorchBatch], torch.Tensor]):
    """
    This is the loss used to train the value network in both A2C (see :class:`PolicyGradientLoss`)
    and PPO (see :class:`ClippedSurrogatePolicyGradientLoss`).

    It requires advantage in the batch, which can be calculated with
    :class:`rlblocks.replay.TransitionDatasetWithAdvantage`.
    """

    def __init__(
        self,
        state_value_fn: Callable[[Observation], Value],
        old_state_value_fn: Callable[[Observation], Value],
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        self.state_value_fn = state_value_fn
        self.old_state_value_fn = old_state_value_fn
        self.criterion = criterion

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            # target_values a.k.a returns
            old_values = self.old_state_value_fn(batch.observation)
            target_values = batch.advantage.reshape(old_values.shape) + old_values

        values = self.state_value_fn(batch.observation)
        return self.criterion(values, target_values)


class ElasticWeightConsolidationLoss:
    """
    EWC from [1].

    Example usage:
    ```python
    model: nn.Module = ...

    ewc_loss = ElasticWeightConsolidationLoss(model)

    # training:
    loss = ... + ewc_loss()

    # updating EWC anchors requires calling loss.backward() on a number of batches
    ewc_loss.set_anchors(task="task1")
    for batch in ...:
        self.optimizer.zero_grad()
        loss = ...
        loss.backward()
        ewc_loss.sample_fisher_information(task="task1")
    ```

    [1] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks."
        Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.
    """

    def __init__(self, model: nn.Module):
        self._model = model

        # Start with no anchors
        self._anchors = {}
        self._fisher_information = {}
        self._num_samples = {}

    def __call__(self) -> torch.Tensor:
        loss = 0
        for key, anchors in self._anchors.items():
            for name, curr_param in self._model.named_parameters():
                f = self._fisher_information[key][name] / self._num_samples[key]
                loss += (f * (anchors[name] - curr_param).square()).sum()
        return 0.5 * loss

    def set_anchors(self, task: Hashable):
        # This is the per-task version of EWC. The online alternative replaces
        # these values (with optional relaxation) instead of extending them
        self._anchors[task] = {
            name: layer.data.clone() for name, layer in self._model.named_parameters()
        }

    def sample_fisher_information(self, task: Hashable):
        # Importance here is computed from parameter gradients
        # Before calling this method, the user must compute those gradients using something like:
        #   `loss_fn(batch_data).backward()`
        f_sample = {
            name: layer.grad.data.clone().square()
            for name, layer in self._model.named_parameters()
        }
        if task not in self._fisher_information:
            self._fisher_information[task] = f_sample
            self._num_samples[task] = 0
        else:
            self._num_samples[task] += 1
            for name, value in f_sample.items():
                self._fisher_information[task][name] += value

    def remove_anchors(self, task: Hashable):
        self._anchors.pop(task, None)
        self._fisher_information.pop(task, None)


class SlicedCramerPreservation:
    """
    SCP from [1].

    Example usage:
    ```python
    model: nn.Module = ...

    scp_loss = SlicedCramerPreservation(model, projections=10)

    # training:
    loss = ... + scp_loss()

    # updating EWC anchors requires calling loss.backward() on a number of batches
    scp_loss.set_anchors(task="task1")
    for batch in ...:
        self.optimizer.zero_grad()
        output = model(batch.x)
        loss = ...
        loss.backward()
        scp_loss.store_synaptic_response(task="task1", batch.x)
    ```

    [1] Kolouri, Soheil, et al. "Sliced cramer synaptic consolidation for preserving deeply learned representations."
        International Conference on Learning Representations. 2019.
    """

    def __init__(self, model: nn.Module, projections: int, rng_seed: int = None):
        self._model = model
        self.rng = np.random.default_rng(rng_seed)

        # Start with no anchors
        self._anchors = {}
        self._num_samples = {}
        self._projections = projections
        self._synaptic_response = {}

    def __call__(self) -> torch.Tensor:
        loss = 0
        for key, anchors in self._anchors.items():
            for name, curr_param in self._model.named_parameters():
                loss += (
                    self._synaptic_response[key][name]
                    * (anchors[name] - curr_param).square()
                ).sum()
        return loss

    def store_synaptic_response(self, key, batch_state):
        ## Initialize the Synaptic matrix per task
        if key not in self._synaptic_response.keys():
            self._synaptic_response[key] = {}
        for name, curr_param in self._model.named_parameters():
            self._synaptic_response[key][name] = torch.zeros(curr_param.shape)

        # Initialize the reponse matrix
        mean_response = self._model(batch_state).mean(axis=0)
        for l in range(self._projections):
            # Slice the mean response
            self._model.zero_grad()
            psi = torch.tensor(
                self._sample_unit_sphere(mean_response.shape[0]), dtype=torch.float32
            )
            unit_slice = torch.dot(psi, mean_response)
            unit_slice.backward(retain_graph=True)
            for name, curr_param in self._model.named_parameters():
                self._synaptic_response[key][name] += (
                    1 / self._projections
                ) * curr_param.grad.square()

    def _sample_unit_sphere(self, dim: int):
        u = self.rng.normal(0, 1, dim)
        d = np.linalg.norm(u)
        return u / d

    def set_anchors(self, key: Hashable):
        self._anchors[key] = {
            name: layer.data.clone() for name, layer in self._model.named_parameters()
        }

    def remove_anchors(self, key: Hashable):
        self._anchors.pop(key, None)
