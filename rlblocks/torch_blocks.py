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
from .base import Action, Observation


class TorchBatch(NamedTuple):
    state: torch.FloatTensor
    action: torch.Tensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    next_state: torch.FloatTensor
    advantage: Optional[torch.FloatTensor]

    def torch(self) -> "TorchBatch":
        state = torch.from_numpy(np.array(self.state)).float()
        action = torch.from_numpy(np.array(self.action))
        if action.dtype == torch.int32:
            action = action.type(torch.int64)
        if action.dtype == torch.int64:
            action = action.unsqueeze(1)
        reward = torch.from_numpy(np.array(self.reward)).float().unsqueeze(1)
        done = torch.from_numpy(np.array(self.done)).float().unsqueeze(1)
        next_state = torch.from_numpy(np.array(self.next_state)).float()
        if self.advantage is not None:
            advantage = torch.from_numpy(np.array(self.advantage)).float().unsqueeze(1)
        else:
            advantage = None
        return TorchBatch(state, action, reward, done, next_state, advantage)


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
        return self.old_q_fn(batch.next_state).max(-1)[0].unsqueeze(1)

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            max_next_q = self._get_max_next_q(batch)
            target_q = batch.reward + self.discount * max_next_q * (1.0 - batch.done)
        current_q = self.q_fn(batch.state).gather(dim=1, index=batch.action)
        return self.criterion(current_q, target_q)


class DoubleQLoss(QLoss):
    def _get_max_next_q(self, batch: TorchBatch) -> torch.Tensor:
        next_action = self.q_fn(batch.next_state).max(-1)[1].unsqueeze(1)
        return self.old_q_fn(batch.next_state).gather(dim=-1, index=next_action)


class EntropyLoss(Callable[[TorchBatch], torch.Tensor]):
    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        # TODO this potentially invokes distribution_fn twice, is that different from reusing?
        dist = self.distribution_fn(batch.state)
        return -dist.entropy().mean()


class PolicyGradientLoss(Callable[[TorchBatch], torch.Tensor]):
    # loss used in A2C https://github.com/deepmind/rlax/blob/66ea2d68a083933a3fb3f76a4e0935339005e1aa/rlax/_src/policy_gradients.py#L89
    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        dist = self.distribution_fn(batch.state)
        log_prob_a = dist.log_prob(batch.action)
        return ((0.0 - log_prob_a) * batch.advantage).mean()


class ClippedPolicyGradientLoss:
    # PPO https://github.com/deepmind/rlax/blob/66ea2d68a083933a3fb3f76a4e0935339005e1aa/rlax/_src/policy_gradients.py#L258
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
            old_log_prob_a = self.old_distribution_fn(batch.state).log_prob(
                batch.action
            )
        log_prob_a = self.distribution_fn(batch.state).log_prob(batch.action)

        ratio = torch.exp(log_prob_a - old_log_prob_a)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        return (
            0.0 - torch.min(ratio * batch.advantage, clipped_ratio * batch.advantage)
        ).mean()


class TDAdvantageLoss(Callable[[TorchBatch], torch.Tensor]):
    # used in both A2C and PPO
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
            target_values = batch.advantage + self.old_state_value_fn(batch.state)

        values = self.state_value_fn(batch.state)
        return self.criterion(values, target_values)


class ElasticWeightConsolidationLoss:
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


class OnlineElasticWeightConsolidationLoss(ElasticWeightConsolidationLoss):
    def __init__(self, model: nn.Module, ewc_lambda: float, update_relaxation: float):
        super().__init__(model, ewc_lambda)
        self.update_relaxation = update_relaxation

    def add_anchors(self):
        # Importance here is computed from parameter gradients
        # Before calling this method, the user must compute those gradients using something like:
        #   `loss_fn(batch_data).backward()`
        importance = [
            layer.grad.detach().clone() ** 2
            if layer.grad is not None
            else torch.zeros_like(layer)
            for layer in self._model.parameters()
        ]

        if not self._anchor_values:
            self._anchor_values = [
                layer.detach().clone() for layer in self._model.parameters()
            ]
            self._importance = importance
        else:
            self._anchor_values = [
                old * self.update_relaxation
                + new.detach().clone() * (1 - self.update_relaxation)
                for old, new in zip(self._anchor_values, self._model.parameters())
            ]
            self._importance = [
                old * self.update_relaxation + new * (1 - self.update_relaxation)
                for old, new in zip(self._importance, importance)
            ]


class SlicedCramerPreservation:
    def __init__(self, model: nn.Module, projections: int):
        self._model = model

        # Start with no anchors
        self._anchors = {}
        self._num_samples = {}
        self._projections = projections
        self._synaptic_response = {}
        ## Initialize to zeros
        for name, curr_param in self._model.named_parameters():
            if not name.endswith('bias'):
                self._synaptic_response[name] = torch.zeros(curr_param.shape[1],curr_param.shape[1]) 

    def __call__(self) -> torch.Tensor:
        loss = 0
        for key, anchors in self._anchors.items():
            for name, curr_param in self._model.named_parameters():
                if not name.endswith('bias'):
                    syn_prod = self._synaptic_response[name].diagonal()
                    loss += (syn_prod * (anchors[name] - curr_param).square()).sum()
        return loss

    def store_synaptic_response(self, mean_response):
        dim = mean_response.shape[0]
        # Initialize the reponse matrix
        for l in range(self._projections):
            # Slice the mean response
            psi = torch.tensor(self.sample_unit_sphere(dim), dtype=torch.float32)
            unit_slice = torch.dot(psi, mean_response)
            unit_slice.backward(retain_graph=True)
            for name, curr_param in self._model.named_parameters():
                if not name.endswith('bias'):
                    param_grad = curr_param.grad.squeeze()
                    self._synaptic_response[name] += (1 / self._projections) * \
                        torch.mm(param_grad.T, param_grad)

    def sample_unit_sphere(sefl, dim: int):
        u = np.random.normal(0, 1, dim)
        d = np.sum(u**2) ** (0.5)
        return u / d

    def set_anchors(self, key: Hashable):
        # This is the per-task version of EWC. The online alternative replaces
        # these values (with optional relaxation) instead of extending them
        self._anchors[key] = {
            name: layer.data.clone() for name, layer in self._model.named_parameters()
        }

    def remove_anchors(self, key: Hashable):
        self._anchors.pop(key, None)
        # self._fisher_information.pop(key, None)


class SampleAction(Callable[[Observation], Action]):
    def __init__(self, distribution_fn: Callable[[Observation], Distribution]) -> None:
        self.distribution_fn = distribution_fn

    def __call__(self, observation: Observation) -> Action:
        return self.distribution_fn(observation).sample()


class NumpyToTorchConverter(Callable[[np.ndarray], np.ndarray]):
    def __init__(self, torch_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.torch_fn = torch_fn

    def __call__(self, x_numpy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_torch = torch.from_numpy(x_numpy).float()
            y_torch = self.torch_fn(x_torch)
            y_numpy = y_torch.numpy()
            return y_numpy


class ArgmaxAction(Callable[[Observation], Action]):
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
        batch.state,
        batch.action,
        batch.reward,
        batch.done,
        batch.next_state,
        (batch.advantage - batch.advantage.mean()) / (1e-6 + batch.advantage.std()),
    )
