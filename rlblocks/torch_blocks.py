from typing import *
import torch
from torch import nn
from torch.distributions import Distribution
import numpy as np
import scipy.signal
from .base import (
    Action,
    Observation,
    Transition,
    TransitionObserver,
)


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
        self.criterion = criterion()

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
    def __init__(self, dist_fn: Callable[[Observation], Distribution]) -> None:
        self.dist_fn = dist_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        # TODO this potentially invokes dist_fn twice, is that different from reusing?
        dist = self.dist_fn(batch.state)
        return -dist.entropy().mean()


class PolicyGradientLoss(Callable[[TorchBatch], torch.Tensor]):
    # loss used in A2C https://github.com/deepmind/rlax/blob/66ea2d68a083933a3fb3f76a4e0935339005e1aa/rlax/_src/policy_gradients.py#L89
    def __init__(self, dist_fn: Callable[[Observation], Distribution]) -> None:
        self.dist_fn = dist_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        dist = self.dist_fn(batch.state)
        log_prob_a = dist.log_prob(batch.action)
        return ((0.0 - log_prob_a) * batch.advantage).mean()


class ClippedSurrogatePolicyGradientLoss:
    # PPO https://github.com/deepmind/rlax/blob/66ea2d68a083933a3fb3f76a4e0935339005e1aa/rlax/_src/policy_gradients.py#L258
    def __init__(
        self,
        dist_fn: Callable[[Observation], Distribution],
        old_dist_fn: Callable[[Observation], Distribution],
        clip_range: float = 0.2,
    ) -> None:
        self.dist_fn = dist_fn
        self.old_dist_fn = old_dist_fn
        self.clip_range = clip_range

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            old_log_prob_a = self.old_dist_fn(batch.state).log_prob(batch.action)
        log_prob_a = self.dist_fn(batch.state).log_prob(batch.action)

        ratio = torch.exp(log_prob_a - old_log_prob_a)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        return (
            0.0 - torch.min(ratio * batch.advantage, clipped_ratio * batch.advantage)
        ).mean()


class DeterministicPolicyGradientLoss(Callable[[TorchBatch], torch.Tensor]):
    # loss used in DDPG
    def __init__(
        self,
        action_fn: Callable[[Observation], Action],
        value_fn: Callable[[Observation, Action], Value],
    ) -> None:
        self.action_fn = action_fn
        self.value_fn = value_fn

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        action = self.action_fn(batch.state)
        return (0.0 - self.value_fn(batch.state, action)).mean()


class TDAdvantageLoss(Callable[[TorchBatch], torch.Tensor]):
    # used in both A2C and PPO
    def __init__(
        self,
        state_value_fn: Callable[[Observation], Value],
        old_state_value_fn: Callable[[Observation], Value],
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()
        self.state_value_fn = state_value_fn
        self.old_state_value_fn = old_state_value_fn
        self.criterion = criterion()

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            # target_values a.k.a returns
            target_values = batch.advantage + self.old_state_value_fn(batch.state)

        values = self.state_value_fn(batch.state)
        return self.criterion(values, target_values)


class TDActionValueLoss(Callable[[TorchBatch], torch.Tensor]):
    # Used in DDPG to train critic
    def __init__(
        self,
        action_value_fn: Callable[[Observation, Action], Value],
        old_action_fn: Callable[[Observation], Action],
        old_action_value_fn: Callable[[Observation, Action], Value],
        criterion: nn.Module = nn.MSELoss(),
        discount: float = 0.99,
    ) -> None:
        self.action_value_fn = action_value_fn
        self.old_action_fn = old_action_fn
        self.old_action_value_fn = old_action_value_fn
        self.criterion = criterion
        self.discount = discount

    def __call__(self, batch: TorchBatch) -> torch.Tensor:
        with torch.no_grad():
            next_action = self.old_action_fn(batch.next_state)
            next_v = self.old_action_value_fn(batch.next_state, next_action)
            target_v = batch.reward + self.discount * next_v * (1.0 - batch.done)
        v = self.action_value_fn(batch.state, batch.action)
        return self.criterion(v, target_v)


class SampleAction(Callable[[Observation], Action]):
    def __init__(
        self, dist_fn: Callable[[Observation], torch.distributions.Distribution]
    ) -> None:
        super().__init__()
        self.dist_fn = dist_fn

    def __call__(self, observation: Observation) -> Action:
        dist = self.dist_fn(observation)
        return dist.sample()


class Numpy(Callable[[Observation], np.ndarray]):
    def __init__(self, action_fn: Callable[[Observation], torch.Tensor]) -> None:
        super().__init__()
        self.action_fn = action_fn

    @torch.no_grad()
    def __call__(self, observation: Observation) -> np.ndarray:
        return self.action_fn(torch.from_numpy(observation).float()).numpy()


class ArgmaxAction(Callable[[Observation], Action]):
    def __init__(self, score_fn: Callable[[Observation], torch.Tensor]) -> None:
        super().__init__()
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
        decision_p = self.prob_fn()
        if self.rng.random() <= decision_p:
            return self.action_fn_le(observation)
        else:
            return self.action_fn_g(observation)


class Interpolate(Callable[[], float]):
    def __init__(self, start: float, end: float, n: int) -> None:
        super().__init__()
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


class OptimizerStep(Callable[[torch.Tensor], None]):
    # TODO: add gradient clipping?
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def __call__(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SoftParameterUpdate(Callable[[], None]):
    def __init__(
        self, model: torch.nn.Module, target: torch.nn.Module, tau: float
    ) -> None:
        self.model = model
        self.target = target
        self.tau = tau

    @torch.no_grad()
    def __call__(self):
        for p, target_p in zip(self.model.parameters(), self.target.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)


PolyakParameterUpdate = SoftParameterUpdate


class HardParameterUpdate(SoftParameterUpdate):
    def __init__(self, model: torch.nn.Module, target: torch.nn.Module) -> None:
        super().__init__(model, target, 1.0)


class BatchIterator:
    def __init__(self, buffer, batch_size: int, rng: np.random.Generator) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.rng = rng

    def __call__(self) -> Generator[TorchBatch, None, None]:
        n = len(self.buffer)
        inds = self.rng.permutation(n)
        i = 0
        while i + self.batch_size < n:
            yield self.buffer.get_batch(inds[i : i + self.batch_size])
            i += self.batch_size


class FIFOBuffer(TransitionObserver):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self.max_size = max_size
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def receive_transitions(self, transitions: Sequence[Transition]) -> None:
        self.transitions.extend(transitions)
        while len(self.transitions) > self.max_size:
            self.transitions.pop(0)
        assert len(self.transitions) <= self.max_size

    def get_batch(self, indices: List[int]) -> TorchBatch:
        ts = [self.transitions[i] for i in indices]
        return TorchBatch(*list(zip(*ts)), None).torch()


class GeneralizedAdvantageEstimatingBuffer(TransitionObserver):
    def __init__(
        self,
        value_fn: Callable[[Observation], np.ndarray],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        super().__init__()
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam
        self.episodes = []
        self.episode_values = []
        self.episode_rewards = []
        self.transitions = []

    def receive_transitions(self, transitions: Sequence[Transition]) -> None:
        if len(self.episodes) == 0:
            for _ in range(len(transitions)):
                self.episodes.append([])
                self.episode_values.append([])
                self.episode_rewards.append([])
        assert len(self.episodes) == len(transitions)
        for i_ep, t in enumerate(transitions):
            self.episodes[i_ep].append(t)
            self.episode_rewards[i_ep].append(t.reward)
            self.episode_values[i_ep].append(self.value_fn(t.state))
            if t.done:
                self._consume_episode(i_ep, np.array([0.0]))

    def _consume_episode(self, i_ep: int, final_value: np.ndarray):
        self.episode_values[i_ep].append(final_value)

        values = np.concatenate(self.episode_values[i_ep])
        rewards = np.array(self.episode_rewards[i_ep])
        assert len(values) == len(rewards) + 1

        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = scipy.signal.lfilter(
            [1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0
        )[
            ::-1
        ]  # copied from rllib & others

        assert len(advantages) == len(self.episodes[i_ep])
        for i_step, t in enumerate(self.episodes[i_ep]):
            self.transitions.append((*t, advantages[i_step]))

        self.episodes[i_ep].clear()
        self.episode_rewards[i_ep].clear()
        self.episode_values[i_ep].clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self):
        self.episodes = []
        self.episode_values = []
        self.episode_rewards = []
        self.transitions = []

    def complete_partial_trajectories(self):
        for i_ep, partial_trajectory in enumerate(self.episodes):
            if len(partial_trajectory) > 0:
                final_transition = partial_trajectory[-1]
                self._consume_episode(
                    i_ep,
                    final_value=self.value_fn(final_transition.next_state),
                )

    def get_batch(self, indices: List[int]) -> TorchBatch:
        ts = [self.transitions[i] for i in indices]
        return TorchBatch(*list(zip(*ts))).torch()


def buffer_normalize_advantage(buffer: GeneralizedAdvantageEstimatingBuffer):
    ...


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
