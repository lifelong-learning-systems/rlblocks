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

import abc
import collections
import enum
from typing import *
import gym

Observation = TypeVar("Observation")
Action = TypeVar("Action")
Reward = float
Done = bool


class Transition(NamedTuple):
    observation: Observation
    action: Action
    reward: float
    done: bool
    next_observation: Observation


Vectorized = Iterable


class TransitionObserver(abc.ABC):
    """
    Components which observe the environment can inherit from this class to standardize passing data
    """
    @abc.abstractmethod
    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        pass


class RLTimeUnit(enum.Enum):
    STEPS = enum.auto()
    EPISODES = enum.auto()


Steps = RLTimeUnit.STEPS
Episodes = RLTimeUnit.EPISODES


class Timer(TransitionObserver):
    """
    Count number of steps or episodes completed. Compare to threshold with `.is_finished` method.
    """
    def __init__(self, n: int, unit: RLTimeUnit, *, offset: int = 0) -> None:
        self.n = n
        self.unit = unit
        self.num_steps_taken = 0
        self.num_episodes_completed = 0
        self.delay = offset

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        for _obs, _action, _reward, done, _next_obs in transitions:
            self.num_steps_taken += 1
            self.num_episodes_completed += int(done)

    def restart(self) -> None:
        self.num_steps_taken = 0
        self.num_episodes_completed = 0
        self.delay = 0

    def is_finished(self) -> bool:
        value = (
            self.num_steps_taken if self.unit == Steps else self.num_episodes_completed
        )
        return value >= self.n + self.delay


Duration = Timer
Every = Timer


class PeriodicCallbacks(TransitionObserver):
    """
    Pair Timer instances with callback functions to run those callbacks on each timer completion.
    """
    def __init__(self, *callback_by_timer: Tuple[Timer, Callable[[], None]]) -> None:
        super().__init__()
        self.callback_by_timer = callback_by_timer

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        for timer, callback in self.callback_by_timer:
            timer.receive_transitions(transitions)
            if timer.is_finished():
                callback()
                timer.restart()


class RewardTracker(TransitionObserver):
    """
    Tracks total reward received for high-level performance monitoring
    """
    def __init__(self) -> None:
        super().__init__()
        self.total_reward = 0.0
        self.total_steps = 0
        self.total_episodes = 0
        self.num_envs = None

    def clear(self):
        self.total_reward = 0.0
        self.total_steps = 0
        self.total_episodes = 0

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        self.num_envs = len(transitions)
        for t in transitions:
            self.total_reward += t.reward
            self.total_steps += 1
            self.total_episodes += int(t.done)

    def __str__(self) -> str:
        return str(
            {
                "total_reward": self.total_reward,
                "num_episodes": self.total_episodes + self.num_envs,
                "num_steps": self.total_steps,
                "mean_episode_reward": self.total_reward
                / (self.total_episodes + self.num_envs),
                "mean_episode_length": self.total_steps
                / (self.total_episodes + self.num_envs),
            }
        )


class ActionTracker(TransitionObserver):
    """
    Tracks actions taken for high-level performance monitoring
    """
    def __init__(self) -> None:
        super().__init__()
        self.action_count = collections.Counter()
        self.total_steps = 0
        self.num_envs = None

    def clear(self):
        self.action_count.clear()
        self.total_steps = 0

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        for t in transitions:
            self.total_steps += 1
            self.action_count[t.action] += 1

    def __str__(self) -> str:
        return str(
            {
                act: cnt / self.total_steps
                for act, cnt in self.action_count.most_common()
            }
        )


def run_env_interaction(
    *,
    env_fn: Callable[[], gym.Env],
    num_envs: int = 1,
    choose_action_fn: Callable[[Sequence[Observation]], Action],
    transition_observers: List[TransitionObserver],
    duration: Duration,
) -> None:
    """
    Run an agent through an environment

    :param env_fn: Callable which creates a gym.Env environment
    :param num_envs: Optional number of parallel environments to run. Default 1
    :param choose_action_fn: Callable function by which the agent chooses actions
    :param transition_observers: List of objects which each will be passed data from the environment
    :param duration: Timer instance which ends the interaction when complete
    :return: None
    """
    vec_env: gym.vector.VectorEnv = gym.vector.SyncVectorEnv(
        [env_fn for _ in range(num_envs)]
    )

    if duration not in transition_observers:
        transition_observers.append(duration)

    obss = vec_env.reset()
    while not duration.is_finished():
        actions = choose_action_fn(obss)
        new_obss, rewards, dones, infos = vec_env.step(actions)
        next_obss = [
            n if not d else i["terminal_observation"]
            for n, d, i in zip(new_obss, dones, infos)
        ]
        transitions = [
            Transition(*values)
            for values in zip(obss, actions, rewards, dones, next_obss)
        ]
        for observer in transition_observers:
            observer.receive_transitions(transitions)
        obss = new_obss
