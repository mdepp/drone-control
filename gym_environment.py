import gym
from numbers import Real
from typing import Any, Tuple, TypeVar

from environment import Environment

State = TypeVar('State')
Action = TypeVar('Action')


class GymEnvironment(Environment):
    def __init__(self, env: gym.Env):
        self.env = env

    def start_run(self) -> None:
        pass

    def start_episode(self) -> State:
        return self.env.reset()

    def step(self, action: Action) -> Tuple[Real, State, bool]:
        self.env.render()
        observation, reward, done, info = self.env.step(action)  # type: (State, Real, bool, Any)
        return reward, observation, done
