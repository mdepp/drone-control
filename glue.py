# Inspired by rl-glue (e.g. https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue)

from abc import ABC, abstractmethod
from numbers import Real
from typing import Optional, Tuple, TypeVar
from agent import Agent
from environment import Environment

State = TypeVar('State')
Action = TypeVar('Action')


class Glue:
    def __init__(self, agent: Agent, environment: Environment) -> None:
        self.agent = agent
        self.environment = environment
        self._prev_action: Action = None
        self._cumulative_episode_reward: Real = 0
        self._cumulative_run_reward: Real = 0

    def start_run(self):
        self.agent.start_run()
        self.environment.start_run()
        self._prev_action = None
        self._cumulative_run_reward = 0

    def start_episode(self):
        initial_state = self.environment.start_episode()
        self._prev_action = self.agent.start_episode(initial_state)
        self._cumulative_episode_reward = 0

    def step(self, force_terminal: bool = False) -> Tuple[Action, Real, State, bool]:
        assert self._prev_action is not None
        reward, next_state, terminal = self.environment.step(self._prev_action)
        terminal = terminal or force_terminal
        if not terminal:
            self._prev_action = self.agent.step(reward, next_state)
        else:
            self.agent.end_episode(reward, next_state)
        self._cumulative_episode_reward += reward
        self._cumulative_run_reward += reward
        return self._prev_action, reward, next_state, terminal

    @property
    def cumulative_episode_reward(self):
        return self._cumulative_episode_reward

    @property
    def cumulative_run_reward(self):
        return self._cumulative_run_reward

    def do_episode(self, max_length: Optional[int] = None) -> None:
        self.start_episode()
        steps = 0
        while True:
            force_terminal = max_length is not None and steps >= max_length
            steps += 1
            _, _, _, terminal = self.step(force_terminal=force_terminal)
            if terminal:
                break
