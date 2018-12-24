# Inspired by rl-glue (e.g. https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue)

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')


class Agent(ABC):
    @abstractmethod
    def start_run(self) -> None:
        """
        Initialize the agent to start a new run, possibly after completing a previous one.

        """
        pass

    @abstractmethod
    def start_episode(self, state: State) -> Action:
        """
        Initialize the agent to start a new episode

        Args:
            state: The (non-terminal) initial state of the new episode

        Returns:
            The first action taken by the agent in the new episode

        """
        pass

    @abstractmethod
    def step(self, reward: Real, state: State) -> Action:
        """
        In response to the reward and new state due to the previous action, take a new action.

        Args:
            reward: The reward obtained for taking the previous action
            state: The (non-terminal) current state of the environment after taking the previous action

        Returns:
            The action taken by the agent in the current state

        """
        pass

    @abstractmethod
    def end_episode(self, reward: Real, state: State) -> None:
        """
        Response to a terminal state being reached by the environment.

        Args:
            reward: The reward obtained for taking the previous action
            state: The current (terminal) state of the environment after taking the previous action

        """
        pass
