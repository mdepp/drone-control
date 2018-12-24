# Inspired by rl-glue (e.g. https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue)

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Tuple, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')


class Environment(ABC):
    @abstractmethod
    def start_run(self) -> None:
        """
        Initialize the environment to start a new run

        """
        pass

    @abstractmethod
    def start_episode(self) -> State:
        """
        Initialize the environment to start a new episode

        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Real, State, bool]:
        """
        Calculate and return the response of the environment to an action of the agent

        Args:
            action: The action taken by the agent in response to the current state

        Returns:
            A tuple (reward, state, terminal), where `reward` is the reward obtained by the agent in response to its
                action; `state` is the new state of the environment in response to the agent's action; `terminal` is
                `True` if and only if the new state is terminal.
        """
        pass
