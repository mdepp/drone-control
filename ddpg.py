from numbers import Real
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from agent import Agent, State, Action
from replay_buffer import ReplayBuffer


class DDPGAgent(Agent):
    def __init__(self,
                 actor_model: keras.models.Model,
                 critic_model: keras.models.Model,
                 state_shape: Tuple,
                 action_shape: Tuple,
                 max_buffer_size, N):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.N = N  # minibatch size
        # Elements of the relay buffer are flattened
        self.replay_buffer = ReplayBuffer((2*sum(state_shape)+sum(action_shape)+1,), max_buffer_size, N)

        self.Q = critic_model
        self.mu = actor_model
        self.Q_target = keras.models.clone_model(self.Q)
        self.mu_target = keras.mmodels.clone_model(self.mu)

        self.random_process = None

    def start_run(self) -> None:
        # Lots of things to do here
        pass

    def start_episode(self, state: State) -> Action:
        pass

    def step(self, reward: Real, state: State) -> Action:
        pass

    def end_episode(self, reward: Real, state: State) -> None:
        pass