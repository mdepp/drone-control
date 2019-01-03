from numbers import Real
from typing import Callable, Tuple, Union

from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer

State = np.ndarray
Action = np.ndarray


# Based on https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process,
#          https://www.wolframalpha.com/input/?i=Wiener+process
class OrnsteinUhlenbeckProcess:
    def __init__(self,
                 shape: Tuple[int, ...],
                 theta: float = 0.15,
                 mu: Union[float, np.ndarray] = 0,
                 sigma: float = 0.2, dt: float = 1.0):
        self.x = np.zeros(shape)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def reset(self) -> None:
        self.x.fill(0.0)

    def get(self) -> np.ndarray:
        self.x = self.theta*(self.mu - self.x)*self.dt + self.sigma*np.random.normal(0, np.sqrt(self.dt))
        return self.x


class DDPGAgent(Agent):
    """
    Implements DDPG (https://arxiv.org/abs/1509.02971). This is a actor-critic method for continuous states and actions
    which produces a deterministic policy.
    """
    def __init__(self,
                 session: tf.Session,
                 global_step: tf.Variable,
                 state_min: np.ndarray,
                 state_max: np.ndarray,
                 action_min: np.ndarray,
                 action_max: np.ndarray,
                 make_critic: Callable[[], Model],
                 make_actor: Callable[[], Model],
                 gamma: float = 0.95,
                 tau: float = 1e-3,
                 minibatch_size: int = 64,
                 replay_buffer_size: int = int(1e6),
                 ):
        # Set up constants
        self.action_min = action_min
        self.action_max = action_max
        assert action_min.shape == action_max.shape
        self.action_shape = action_min.shape
        self.state_min = state_min
        self.state_max = state_max
        assert state_min.shape == state_max.shape
        self.state_shape = state_min.shape
        self.gamma = gamma
        self.tau = tau

        self.sess = session
        self.global_step = global_step
        self.noise = OrnsteinUhlenbeckProcess(shape=self.action_shape)

        with tf.name_scope('ddpg-root') as ddpg_root:
            self.root_scope = ddpg_root
            self.update_global_step = tf.assign_add(self.global_step, 1)

            # Replay buffer, which stores *normalized* states and actions
            self.replay_buffer = ReplayBuffer(
                item_shapes=[self.state_shape, self.action_shape, (), self.state_shape],  # state, action, reward, next state
                max_elements=replay_buffer_size,
                minibatch_size=minibatch_size,
            )

            # Construct actor and critic models
            with tf.name_scope('critic-model'):
                self.critic = make_critic()
            with tf.name_scope('critic-target-model'):
                self.critic_target = make_critic()
            with tf.name_scope('actor-model'):
                self.actor = make_actor()
            with tf.name_scope('actor-target-model'):
                self.actor_target = make_actor()

            # Commands to reset target weights to model weights
            with tf.name_scope('reset-target-weights'):
                reset_critic_target = tf.group([
                    tf.assign(target_var, var) for target_var, var in
                    zip(self.critic_target.trainable_variables, self.critic.trainable_variables)
                ])
                reset_actor_target = tf.group([
                    tf.assign(target_var, var) for target_var, var in
                    zip(self.actor_target.trainable_variables, self.actor.trainable_variables)
                ])
                self.reset_targets = tf.group([reset_critic_target, reset_actor_target])

            # Define placeholders which store state-action-reward-state data from replay buffer
            with tf.name_scope('sarsa-inputs'):
                self.state = tf.placeholder(dtype=tf.float32, shape=(None, *self.state_shape), name='state')
                self.action = tf.placeholder(dtype=tf.float32, shape=(None, *self.action_shape), name='action')
                self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
                self.next_state = tf.placeholder(dtype=tf.float32, shape=(None, *self.state_shape), name='next_state')

            # Execute the policy (with exploration noise) on a state
            with tf.name_scope('act'):
                self.observed_state = tf.placeholder(dtype=tf.float32, shape=self.state_shape, name='observed-state')
                self.exploration_noise = tf.placeholder(dtype=tf.float32, shape=self.action_shape, name='exploration-noise')
                noiseless_action = self.actor(tf.expand_dims(self.observed_state, 0))[0]
                self.taken_action = noiseless_action + self.exploration_noise

            # Critic training update
            with tf.name_scope('critic-update'):
                y = self.reward + self.gamma * self.critic_target([self.next_state, self.actor_target(self.next_state)])
                target = self.critic([self.state, self.action])
                critic_loss = tf.losses.mean_squared_error(labels=y, predictions=target)
                regularizer = tf.contrib.layers.l2_regularizer(scale=1e-2)
                reg_critic_loss = critic_loss + tf.contrib.layers.apply_regularization(regularizer,
                                                                                       self.critic.trainable_weights)
                critic_optimizer = tf.train.AdamOptimizer(1e-3)
                self.critic_train_op = critic_optimizer.minimize(reg_critic_loss,
                                                                 var_list=self.critic.trainable_variables)

            # Actor training update
            with tf.name_scope('actor-update'):
                with tf.control_dependencies([self.critic_train_op]):
                    predicted_action_value = self.critic([self.state, self.actor(self.state)])
                    mean_predicted_action_value = tf.reduce_mean(predicted_action_value)
                    actor_loss = -mean_predicted_action_value
                    self.actor_train_op = tf.train.AdamOptimizer(1e-4).minimize(actor_loss,
                                                                                var_list=self.actor.trainable_variables)

                    grads = tf.gradients(actor_loss, self.actor.trainable_variables)
                    self.grads_norm = sum(tf.norm(grad) for grad in grads)

            # Update target weights toward trained weights
            with tf.name_scope('update-target-weights'):
                with tf.control_dependencies([self.actor_train_op, self.critic_train_op]):

                    def update_target_weights(model, target_model):
                        new_target_weights = [tau * weight + (1 - tau) * target_weight for weight, target_weight
                                              in zip(model.trainable_variables, target_model.trainable_variables)]
                        return tf.group([
                            tf.assign(weight, new_weight) for weight, new_weight
                            in zip(target_model.trainable_variables, tf.tuple(new_target_weights))
                        ])

                    self.update_targets = tf.group([
                        update_target_weights(model=self.critic, target_model=self.critic_target),
                        update_target_weights(model=self.actor, target_model=self.actor_target),
                    ])

            # Initialize all DDPG variables
            with tf.name_scope('initialize-variables'):
                global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.root_scope)
                self.initialize_variables = tf.group([var.initializer for var in global_variables])

            # Tensorboard stuff
            self.observed_reward = tf.placeholder(dtype=tf.float32, shape=())
            self.observed_prev_state = tf.placeholder(dtype=tf.float32, shape=self.state_shape, name='observed_prev_state')
            self.observed_prev_action = tf.placeholder(dtype=tf.float32, shape=self.action_shape, name='observed_prev_action')
            self.predicted_return = tf.squeeze(self.critic([
                tf.expand_dims(self.observed_prev_state, 0),
                tf.expand_dims(self.observed_prev_action, 0),
            ]))
            self.summaries = tf.summary.merge([
                tf.summary.scalar('critic-loss', critic_loss),
                tf.summary.scalar('critic-loss-regularized', reg_critic_loss),
                tf.summary.scalar('actor-"loss"', actor_loss),
                tf.summary.scalar('observed-reward', self.observed_reward),
                tf.summary.scalar('predicted-return', self.predicted_return),
            ])
            self.summary_writer = tf.summary.FileWriter('logdir', self.sess.graph)

        self.prev_action: np.ndarray = None
        self.prev_state: np.ndarray = None

    def start_run(self) -> None:
        self.sess.run(self.initialize_variables)
        self.sess.run(self.reset_targets)
        self.prev_state = None
        self.prev_action = None
        self.replay_buffer.clear()

    def start_episode(self, denormalized_state: State) -> Action:
        state = self._normalize_state(denormalized_state)
        self.prev_state = self._normalize_state(state)
        self.prev_action = self.sess.run(self.taken_action, {
            self.observed_state: state,
            self.exploration_noise: self.noise.get(),
        })
        self.noise.reset()
        self.sess.run(self.update_global_step)
        return self._denormalize_action(self.prev_action)

    def step(self, reward: Real, denormalized_state: State) -> Action:
        state = self._normalize_state(denormalized_state)
        self.replay_buffer.add(self.prev_state, self.prev_action, np.asarray(reward), state)
        self.prev_action = self._update_models_and_select_action(
            self.prev_state, self.prev_action, reward, state
        )
        self.prev_state = state
        return self._scale_action(self.prev_action)

    def end_episode(self, reward: Real, state: State) -> None:
        self.replay_buffer.add(self.prev_state, self.prev_action, np.asarray(reward), state)
        self._update_models_and_select_action(
            self.prev_state, self.prev_action, reward, state
        )
        self.prev_state = state
        self.prev_action = None

    def _update_models_and_select_action(self,
                                         prev_state: State,
                                         prev_action: Action,
                                         reward: Real,
                                         state: State) -> Action:
        states, actions, rewards, next_states = self.replay_buffer.select_minibatch()
        step, g, summary, action, *_ = self.sess.run(
            [self.global_step, self.grads_norm, self.summaries, self.taken_action, self.critic_train_op,
             self.actor_train_op, self.update_targets],
            {
                self.state: states,
                self.action: actions,
                self.reward: rewards,
                self.next_state: next_states,
                self.observed_state: state,
                self.exploration_noise: self.noise.get(),
                self.observed_reward: reward,
                self.observed_prev_state: prev_state,
                self.observed_prev_action: prev_action,
            }
        )
        print(g)
        self.summary_writer.add_summary(summary, step)
        return action

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        return self.action_min + (self.action_max - self.action_min) * (action+1)/2

    def _normalize_state(self, state: State) -> State:
        return (((state - self.state_min) / (self.state_max - self.state_min)) - 0.5) * 2
