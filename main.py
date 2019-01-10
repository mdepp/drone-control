from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Reshape, add, subtract, multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from tensorflow import Tensor
from tensorflow import keras
import tensorflow as tf

from ddpg_agent import DDPGAgent
from gym_environment import GymEnvironment
from glue import Glue
import gym

from functools import reduce
from operator import mul


def make_critic(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Model:
    """
    Creates the critic model, which is a keras Model with two inputs (called 'state' and 'action') and one output
    called 'value'.

    Args:
        state_shape: shape of state input
        action_shape: shape of action input
        scope_name: name of the scope containing the model operations

    Returns:
        The critic model
    """
    final_initializer = RandomUniform(-3e-3, 3e-3)
    initializer = VarianceScaling(scale=1/3, mode='fan_in')

    state_input: Tensor = Input(shape=state_shape, name='state')
    action_input: Tensor = Input(shape=action_shape, name='action')
    temp = Flatten()(state_input)
    temp = BatchNormalization()(temp)
    temp = Concatenate()([temp, Flatten()(action_input)])
    temp = Dense(400, activation='relu', kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    temp = BatchNormalization()(temp)
    temp = Concatenate()([temp, Flatten()(action_input)])
    temp = Dense(300, activation='relu', kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    output = Dense(1, name='value', kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
    model = Model(inputs=[state_input, action_input], outputs=output, name='critic')
    model.compile(optimizer=Adam(1e-3),
                  loss='mean_squared_error')
    return model


def make_actor(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Model:
    final_initializer = RandomUniform(-3e-3, 3e-3)
    initializer = VarianceScaling(scale=1/3, mode='fan_in')

    state_input = Input(shape=state_shape)
    temp = Flatten()(state_input)
    temp = BatchNormalization()(temp)
    temp = Dense(400, activation='relu', kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    temp = BatchNormalization()(temp)
    temp = Dense(300, activation='relu', kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    temp = BatchNormalization()(temp)
    temp = Dense(reduce(mul, action_shape, 1), activation='tanh',
                 kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
    output = Reshape(action_shape, name='action')(temp)
    model = Model(inputs=state_input, outputs=output, name='actor')
    model.compile(optimizer=Adam(1e-4),
                  loss='mean_squared_error')
    return model


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--force-cpu', action='store_true')
    args = parser.parse_args()

    if args.force_cpu:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    else:
        config = None

    sess = tf.Session(config=config)
    global_step = tf.train.create_global_step(sess.graph)
    sess.run(global_step.initializer)

    environment = GymEnvironment(
        gym.make('Pendulum-v0'),
        quiet=args.quiet,
    )
    environment.env.metadata['video.frames_per_second'] = 120
    action_shape = environment.env.action_space.shape
    state_shape = environment.env.observation_space.shape
    agent = DDPGAgent(
        session=sess,
        global_step=global_step,
        state_min=environment.env.observation_space.low,
        state_max=environment.env.observation_space.high,
        action_min=environment.env.action_space.low,
        action_max=environment.env.action_space.high,
        make_critic=lambda: make_critic(state_shape, action_shape),
        make_actor=lambda: make_actor(state_shape, action_shape),
        quiet=args.quiet,
    )
    glue = Glue(agent, environment)

    glue.start_run()
    while True:
        glue.do_episode()


if __name__ == '__main__':
    main()
