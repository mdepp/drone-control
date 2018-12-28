from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Reshape, add, subtract, multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from tensorflow import Tensor
from tensorflow import keras
import tensorflow as tf


def make_critic(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Model:
    """
    Creates the critic model, which is a keras Model with two inputs (called 'state' and 'action') and one output
    called 'value'.

    Args:
        state_shape: shape of state input
        action_shape: shape of action input

    Returns:
        The critic model
    """
    with tf.variable_scope('critic-model'):
        regularizer = l2(1e-2)  # kernal regularizer?
        final_initializer = RandomUniform(-3e-3, 3e-3)
        initializer = VarianceScaling(scale=1/3, mode='fan_in')

        state_input: Tensor = Input(shape=state_shape, name='state')
        action_input: Tensor = Input(shape=action_shape, name='action')
        temp = Flatten()(state_input)
        temp = BatchNormalization()(temp)
        temp = Dense(400, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                     bias_initializer=initializer)(temp)
        temp = BatchNormalization()(temp)
        temp = Concatenate()([temp, Flatten()(action_input)])
        temp = Dense(300, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                     bias_initializer=initializer)(temp)
        output = Dense(1, name='value', kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
        model = Model(inputs=[state_input, action_input], outputs=output, name='critic')
        model.compile(optimizer=Adam(1e-3),
                      loss='mean_squared_error')
        return model


def make_actor(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Model:
    with tf.variable_scope('actor-model'):
        regularizer = l2(1e-2)
        final_initializer = RandomUniform(-3e-3, 3e-3)
        initializer = VarianceScaling(scale=1/3, mode='fan_in')

        state_input = Input(shape=state_shape)
        temp = Flatten()(state_input)
        temp = BatchNormalization()(temp)
        temp = Dense(400, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                     bias_initializer=initializer)(temp)
        temp = BatchNormalization()(temp)
        temp = Dense(300, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                     bias_initializer=initializer)(temp)
        temp = BatchNormalization()(temp)
        temp = Dense(sum(action_shape), activation='tanh', kernel_regularizer=regularizer,
                     kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
        output = Reshape(action_shape, name='action')(temp)
        model = Model(inputs=state_input, outputs=output, name='actor')
        model.compile(optimizer=Adam(1e-4),
                      loss='mean_squared_error')
        return model


def main():
    print(keras.__version__)
    # TODO: consider using tf layers instead of keras layers

    state_shape = (2,)
    action_shape = (2,)

    critic = make_critic(state_shape, action_shape)
    keras.utils.plot_model(critic, to_file='graphs/q-model.png', show_shapes=True)

    actor = make_actor(state_shape, action_shape)
    keras.utils.plot_model(actor, to_file='graphs/mu-model.png', show_shapes=True)

    minibatch_size: int = 64
    gamma: float = 0.95

    with tf.variable_scope('sarsa-inputs'):
        state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape), name='state')
        action = tf.placeholder(dtype=tf.float32, shape=(None, *action_shape), name='action')
        reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
        next_state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape), name='next_state')
        next_action = tf.placeholder(dtype=tf.float32, shape=(None, *action_shape), name='next_action')

    state_in = Input(tensor=state, name='state')
    action_in = Input(tensor=action, name='action')
    reward_in = Input(tensor=reward, name='reward')
    next_state_in = Input(tensor=next_state, name='next_state')
    next_action_in = Input(tensor=next_action, name='next_action')

    # Critic update
    with tf.variable_scope('critic-update'):
        y = reward + gamma*critic([next_state, actor(next_state)])
        target = critic([state, action])
        print('y:', y)
        critic_loss = tf.losses.mean_squared_error(labels=y, predictions=target)  # TODO: add batch normalization
        print('critic_loss:', critic_loss)
        critic_optimizer = tf.train.AdamOptimizer(1e-3)
        critic_opt_ops = critic_optimizer.minimize(critic_loss, var_list=critic.trainable_variables)
        print('critic_opt_ops:', critic_opt_ops)

    # Actor update
    with tf.variable_scope('actor-update'):
        # Average action-value estimate under actor behaviour. Gradient of this should be the DDPG policy gradient.
        predicted_action_value = tf.reduce_mean(critic([state, actor(action)]), axis=0)[0]
        actor_optimizer = tf.train.AdamOptimizer(1e-4)
        actor_opt_ops = actor_optimizer.minimize(-predicted_action_value, var_list=actor.trainable_weights)

    tf.summary.scalar('critic_loss', critic_loss)
    tf.summary.scalar('predicted_action_value', predicted_action_value)
    tf.summary.histogram('critic_loss_hist', critic_loss)

    merged_summaries = tf.summary.merge_all()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter('logdir', sess.graph)
    sess.run(tf.global_variables_initializer())

    while True:
        summary, _, _ = sess.run([merged_summaries, critic_opt_ops, actor_opt_ops], {
            state: [[0.5, 0.5], [0.2, 0.2]],
            action: [[0.5, 0.5], [0.2, 0.2]],
            reward: [1.0, 2.0],
            next_state: [[0.5, 0.5], [0.2, 0.2]],
            next_action: [[0.5, 0.5], [0.2, 0.2]],
        })
        summary_writer.add_summary(summary)


if __name__ == '__main__':
    main()
