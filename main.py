from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Reshape, add, subtract, multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from tensorflow import Tensor
from tensorflow import keras
import tensorflow as tf


def make_critic(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...], scope_name: str) -> Model:
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
    with tf.variable_scope(scope_name):
        final_initializer = RandomUniform(-3e-3, 3e-3)
        initializer = VarianceScaling(scale=1/3, mode='fan_in')

        state_input: Tensor = Input(shape=state_shape, name='state')
        action_input: Tensor = Input(shape=action_shape, name='action')
        temp = Flatten()(state_input)
        temp = BatchNormalization()(temp)
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


def make_actor(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...], scope_name: str) -> Model:
    with tf.variable_scope(scope_name):
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
        temp = Dense(sum(action_shape), activation='tanh',
                     kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
        output = Reshape(action_shape, name='action')(temp)
        model = Model(inputs=state_input, outputs=output, name='actor')
        model.compile(optimizer=Adam(1e-4),
                      loss='mean_squared_error')
        return model


def main():
    print(keras.__version__)
    # TODO: consider using tf layers instead of keras layers

    state_shape = (1,)
    action_shape = (1,)

    critic = make_critic(state_shape, action_shape, 'critic-model')
    keras.utils.plot_model(critic, to_file='graphs/q-model.png', show_shapes=True)

    actor = make_actor(state_shape, action_shape, 'actor-model')
    keras.utils.plot_model(actor, to_file='graphs/mu-model.png', show_shapes=True)

    critic_target = make_critic(state_shape, action_shape, 'critic-target-model')
    reset_critic_target = tf.group([
        tf.assign(target_var, var) for target_var, var in
        zip(critic_target.trainable_variables, critic.trainable_variables)
    ])
    actor_target = make_actor(state_shape, action_shape, 'actor-target-model')
    reset_actor_target = tf.group([
        tf.assign(target_var, var) for target_var, var in
        zip(actor_target.trainable_variables, actor.trainable_variables)
    ])

    minibatch_size: int = 64
    gamma: float = 0.95
    tau = 1e-3

    with tf.variable_scope('sarsa-inputs'):
        state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape), name='state')
        action = tf.placeholder(dtype=tf.float32, shape=(None, *action_shape), name='action')
        reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
        next_state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape), name='next_state')
        next_action = tf.placeholder(dtype=tf.float32, shape=(None, *action_shape), name='next_action')

    # Step increment https://stackoverflow.com/a/39671977
    global_step = tf.train.create_global_step()
    increment_global_step_op = tf.assign_add(global_step, 1)

    # Critic update
    with tf.variable_scope('critic-update'):
        y = reward + gamma*critic_target([next_state, actor_target(next_state)])
        target = critic([state, action])
        critic_loss = tf.losses.mean_squared_error(labels=y, predictions=target)
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-2)
        reg_critic_loss = critic_loss + tf.contrib.layers.apply_regularization(regularizer, critic.trainable_weights)
        critic_optimizer = tf.train.AdamOptimizer(1e-3)
        critic_opt_ops = critic_optimizer.minimize(reg_critic_loss, var_list=critic.trainable_variables)

    # Actor update
    with tf.variable_scope('actor-update'):
        # Average action-value estimate under actor behaviour. Gradient of this should be the DDPG policy gradient.
        predicted_action_value = tf.reduce_mean(critic([state, actor(action)]), axis=0)[0]
        grads = tf.gradients(predicted_action_value, actor.trainable_weights)
        grads_and_vars = list(zip((-grad for grad in grads), actor.trainable_variables))
        actor_opt_ops = tf.train.AdamOptimizer(1e-4).apply_gradients(grads_and_vars)
        # actor_opt_ops = tf.group(
        #     [tf.assign_add(weight, 1e-4*grad) for weight, grad in zip(actor.trainable_weights, tf.tuple(grads))]
        # )

    with tf.variable_scope('update-targets'):
        with tf.control_dependencies([critic_opt_ops, actor_opt_ops]):

            def update_target_weights(model, target_model):
                new_target_weights = [tau*weight + (1-tau)*target_weight for weight, target_weight
                                      in zip(model.trainable_variables, target_model.trainable_variables)]
                return tf.group([
                    tf.assign(weight, new_weight) for weight, new_weight
                    in zip(target_model.trainable_variables, tf.tuple(new_target_weights))
                ])

            update_targets = tf.group([
                update_target_weights(model=critic, target_model=critic_target),
                update_target_weights(model=actor, target_model=actor_target),
            ])

    tf.summary.scalar('critic_loss', critic_loss)
    tf.summary.scalar('regularized_critic_loss', reg_critic_loss)
    tf.summary.scalar('predicted_action_value', predicted_action_value)
    tf.summary.histogram('critic_loss_hist', critic_loss)

    merged_summaries = tf.summary.merge_all()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter('logdir', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run([reset_actor_target, reset_critic_target])

    print('Running...')
    while True:
        summary, step, *_ = sess.run([merged_summaries, global_step, increment_global_step_op, actor_opt_ops, critic_opt_ops, update_targets], {
            state: [[1.0], [1.0]],
            action: [[1.0], [2.0]],
            reward: [1.0, 2.0],
            next_state: [[1.0], [1.0]],
            next_action: [[0.0], [0.0]],
        })
        summary_writer.add_summary(summary, step)


if __name__ == '__main__':
    main()
