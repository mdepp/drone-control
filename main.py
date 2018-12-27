from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from tensorflow import Tensor
from tensorflow import keras


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
    regularizer = l2(1e-2)  # kernal regularizer?
    final_initializer = RandomUniform(-3e-3, 3e-3)
    initializer = VarianceScaling(scale=1/3, mode='fan_in')

    state_input: Tensor = Input(shape=state_shape, name='state')
    action_input: Tensor = Input(shape=action_shape, name='action')
    temp = BatchNormalization()(Flatten()(state_input))
    temp = Dense(400, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    temp = BatchNormalization()(temp)
    temp = Concatenate()([temp, Flatten()(action_input)])
    temp = Dense(300, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initializer,
                 bias_initializer=initializer)(temp)
    output = Dense(1, name='value', kernel_initializer=final_initializer, bias_initializer=final_initializer)(temp)
    model = Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer=Adam(1e-3),
                  loss='mean_squared_error')
    return model


def make_actor(state_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Model:
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
    model = Model(inputs=state_input, outputs=output)
    model.compile(optimizer=Adam(1e-4),
                  loss='mean_squared_error')
    return model


def main():
    print(keras.__version__)
    # TODO: consider using tf layers instead of keras layers

    state_shape = (2, 2)
    action_shape = (3, 3)

    critic = make_critic(state_shape, action_shape)
    keras.utils.plot_model(critic, to_file='graphs/q-model.png', show_shapes=True)

    actor = make_actor(state_shape, action_shape)
    keras.utils.plot_model(actor, to_file='graphs/mu-model.png', show_shapes=True)


if __name__ == '__main__':
    main()
