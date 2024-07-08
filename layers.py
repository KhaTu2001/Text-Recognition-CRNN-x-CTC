import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution2D, MaxPooling2D, BatchNormalization, Dense, Multiply,
    Activation, LeakyReLU, Reshape, Permute, Lambda, RepeatVector
)


def custom_cnn(config, image_input, alpha=0):
    for idx, (block_name, block_config) in enumerate(config.items()):
        num_conv, filters, pool_size = block_config.values()
        for conv_idx in range(1, num_conv + 1):
            x = Convolution2D(
                filters = filters,
                kernel_size = (3, 3) if pool_size else (2, 2),
                padding = 'same' if pool_size else 'valid',
                kernel_initializer = 'he_uniform',
                name = f'{block_name}_conv{conv_idx}'
            )(image_input if idx + conv_idx == 1 else x)

            x = BatchNormalization(name=f'{block_name}_bn{conv_idx}')(x)
            if alpha > 0: x = LeakyReLU(alpha, name=f'{block_name}_act{conv_idx}')(x)
            else: x = Activation('relu', name=f'{block_name}_relu{conv_idx}')(x)

        if pool_size is not None: 
            x = MaxPooling2D(pool_size, name=f'{block_name}_pool')(x)
    return x


def reshape_features(last_cnn_layer, dim_to_keep=-1, name='cnn_features'):
    _, height, width, channel = last_cnn_layer.get_shape()
    if dim_to_keep == 1:
        target_shape = (height, width * channel)
    elif dim_to_keep == 2:
        target_shape = (width, height * channel)
    elif dim_to_keep == 3 or dim_to_keep == -1:
        target_shape = (height * width, channel)
    else:
        raise ValueError('Invalid dim_to_keep value')
    return Reshape(target_shape=(target_shape), name=name)(last_cnn_layer)
