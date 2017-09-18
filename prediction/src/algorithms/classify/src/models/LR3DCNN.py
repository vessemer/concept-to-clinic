import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Input, LeakyReLU
from keras.layers.merge import add, average
from keras.models import Model


def identity_block(input_tensor, kernel, filters, bn_axis, dropout=None):
    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Conv3D(filters[0], (kernel, kernel, kernel), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Conv3D(filters[1], (kernel, kernel, kernel), padding='same')(x)
    x = add([x, input_tensor])

    return x


def conv_block(input_tensor, kernel, filters, bn_axis, dropout=None,
               stride=(1, 1, 1), mode='same'):
    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    if stride != (1, 1, 1):
        mode = 'valid'
    x = Conv3D(filters[0], (kernel, kernel, kernel), strides=stride, padding=mode)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Conv3D(filters[1], (kernel, kernel, kernel), padding='same')(x)
    shortcut = Conv3D(filters[1], (kernel, kernel, kernel),
                      strides=stride, padding=mode)(input_tensor)
    x = add([x, shortcut])

    return x


def coder(in_tensor, stride, kernel, bn_axis, dropout=.2):
    x = Dropout(dropout)(in_tensor)
    x = Conv3D(32, (kernel[0], kernel[1], kernel[2]), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = identity_block(x, 3, [32, 32], dropout=dropout, bn_axis=bn_axis)
    x = conv_block(x, 3, [64, 64], stride=stride, dropout=dropout, bn_axis=bn_axis)
    x = identity_block(x, 3, [64, 64], dropout=dropout, bn_axis=bn_axis)
    x = conv_block(x, 3, [128, 128], dropout=dropout, bn_axis=bn_axis)
    x = identity_block(x, 3, [128, 128], dropout=dropout, bn_axis=bn_axis)

    return x


def predictor(in_shape=None, strides=None, kernels=None,
              dropout_conv=.2, dropout_dence=.2):
    """

    """
    # Determine proper input shape
    if in_shape is None:
        in_shape = [(24, 42, 42, 1),
                    (42, 24, 42, 1),
                    (42, 42, 24, 1)]
    if strides is None:
        strides = [(1, 2, 2),
                   (2, 1, 2),
                   (2, 2, 1)]
    if kernels is None:
        kernels = [(5, 5, 5),
                   (5, 5, 5),
                   (5, 5, 5)]
    bn_axis = 1
    if K.image_data_format() == 'channels_last':
        bn_axis = 3

    inputs = [Input(shape=inx) for inx in in_shape]
    coders = [coder(in_tensor=tnsr, stride=stride, bn_axis=bn_axis,
                    kernel=kernel, dropout=dropout_conv)
              for tnsr, stride, kernel in zip(inputs, strides, kernels)]
    x = average(coders)

    #   shape:  128, 9, 10, 10
    x = conv_block(x, 3, [128, 128], dropout=dropout_conv, bn_axis=bn_axis)
    x = identity_block(x, 3, [128, 128], dropout=dropout_conv, bn_axis=bn_axis)
    x = conv_block(x, 3, [256, 256], stride=(2, 2, 2), dropout=dropout_conv, bn_axis=bn_axis)
    x = identity_block(x, 3, [256, 256], dropout=dropout_conv, bn_axis=bn_axis)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((2, 2, 2))(x)

    x = Flatten()(x)
    x = Dropout(dropout_dence)(x)
    x = Dense(256)(x)
    x = LeakyReLU(.3)(x)
    x = Dropout(dropout_dence)(x)
    x = Dense(2, activation='softmax', name='is_nodule')(x)

    return Model(inputs, x)

