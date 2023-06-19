from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add

from tensorflow.keras.regularizers import l1_l2
from DepthwiseConv3D import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D


def Xception(input_shape):

	# Determine proper input shape
    vid_input = Input(shape=input_shape)

    # Block 1
    x = Conv3D(8, (3, 3, 3), strides=(2, 2, 2), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3, 3, 3), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv3D(32, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)

    # Block 4 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    # Block 14
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(1024, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='linear')(x)

    inputs = vid_input

    # Create model
    model = Model(inputs, x, name='xception')

    return model

