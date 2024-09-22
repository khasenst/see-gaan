import math
from tensorflow.keras import Model
from tensorflow.keras.layers import *

from settings import *
from layers import *


def normalize_noise(noise):
    mean = tf.reduce_mean(noise, axis = (1,2,3))[:,tf.newaxis,tf.newaxis,tf.newaxis]
    sd   = tf.math.reduce_std(noise, axis = (1,2,3))[:,tf.newaxis,tf.newaxis,tf.newaxis]
    return (noise - mean) / sd

# Pixel space to feature space
def from_rgb(input, filters):

    model = EqualizedConv2D(filters, 1)(input)
    model = LeakyReLU(ALPHA)(model)

    return model


# Downsample
def downsample(input):
    return AveragePooling2D()(input)


# Build a block
def build_block(input, filters):

    residual = EqualizedConv2D(filters, 1, use_bias = False)(input)
    residual = downsample(residual)

    model = EqualizedConv2D(filters, KERNEL_SIZE)(input)
    model = LeakyReLU(ALPHA)(model)

    model = EqualizedConv2D(filters, KERNEL_SIZE)(model)
    model = LeakyReLU(ALPHA)(model)

    model = downsample(model)
    model = Add()([model, residual])
    model = Lambda(lambda x: x / math.sqrt(2.))(model)

    return model


def build_noise(input):
    noise = EqualizedDense(NOISE_REDUCE_DIM)(input)
    noise = LeakyReLU(ALPHA)(noise)
        
    noise = EqualizedDense(IMAGE_SIZE * IMAGE_SIZE * NB_CHANNELS)(noise)
    noise = Reshape((IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS))(noise)
    noise = Lambda(normalize_noise)(noise)
    
    return noise


# Build the encoder
def build_model():

    filters = get_filters(ENC_MIN_FILTERS, ENC_MAX_FILTERS, False)

    model_input = Input(shape = (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS))
    model = from_rgb(model_input, filters[0])

    for i in range(NB_BLOCKS - 1):
        model = build_block(model, filters[i])

    model = EqualizedConv2D(filters[-1], KERNEL_SIZE)(model)
    model = LeakyReLU(ALPHA)(model)

    model = Flatten()(model)
    model = EqualizedDense(filters[-1])(model)
    model = LeakyReLU(ALPHA)(model)

    model = EqualizedDense(LATENT_DIM)(model)

    return Model(model_input, model)


