#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:38:24 2023

@author: khasenstab
"""

import os
import math
import tensorflow as tf

#%%
# Paths
DATA_DIR    = os.path.join(os.getcwd(), 'data')
LOAD_DIR    = os.path.join(os.getcwd(), 'output')
OUTPUT_DIR  = os.path.join(os.getcwd(), 'output')
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "images")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")


# Dataset Options
IMAGE_SIZE  = 256 # Width and height of the images
NB_CHANNELS = 1   # Number of channels in the images


# Output Options
OUTPUT_SHAPE   = (4, 4) # Shape of the output image (columns, rows)
SAVE_FREQUENCY = 1000   # Save frequency for weights (steps)
PLOT_FREQUENCY = 100    # save frequency for plots (steps)


# Architecture Options
LATENT_DIM      = 512  # Dimension of the latent space
MAPPING_LAYERS  = 8    # Number of layers in the mapping network
MIN_IMAGE_SIZE  = 4    # The smallest size of convolutional layers
ENC_MIN_FILTERS = 32   # The smallest number of filters in the encoder
ENC_MAX_FILTERS = 512  # The largest number of filters in the encoder
GEN_MIN_FILTERS = 64   # The smallest number of filters in the generator
GEN_MAX_FILTERS = 512  # The largest number of filters in the generator
DIS_MIN_FILTERS = 32   # The smallest number of filters in the discriminator
DIS_MAX_FILTERS = 512  # The largest number of filters in the discriminator
KERNEL_SIZE     = 3    # Size of the convolutional kernels
ALPHA           = 0.2  # LeakyReLU slope
GAIN            = 1.2  # Equalized layers gain
NB_BLOCKS  = int(math.log(IMAGE_SIZE, 2)) - int(math.log(MIN_IMAGE_SIZE, 2)) + 1


# Training Hyperparameters
NB_GPUS          = len(tf.config.list_physical_devices('GPU')) # For distributed training
BATCH_SIZE       = 4      # Batch size
NB_EPOCHS        = 10000  # Number of epochs
LEARNING_RATE    = 0.001  # Learning rate
MAPPING_LR_RATIO = 0.01   # Learning rate ratio of the mapping network
BETA_1           = 0.     # Adam beta 1
BETA_2           = 0.99   # Adam beta 2
EPSILON          = 1e-8   # Adam epsilon
MA_BETA          = 0.999  # Weight for moving average of network weights


# Regularization
STYLE_MIX_PROBA           = 0.9  # Probability of mixing styles
GRADIENT_PENALTY_COEF     = 10.  # Gradient penalty coefficient
GRADIENT_PENALTY_INTERVAL = 4    # Interval of gradient penalty
LAMBDA_ADV                = 0.1  # weight of adversarial loss
LAMBDA_VGG                = 0.01 # weight of perceptual loss
LAMBDA_VAR                = 0.   # penalize variance of style vectors
LAMBDA_VAR_COMPARE        = 1.   # similar latent variance - mapping and encoder


# Adaptive Discriminator Augmentation
AUGMENTATION_PROB = 0.2
MAX_TRANSLATION   = 0.05
MAX_ROTATION      = 0.02
MAX_ZOOM          = 0.15
MAX_BRIGHTNESS    = 0.25
TARGET_ACCURACY   = 0.95
INTEGRATION_STEPS = 1000



