#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:05:24 2023

@author: khasenstab
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from settings import *


# "hard sigmoid", useful for binary accuracy calculation from logits
def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


# augments images with a probability that is dynamically updated during training
class AdaptiveAugmenter(keras.Model):
    def __init__(self):
        super().__init__()

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0)

        # the corresponding augmentation names from the paper are shown above each layer
        # the authors show (see figure 4), that the blitting and geometric augmentations
        # are the most helpful in the low-data regime
        self.augmenter = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS*2)),
                layers.RandomTranslation(height_factor=MAX_TRANSLATION,
                                         width_factor=MAX_TRANSLATION, 
                                         interpolation="nearest",
                                         fill_mode = 'constant',
                                         fill_value = -1.),
                layers.RandomRotation(factor=MAX_ROTATION,
                                      interpolation="nearest",
                                      fill_mode = 'constant',
                                      fill_value = -1.),
                layers.RandomZoom(height_factor=MAX_ZOOM,
                                  interpolation="nearest",
                                  fill_mode = 'constant',
                                  fill_value = -1.),
                
                layers.RandomBrightness(factor=(-MAX_BRIGHTNESS, MAX_BRIGHTNESS), 
                                        value_range=(-1, 1)),
                
                #tf.keras.layers.Lambda(lambda x: (x + 1.)/2.),
                #layers.RandomContrast(factor=MAX_CONTRAST),                
                tf.keras.layers.Lambda(lambda x: (((x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) ) * 2) - 1 ),
                

            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        if training:
            augmented_images = self.augmenter(images, training)

            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = tf.random.uniform(
                shape=(BATCH_SIZE * NB_GPUS, 1, 1, 1), minval=0.0, maxval=1.0
            )
            if AUGMENTATION_PROB:
                augmentation_bools = tf.math.less(augmentation_values, AUGMENTATION_PROB)
            else:
                augmentation_bools = tf.math.less(augmentation_values, self.probability)
            
            images = tf.where(augmentation_bools[3], augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - TARGET_ACCURACY
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / INTEGRATION_STEPS, 0.0, 1.0
            )
        )
