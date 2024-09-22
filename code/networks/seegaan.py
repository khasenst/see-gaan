#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:29:42 2023

@author: khasenstab
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

from settings import *
import utils
import mapping, generator, discriminator, encoder
from augmenter import AdaptiveAugmenter


class SEEGAAN(Model):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # mapping network
        self.mapping = mapping.build_model()
        self.ma_mapping = clone_model(self.mapping)
        self.ma_mapping.set_weights(self.mapping.get_weights())
        
        # generator
        self.generator = generator.build_model()
        self.ma_generator = clone_model(self.generator)
        self.ma_generator.set_weights(self.generator.get_weights())
        
        # discriminator
        self.discriminator = discriminator.build_model()
        
        # encoder
        self.encoder = encoder.build_model()
        self.ma_encoder = clone_model(self.encoder)
        self.ma_encoder.set_weights(self.encoder.get_weights())
        
        # vgg for perceptual loss
        self.vgg = tf.keras.applications.vgg16.VGG16(include_top = False, weights = "imagenet")
        self.vgg12 = tf.keras.models.Model(self.vgg.input, self.vgg.get_layer('block1_conv2').output)
        self.vgg22 = tf.keras.models.Model(self.vgg.input, self.vgg.get_layer('block2_conv2').output)
        self.vgg33 = tf.keras.models.Model(self.vgg.input, self.vgg.get_layer('block3_conv3').output)
        self.vgg43 = tf.keras.models.Model(self.vgg.input, self.vgg.get_layer('block4_conv3').output)
        self.vgg53 = tf.keras.models.Model(self.vgg.input, self.vgg.get_layer('block5_conv3').output)

        # adaptive data augmenter
        self.augmenter = AdaptiveAugmenter()
        
        
        self.step = 0
        self.tf_step = tf.Variable(self.step, dtype = tf.int32, trainable = False, name = "step")


    # Compile the model
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.mapping_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)
        self.encoder_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)
        self.discriminator_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)


    # Print a summary of the model
    def summary(self, line_length = None, positions = None, print_fn = None):

        print("Mapping: {} -> {} | {:,} parameters".format(self.mapping.input_shape, self.mapping.output_shape, self.mapping.count_params()))
        
        print("Encoder: {} -> {} | {:,} parameters".format(self.encoder.input_shape, self.encoder.output_shape, self.encoder.count_params()))

        print("Generator: [{}, {} * {}, {} * {}] -> {} | {:,} parameters".format(
            self.generator.input_shape[0],
            self.generator.input_shape[1], NB_BLOCKS,
            self.generator.input_shape[-1], (NB_BLOCKS * 2) - 1,
            self.generator.output_shape,
            self.generator.count_params()))

        print("Discriminator: {} -> {} | {:,} parameters".format(self.discriminator.input_shape, self.discriminator.output_shape, self.discriminator.count_params()))


    # Save the model
    def save_weights(self, dir):
        
        if self.step == 1:
            path = os.path.join(dir, "model_" + str(self.step - 1))
        else:
            path = os.path.join(dir, "model_" + str(self.step))

        if not os.path.exists(path):
            os.makedirs(path)

        self.mapping.save_weights(os.path.join(path, "mapping.h5"))
        self.ma_mapping.save_weights(os.path.join(path, "ma_mapping.h5"))
        
        self.generator.save_weights(os.path.join(path, "generator.h5"))
        self.ma_generator.save_weights(os.path.join(path, "ma_generator.h5"))
        
        self.discriminator.save_weights(os.path.join(path, "discriminator.h5"))
        
        self.encoder.save_weights(os.path.join(path, "encoder.h5"))
        self.ma_encoder.save_weights(os.path.join(path, "ma_encoder.h5"))



    # Load the model
    def load_weights(self, dir):

        folder = ""
        i = 0

        while True:

            if os.path.exists(os.path.join(dir, "model_" + str(i))):
                folder = os.path.join(dir, "model_" + str(i))

            else:
                break

            i += SAVE_FREQUENCY

        if folder != "":

            self.mapping.load_weights(os.path.join(folder, "mapping.h5"))
            self.ma_mapping.load_weights(os.path.join(folder, "ma_mapping.h5"))
            
            self.generator.load_weights(os.path.join(folder, "generator.h5"))
            self.ma_generator.load_weights(os.path.join(folder, "ma_generator.h5"))
            
            self.discriminator.load_weights(os.path.join(folder, "discriminator.h5"))
            
            if os.path.exists(os.path.join(folder, "encoder.h5")):
                self.encoder.load_weights(os.path.join(folder, "encoder.h5"))
                self.ma_encoder.load_weights(os.path.join(folder, "ma_encoder.h5"))

            self.step = i - SAVE_FREQUENCY
            self.tf_step.assign(self.step)

        return folder != ""


    # Apply moving average
    def moving_average(self):
            
        for i in range(len(self.mapping.layers)):

            weights = self.mapping.layers[i].get_weights()
            old_weights = self.ma_mapping.layers[i].get_weights()
            new_weights = []

            for j in range(len(weights)):
                new_weights.append(old_weights[j] * MA_BETA + (1. - MA_BETA) * weights[j])

            self.ma_mapping.layers[i].set_weights(new_weights)

        for i in range(len(self.generator.layers)):

            weights = self.generator.layers[i].get_weights()
            old_weights = self.ma_generator.layers[i].get_weights()
            new_weights = []

            for j in range(len(weights)):
                new_weights.append(old_weights[j] * MA_BETA + (1. - MA_BETA) * weights[j])

            self.ma_generator.layers[i].set_weights(new_weights)
            
        
        for i in range(len(self.encoder.layers)):

            weights = self.encoder.layers[i].get_weights()
            old_weights = self.ma_encoder.layers[i].get_weights()
            new_weights = []

            for j in range(len(weights)):
                new_weights.append(old_weights[j] * MA_BETA + (1. - MA_BETA) * weights[j])

            self.ma_encoder.layers[i].set_weights(new_weights)



    ###########################################################################
    """ Generate images for mapping network """
    def generate(self, nb):

        z = np.random.normal(0., 1., (nb, LATENT_DIM))
        noise = np.random.normal(0., 1., ((NB_BLOCKS * 2) - 1, nb, IMAGE_SIZE, IMAGE_SIZE, 1))

        return self.predict(z, noise)
    
    
    # Give the output of the model from the inputs
    def predict(self, z, noise, denorm = True):

        generations = np.zeros((z.shape[0], IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS), dtype = np.uint8)

        for i in range(0, z.shape[0], BATCH_SIZE):

            size = min(BATCH_SIZE, z.shape[0] - i)
            const_input = [tf.ones((size, 1))]
            w = tf.convert_to_tensor(self.ma_mapping(z[i:i + size]))
            n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
            gen = self.ma_generator(const_input + ([w] * NB_BLOCKS) + n)
            if denorm:
                generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())
            else:
                generations[i:i + size, :, :, :] = gen.numpy()

        return generations
    
    
    ###########################################################################
    """ Generate images for encoder network"""
    def predict_recon(self, images, denorm = True):
        
        #noise = np.zeros(((NB_BLOCKS * 2) - 1, images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        noise = np.random.normal(0., 1., ((NB_BLOCKS * 2) - 1, images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        generations = np.zeros((images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS), dtype = np.uint8)

        for i in range(0, images.shape[0], BATCH_SIZE):

            size = min(BATCH_SIZE, images.shape[0] - i)
            const_input = [tf.ones((size, 1))]
            w = tf.convert_to_tensor(self.encoder(images[i:i + size]))
            n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
            gen = self.generator(const_input + ([w] * NB_BLOCKS) + n)
            if denorm:
                generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())
            else:
                generations[i:i + size, :, :, :] = gen.numpy()

        return generations
    
    
    def predict_recon_ma(self, images, denorm = True):
        
        #noise = np.zeros(((NB_BLOCKS * 2) - 1, images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        noise = np.random.normal(0., 1., ((NB_BLOCKS * 2) - 1, images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        generations = np.zeros((images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS), dtype = np.uint8)

        for i in range(0, images.shape[0], BATCH_SIZE):

            size = min(BATCH_SIZE, images.shape[0] - i)
            const_input = [tf.ones((size, 1))]
            w = tf.convert_to_tensor(self.ma_encoder(images[i:i + size]))
            n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
            gen = self.ma_generator(const_input + ([w] * NB_BLOCKS) + n)
            if denorm:
                generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())
            else:
                generations[i:i + size, :, :, :] = gen.numpy()

        return generations
    

    ###########################################################################
    """ Create latent w and noise injections """
    @tf.function
    def get_w(self, batch_size):

        rand = tf.random.uniform(shape = (), minval = 0., maxval = 1., dtype = tf.float32)

        # Style mixing
        if rand < STYLE_MIX_PROBA:

            cross_over_point = tf.random.uniform(shape = (), minval = 1, maxval = NB_BLOCKS, dtype = tf.int32)

            z1 = tf.random.normal(shape = (batch_size, LATENT_DIM))
            z2 = tf.random.normal(shape = (batch_size, LATENT_DIM))

            w1 = self.mapping(z1, training = True)
            w2 = self.mapping(z2, training = True)
            w = []

            for i in range(NB_BLOCKS):

                if i < cross_over_point:
                    w_i = w1

                else:
                    w_i = w2

                w.append(w_i)

            return w

        # No style mixing
        else:

            z = tf.random.normal(shape = (batch_size, LATENT_DIM))
            w = self.mapping(z, training = True)

            return [w] * NB_BLOCKS


    # Give noise for training
    def get_noise(self, batch_size, zeros = False):
        if zeros:
            return [tf.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]
        else:
            return [tf.random.normal((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]
    


    ###########################################################################
    """ Loss functions """
    
    # Generator adversarial loss
    def generator_adv_loss(self, fake_output):
        return tf.reduce_mean(tf.nn.softplus(-fake_output))
    
    # L2 reconstruction loss
    def l2_recon_loss(self, real_images, fake_images):
        return tf.reduce_mean(tf.square(real_images - fake_images))
    
    # vgg perceptual loss
    def vgg_loss(self, real, fake, resize = (256,256)):
        
        # scale for VGG
        real = real * 127.5
        fake = fake * 127.5
        
        if resize:
            real = tf.image.resize(real, resize)
            fake = tf.image.resize(fake, resize)
        
        # repeat for 3 channels
        real = tf.repeat(real, 3, -1)
        fake = tf.repeat(fake, 3, -1)
        
        # enforce similar scale of vgg features across layers
        feat12 = self.normalize(self.vgg12(real), self.vgg12(fake))
        feat22 = self.normalize(self.vgg22(real), self.vgg22(fake))
        feat33 = self.normalize(self.vgg33(real), self.vgg33(fake))
        feat43 = self.normalize(self.vgg43(real), self.vgg43(fake))
        feat53 = self.normalize(self.vgg53(real), self.vgg53(fake))
        
        loss_vgg = tf.reduce_mean([self.l2_recon_loss(feat12[0], feat12[1])  + 
                                    self.l2_recon_loss(feat22[0], feat22[1]) +
                                    self.l2_recon_loss(feat33[0], feat33[1]) +
                                    self.l2_recon_loss(feat43[0], feat43[1]) +  
                                    self.l2_recon_loss(feat53[0], feat53[1])])
        
        return loss_vgg

    def normalize(self, real, fake):
        mean = tf.reduce_mean(real)
        sd   = tf.math.reduce_std(real)
        real = (real - mean) / sd
        fake = (fake - mean) / sd
        return real,fake
        
    # discriminator loss
    def discriminator_loss(self, real_output, fake_output_mapping, fake_output_encoder):
        lreal    = tf.reduce_mean(tf.nn.softplus(-real_output))
        lmapping = tf.reduce_mean(tf.nn.softplus(fake_output_mapping))
        lencoder = tf.reduce_mean(tf.nn.softplus(fake_output_encoder))
        return lreal + 0.5*lmapping + 0.5*lencoder


    ###########################################################################
    """ Penalties and Regularization """
    
    # Discriminator gradient penalty
    def gradient_penalty(self, real_output, real_images):
        gradients = tf.gradients(tf.reduce_sum(real_output), [real_images])[0]
        gradient_penalty = tf.reduce_sum(tf.square(gradients), axis = [1, 2, 3])
        return tf.reduce_mean(gradient_penalty) * GRADIENT_PENALTY_COEF * 0.5 * GRADIENT_PENALTY_INTERVAL
    
    
    # penalize style variance
    def style_penalty(self, w, mapping = True):
        w = tf.stack(w, -1)
        penalty = tf.reduce_mean(tf.square(w))    
        var_penalty = penalty * LAMBDA_VAR
        return var_penalty
    
    
    def style_compare(self, w_map, w_enc):
        
        w_map = tf.stack(w_map, -1); w_enc = tf.stack(w_enc, -1)
        
        norm_map = tf.reduce_mean(tf.square(w_map))
        norm_enc = tf.reduce_mean(tf.square(w_enc))
        
        penalty = tf.square(norm_map - norm_enc)
        
        return penalty * LAMBDA_VAR_COMPARE
        
    
    ###########################################################################
    """ Training """
    
    
    def train_step(self, data):
        
        batch_size = tf.shape(data)[0]
        const_input = [tf.ones((batch_size, 1))]
        noise1 = self.get_noise(batch_size)
        noise2 = self.get_noise(batch_size)
        gradient_penalty = 0.
        style_penalty_map = 0.
        style_penalty_enc = 0.
        style_penalty_comp = 0.

        with tf.GradientTape() as map_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
            
            # augment original data
            data = self.augmenter(data, training=True)

            # Generate images from mapping network
            w_map = self.get_w(batch_size)
            g_w_map = self.generator(const_input + w_map + noise1, training = True)
            
            # Encode real images
            w_enc = [self.encoder(data)] * NB_BLOCKS
            g_w_enc = self.generator(const_input + w_enc + noise2, training = True)     
            
            # augment mapping output
            #g_w_map = self.augmenter(g_w_map, training=True)
            
            # Discriminator outputs
            d_real = self.discriminator(data, training = True)
            d_fake_map = self.discriminator(g_w_map, training = True)
            d_fake_enc = self.discriminator(g_w_enc, training = True)
            
            # mapping loss
            advloss_map = self.generator_adv_loss(d_fake_map)
            
            # encoder loss
            reconloss_enc = self.l2_recon_loss(data, g_w_enc)
            reconloss_vgg = self.vgg_loss(data, g_w_enc)
            advloss_enc   = self.generator_adv_loss(d_fake_enc)
            enc_loss      = reconloss_enc + LAMBDA_VGG*reconloss_vgg + LAMBDA_ADV*advloss_enc
            
            # discriminator loss
            disc_loss = self.discriminator_loss(d_real, d_fake_map, d_fake_enc)

            # Compute penalties
            penalties = tf.cond(self.tf_step % GRADIENT_PENALTY_INTERVAL == 0, 
                    lambda: [self.gradient_penalty(d_real, data),
                             self.style_penalty(w_map, mapping = True),
                             self.style_penalty(w_enc, mapping = False),
                             self.style_compare(w_enc, w_map)] ,
                    lambda: [0., 0., 0., 0.])
            gradient_penalty, style_penalty_map, style_penalty_enc, style_penalty_comp = penalties

            # weight updates
            map_weights = (self.mapping.trainable_weights + self.generator.trainable_weights)
            map_grad = map_tape.gradient(advloss_map + style_penalty_map + style_penalty_comp, map_weights)
            
            enc_weights = (self.encoder.trainable_weights + self.generator.trainable_weights)
            enc_grad = enc_tape.gradient(enc_loss + style_penalty_enc + style_penalty_comp, enc_weights)
            
            # Get discriminator gradients
            disc_grad = disc_tape.gradient(disc_loss + gradient_penalty, self.discriminator.trainable_variables)
            
            # apply gradients
            self.mapping_optimizer.apply_gradients(zip(map_grad, map_weights))
            self.encoder_optimizer.apply_gradients(zip(enc_grad, enc_weights))
            self.discriminator_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
            
            # update augmentation probability
            self.augmenter.update(d_real)

            return {
                "enc"            : enc_loss,
                "recon"          : reconloss_enc,
                "vgg"            : LAMBDA_VGG * reconloss_vgg,
                "adv_map"        : advloss_map,
                "adv_enc"        : advloss_enc,
                "disc"           : disc_loss,
                "grad_penalty"   : gradient_penalty,
                "penalty_map"    : style_penalty_map,
                "penalty_enc"    : style_penalty_enc,
                "penalty_comp"   : style_penalty_comp,
                "aug_prob"       : self.augmenter.probability
            }


    ###########################################################################
    """ Validation """
    
    def test_step(self, data):
        
        batch_size = tf.shape(data)[0]
        const_input = [tf.ones((batch_size, 1))]
        noise = self.get_noise(batch_size)
        
        # Encode real images
        w_enc = [self.ma_encoder(data)] * NB_BLOCKS
        g_w_enc = self.ma_generator(const_input + w_enc + noise, training = False)     
        
        # encoder loss
        reconloss_enc = self.l2_recon_loss(data, g_w_enc)
        reconloss_vgg = self.vgg_loss(data, g_w_enc)
    
        return {
            "recon_enc"      : reconloss_enc,
            "vgg_enc"        : LAMBDA_VGG * reconloss_vgg,
        }

    
    

