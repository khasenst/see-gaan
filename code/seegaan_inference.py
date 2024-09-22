#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:17:57 2023

@author: khasenstab
"""



import os
os.chdir('/data-synology/khasenstab/projects/seegaan_packaged/')


#%% Append SEE-GAAN paths
import os
import sys


sys.path.append("./code")
sys.path.append("./code/utils")
sys.path.append("./code/networks")


# Specify GPU Device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


#%% Load libraries

import numpy as np
import tensorflow as tf


# SEE-GAAN libraries
from settings import *
import utils
from callbacks import *
from seegaan import SEEGAAN


#%% instantiate SEE-GAAN and load weights
gan = SEEGAAN()
gan.ma_mapping.load_weights(os.path.join(LOAD_DIR, 'models/model_1000000/ma_mapping.h5'))
gan.ma_encoder.load_weights(os.path.join(LOAD_DIR, 'models/model_1000000/ma_encoder.h5'))
gan.ma_generator.load_weights(os.path.join(LOAD_DIR, 'models/model_1000000/ma_generator.h5'))


#%%
# create constant and noise generator inputs
const = [tf.ones((1, 1))]
noise = [np.random.normal(size = (1,256,256,1))] * 13


#%% generate random images
latents = gan.ma_mapping.predict(np.clip(np.random.normal(size = (1, 512)), -1.96, 1.96), verbose = False)
images  = gan.ma_generator.predict(const + [latents] * 7 + noise, verbose = False)
plt.imshow(images[0], cmap = 'gray'); plt.axis('off')


#%% encode and decode real images

# load example image
img = np.load(os.path.join(LOAD_DIR, "samples/sample_xray.npy"))[np.newaxis,:,:,np.newaxis]
# map to -1, 1
img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1

# calculate latent representation of real image
latent = gan.ma_encoder.predict(img, verbose = False)
# reconstruct image from latent representation
recon  = gan.ma_generator.predict(const + [latent] * 7 + noise, verbose = False)

# plot image and reconstruction
f,(ax1,ax2) = plt.subplots(1, 2, figsize = (15, 15))
ax1.imshow(img[0], cmap = 'gray')
ax2.imshow(recon[0], cmap = 'gray')



#%% latent space manipulation

# calculate latent representation of real image
hf_latents = np.load(os.path.join(LOAD_DIR, "samples/HeartFailure_average_latents.npy"))
w_lambda   = latent + (hf_latents - hf_latents[0:1])

# SEE-GAAN synthetic sequence
const    = [tf.ones((5, 1))]
noise    = [np.random.normal(size = (5,256,256,1))] * 13
sequence = gan.ma_generator.predict(const + [w_lambda] * 7 + noise, verbose = False)

# SEE-GAAN subtractions
sequence_diff = sequence - sequence[0:1]

# plot see-gaan sequences
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, img in zip(axes.ravel()[0:5], sequence):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
for ax, img in zip(axes.ravel()[5:], sequence_diff):
    ax.imshow(img, cmap='Spectral_r', vmin = -0.5, vmax = 0.5)
    ax.axis('off')
plt.tight_layout()
plt.show()



