#%% Append SEE-GAAN paths
import os
import sys


sys.path.append("./code")
sys.path.append("./code/utils")
sys.path.append("./code/networks")


#%% Load libraries

import numpy as np
import tensorflow as tf


# SEE-GAAN libraries
from settings import *
import utils
from callbacks import *
from seegaan import SEEGAAN



#%% Define and instantiate data generator

def data_generator(data_dir, fnames, batch_size):
  while True:
    samp = np.random.choice(range(len(fnames)), size = batch_size, replace = False)

    fnames_batch = fnames[samp]
    X = list()
    for fname in fnames_batch:
      img = np.load(os.path.join(data_dir, fname))
      img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
      X.append(img)
      
    if IMAGE_SIZE == 256:
        X = np.expand_dims(X, -1)
    X = np.array(X)
    
    yield X
    
# define data generators
train_ids = np.array(os.listdir(os.path.join(DATA_DIR, 'train')))
valid_ids = np.array(os.listdir(os.path.join(DATA_DIR, 'val')))
training_generator = data_generator(os.path.join(DATA_DIR, 'train'), train_ids, BATCH_SIZE)
validation_generator = data_generator(os.path.join(DATA_DIR, 'val'), valid_ids, BATCH_SIZE)



#%% samples to monitor training
samples_z = np.load(os.path.join(LOAD_DIR, "samples/samples_z.npy"))
samples_noise = np.random.normal(size = (13,len(samples_z),256,256,1))
samples_images = np.load(os.path.join(LOAD_DIR, "samples/samples_images.npy"))



#%% instantiate SEE-GAAN
gan = SEEGAAN()
#save_found = gan.load_weights(os.path.join(LOAD_DIR, 'models'))
gan.compile()


#%% train SEE-GAAN
history = gan.fit(
    training_generator,
    validation_data = validation_generator,
    batch_size = BATCH_SIZE,
    epochs = 1000,
    steps_per_epoch = 1000,
    validation_batch_size = BATCH_SIZE,
    validation_steps = 100,
    shuffle = True,
    callbacks = [
        Updates(),
        SaveSamplesMapping(samples_z, samples_noise),
        SaveSamplesEncoder(samples_images),
        SaveModels()
    ]
)


