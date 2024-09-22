import os
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from settings import *


# Update variables and apply moving average
class Updates(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def on_batch_begin(self, batch, logs = None):

		self.model.step += 1
		self.model.tf_step.assign(self.model.step)


	def on_batch_end(self, batch, logs = None):

		self.model.moving_average()

# Save samples
class SaveSamplesMapping(Callback):

    def __init__(self, z, noise, **kwargs):

        super().__init__(**kwargs)
        self.z = z
        self.noise = noise
        self.epoch = 0


    def on_batch_end(self, batch, logs = None):

        if self.model.step % PLOT_FREQUENCY == 0 or self.model.step == 1:

            generations = self.model.predict(self.z, list(self.noise))
            
            index = 0
            fig, axs = plt.subplots(OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], figsize = (15,15))
            for rows in range(OUTPUT_SHAPE[0]):
                for cols in range(OUTPUT_SHAPE[1]):
                    axs[rows, cols].imshow(generations[index,:,:,0], cmap = 'gray')
                    axs[rows, cols].axis('off')
                    axs[rows, cols].set_aspect('equal')
                    index += 1
            
            if not os.path.exists(os.path.join(SAMPLES_DIR, 'mapping')):
                os.makedirs(os.path.join(SAMPLES_DIR, 'mapping'))
                
            plt.savefig(os.path.join(SAMPLES_DIR, 'mapping', "image_" + str(self.model.step) + ".png"), bbox_inches = 'tight')
            plt.savefig(os.path.join(OUTPUT_DIR, "last_image.png"), bbox_inches = 'tight')
            
            plt.close()


    def on_epoch_begin(self, epoch, logs = None):

        self.epoch = epoch


# Save samples
class SaveSamplesEncoder(Callback):

    def __init__(self, images, **kwargs):

        super().__init__(**kwargs)
        self.images = images
        self.epoch = 0


    def on_batch_end(self, batch, logs = None):

        if self.model.step % PLOT_FREQUENCY == 0 or self.model.step == 1:

            generations = self.model.predict_recon_ma(self.images)
            
            index = 0
            fig, axs = plt.subplots(self.images.shape[0], 2, figsize = (3,15))
            for rows in range(self.images.shape[0]):
                axs[rows, 0].imshow(self.images[index,:,:,0], cmap = 'gray')
                axs[rows, 0].axis('off')
                axs[rows, 0].set_aspect('equal')
                
                axs[rows, 1].imshow(generations[index,:,:,0], cmap = 'gray')
                axs[rows, 1].axis('off')
                axs[rows, 1].set_aspect('equal')
                index += 1
            
            if not os.path.exists(os.path.join(SAMPLES_DIR, 'encoder_ma')):
                os.makedirs(os.path.join(SAMPLES_DIR, 'encoder_ma'))
                
            plt.savefig(os.path.join(SAMPLES_DIR, 'encoder_ma', "recon_" + str(self.model.step) + ".png"), bbox_inches = 'tight')
            plt.savefig(os.path.join(OUTPUT_DIR, "last_recon_ma.png"), bbox_inches = 'tight')
            
            plt.close()
            


    def on_epoch_begin(self, epoch, logs = None):
        self.epoch = epoch




# Save models
class SaveModels(Callback):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_batch_end(self, batch, logs = None):

        if self.model.step % SAVE_FREQUENCY == 0 or self.model.step == 1:
            self.model.save_weights(MODELS_DIR)
