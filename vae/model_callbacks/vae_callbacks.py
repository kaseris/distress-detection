from __future__ import absolute_import
import os

import keras
from keras.callbacks import Callback

class VAEProgressMonitor(Callback):
    def __init__(self, train_examples):
        super(VAEProgressMonitor, self).__init__()
        self.train_samples = train_examples

    def on_epoch_end(self, epoch, logs=None):
        pass

