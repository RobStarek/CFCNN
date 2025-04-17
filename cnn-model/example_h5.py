"""
A simple example on how to use the CNN model to tranform all images in the h5 dataset.
"""

import tensorflow as tf
import numpy as np
import h5py
#One should set the following environment variable if keras 3 is used:
#`export TF_USE_LEGACY_KERAS=1` into console.")

#Set paths to input and output
INPUT = 'test_in.h5'
OUTPUT = 'test_out.h5'

if __name__ == '__main__':
  #Replace custom metrics with default one. For running the code, training helper function
  #are irrelevant anyway.
  customs = {'Custom_mse_conv_func' : tf.keras.losses.mse, 'Custom_mae_conv_func' : tf.keras.losses.mae}
  #Load Keras model.
  model = tf.keras.models.load_model('model.h5', custom_objects = customs)
  print(model.summary())
  #Iterate over all datasets in input file and transform them
  #using the neural network.
  with h5py.File(INPUT, 'r') as h5fi, h5py.File(OUTPUT, 'w') as h5fo:
      for dsetkey in h5fi.keys():
          print(dsetkey, h5fi[dsetkey].shape, h5fi[dsetkey].dtype)
          images = np.array(h5fi[dsetkey])
          transformed = model.predict(images)
          h5fo.create_dataset(dsetkey, data = transformed)
  print('done')
