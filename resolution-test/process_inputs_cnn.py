import os
import tensorflow as tf
import numpy as np
import h5py
#One should set the following environment variable if keras 3 is used:
#`export TF_USE_LEGACY_KERAS=1` into console.")

#Set paths to input and output
INPUT = 'input_res_for_cnn_psf2.h5'
OUTPUT = 'cnn_output_res_psf2.h5'

if __name__ == '__main'__:
  customs = {'Custom_mse_conv_func' : tf.keras.losses.mse, 'Custom_mae_conv_func' : tf.keras.losses.mae}
  model = tf.keras.models.load_model('../cnn-model/model_v3.h5', custom_objects = customs)
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
