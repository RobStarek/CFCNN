"""
process_inputs_cnn.py

This script processes input datasets using a pre-trained convolutional neural network (CNN).
The input datasets are read from an HDF5 file, transformed using the CNN, and the results
are saved into a new HDF5 file.

Key Features:
- Loads a pre-trained CNN model.
- Processes datasets in an input HDF5 file.
- Saves the transformed datasets into an output HDF5 file.

Inputs & outputs:
- see constants


Outputs are later processed by another notebook.
"""

import tensorflow as tf
import numpy as np
import h5py

# One should set the following environment variable if keras 3 is used:
# `export TF_USE_LEGACY_KERAS=1` into console.

# Set paths to input and output
INPUTS = ['data/input_for_cnn_star_w_psfs.h5', 'data/single_point_psf_for_cnn.h5']
#'cnn_output_star.h5'
OUTPUTS = ['data/cnn_output_star.h5', 'data/cnn_output_single_psf.h5']

if __name__ == '__main__':
    # Define custom objects for the model
    customs = {
        'Custom_mse_conv_func': tf.keras.losses.mse,
        'Custom_mae_conv_func': tf.keras.losses.mae
    }
    # Load the pre-trained CNN model
    model = tf.keras.models.load_model('../cnn-model/model.h5', custom_objects=customs)
    # model = tf.keras.models.load_model('Mol_Attempt_v3_Best_Gauss_blur_on_fly_v21.h5', custom_objects=customs)    
    print(model.summary())
    # Set paths to input and output
    for IN, OUT in zip(INPUTS, OUTPUTS):
        print(f'{IN}->{OUT}')
        # Iterate over all datasets in the input file and transform them using the CNN if the dataset name starts with A_,
        # which indicates images to be transformed, otherwise just copy the dataset
        with h5py.File(IN, 'r') as h5fi, h5py.File(OUT, 'w') as h5fo:
            for key in h5fi.keys():
                print(f"--- {key} ---")
                if key[0:2] == 'A_':
                    images = np.array(h5fi[key])
                    images = images.reshape((*images.shape, 1))
                    print(images.shape)
                    transformed = model.predict(images)            
                    h5fo.create_dataset(key, data=transformed)
                else:
                    h5fo.create_dataset(key, data=h5fi[key])
    print('done')
