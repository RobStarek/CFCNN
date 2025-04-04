"""
process_inputs_cnn.py

This script processes input datasets using a pre-trained convolutional neural network (CNN).
The input datasets are read from an HDF5 file, transformed using the CNN, and the results
are saved into a new HDF5 file.

Key Features:
- Loads a pre-trained CNN model.
- Processes datasets in an input HDF5 file.
- Saves the transformed datasets into an output HDF5 file.

Inputs:
- `input_res_for_cnn_psf2.h5`: HDF5 file containing input datasets.

Outputs:
- `cnn_output_res_psf2.h5`: HDF5 file containing transformed datasets.

"""

import os
import tensorflow as tf
import numpy as np
import h5py

# One should set the following environment variable if keras 3 is used:
# `export TF_USE_LEGACY_KERAS=1` into console.

# Set paths to input and output
INPUT = 'input_res_for_cnn_psf2.h5'
OUTPUT = 'cnn_output_res_psf2.h5'

if __name__ == '__main__':
    """
    Main script execution for processing input datasets using a pre-trained CNN model.
    """
    # Define custom objects for the model
    customs = {
        'Custom_mse_conv_func': tf.keras.losses.mse,
        'Custom_mae_conv_func': tf.keras.losses.mae
    }

    # Load the pre-trained CNN model
    model = tf.keras.models.load_model('../cnn-model/model_v3.h5', custom_objects=customs)
    print(model.summary())

    # Iterate over all datasets in the input file and transform them using the CNN
    with h5py.File(INPUT, 'r') as h5fi, h5py.File(OUTPUT, 'w') as h5fo:
        for dsetkey, dset in h5fi.items():
            print(dsetkey, dset.shape, dset.dtype)
            images = np.array(dset)
            transformed = model.predict(images)
            h5fo.create_dataset(dsetkey, data=transformed)

    print('done')
