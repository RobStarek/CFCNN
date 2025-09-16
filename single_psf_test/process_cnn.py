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
- `input_for_cnn_fixed_rng.h5`: HDF5 file containing input datasets.

Outputs:
- `cnn_output_star.h5`: HDF5 file containing transformed datasets.

Outputs are later process another notebook.
"""

import tensorflow as tf
import numpy as np
import h5py

# One should set the following environment variable if keras 3 is used:
# `export TF_USE_LEGACY_KERAS=1` into console.

# Set paths to input and output
INPUT = 'single_psf_airy2.npy'
OUTPUT = 'single_psf_airy_cnn2.npy'

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

    # Iterate over all datasets in the input file and transform them using the CNN
    img = np.load(INPUT).reshape((-1,50,50))    
    transformed = model.predict(img)
    np.save(OUTPUT, transformed)
    print('done')
