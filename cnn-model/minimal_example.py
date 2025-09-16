"""
A simple example on how to use the CNN model on stack of images.
"""

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

# One should set the following environment variable if keras 3 is used:
# `export TF_USE_LEGACY_KERAS=1` into console.")
# Set paths to input and output


if __name__ == "__main__":
    # Replace custom metrics, as they are not needed for inference.
    customs = {
        "Custom_mse_conv_func": tf.keras.losses.mse,
        "Custom_mae_conv_func": tf.keras.losses.mae,
    }
    # Load Keras model.
    model = tf.keras.models.load_model("model.h5", custom_objects=customs)
    # Load test images
    with h5py.File("test_in.h5", "r") as h5fi:
        images = np.array(h5fi["dataA"])
    # Transform test images
    transformed = model.predict(images)
    np.save('transformed_example.npy', transformed)
