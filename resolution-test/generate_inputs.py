# -*- coding: utf-8 -*-
"""
generate_inputs.py

This script generates synthetic data for resolution testing. The generated data includes
input images with varying signal-to-noise ratios (SNRs) and corresponding reference images.
The outputs are stored in HDF5 files and can be processed further by `process_inputs.py`.

Key Features:
- Simulates images with Gaussian point spread functions (PSFs).
- Generates ground truth reference images with upscaled resolution.
- Supports Monte Carlo sampling for SNR variations.

Outputs:
- `input_res_for_cnn_psf2.h5`: Contains input images for different SNRs.
- `synth_res_reference_psf2.h5`: Contains reference images for ground truth.

"""

import numpy as np
import h5py

# Parameters of simulation
# Random number generator with fixed seed
RNG = np.random.default_rng(seed=20250113)
MONTE_CARLO_SAMPLES = 10  # Number of Monte Carlo samples per SNR
#MONTE_CARLO_SAMPLES = 100 #for final test, do this at your own risk, generated input have 3 GB
#and the corresponding CFCNN output file around 45 GB.

PSFW = 2  # Width of the PSF (Gaussian standard deviation)
RAYLEIGH = 3 * PSFW  # Approximate Rayleigh limit distance
WIDTH = 50  # Image width
HEIGHT = 50  # Image height
UPSCALE = 4  # Upscaling factor for rendering reference images
snrs = np.logspace(0, 3, 37)  # Range of tested SNRs
distances = np.linspace(0, 1, 81) * RAYLEIGH  # Distances for emitter positions
xys = np.array([((25 - d / 2, 25), (25 + d / 2, 25)) for d in distances])
snrs = list(snrs) + [-1]  # -1 is a special SNR indicating noiseless data


def gauss_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Generate a 2D Gaussian function.

    Parameters:
        xy (tuple): Meshgrid of x and y coordinates.
        amplitude (float): Amplitude of the Gaussian.
        xo (float): X-coordinate of the Gaussian center.
        yo (float): Y-coordinate of the Gaussian center.
        sigma_x (float): Standard deviation in the x-direction.
        sigma_y (float): Standard deviation in the y-direction.
        theta (float): Rotation angle of the Gaussian.
        offset (float): Offset value for the Gaussian.

    Returns:
        np.ndarray: Flattened 2D Gaussian values.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + \
        (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + \
        (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + \
        (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo)**2)))
    return g.ravel()


def generate_image(xyi, noise_mean, noise_std, psf_width, h, w):
    """
    Generate a synthetic image with Gaussian PSFs and noise.

    Parameters:
        xyi (np.ndarray): Array of emitter positions and intensities.
        noise_mean (float): Mean of the background noise.
        noise_std (float): Standard deviation of the background noise.
        psf_width (float): Width of the Gaussian PSF.
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        np.ndarray: Normalized synthetic image.
    """
    img = RNG.normal(noise_mean, noise_std, h * w).reshape(h, w)
    ys = np.arange(h)
    xs = np.arange(w)
    xy = np.meshgrid(xs, ys)
    for x, y, intensity in xyi:
        img += gauss_2d(xy, intensity, x, y, psf_width,
                        psf_width, 0, 0).reshape((h, w))
    img = img - np.min(img)
    img = img / np.sum(img)
    return img


def generate_gt_image(xyi, h, w):
    """
    Generate a ground truth reference image.

    Parameters:
        xyi (np.ndarray): Array of emitter positions and intensities.
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        np.ndarray: Normalized ground truth image.
    """
    img = np.zeros((h, w))
    for x, y, intensity in xyi:
        x_nearest = int(np.round(x))
        y_nearest = int(np.round(y))
        if (x_nearest <= w - 1) and (y_nearest <= h - 1):
            img[y_nearest, x_nearest] += intensity
    return img / np.sum(img)


if __name__ == '__main__':
    """
    Main script execution for generating synthetic data and saving it to HDF5 files.
    """
    snrs_mc = np.kron(snrs, np.ones(MONTE_CARLO_SAMPLES))
    datasets_inp = dict()
    datasets_ref = dict()

    for counter, snr in enumerate(snrs_mc):
        key = f'snr_{int(round(snr)):d}_{counter}' if snr > 0 else 'noiseless'
        print(key, snr)
        # Background level offset is 1 for all standard SNRs
        # For noiseless data, background level is 0, SNR is set to 1, and no noise is applied.
        _bckgstd = 1 if snr > 0 else 0
        _snr = snr if snr > 0 else 1
        images_in = []
        for xy in xys:
            xyi = np.vstack([xy[:, 0], xy[:, 1], np.ones(2) * _snr]).T
            images_in.append(generate_image(
                xyi, 3, _bckgstd, PSFW, WIDTH, HEIGHT))
        datasets_inp[key] = np.array(images_in)

    key = 'reference'
    images_ref = []
    lateral_offset = (UPSCALE - 1) / 2
    for xy in xys:
        xyi = np.vstack([UPSCALE * xy[:, 0] + lateral_offset,
                        UPSCALE * xy[:, 1] + lateral_offset, np.ones(2) * 1]).T
        images_ref.append(generate_gt_image(xyi, 200, 200))

    datasets_ref[key] = np.array(images_ref)

    print("Writing inputs...")
    with h5py.File('input_res_for_cnn_psf2hr.h5', 'w') as h5f:
        for key, data_item in datasets_inp.items():
            print(key)
            h5f.create_dataset(key, data=data_item.astype(
                np.float32), dtype=np.float32)

    print("Writing references...")
    with h5py.File('synth_res_reference_psf2hr.h5', 'w') as h5f:
        for key, data_item in datasets_ref.items():
            print(key)
            h5f.create_dataset(key, data=data_item.astype(
                np.float32), dtype=np.float32)

    print("Done")
