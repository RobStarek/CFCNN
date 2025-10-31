# Calibration-free single-frame super-resolution fluorescence microscopy

This repository contains data and scripts required for reproducing the results presented in the paper **Calibration-free single-frame super-resolution fluorescence microscopy** by Anežka Dostálová, Dominik Vašinka, Robert Stárek, and Miroslav Ježek.

The paper is available on:
* arXiv: [https://arxiv.org/abs/2505.13293](https://arxiv.org/abs/2505.13293).
* bioRxiv: [https://www.biorxiv.org/content/10.1101/2025.05.20.655080v1](https://www.biorxiv.org/content/10.1101/2025.05.20.655080v1)

[![DOI](https://zenodo.org/badge/960423522.svg)](https://doi.org/10.5281/zenodo.15470389)

## Python environment
We have used two Python environments. For all scripts, except the DECODE-related one we used the python 3.11. environemnet specified by `requirements-general.txt`. For DECODE-related scripts we used python 3.10. and the packages specified in `decode-model/requirements.txt`. 

# Getting Started

This guide will help you set up your environment and run your first example using the CFCNN model.

## 1. Python Environment Setup

- For most scripts, use Python 3.11 and install dependencies from `requirements-general.txt`:

    ```sh
    pip install -r requirements-general.txt
    ```

- For DECODE-related scripts, use Python 3.10 and see the file `decode-model/requirements.txt` which specifies our installation. Refer to the [original DECODE documentation](https://github.com/TuragaLab/DECODE) for more information on how install DECODE.

## 2. Running the CFCNN Example

Navigate to the `cnn-model` directory. You can run the provided example notebook or script:

- **Jupyter Notebook:**  
  Open `example_h5.ipynb` in Jupyter and run the cells to see how to load the model and process test data.
  Make sure you have activated legacy support for Keras 2 in your jupyter environment. If this is an issue, plain python in combination with setting environmental variables usually do the trick. To set environement variable in linux console, one doest this:
  ```bash
  export TF_USE_LEGACY_KERAS=1
  ```

- **Python Script:**  
  Run the script directly:

    ```sh
    python cnn-model/example_h5.py
    ```

This will load the pretrained model ([model.h5](cnn-model/model.h5)), process the input data ([test_in.h5](cnn-model/test_in.h5)), and demonstrate basic usage.

## 3. Data and Results

- Input and output data are stored in HDF5 format.
- For more details, see the scripts and notebooks in the [`experimental-data`](experimental-data), [`resolution-test`](resolution-test), and [`star-test`](star-test) folders.

## 4. Additional Information

- For details on the project structure and available scripts, see [README.md](README.md).
- For reproducing figures and tables from the paper, refer to the notebooks in [`experimental-data`](experimental-data) and [`star-test`](star-test).

## 5. Notes
- Due to the file size, some synthetic datasets are not provided. Instead, please use the attached scripts to generate the data and process them to obtain the final data.

---

If you have any questions, please refer to the documentation in each folder or open the relevant notebook/script for more details.

# Directory structure

## cnn-model

This directory contains the developed calibration-free convolutional neural network (**CFCNN**) for super-resolving image reconstruction from a single intensity frame, and a usage example.


## decode-model & reference_decode_training

Here we store DECODE models trained to fit our experimental and synthetic data as a reference to compare against CFCNN. They are used to reconstruct experimental data and the synthetic data. These models originate in training notebooks placed in `reference_decode_training` directory. The environment used for DECODE training and inference is specified by `requirements.txt` file. The training parameters are specified by .yaml files. If you would like to train DECODE yourself, then please update models in `decode-model` directory, as the other script rely on this structure.

## decode-model

Here we store DECODE models trained to fit our experimental and synthetic data as a reference to compare against CFCNN. They are used to reconstruct experimental data and the synthetic data. The environment used for DECODE training and inference is specified by `requirements.txt` file. The training parameters are specified by .yaml files saved in the model directory.

## experimental-data

In this folder, the data and script necessary to recreate Figure 2 and Table I of the paper are provided. **Fig. 2** shows the experimentally acquired fluorescence microscopy images together with their ground truth, and the visual comparison of the reconstructed outputs from the Richardson-Lucy (R-L) deconvolution algorithm, multi-emitter fitting (MEF) using ThunderSTORM, DECODE, and our CFCNN. These are stored in HDF5 format, respectively, with separate files for each experimental image ("Image1.h5", "Image2.h5", "Image3.h5").<br />
**Table I** provides a quantitative comparison of the reconstruction quality in terms of mean absolute error and Kullback-Leibler divergence. These metrics are computed between the output of each reconstruction method and the ground truth for each experimental image.

## resolution-test

This folder contains scripts for the analysis of the resolution achievable by the CFCNN. More detailed comments are included within the scripts.
* `generate_inputs.py` generates synthetic data for resolution testing, including input images with varying signal-to-noise ratios (SNRs) and corresponding reference images, and stores the outputs in HDF5 files.
* `process_cnn_outputs.ipynb` evaluates the resolving ability of the CFCNN model on the generated synthetic data.
* `process_inputs_cnn.py` processes the input datasets by the CFCNN and saves the results in HDF5 files.

## star-test

This folder contains scripts for the recreation of Figure 3 of the paper. Synthetic data are generated and analyzed by our CFCNN, the R-L deconvolution, and MEF using ThunderSTORM for a broader and more systematic evaluation of the performance beyond the presented experimental images. More detailed comments are included within the scripts.
* `generate_figure.ipynb` recreates the Fig. 3.
* `generate_inputs.py` generates synthetic 2D images of a star-shaped pattern with varying SNRs and corresponding ground truth images.
* `process_inputs_cnn.py` processes the input datasets by the CFCNN and saves the results in HDF5 files.
* `rl_module.py` provides functions for generating Gaussian kernels and performing Richardson-Lucy deconvolution.
* `rl_process.py` applies the R-L algorithm.
* `process_decode.ipynb` uses DECODE model to process the test dataset.

## aberration-tolerance

Here we test tolerance of CFCNN model to optical aberrations in the microscope. We parameterize the aberration using Zernike polynomials. We test the resilience agains defocus, spherical aberration, coma, and astigmatism. We also test various choices of numerical aperture.

