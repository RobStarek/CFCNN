# Calibration-free single-frame super-resolving fluorescence microscopy**

This repository contains data and scripts required for reproducing the results presented in the paper **Calibration-free single-frame super-resolving fluorescence microscopy** by Anežka Dostálová, Dominik Vašinka, Robert Stárek, and Miroslav Ježek.
The paper is available on arXiv.

**cnn-model**
This folder contains the developed calibration-free convolutional neural network (CFCNN) for super-resolving image reconstruction from a single intensity frame, *and a usage example (?)*.

**experimental-data**
In this folder, the data and script necessary to recreate Figure 2 and Table I of the paper are provided. Fig. 2 shows the experimentally acquired fluorescence microscopy images together with their ground truth, and the visual comparison of the reconstruted outputs from Richardson-Lucy (R-L) deconvolution algorithm, multi-emitter fitting (MEF) using ThunderSTORM, and our CFCNN. These are stored in HDF5 format, respectively, with separate files for each experimental image ("Image1.h5", "Image2.h5", "Image3.h5").
Table I provides a quantitative comparison of the reconstruction quality in terms of mean absolute error and Kullback-Leibler divergence. These metrics are computed between the output of each reconstruction method and the ground truth for each experimental image.

**resolution-test**
This folder contains scripts for the analysis of the resolution achievable by the CFCNN. More detailed comments are included within the scripts.
"generate_inputs.py" generates synthetic data for resolution testing, including input images with varying signal-to-noise ratios (SNRs) and corresponding reference images, and stores the outputs in HDF5 files.
"process_cnn_outputs.ipynb" evaluates the resolving ability of the CFCNN model on the generated synthetic data.
"process_inputs_cnn.py" processes the input datasets by the CFCNN and saves the results in HDF5 files.

**star-test**
This folder contains scripts for the recreation of Figure 3 of the paper. Synthetic data are generated and analyzed by our CFCNN, the R-L deconvolution, and MEF using ThunderSTORM for a broader and more systematic evaluation of the performance beyond the presented experimental images. More detailed comments are included within the scripts.
"generate_figure.ipynb" recreates the Fig. 3.
"generate_inputs.py" generates synthetic 2D images of a star-shaped pattern with varying SNRs and corresponding ground truth images.
"process_inputs_cnn.py" processes the input datasets by the CFCNN and saves the results in HDF5 files.
"rl_module.py" provides functions for generating Gaussian kernels and performing Richardson-Lucy deconvolution.
"rl_process.py" applies the R-L algorithm.