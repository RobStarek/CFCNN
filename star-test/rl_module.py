"""
This module provides functions for generating Gaussian kernels and performing 
Richardson-Lucy deconvolution. The main functionalities include:
1. `Gauss_function`: Computes a 1D Gaussian function.
2. `Gauss_kernel`: Generates a normalized 2D Gaussian kernel matrix.
3. `RL_iteration_for_concurrent`: Performs the Richardson-Lucy deconvolution 
    algorithm for concurrent processing.
Functions:
-----------
- Gauss_function(x_, sigma_):
     Computes a 1D Gaussian function using the 1/sigma^2 convention.
- Gauss_kernel(sigma_):
     Generates a normalized 2D Gaussian kernel matrix based on the given 
     standard deviation (sigma). The kernel size is determined using the 
     3-sigma rule, ensuring that the kernel captures most of the Gaussian 
     distribution.
- RL_iteration_for_concurrent(kernel_, input_sample_, max_iter_=1e6, min_change_=1e-10):
     Performs the Richardson-Lucy deconvolution algorithm iteratively to 
     estimate the original image from a blurred input sample using a given 
     kernel. The process stops based on specified stopping criteria, such as 
     the maximum number of iterations or the minimal average change per pixel.
"""

import numpy as np
import scipy

def Gauss_function(x_, sigma_):
    """
    1D Gaussian function. With 1/sigma^2 convention.
    """
    return np.exp(-(x_)**2/(sigma_**2))

def Gauss_kernel(sigma_):
    """
    Generates a normalized 2D Gaussian kernel matrix based on the given standard deviation (sigma).
    The kernel size is determined using the 3-sigma rule, ensuring that the kernel captures
    most of the Gaussian distribution. The kernel is normalized so that the sum of all its
    elements equals 1.
    Args:
        sigma_ (float): The standard deviation of the Gaussian distribution.
    Returns:
        numpy.ndarray: A 2D array representing the normalized Gaussian kernel.
    """
    radius = int(np.ceil(3*sigma_))
    k = int(2*radius+1)
    #------------------------------------------------------------------------
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx = np.sqrt(x**2 + np.transpose(x)**2)                                         #Field for kernel; size based on 3sigma rule
    #------------------------------------------------------------------------
    unnormed_psf_matrix = Gauss_function(xx, sigma_)                                #Gaussian kernel; size based on 3sigma rule
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    
    return normed_psf_matrix

def RL_iteration_for_concurrent(kernel_, input_sample_, max_iter_ = 1e6, min_change_ = 1e-10):    
    """
    Perform the Richardson-Lucy deconvolution algorithm for concurrent processing.
    This function iteratively applies the Richardson-Lucy deconvolution algorithm
    to estimate the original image from a blurred input sample using a given kernel.
    The process stops when either the maximum number of iterations is reached or
    the change per pixel falls below a specified threshold.
    Parameters:
        kernel_ (numpy.ndarray): The convolution kernel (point spread function) used for deconvolution.
        input_sample_ (numpy.ndarray): The input image or sample to be deconvolved.
        max_iter_ (int, optional): The maximum number of iterations to perform. The default is 10^6.
        min_change_ (float, optional): The minimum change per pixel to stop the iteration. Default is 10^(-10).
    Returns:
        numpy.ndarray: The deconvolved image with the same total intensity as the input sample.
    Notes:
        - The stopping criteria are based on the maximum number of iterations and the minimal
          average change per pixel between consecutive iterations.
        - The function ensures numerical stability by handling division by zero in the deconvolution step.
        - The input sample is normalized to have a sum of 1 before processing, and the output
          is scaled back to match the original intensity of the input sample.
    Prints:
        A message indicating the reason for stopping (precision achieved or iteration limit reached),
        the number of iterations performed, and the final average change per pixel.
    """
    #Stopping criteria: Max number of iterations & Minimal difference per pixel
    
    d = input_sample_ / np.sum(input_sample_)
    u_new = d
    u = np.zeros(u_new.shape)
    
    iteration = 0
    reason = 'precision achieved'
    while (np.sum(np.abs(u - u_new)) / (d.shape[0]*d.shape[1])) > min_change_:      
        u = u_new
        convolution = scipy.signal.convolve(u, kernel_, mode="same")
        division = np.divide(d, convolution, out=np.zeros_like(d), where=convolution!=0)        
        u_new = u * scipy.signal.convolve(division, kernel_, mode="same")
        if iteration > max_iter_:
            reason = 'limit reached'
            break
        else:
            iteration += 1
    print('Iteration stop reason:', reason, 'iterations:', iteration, 'change:', (np.sum(np.abs(u - u_new)) / (d.shape[0]*d.shape[1])))    
    return u_new * np.sum(input_sample_)
