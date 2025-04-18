"""
Process resampled experimental images using Richardson-Lucy algorithm.
"""

import numpy as np
from rl_module import Gauss_kernel, RL_iteration_for_concurrent
from concurrent.futures import ProcessPoolExecutor
import cv2
import h5py



INPUTS = [f'mc_img{i}.h5' for i in (1, 2, 3)]
OUTPUTS = [f'mc_img{i}_rl.h5' for i in (1, 2, 3)]
M = 17 #~ 3.*4*sqrt(2)
KERNEL_X = Gauss_kernel(M)
ITERS = 100e3
CPU_units_to_use = 20 #rewrite according to your needs

def norm(img):
    im = img - np.min(img)
    return im/np.sum(im)

def rl_func(img_pair):    
    """
    Applies the Richardson-Lucy (RL) deconvolution algorithm to an up-sized image.

    Args:
        img_pair (tuple): A tuple containing:
            - index (int): The index of the image.
            - img (numpy.ndarray): The image to process.

    Returns:
        numpy.ndarray: The deconvolved image after applying the RL algorithm.

    Notes:
        - The function prints a message indicating the index of the image being processed.
        - The image is resized to 200x200 using cubic interpolation before processing.
        - The RL algorithm is applied with a maximum number of iterations (`max_iter_`) 
          and a minimum change threshold (`min_change_`).
    """
    index, img = img_pair
    print(f"calling RL @img{index}...")
    return RL_iteration_for_concurrent(
        KERNEL_X, 
        norm(cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)),
        max_iter_=ITERS, min_change_= 1e-10)    



if __name__ == '__main__':
    for file_in, file_out in zip(INPUTS, OUTPUTS):
        with h5py.File(file_in, 'r') as h5fi, h5py.File(file_out, 'w') as h5fo:
            for key in list(h5fi.keys()):
                print(key, "...")
                imgs = ((i, img) for i, img in enumerate(h5fi[key]))
                with ProcessPoolExecutor(CPU_units_to_use) as pool:
                    timgs = np.array(list(pool.map(rl_func, imgs)))            
                h5fo.create_dataset(key, data = timgs)
        print("Done.")
