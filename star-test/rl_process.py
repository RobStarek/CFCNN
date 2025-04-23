import numpy as np
from rl_module import Gauss_kernel, RL_iteration_for_concurrent
from concurrent.futures import ProcessPoolExecutor
import cv2
import h5py


def norm(img):
    im = img - np.min(img)
    return im/np.sum(im)

M = 11 #~ 2.*4*sqrt(2)
KERNEL_X = Gauss_kernel(M)
ITERS = 100e3
CPU_units_to_use = 10 #rewrite according to your needs

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


INPUT = 'input_for_cnn_star.h5'
OUTPUT = 'rl_star.h5'
if __name__ == '__main__':
    with h5py.File(INPUT, 'r') as h5fi, h5py.File(OUTPUT, 'w') as h5fo:
        for key in list(h5fi.keys()):
            print(key, "...")
            imgs = ((i, img) for i, img in enumerate(h5fi[key]))
            with ProcessPoolExecutor(CPU_units_to_use) as pool:
                timgs = np.array(list(pool.map(rl_func, imgs)))            
            h5fo.create_dataset(key, data = timgs)
    print("Done.")
