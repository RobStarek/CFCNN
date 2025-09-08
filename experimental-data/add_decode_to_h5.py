import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import h5py
from scipy.stats import entropy


h5_files = {
    1: 'Image1.h5',
    2: 'Image2.h5',
    3: 'Image3.h5'
}

if __name__ == '__main__':
    for n, h5f in h5_files.items():
        with h5py.File("decode_output.h5", 'r') as decodeh5, h5py.File(h5f, 'a') as h5o:
            key = f'img{n}_decode'
            print(key)
            imgs = np.array(decodeh5[key])
            if key in list(h5o.keys()):
                del h5o[key]
            h5o.create_dataset(key, data = imgs)
        