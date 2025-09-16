import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import scipy.ndimage as ndi
import skimage as ski
from scipy.stats import entropy

import h5py


def generate_storm_image(storm_table, h, w):    
    """Generate it by rounding to nearest neighbour."""
    #"id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","bkgstd [photon]","chi2","uncertainty [nm]"
    img = np.zeros((h,w), dtype=np.float64)
    kappa = 1/60 #nm/px
    #storm_table
    for id, frame, x, y, sigma, intensity, offset, bkgstd, _a, _b in storm_table:
        xnear = int(np.round(x*4*kappa)+1.5)
        ynear = int(np.round(y*4*kappa)+1.5)
        if (xnear >= w) or (xnear < 0) or (ynear >= h) or (ynear < 0):
            continue
        if img[ynear, xnear] != 0:
            print("Ooops!", xnear, ynear, 'is already there')
        img[ynear, xnear] += (intensity + offset)
    return img

new_to_orig_numbering = {1 : 1, 2 : 3, 3 : 5}

if __name__ == '__main__':
    i = 1
    fn = f'resampled/mc_img{i}.h5'
    with h5py.File(fn, 'r') as h5f:
        mc_samples = h5f['resampled'].shape[0]    


    for i in [1,2,3]:
        stormdata = np.zeros((mc_samples,200,200), dtype=np.float32)
        for j in range(mc_samples):
            path = f'resampled/thunderstorm_outputs/dset{i}_{j:03d}.csv'
            storm_table = np.genfromtxt(path, skip_header=1, delimiter=',')
            if storm_table.size == 0:
                print("skipping, keeping zeros")            
                continue
            elif len(storm_table.shape)==1:
                storm_table = storm_table.reshape((1,-1))
            stormdata[j,:,:] = generate_storm_image(storm_table, 200, 200)
        plt.matshow(ndi.gaussian_filter(stormdata[-1],1))
        plt.show()

        with h5py.File(f'resampled/mc_img{i}_mef.h5', 'w') as h5o:
            old_number = new_to_orig_numbering[i]
            h5o.create_dataset(f'mc_tia{old_number}', data = stormdata)
